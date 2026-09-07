import os
import sys
import ast
import json
import dotenv
from google import genai
from pydantic import BaseModel
from app.utils.structured_generation import generate_structured

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"

_judge_client = genai.Client(api_key=GEMINI_API_KEY)


async def llm_judge(prompt: str, schema: type[BaseModel], model: str = DEFAULT_JUDGE_MODEL, max_retries: int = 3):
    """Shared retry-until-schema-valid LLM-as-judge call. Every bespoke judge evaluator in this
    suite needs exactly this: send a grading prompt, parse the structured verdict, retry on
    schema-validation failure. Centralized here instead of re-implementing the retry loop per file."""
    return await generate_structured(_judge_client, model, prompt, schema, max_retries=max_retries)


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def field_correctness_evaluator(key: str, fields: list[str]):
    def field_correctness(outputs: dict, reference_outputs: dict):
        actual = outputs[key]
        expected = reference_outputs[key]

        results = [
            {'key': f"{field}_correctness", 'score': actual[field] == expected[field]}
            for field in fields if expected[field] is not None
        ]

        return results

    return field_correctness


def set_overlap_evaluator(key: str, fields: list[str]):
    def set_overlap(outputs: dict, reference_outputs: dict):
        actual = outputs[key]
        expected = reference_outputs[key]

        results = [
            {'key': f'{field}_overlap', 'score': jaccard(actual[field], expected[field])}
            for field in fields
        ]

        return results

    return set_overlap


def hallucination_rate_evaluator(key: str, fields: list[str]):
    def hallucination_rate(outputs: dict, reference_outputs: dict):
        actual = outputs[key]
        expected = reference_outputs[key]

        null_fields = [field for field in fields if expected[field] is None]

        if not null_fields:
            return

        return len([field for field in null_fields if actual[field] is None]) / len(null_fields)

    return hallucination_rate


class FieldJudgment(BaseModel):
    field: str
    is_correct: bool
    reasoning: str


class JudgmentBatch(BaseModel):
    judgments: list[FieldJudgment]


def semantic_correctness_evaluator(key: str, fields: list[str], model: str = DEFAULT_JUDGE_MODEL, max_retries: int = 3):
    async def semantic_correctness(outputs: dict, reference_outputs: dict):
        actual = outputs[key]
        expected = reference_outputs[key]

        pairs = {field: {'actual': actual.get(field), 'expected': expected.get(field)} for field in fields}

        prompt = f"""
        For each field below, judge whether the actual value is an acceptable match for the expected value.

        Fields:
        {json.dumps(pairs, indent=2)}

        Guidance:
        - For free-text fields (such as descriptions), judge semantic equivalence rather than exact wording.
        - For list fields (such as feature lists, constraints, preferences, or metrics), judge whether the actual list captures the same intent as the expected list: no missing requirements, nothing invented, phrasing and ordering may differ.
        - If expected is null/empty, actual should also be null/empty to be correct.
        - If expected has a value, actual must express the same meaning to be correct.

        Return exactly one judgment for each field listed above, using the field name as given.
        """

        batch = await llm_judge(prompt, JudgmentBatch, model=model, max_retries=max_retries)

        return [
            {'key': f'{judgment.field}_correctness', 'score': judgment.is_correct, 'comment': judgment.reasoning}
            for judgment in batch.judgments
        ]

    return semantic_correctness


class ListFieldMatch(BaseModel):
    field: str
    matched_count: int
    reasoning: str


class MatchBatch(BaseModel):
    matches: list[ListFieldMatch]


def semantic_overlap_evaluator(key: str, fields: list[str], model: str = DEFAULT_JUDGE_MODEL, max_retries: int = 3):
    async def semantic_overlap(outputs: dict, reference_outputs: dict):
        actual = outputs[key]
        expected = reference_outputs[key]

        pairs = {field: {'actual': actual.get(field, []), 'expected': expected.get(field, [])} for field in fields}

        prompt = f"""
        For each list field below, count how many items in "expected" have a semantically equivalent
        counterpart in "actual" (phrasing may differ, but they must express the same requirement or intent).
        Do not count an actual item as matching more than one expected item, or vice versa.

        Fields:
        {json.dumps(pairs, indent=2)}

        Return exactly one match count for each field listed above, using the field name as given.
        """

        batch = await llm_judge(prompt, MatchBatch, model=model, max_retries=max_retries)

        results = []
        for match in batch.matches:
            expected_count = len(expected.get(match.field, []))
            actual_count = len(actual.get(match.field, []))

            if expected_count == 0 and actual_count == 0:
                score = 1.0
            else:
                matched = min(match.matched_count, expected_count, actual_count)
                union = expected_count + actual_count - matched
                score = matched / union if union else 1.0

            results.append({'key': f'{match.field}_overlap', 'score': score, 'comment': match.reasoning})

        return results

    return semantic_overlap


_STDLIB_MODULES = set(sys.stdlib_module_names)

# Import name -> substring expected in the declared pip dependency name, for
# packages whose import name doesn't match their pip name.
_IMPORT_TO_PACKAGE_ALIAS = {
    'sklearn': 'scikit-learn',
    'skimage': 'scikit-image',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'yaml': 'pyyaml',
    'bs4': 'beautifulsoup4',
}


def _imported_top_level_modules(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.add(node.module.split('.')[0])

    return modules


def dependency_consistency_evaluator(output_key: str, code_fields: list[str], dependencies_field: str = 'dependencies'):
    """Deterministic, execution-free stand-in for 'does this actually run': every third-party
    module imported by the generated code must be declared in its dependencies list. Shared by
    every eval whose LLM call produces (code, dependencies) pairs — experiment implementation/repair
    and output-script generation/repair — instead of re-deriving this AST walk in each file."""

    def dependency_consistency(outputs: dict):
        payload = outputs[output_key]
        declared = [dep.lower() for dep in payload.get(dependencies_field, [])]

        undeclared = set()
        for field in code_fields:
            for module in _imported_top_level_modules(payload.get(field, '')):
                if module in _STDLIB_MODULES:
                    continue
                expected_name = _IMPORT_TO_PACKAGE_ALIAS.get(module, module).lower()
                if not any(expected_name in dep or dep in expected_name for dep in declared):
                    undeclared.add(module)

        return {
            'key': 'dependency_consistency',
            'score': 0.0 if undeclared else 1.0,
            'comment': f'Imported but not declared as a dependency: {sorted(undeclared)}' if undeclared else 'All third-party imports are declared as dependencies.'
        }

    return dependency_consistency
