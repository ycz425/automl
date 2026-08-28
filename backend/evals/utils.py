import os
import json
import dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


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


def semantic_correctness_evaluator(key: str, fields: list[str], model: str = "gemini-3.1-flash-lite", max_retries: int = 3):
    client = genai.Client(api_key=GEMINI_API_KEY)

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

        validation_error = None
        for attempt in range(max_retries + 1):
            attempt_prompt = prompt
            if validation_error:
                attempt_prompt += (
                    f"\n\nYour previous output failed schema validation:\n\n{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema."
                )

            interaction = await traced_interactions_create(
                client,
                model=model,
                input=attempt_prompt,
                generation_config={'thinking_level': 'low', 'temperature': 0},
                response_format={'mime_type': 'application/json', 'schema': JudgmentBatch.model_json_schema()}
            )

            try:
                batch = JudgmentBatch.model_validate_json(interaction.output_text)
                break
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)

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


def semantic_overlap_evaluator(key: str, fields: list[str], model: str = "gemini-3.1-flash-lite", max_retries: int = 3):
    client = genai.Client(api_key=GEMINI_API_KEY)

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

        validation_error = None
        for attempt in range(max_retries + 1):
            attempt_prompt = prompt
            if validation_error:
                attempt_prompt += (
                    f"\n\nYour previous output failed schema validation:\n\n{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema."
                )

            interaction = await traced_interactions_create(
                client,
                model=model,
                input=attempt_prompt,
                generation_config={'thinking_level': 'low', 'temperature': 0},
                response_format={'mime_type': 'application/json', 'schema': MatchBatch.model_json_schema()}
            )

            try:
                batch = MatchBatch.model_validate_json(interaction.output_text)
                break
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)

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
