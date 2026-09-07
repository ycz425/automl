import json
from pydantic import BaseModel
from app.agents.clarification_agent import ClarificationAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetAnalysis
from langsmith import aevaluate
from evals.utils import llm_judge


async def run_agent(inputs: dict):
    clarification_agent = ClarificationAgent()

    if inputs['data_type'] == 'UserRequest':
        data = UserRequest.model_validate(inputs['data'])
    elif inputs['data_type'] == 'DatasetAnalysis':
        data = DatasetAnalysis.model_validate(inputs['data'])
    else:
        raise ValueError

    # Continuation behavior (previous_interaction_id) isn't covered here —
    # that threads through the live Gemini interactions API and isn't
    # meaningfully fabricable from a static eval example. This tests the
    # first-time-asking behavior only.
    question, _ = await clarification_agent.request_clarification(data, inputs['problems'])

    return {'question': question}


class ProblemCoverageJudgment(BaseModel):
    problem: str
    addressed: bool
    reasoning: str


class ClarificationQuestionJudgment(BaseModel):
    problem_coverage: list[ProblemCoverageJudgment]
    introduces_unrelated_asks: bool
    unrelated_asks_reasoning: str
    is_clear_and_concise: bool
    clarity_reasoning: str


async def clarification_quality(inputs: dict, outputs: dict):
    data = inputs['data']
    problems = inputs['problems']
    question = outputs['question']

    prompt = f"""
    Judge the quality of the following clarification question, generated to resolve specific
    detected problems with a piece of structured data.

    Existing data:
    {json.dumps(data, indent=2)}

    Detected problems:
    {json.dumps(problems, indent=2)}

    Generated clarification question:
    {question}

    Instructions:
    - For each detected problem listed above, judge whether the question actually asks about it.
    - Judge whether the question introduces asks unrelated to any of the detected problems.
    - Judge whether the question is clear, concise, and answerable in plain language, not overly
      technical or vague.
    """

    judgment = await llm_judge(prompt, ClarificationQuestionJudgment)

    results = []

    if judgment.problem_coverage:
        score = sum(p.addressed for p in judgment.problem_coverage) / len(judgment.problem_coverage)
        missing = [p.problem for p in judgment.problem_coverage if not p.addressed]
        results.append({
            'key': 'problem_coverage',
            'score': score,
            'comment': f'Not addressed: {missing}' if missing else 'All problems addressed.'
        })

    results.append({
        'key': 'no_unrelated_asks',
        'score': not judgment.introduces_unrelated_asks,
        'comment': judgment.unrelated_asks_reasoning
    })
    results.append({
        'key': 'is_clear_and_concise',
        'score': judgment.is_clear_and_concise,
        'comment': judgment.clarity_reasoning
    })

    return results


# Every field name across both clarifiable schemas (UserRequest, DatasetAnalysis) — the prompt
# explicitly forbids surfacing these (or raw problem wording) to the user, so this is checked
# directly rather than relying on the LLM judge to catch a leak.
INTERNAL_FIELD_NAMES = [
    'task_type', 'target_description', 'group_description', 'include_features',
    'exclude_features', 'constraints', 'preferences', 'primary_metric',
    'secondary_metrics', 'evaluation_method', 'validation_size', 'num_folds',
    'stratify', 'target_column', 'feature_columns', 'excluded_columns',
    'group_column', 'positive_class',
]


def no_field_name_leakage(outputs: dict):
    question = outputs['question']
    leaked = [name for name in INTERNAL_FIELD_NAMES if name in question]

    return {
        'key': 'no_field_name_leakage',
        'score': 0.0 if leaked else 1.0,
        'comment': f'Leaked internal field name(s): {leaked}' if leaked else 'No internal field/schema names leaked into the question.'
    }


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_request_clarification',
        evaluators=[clarification_quality, no_field_name_leakage],
        experiment_prefix='automl_request_clarification_eval'
    ))
