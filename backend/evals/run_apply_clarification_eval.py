import json
from pydantic import BaseModel
from app.agents.clarification_agent import ClarificationAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetAnalysis
from langsmith import aevaluate
from evals.utils import llm_judge


async def run_agent(inputs: dict):
    if inputs['data_type'] == 'UserRequest':
        data = UserRequest.model_validate(inputs['data'])
    elif inputs['data_type'] == 'DatasetAnalysis':
        data = DatasetAnalysis.model_validate(inputs['data'])
    else:
        raise ValueError

    clarification_agent = ClarificationAgent()

    try:
        new_data = await clarification_agent.apply_clarification(
            data,
            inputs['question'],
            inputs['problems'],
            inputs['clarification']
        )
    except Exception as e:
        # The agent failing to produce a valid update (bad output, content-filter block, ...) is a
        # failed example, not a missing one — a real pipeline run would fail at this step too. Return
        # it so the evaluators score it 0 instead of the harness silently dropping it from the averages.
        # Billing / quota errors are infrastructure, not agent quality, so those still abort the example.
        if getattr(e, 'status_code', None) in (402, 429):
            raise
        return {'data': None, 'error': f'{type(e).__name__}: {str(e)[:300]}'}

    return {'data': new_data.model_dump()}


# Fields whose value is free-form prose: a correct update can be worded differently from the reference.
FREE_TEXT_FIELDS = {'target_description'}


def _normalize(value):
    """Canonical form for comparison: ignores case/whitespace in strings and ordering in lists
    (feature/column lists are sets in practice), so only real differences count as changes."""
    if isinstance(value, str):
        return ' '.join(value.lower().split())
    if isinstance(value, list):
        return sorted(json.dumps(_normalize(v), sort_keys=True) for v in value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _same(a, b) -> bool:
    return _normalize(a) == _normalize(b)


class SameMeaningJudgment(BaseModel):
    reasoning: str
    same_meaning: bool


async def _same_meaning(field: str, expected, actual) -> bool:
    prompt = f"""
    Two versions of the "{field}" field of an ML task specification are shown below. Judge whether
    they say the same thing: they identify the same thing and carry the same information, even if worded
    differently. They differ if one is more specific, contradicts the other, or refers to something else.

    Reference version:
    {json.dumps(expected)}

    Produced version:
    {json.dumps(actual)}
    """
    return (await llm_judge(prompt, SameMeaningJudgment)).same_meaning


def _failed(key: str, outputs: dict):
    return {'key': key, 'score': 0.0, 'comment': f"Agent failed to produce an update: {outputs.get('error')}"}


def apply_succeeded(outputs: dict):
    failed = outputs.get('data') is None
    return {
        'key': 'apply_succeeded',
        'score': 0.0 if failed else 1.0,
        'comment': outputs.get('error') if failed else None
    }


async def field_resolution(inputs: dict, outputs: dict, reference_outputs: dict):
    """Of the fields the reference changed, the fraction the agent updated to the reference value."""
    if outputs.get('data') is None:
        return _failed('field_resolution', outputs)

    before = inputs['data']
    actual = outputs['data']
    expected = reference_outputs['data']

    relevant_fields = [field for field in before if not _same(before[field], expected[field])]

    # Nothing was supposed to change: there is nothing to resolve (field_preservation covers it).
    if not relevant_fields:
        return None

    unresolved = []
    for field in relevant_fields:
        if _same(actual[field], expected[field]):
            continue
        if field in FREE_TEXT_FIELDS and await _same_meaning(field, expected[field], actual[field]):
            continue
        unresolved.append(field)

    return {
        'key': 'field_resolution',
        'score': 1 - len(unresolved) / len(relevant_fields),
        'comment': f'Not resolved to the reference value: {unresolved}' if unresolved else None
    }


def field_preservation(inputs: dict, outputs: dict, reference_outputs: dict):
    """Of the fields the reference left alone, the fraction the agent also left alone."""
    if outputs.get('data') is None:
        return _failed('field_preservation', outputs)

    before = inputs['data']
    actual = outputs['data']
    expected = reference_outputs['data']

    irrelevant_fields = [field for field in before if _same(before[field], expected[field])]

    if not irrelevant_fields:
        return None

    changed = [field for field in irrelevant_fields if not _same(actual[field], before[field])]

    return {
        'key': 'field_preservation',
        'score': 1 - len(changed) / len(irrelevant_fields),
        'comment': f'Changed although it should not have been: {changed}' if changed else None
    }


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data="automl_apply_clarification",
        evaluators=[apply_succeeded, field_resolution, field_preservation],
        experiment_prefix="automl_apply_clarification_eval"
    ))
