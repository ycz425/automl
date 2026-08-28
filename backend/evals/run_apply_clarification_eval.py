from app.agents.clarification_agent import ClarificationAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetAnalysis
from langsmith import aevaluate


async def run_agent(inputs: dict):
    if inputs['data_type'] == 'UserRequest':
        data = UserRequest.model_validate(inputs['data'])
    elif inputs['data_type'] == 'DatasetAnalysis':
        data = DatasetAnalysis.model_validate(inputs['data'])
    else:
        raise ValueError

    clarification_agent = ClarificationAgent()
    new_data = await clarification_agent.apply_clarification(
        data,
        inputs['question'],
        inputs['problems'],
        inputs['clarification']
    )

    return {'data': new_data.model_dump()}


def field_resolution(inputs: dict, outputs: dict, reference_outputs: dict):
    before = inputs['data']
    actual = outputs['data']
    expected = reference_outputs['data']

    relevant_fields = [field for field in before if before[field] != expected[field]]

    if not relevant_fields:
        return 1.0

    return sum(actual[field] == expected[field] for field in relevant_fields) / len(relevant_fields)


def field_preservation(inputs: dict, outputs: dict, reference_outputs: dict):
    before = inputs['data']
    actual = outputs['data']
    expected = reference_outputs['data']

    irrelevant_fields = [field for field in before if before[field] == expected[field]]

    if not irrelevant_fields:
        return 1.0

    return sum(actual[field] == before[field] for field in irrelevant_fields) / len(irrelevant_fields)


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data="automl_apply_clarification",
        evaluators=[field_resolution, field_preservation],
        experiment_prefix="automl_apply_clarification_eval"
    ))