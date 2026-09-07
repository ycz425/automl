from app.agents.data_agent import DataAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile
from langsmith import aevaluate
from evals.utils import field_correctness_evaluator, set_overlap_evaluator, hallucination_rate_evaluator


async def run_agent(inputs: dict):
    data_agent = DataAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    dataset_profile = DatasetProfile.model_validate(inputs['dataset_profile'])

    dataset_analysis = await data_agent.analyse_profile(user_request, dataset_profile)

    return {'dataset_analysis': dataset_analysis.model_dump()}

key = 'dataset_analysis'

fields = [
    'target_column',
    'group_column',
    'positive_class'
]

list_fields = [
    'excluded_columns',
    'feature_columns'
]


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_dataset_analysis',
        evaluators=[
            field_correctness_evaluator(key, fields),
            set_overlap_evaluator(key, list_fields),
            hallucination_rate_evaluator(key, fields)
        ],
        experiment_prefix='automl_dataset_analysis_eval'
    ))