from langsmith import aevaluate
from app.agents.prompt_agent import PromptAgent
from evals.utils import field_correctness_evaluator, hallucination_rate_evaluator, semantic_correctness_evaluator, semantic_overlap_evaluator


async def run_agent(inputs: dict):
    prompt_agent = PromptAgent()
    user_request = await prompt_agent.parse(inputs['user_input'])
    return {'user_request': user_request.model_dump()}


key = 'user_request'

deterministic_fields = [
    'evaluation_method',
    'task_type',
    'num_folds',
    'primary_metric',
    'validation_size',
    'stratify'
]

semantic_fields = [
    'target_description',
    'group_description'
]

semantic_list_fields = [
    'include_features',
    'exclude_features',
    'constraints',
    'preferences',
    'secondary_metrics'
]


if __name__ == "__main__":
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data="automl_user_request",
        evaluators=[
            field_correctness_evaluator(key, deterministic_fields),
            hallucination_rate_evaluator(key, deterministic_fields + semantic_fields),
            semantic_correctness_evaluator(key, semantic_fields),
            semantic_overlap_evaluator(key, semantic_list_fields)
        ],
        experiment_prefix="automl_user_request_eval"
    ))
