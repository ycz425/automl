from pydantic import BaseModel
from app.agents.experiment_agent import ExperimentAgent
from app.graph.schemas.experiment import ExperimentImplementation
from app.graph.schemas.plan import Plan
from langsmith import aevaluate
from evals.utils import llm_judge, dependency_consistency_evaluator
from evals.run_experiment_implementation_eval import plan_adherence, no_rogue_files


async def run_agent(inputs: dict):
    experiment_agent = ExperimentAgent()

    implementation = ExperimentImplementation.model_validate(inputs['implementation'])
    plan = Plan.model_validate(inputs['plan'])

    repaired = await experiment_agent.repair_implementation(
        implementation,
        inputs['error_message'],
        plan,
        inputs['output_path']
    )

    return {'implementation': repaired.model_dump()}


class ErrorFixJudgment(BaseModel):
    fixes_reported_error: bool
    reasoning: str


async def addresses_reported_error(inputs: dict, outputs: dict):
    error_message = inputs['error_message']
    original_code = inputs['implementation']['code']
    repaired_code = outputs['implementation']['code']

    prompt = f"""
    The following code failed with this error:
    {error_message}

    Original (broken) code:
    {original_code}

    Repaired code:
    {repaired_code}

    Judge whether the repaired code plausibly fixes the specific reported error, rather than
    leaving the same issue in place or making an unrelated change that doesn't address it.
    """

    judgment = await llm_judge(prompt, ErrorFixJudgment)

    return {'key': 'addresses_reported_error', 'score': judgment.fixes_reported_error, 'comment': judgment.reasoning}


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_experiment_repair',
        evaluators=[
            plan_adherence,
            no_rogue_files,
            dependency_consistency_evaluator('implementation', ['code']),
            addresses_reported_error,
        ],
        experiment_prefix='automl_experiment_repair_eval'
    ))
