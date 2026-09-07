from pydantic import BaseModel
from app.agents.output_agent import OutputAgent
from app.graph.schemas.output import OutputScripts
from app.graph.schemas.experiment import Experiment
from app.graph.schemas.data_info import DatasetAnalysis
from langsmith import aevaluate
from evals.utils import llm_judge, dependency_consistency_evaluator
from evals.run_output_eval import (
    experiment_adherence,
    trains_on_full_dataset,
    feature_column_usage,
    cli_interface_compliance,
)


async def run_agent(inputs: dict):
    output_agent = OutputAgent()

    scripts = OutputScripts.model_validate(inputs['scripts'])
    experiment = Experiment.model_validate(inputs['experiment'])
    dataset_analysis = DatasetAnalysis.model_validate(inputs['dataset_analysis'])

    repaired = await output_agent.repair_scripts(
        scripts,
        inputs['error_message'],
        experiment,
        dataset_analysis
    )

    return {'output_scripts': repaired.model_dump()}


class ErrorFixJudgment(BaseModel):
    fixes_reported_error: bool
    reasoning: str


async def addresses_reported_error(inputs: dict, outputs: dict):
    error_message = inputs['error_message']
    original_scripts = inputs['scripts']
    repaired_scripts = outputs['output_scripts']

    prompt = f"""
    The following deployment scripts failed with this error:
    {error_message}

    Original (broken) train_script:
    {original_scripts['train_script']}

    Original (broken) predict_script:
    {original_scripts['predict_script']}

    Repaired train_script:
    {repaired_scripts['train_script']}

    Repaired predict_script:
    {repaired_scripts['predict_script']}

    Judge whether the repaired scripts plausibly fix the specific reported error, rather than
    leaving the same issue in place or making an unrelated change that doesn't address it.
    """

    judgment = await llm_judge(prompt, ErrorFixJudgment)

    return {'key': 'addresses_reported_error', 'score': judgment.fixes_reported_error, 'comment': judgment.reasoning}


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_output_repair',
        evaluators=[
            experiment_adherence,
            trains_on_full_dataset,
            feature_column_usage,
            cli_interface_compliance,
            dependency_consistency_evaluator('output_scripts', ['train_script', 'predict_script']),
            addresses_reported_error,
        ],
        experiment_prefix='automl_output_repair_eval'
    ))
