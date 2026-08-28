import os
import dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create
from app.agents.output_agent import OutputAgent
from app.graph.schemas.output import OutputScripts
from app.graph.schemas.experiment import Experiment
from app.graph.schemas.data_info import DatasetAnalysis
from langsmith import aevaluate
from evals.run_output_eval import experiment_adherence, trains_on_full_dataset, feature_column_usage

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3


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

    validation_error = None
    for attempt in range(MAX_RETRIES + 1):
        attempt_prompt = prompt
        if validation_error:
            attempt_prompt += (
                f"\n\nYour previous output failed schema validation:\n\n{validation_error}\n\n"
                "Return a corrected response that strictly matches the required schema."
            )

        interaction = await traced_interactions_create(
            client,
            model=MODEL,
            input=attempt_prompt,
            generation_config={'thinking_level': 'low', 'temperature': 0},
            response_format={'mime_type': 'application/json', 'schema': ErrorFixJudgment.model_json_schema()}
        )

        try:
            judgment = ErrorFixJudgment.model_validate_json(interaction.output_text)
            break
        except ValidationError as e:
            if attempt == MAX_RETRIES:
                raise
            validation_error = str(e)

    return {'key': 'addresses_reported_error', 'score': judgment.fixes_reported_error, 'comment': judgment.reasoning}


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_output_repair',
        evaluators=[experiment_adherence, trains_on_full_dataset, feature_column_usage, addresses_reported_error],
        experiment_prefix='automl_output_repair_eval'
    ))
