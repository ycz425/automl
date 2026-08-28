import os
import dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create
from app.agents.experiment_agent import ExperimentAgent
from app.graph.schemas.experiment import ExperimentImplementation
from app.graph.schemas.plan import Plan
from langsmith import aevaluate
from evals.run_experiment_implementation_eval import plan_adherence, no_rogue_files

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3


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
        data='automl_experiment_repair',
        evaluators=[plan_adherence, no_rogue_files, addresses_reported_error],
        experiment_prefix='automl_experiment_repair_eval'
    ))
