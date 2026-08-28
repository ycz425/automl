import os
import json
import dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create
from app.agents.output_agent import OutputAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.experiment import Experiment
from langsmith import aevaluate

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3


async def run_agent(inputs: dict):
    output_agent = OutputAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    experiments = [Experiment.model_validate(e) for e in inputs['experiments']]

    summary = await output_agent.generate_summary(user_request, experiments)

    return {'summary': summary}


def _best_experiment_index(user_request: dict, experiments: list[dict]):
    primary_metric = user_request['primary_metric']
    direction = primary_metric['direction']

    best_idx = None
    best_value = None
    for idx, experiment in enumerate(experiments):
        value = next(m['value'] for m in experiment['result']['metrics'] if m['metric'] == primary_metric['name'])
        if best_value is None or (direction == 'max' and value > best_value) or (direction == 'min' and value < best_value):
            best_value = value
            best_idx = idx

    return best_idx, best_value


class SummaryAccuracyJudgment(BaseModel):
    identifies_correct_best_experiment: bool
    identifies_correct_best_experiment_reasoning: str
    states_correct_metric_value: bool
    states_correct_metric_value_reasoning: str
    no_fabrication: bool
    no_fabrication_reasoning: str
    correct_ordering_and_numbering: bool
    correct_ordering_and_numbering_reasoning: str


async def summary_accuracy(inputs: dict, outputs: dict):
    user_request = inputs['user_request']
    experiments = inputs['experiments']
    summary = outputs['summary']

    best_idx, best_value = _best_experiment_index(user_request, experiments)
    primary_metric = user_request['primary_metric']

    prompt = f"""
    Judge whether the following summary of a machine-learning experimentation process is
    factually accurate, given the ground truth below.

    Ground truth:
    - Primary metric: {primary_metric['name']} ({primary_metric['direction']})
    - Experiments, in chronological order (zero-based list here, but the summary should refer
      to them using one-based numbering, e.g. index 0 is "Experiment 1"):
      {json.dumps(experiments, indent=2)}
    - The actual best experiment is at zero-based index {best_idx} (i.e. "Experiment {best_idx + 1}"),
      with {primary_metric['name']} = {best_value}.

    Summary to judge:
    {summary}

    Instructions:
    - Judge whether the summary correctly identifies Experiment {best_idx + 1} as the best experiment.
    - Judge whether the summary states the correct {primary_metric['name']} value ({best_value}) for
      the best experiment, allowing for reasonable rounding.
    - Judge whether the summary avoids fabricating any approach, metric, or number not present in
      the experiments list above.
    - Judge whether experiments are described in chronological order using one-based numbering.
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
            response_format={'mime_type': 'application/json', 'schema': SummaryAccuracyJudgment.model_json_schema()}
        )

        try:
            judgment = SummaryAccuracyJudgment.model_validate_json(interaction.output_text)
            break
        except ValidationError as e:
            if attempt == MAX_RETRIES:
                raise
            validation_error = str(e)

    return [
        {
            'key': 'identifies_correct_best_experiment',
            'score': judgment.identifies_correct_best_experiment,
            'comment': judgment.identifies_correct_best_experiment_reasoning
        },
        {
            'key': 'states_correct_metric_value',
            'score': judgment.states_correct_metric_value,
            'comment': judgment.states_correct_metric_value_reasoning
        },
        {
            'key': 'no_fabrication',
            'score': judgment.no_fabrication,
            'comment': judgment.no_fabrication_reasoning
        },
        {
            'key': 'correct_ordering_and_numbering',
            'score': judgment.correct_ordering_and_numbering,
            'comment': judgment.correct_ordering_and_numbering_reasoning
        },
    ]


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_generate_summary',
        evaluators=[summary_accuracy],
        experiment_prefix='automl_generate_summary_eval'
    ))
