import os
import json
import dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create
from app.agents.clarification_agent import ClarificationAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetAnalysis
from langsmith import aevaluate

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3


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
            response_format={'mime_type': 'application/json', 'schema': ClarificationQuestionJudgment.model_json_schema()}
        )

        try:
            judgment = ClarificationQuestionJudgment.model_validate_json(interaction.output_text)
            break
        except ValidationError as e:
            if attempt == MAX_RETRIES:
                raise
            validation_error = str(e)

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


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_request_clarification',
        evaluators=[clarification_quality],
        experiment_prefix='automl_request_clarification_eval'
    ))
