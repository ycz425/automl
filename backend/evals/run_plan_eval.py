import os
import json
import dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create
from app.agents.plan_agent import PlanAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.experiment import Experiment
from langsmith import aevaluate

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3


async def run_agent(inputs: dict):
    plan_agent = PlanAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    dataset_profile = DatasetProfile.model_validate(inputs['dataset_profile'])
    dataset_analysis = DatasetAnalysis.model_validate(inputs['dataset_analysis'])
    experiments = [Experiment.model_validate(e) for e in inputs.get('experiments', [])]

    plan = await plan_agent.plan(user_request, dataset_profile, dataset_analysis, experiments=experiments)

    return {'plan': plan.model_dump()}


def column_faithfulness(inputs: dict, outputs: dict):
    analysis = inputs['dataset_analysis']
    plan = outputs['plan']

    forbidden = set(analysis.get('excluded_columns', []))
    if analysis.get('target_column'):
        forbidden.add(analysis['target_column'])

    text = ' '.join(plan.get('preprocessing_steps', []))
    text += ' ' + plan.get('architecture_plan', {}).get('architecture_description', '')

    if not forbidden:
        return

    violations = sorted(col for col in forbidden if col in text)
    return {
        'key': 'column_faithfulness',
        'score': 1 - len(violations) / len(forbidden),
        'comment': f'Forbidden columns referenced: {violations}' if violations else 'No forbidden columns referenced.'
    }


class ConstraintJudgment(BaseModel):
    constraint: str
    satisfied: bool
    reasoning: str


class PreferenceJudgment(BaseModel):
    preference: str
    alignment_score: float
    reasoning: str


class PlanJudgment(BaseModel):
    constraint_judgments: list[ConstraintJudgment]
    preference_judgments: list[PreferenceJudgment]
    task_type_consistent: bool
    task_type_reasoning: str
    training_strategy_consistent: bool
    training_strategy_reasoning: str
    avoids_redundant_splitting: bool
    splitting_reasoning: str


class RevisionPlanJudgment(PlanJudgment):
    improves_on_or_justifies_repeat: bool
    improvement_reasoning: str


async def plan_judge(inputs: dict, outputs: dict):
    user_request = inputs['user_request']
    dataset_analysis = inputs['dataset_analysis']
    plan = outputs['plan']
    experiments = inputs.get('experiments', [])
    is_revision = bool(experiments)
    schema_cls = RevisionPlanJudgment if is_revision else PlanJudgment

    prompt = f"""
    Judge whether the following machine-learning experiment plan correctly follows its inputs.

    User request:
    {json.dumps(user_request, indent=2)}

    Dataset analysis:
    {json.dumps(dataset_analysis, indent=2)}
    """

    if is_revision:
        prompt += f"""
    Previous experiment history (this plan is a revision, not an initial plan):
    {json.dumps(experiments, indent=2)}
    """

    prompt += f"""
    Plan:
    {json.dumps(plan, indent=2)}

    Instructions:
    - For each item in the user request's "constraints" list, judge whether the plan satisfies it. These are hard requirements.
    - For each item in the user request's "preferences" list, score from 0 to 1 how well the plan aligns with it. These are soft, non-mandatory.
    - Judge whether the architecture and training strategy are consistent with the request's task_type (e.g. loss function and output structure suit classification vs. regression).
    - Judge whether the training strategy's null/non-null fields are consistent with whether the chosen architecture is gradient-based (e.g. a tree-based model should not specify a learning rate, optimizer, batch size, or scheduler).
    - Judge whether the plan avoids describing data-splitting or cross-validation steps in preprocessing_steps, since splitting is handled by a separate deterministic step upstream and should not be redone or described by the plan.
    """

    if is_revision:
        prompt += """
    - This plan is a revision of a prior attempt. Judge whether it meaningfully changes the
      approach in response to the previous experiment's result (e.g. a different architecture,
      different preprocessing, or a well-justified hyperparameter change), or clearly explains why
      repeating largely the same approach is warranted, rather than repeating the prior plan
      without any justification.
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
            response_format={'mime_type': 'application/json', 'schema': schema_cls.model_json_schema()}
        )

        try:
            judgment = schema_cls.model_validate_json(interaction.output_text)
            break
        except ValidationError as e:
            if attempt == MAX_RETRIES:
                raise
            validation_error = str(e)

    results = []

    if judgment.constraint_judgments:
        score = sum(c.satisfied for c in judgment.constraint_judgments) / len(judgment.constraint_judgments)
        unmet = [c.constraint for c in judgment.constraint_judgments if not c.satisfied]
        results.append({
            'key': 'constraint_adherence',
            'score': score,
            'comment': f'Unmet: {unmet}' if unmet else 'All constraints satisfied.'
        })

    if judgment.preference_judgments:
        score = sum(p.alignment_score for p in judgment.preference_judgments) / len(judgment.preference_judgments)
        results.append({'key': 'preference_alignment', 'score': score})

    results.append({'key': 'task_type_consistency', 'score': judgment.task_type_consistent, 'comment': judgment.task_type_reasoning})
    results.append({'key': 'training_strategy_consistency', 'score': judgment.training_strategy_consistent, 'comment': judgment.training_strategy_reasoning})
    results.append({'key': 'avoids_redundant_splitting', 'score': judgment.avoids_redundant_splitting, 'comment': judgment.splitting_reasoning})

    if is_revision:
        results.append({
            'key': 'improves_on_or_justifies_repeat',
            'score': judgment.improves_on_or_justifies_repeat,
            'comment': judgment.improvement_reasoning
        })

    return results


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_plan',
        evaluators=[
            column_faithfulness,
            plan_judge
        ],
        experiment_prefix='automl_plan_eval'
    ))
