import json
from pydantic import BaseModel
from app.agents.plan_agent import PlanAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.experiment import Experiment
from langsmith import aevaluate
from evals.utils import llm_judge


async def run_agent(inputs: dict):
    plan_agent = PlanAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    dataset_profile = DatasetProfile.model_validate(inputs['dataset_profile'])
    dataset_analysis = DatasetAnalysis.model_validate(inputs['dataset_analysis'])
    experiments = [Experiment.model_validate(e) for e in inputs.get('experiments', [])]

    plan = await plan_agent.plan(user_request, dataset_profile, dataset_analysis, experiments=experiments)

    return {'plan': plan.model_dump()}


# Training-strategy fields by which model families they apply to (per the Plan schema):
#   neural only .......... optimizer, batch_size, epochs, scheduler, gradient_clipping
#   neural + boosted trees  learning_rate, weight_decay (L2 regularization), patience (early stopping)
# Classification uses the architecture *name* only — descriptions routinely mention things like
# "ColumnTransformer" that would falsely read as a neural network.
NEURAL_ONLY_FIELDS = ['optimizer', 'batch_size', 'epochs', 'scheduler', 'gradient_clipping']
BOOSTED_OK_FIELDS = ['learning_rate', 'weight_decay', 'patience']
NEURAL_KEYWORDS = ['mlp', 'neural', 'cnn', 'lstm', 'gru', 'rnn', 'transformer', 'resnet', 'tabnet', 'tabr',
                   'autoencoder', 'perceptron', 'attention', 'deep']
BOOSTING_KEYWORDS = ['xgboost', 'lightgbm', 'catboost', 'gradient boost', 'gradientboost', 'histgradient', 'gbdt', 'gbm', 'adaboost']


def training_strategy_nullness(outputs: dict):
    """Deterministic check on the plan's structured training_strategy: the fields that are filled
    must make sense for the model family (the plan schema says to use null for fields that don't apply)."""
    plan = outputs['plan']
    name = plan['architecture_plan']['architecture_name'].lower()
    strategy = plan['training_strategy']

    is_neural = any(k in name for k in NEURAL_KEYWORDS)
    is_boosting = any(k in name for k in BOOSTING_KEYWORDS) and not is_neural

    problems = []
    if is_neural:
        missing = [f for f in ['optimizer', 'learning_rate', 'batch_size', 'epochs'] if strategy.get(f) is None]
        if missing:
            problems.append(f'neural model is missing training settings: {missing}')
    else:
        stray = [f for f in NEURAL_ONLY_FIELDS if strategy.get(f) is not None]
        if not is_boosting:
            stray += [f for f in ['learning_rate', 'patience'] if strategy.get(f) is not None]
        if stray:
            family = 'boosted-tree' if is_boosting else 'non-neural, non-boosted'
            problems.append(f'{family} model ({name}) sets fields that do not apply: {stray}')

    return {
        'key': 'training_strategy_nullness',
        'score': 0.0 if problems else 1.0,
        'comment': '; '.join(problems) if problems else None
    }


def plan_completeness(outputs: dict):
    """Deterministic sanity check that the required narrative fields are actually present and
    substantive, before spending an LLM call judging their content."""
    plan = outputs['plan']
    issues = []

    if len((plan.get('rationale') or '').strip()) < 20:
        issues.append('rationale is missing or too short')
    if len((plan.get('architecture_plan') or {}).get('architecture_description', '').strip()) < 20:
        issues.append('architecture_description is missing or too short')
    if not plan.get('preprocessing_steps'):
        issues.append('preprocessing_steps is empty')

    return {
        'key': 'plan_completeness',
        'score': 0.0 if issues else 1.0,
        'comment': '; '.join(issues) if issues else 'Plan includes a rationale, architecture description, and preprocessing steps.'
    }


class ConstraintJudgment(BaseModel):
    constraint: str
    reasoning: str
    satisfied: bool


class PreferenceJudgment(BaseModel):
    preference: str
    reasoning: str
    alignment_score: float


class PlanJudgment(BaseModel):
    constraint_judgments: list[ConstraintJudgment]
    preference_judgments: list[PreferenceJudgment]
    task_type_reasoning: str
    task_type_consistent: bool
    input_columns_reasoning: str
    uses_only_allowed_input_columns: bool
    splitting_reasoning: str
    avoids_redundant_splitting: bool


class RevisionPlanJudgment(PlanJudgment):
    improvement_reasoning: str
    improves_on_or_justifies_repeat: bool


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
    - Judge whether the plan uses ONLY legitimate input columns: the target column and every column in the dataset analysis's excluded_columns must NOT be used as model INPUT features (true = no such column is used as an input, false = leakage). Merely mentioning such a column — to drop it, to define the label, to stratify or class-weight by the target, or in a description — is NOT a violation.
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

    judgment = await llm_judge(prompt, schema_cls)

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
    results.append({'key': 'no_column_leakage', 'score': judgment.uses_only_allowed_input_columns, 'comment': judgment.input_columns_reasoning})
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
            plan_completeness,
            training_strategy_nullness,
            plan_judge
        ],
        experiment_prefix='automl_plan_eval'
    ))
