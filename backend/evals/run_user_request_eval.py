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
    'preferences'
]

# secondary_metrics isn't scored by overlap against the reference as a whole: when the user
# doesn't name any, the agent is told to pick sensible defaults, so a different-but-valid set
# (e.g. [accuracy, precision] vs the reference's [accuracy, f1]) is not an error.
METRIC_ALIASES = {
    'auroc': ['auroc', 'auc', 'roc'],
    'rmse': ['rmse', 'root mean'],
    'mae': ['mae', 'mean absolute'],
    'r2': ['r2', 'r-squared', 'r squared'],
}
CLASSIFICATION_METRICS = {'accuracy', 'f1', 'precision', 'recall', 'auroc'}
REGRESSION_METRICS = {'rmse', 'mae', 'r2', 'mse'}


def _metric_names(metrics):
    return {m['name'].lower() for m in (metrics or [])}


def _user_mentioned(metric: str, user_input: str):
    return any(alias in user_input for alias in METRIC_ALIASES.get(metric, [metric]))


def secondary_metrics_explicit_overlap(inputs: dict, outputs: dict, reference_outputs: dict):
    expected = _metric_names(reference_outputs[key]['secondary_metrics'])
    user_input = inputs['user_input'].lower()
    requested = {metric for metric in expected if _user_mentioned(metric, user_input)}

    if not requested:
        return None

    actual = _metric_names(outputs[key]['secondary_metrics'])
    missing = sorted(requested - actual)
    return {
        'key': 'secondary_metrics_explicit_overlap',
        'score': len(requested & actual) / len(requested),
        'comment': f'User-requested metrics missing: {missing}' if missing else None
    }


def secondary_metrics_valid(outputs: dict):
    user_request = outputs[key]
    secondary = _metric_names(user_request['secondary_metrics'])
    primary = (user_request.get('primary_metric') or {}).get('name', '').lower()
    task_type = user_request.get('task_type')

    problems = []
    if not secondary:
        problems.append('no secondary metrics')
    if primary and primary in secondary:
        problems.append('includes the primary metric')
    if task_type == 'regression' and secondary & CLASSIFICATION_METRICS:
        problems.append(f'classification metrics for a regression task: {sorted(secondary & CLASSIFICATION_METRICS)}')
    if task_type and task_type != 'regression' and secondary & REGRESSION_METRICS:
        problems.append(f'regression metrics for a classification task: {sorted(secondary & REGRESSION_METRICS)}')

    return {
        'key': 'secondary_metrics_valid',
        'score': 0.0 if problems else 1.0,
        'comment': '; '.join(problems) if problems else None
    }


if __name__ == "__main__":
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data="automl_user_request",
        evaluators=[
            field_correctness_evaluator(key, deterministic_fields),
            hallucination_rate_evaluator(key, deterministic_fields + semantic_fields),
            semantic_correctness_evaluator(key, semantic_fields),
            semantic_overlap_evaluator(key, semantic_list_fields),
            secondary_metrics_explicit_overlap,
            secondary_metrics_valid
        ],
        experiment_prefix="automl_user_request_eval"
    ))
