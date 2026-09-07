import json
import ast
from pydantic import BaseModel
from app.agents.output_agent import OutputAgent
from app.graph.schemas.data_info import DatasetAnalysis
from app.graph.schemas.experiment import Experiment
from langsmith import aevaluate
from evals.utils import llm_judge, dependency_consistency_evaluator


async def run_agent(inputs: dict):
    output_agent = OutputAgent()

    experiment = Experiment.model_validate(inputs['experiment'])
    dataset_analysis = DatasetAnalysis.model_validate(inputs['dataset_analysis'])

    output_scripts = await output_agent.generate_scripts(experiment, dataset_analysis)

    return {'output_scripts': output_scripts.model_dump()}


class OutputAdherenceJudgment(BaseModel):
    architecture_adherence: bool
    architecture_reasoning: str
    hyperparameter_adherence: bool
    hyperparameter_reasoning: str
    preprocessing_adherence: bool
    preprocessing_reasoning: str
    predict_train_consistency: bool
    predict_train_reasoning: str


async def experiment_adherence(inputs: dict, outputs: dict):
    experiment = inputs['experiment']
    output_scripts = outputs['output_scripts']

    prompt = f"""
    Judge whether the following train/predict scripts correctly reproduce the selected experiment they were derived from.

    Selected experiment's original implementation code:
    {experiment['implementation']['code']}

    Selected experiment's plan (for architecture and hyperparameter intent):
    {json.dumps(experiment['plan'], indent=2)}

    Generated train_script:
    {output_scripts['train_script']}

    Generated predict_script:
    {output_scripts['predict_script']}

    Instructions:
    - Judge whether the model architecture in train_script matches the original implementation's architecture, not a different but plausible choice.
    - Judge whether the hyperparameter values in train_script match the original implementation's hyperparameter values, not just the same class name with different values.
    - Judge whether the preprocessing and feature selection in train_script matches the original implementation exactly.
    - Judge whether predict_script applies inference consistently with how train_script fit the pipeline (e.g. it loads and uses the saved pipeline rather than reimplementing preprocessing separately in a way that could drift from what was trained).
    """

    judgment = await llm_judge(prompt, OutputAdherenceJudgment)

    return [
        {'key': 'architecture_adherence', 'score': judgment.architecture_adherence, 'comment': judgment.architecture_reasoning},
        {'key': 'hyperparameter_adherence', 'score': judgment.hyperparameter_adherence, 'comment': judgment.hyperparameter_reasoning},
        {'key': 'preprocessing_adherence', 'score': judgment.preprocessing_adherence, 'comment': judgment.preprocessing_reasoning},
        {'key': 'predict_train_consistency', 'score': judgment.predict_train_consistency, 'comment': judgment.predict_train_reasoning},
    ]


SPLIT_CALL_NAMES = {
    'train_test_split', 'KFold', 'StratifiedKFold', 'GroupKFold',
    'StratifiedGroupKFold', 'LeaveOneGroupOut', 'GroupShuffleSplit',
    'ShuffleSplit', 'StratifiedShuffleSplit', 'sample'
}


def trains_on_full_dataset(outputs: dict):
    code = outputs['output_scripts']['train_script']

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {'key': 'trains_on_full_dataset', 'score': 0.0, 'comment': f'train_script failed to parse: {e}'}

    violations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', None)
            if func_name in SPLIT_CALL_NAMES:
                violations.add(func_name)

    return {
        'key': 'trains_on_full_dataset',
        'score': 0.0 if violations else 1.0,
        'comment': f'train_script appears to split or subsample data before fitting: {sorted(violations)}' if violations else 'No data splitting or subsampling detected before training.'
    }


def feature_column_usage(inputs: dict, outputs: dict):
    dataset_analysis = inputs['dataset_analysis']
    output_scripts = outputs['output_scripts']

    feature_columns = set(dataset_analysis.get('feature_columns', []))
    forbidden = set(dataset_analysis.get('excluded_columns', []))
    if dataset_analysis.get('target_column'):
        forbidden.add(dataset_analysis['target_column'])
    if dataset_analysis.get('group_column'):
        forbidden.add(dataset_analysis['group_column'])

    text = output_scripts['train_script'] + ' ' + output_scripts['predict_script']

    results = []

    if feature_columns:
        unused = sorted(col for col in feature_columns if col not in text)
        results.append({
            'key': 'feature_column_completeness',
            'score': 1 - len(unused) / len(feature_columns),
            'comment': f'Feature columns never referenced: {unused}' if unused else 'All feature columns referenced.'
        })

    if forbidden:
        violations = sorted(col for col in forbidden if col in text)
        results.append({
            'key': 'feature_column_faithfulness',
            'score': 1 - len(violations) / len(forbidden),
            'comment': f'Forbidden columns referenced: {violations}' if violations else 'No forbidden columns referenced.'
        })

    return results


REQUIRED_TRAIN_FLAGS = ['--input', '--output']
REQUIRED_PREDICT_FLAGS = ['--model', '--input', '--output', '--threshold']


def cli_interface_compliance(outputs: dict):
    """Deterministic, execution-free stand-in for 'the scripts actually run correctly': checks
    the hard CLI-contract and output-format requirements from the generation prompt directly
    against the source text, without needing to invoke either script."""
    scripts = outputs['output_scripts']
    train_script = scripts['train_script']
    predict_script = scripts['predict_script']

    missing = [f'train_script missing {flag}' for flag in REQUIRED_TRAIN_FLAGS if flag not in train_script]
    missing += [f'predict_script missing {flag}' for flag in REQUIRED_PREDICT_FLAGS if flag not in predict_script]

    has_pred_column = "'pred'" in predict_script or '"pred"' in predict_script
    if not has_pred_column:
        missing.append("predict_script never references a 'pred' output column")

    total_checks = len(REQUIRED_TRAIN_FLAGS) + len(REQUIRED_PREDICT_FLAGS) + 1

    return {
        'key': 'cli_interface_compliance',
        'score': 1 - len(missing) / total_checks,
        'comment': '; '.join(missing) if missing else 'train/predict scripts expose the required CLI arguments and pred column.'
    }


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_output_scripts',
        evaluators=[
            experiment_adherence,
            trains_on_full_dataset,
            feature_column_usage,
            cli_interface_compliance,
            dependency_consistency_evaluator('output_scripts', ['train_script', 'predict_script']),
        ],
        experiment_prefix='automl_output_scripts_eval'
    ))
