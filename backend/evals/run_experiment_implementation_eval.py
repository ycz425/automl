import json
import ast
import re
from pydantic import BaseModel
from app.agents.experiment_agent import ExperimentAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.plan import Plan
from langsmith import aevaluate
from evals.utils import llm_judge, dependency_consistency_evaluator


async def run_agent(inputs: dict):
    experiment_agent = ExperimentAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    dataset_profile = DatasetProfile.model_validate(inputs['dataset_profile'])
    dataset_analysis = DatasetAnalysis.model_validate(inputs['dataset_analysis'])
    plan = Plan.model_validate(inputs['plan'])

    implementation = await experiment_agent.generate_implementation(
        inputs['data_path'],
        inputs['split_path'],
        user_request,
        dataset_profile,
        dataset_analysis,
        plan,
        inputs['output_path']
    )

    return {'implementation': implementation.model_dump()}


class PreprocessingStepJudgment(BaseModel):
    step: str
    implemented: bool
    reasoning: str


class CodePlanJudgment(BaseModel):
    preprocessing_step_judgments: list[PreprocessingStepJudgment]
    architecture_adherence: bool
    architecture_reasoning: str
    training_strategy_adherence: bool
    training_strategy_reasoning: str


async def plan_adherence(inputs: dict, outputs: dict):
    plan = inputs['plan']
    implementation = outputs['implementation']

    prompt = f"""
    Judge whether the following generated code correctly implements the given experiment plan.

    Plan:
    {json.dumps(plan, indent=2)}

    Generated code:
    {implementation['code']}

    Dependencies declared:
    {json.dumps(implementation.get('dependencies', []), indent=2)}

    Instructions:
    - For each item in the plan's "preprocessing_steps" list, judge whether the code actually implements it.
    - Judge whether the code's model construction matches the plan's architecture_plan (architecture_name and hyperparameters), not just something plausible for the task.
    - Judge whether the code's training setup matches the plan's training_strategy: non-null fields (optimizer, loss, learning_rate, batch_size, epochs, patience, scheduler, gradient_clipping) should be reflected in the code, and fields the plan left null should not be introduced unrequested.
    """

    judgment = await llm_judge(prompt, CodePlanJudgment)

    results = []

    if judgment.preprocessing_step_judgments:
        score = sum(s.implemented for s in judgment.preprocessing_step_judgments) / len(judgment.preprocessing_step_judgments)
        missing = [s.step for s in judgment.preprocessing_step_judgments if not s.implemented]
        results.append({
            'key': 'preprocessing_adherence',
            'score': score,
            'comment': f'Not implemented: {missing}' if missing else 'All preprocessing steps implemented.'
        })

    results.append({'key': 'architecture_adherence', 'score': judgment.architecture_adherence, 'comment': judgment.architecture_reasoning})
    results.append({'key': 'training_strategy_adherence', 'score': judgment.training_strategy_adherence, 'comment': judgment.training_strategy_reasoning})

    return results


WRITE_METHOD_NAMES = {
    'open', 'to_csv', 'to_json', 'to_pickle', 'to_parquet', 'to_excel',
    'savefig', 'dump', 'save', 'write_text', 'write_bytes', 'imsave'
}

PATH_LIKE_RE = re.compile(
    r'^[\w./\\\-]+\.(csv|json|jsonl|pkl|pickle|parquet|png|jpg|jpeg|txt|log|h5|pt|pth|joblib|npy|npz|xlsx)$',
    re.IGNORECASE
)

WRITE_MODES = {'w', 'a', 'x', 'wb', 'ab', 'xb', 'w+', 'a+', 'x+'}


def no_rogue_files(inputs: dict, outputs: dict):
    implementation = outputs['implementation']
    code = implementation['code']

    expected_paths = {inputs.get('output_path'), inputs.get('data_path'), inputs.get('split_path')}

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {'key': 'no_rogue_files', 'score': 0.0, 'comment': f'Generated code failed to parse: {e}'}

    rogue_paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', None)
        if func_name not in WRITE_METHOD_NAMES:
            continue

        string_args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        string_args += [kw.value.value for kw in node.keywords if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)]

        if func_name == 'open':
            mode = string_args[1] if len(string_args) > 1 else 'r'
            if mode not in WRITE_MODES:
                continue

        for arg in string_args:
            if PATH_LIKE_RE.match(arg.strip()) and arg not in expected_paths:
                rogue_paths.add(arg)

    return {
        'key': 'no_rogue_files',
        'score': 1.0 if not rogue_paths else 0.0,
        'comment': f'Unexpected file writes: {sorted(rogue_paths)}' if rogue_paths else 'No writes outside the given paths detected.'
    }


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_experiment_implementation',
        evaluators=[
            plan_adherence,
            no_rogue_files,
            dependency_consistency_evaluator('implementation', ['code']),
        ],
        experiment_prefix='automl_experiment_implementation_eval'
    ))
