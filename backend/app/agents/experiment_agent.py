from app.agents.base import LLMAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis, DataSplits
from app.graph.schemas.plan import Plan
from app.graph.schemas.experiment import ExperimentImplementation, ExperimentResult, Experiment
from app.utils.model_scripts import create_venv, format_subprocess_error
from langsmith import traceable
from pydantic import ValidationError
import tempfile
from pathlib import Path
from datetime import datetime
import asyncio
import json
import subprocess


class ExperimentAgent(LLMAgent):
    @traceable(name="ExperimentAgent.generate_implementation")
    async def generate_implementation(self, data_path: str, split_path: str, user_request: UserRequest, dataset_profile: DatasetProfile, dataset_analysis: DatasetAnalysis, plan: Plan, output_path: str, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     generating experiment implementation...')

        prompt = f"""
        Generate executable Python code for the machine-learning experiment below.

        Requirements:
        - Follow the experiment plan exactly.
        - Use only columns listed in the dataset analysis.
        - Do not invent column names.
        - Handle missing values and feature types described in the dataset profile.
        - Read the dataset from: {data_path}
        - Read the split indices from {split_path}
        - Expect split indices to follow this JSON schema:
        {json.dumps(DataSplits.model_json_schema(), indent=2)}
        - Do not generate a new dataset split and only use the given split indices.
        - Save the final experiment result to:
            {output_path}
        - Make experiment_result.json conform exactly to this JSON schema:
        {json.dumps(ExperimentResult.model_json_schema(), indent=2)}
        - Use exactly the metric names specified by the user request.
        - If the task is binary classification, tune the decision threshold applied to the predicted probability of the class named in dataset analysis's positive_class field, using only held-out validation data (per fold, then aggregated across folds if cross-validation is used) to optimize the primary metric, and report it as `threshold` in the experiment result. If the primary metric is threshold-invariant (e.g. AUROC/AUC, or any other ranking-based metric unaffected by the decision threshold), tune the threshold to optimize F1 instead. Never tune it using data the model was fit on. Use `threshold: null` if the task is regression or multiclass classification.
        - Produce complete runnable code, not pseudocode.
        - Do not include markdown fences or explanations.

        User request:
        {user_request.model_dump_json(indent=2)}

        Dataset profile:
        {dataset_profile.model_dump_json(indent=2)}

        Dataset analysis:
        {dataset_analysis.model_dump_json(indent=2)}

        Experiment plan:
        {plan.model_dump_json(indent=2)}
        """

        return await self.generate_structured(
            prompt, ExperimentImplementation, max_retries=max_retries, label="ExperimentImplementation"
        )

    @traceable(name="ExperimentAgent.repair_implementation")
    async def repair_implementation(self, implementation: ExperimentImplementation, error_message: str, plan: Plan, output_path: str, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     repairing experiment implementation...')

        prompt = f"""
            Repair the previous ExperimentImplementation.

            Failure details:
            {error_message}

            Previous implementation:
            {implementation.model_dump_json(indent=2)}

            Plan:
            {plan.model_dump_json(indent=2)}

            ExperimentResult JSON schema:
            {json.dumps(ExperimentResult.model_json_schema(), indent=2)}

            Return the complete corrected ExperimentImplementation.

            Rules:
            - Preserve the experiment plan.
            - For dependency failures, correct the dependencies or replace the incompatible library with an equivalent supported approach.
            - For execution failures, repair the Python code.
            - For result validation failures, repair the code that creates experiment_result.json so it correctly follows the ExperimentResult JSON schema.
            - For result file not found errors, make sure the result file is created at the correct path: {output_path}.
            - If the task is binary classification, tune the decision threshold applied to the predicted probability of the class named in dataset analysis's positive_class field, using only held-out validation data (per fold, then aggregated across folds if cross-validation is used) to optimize the primary metric, and report it as `threshold` in the experiment result. If the primary metric is threshold-invariant (e.g. AUROC/AUC, or any other ranking-based metric unaffected by the decision threshold), tune the threshold to optimize F1 instead. Never tune it using data the model was fit on. Use `threshold: null` if the task is regression or multiclass classification.
            - Do not hide errors or fabricate successful results.
            - Do not return a patch.
        """

        return await self.generate_structured(
            prompt, ExperimentImplementation, max_retries=max_retries, label="ExperimentImplementation"
        )

    @traceable(name="ExperimentAgent.execute_plan")
    async def execute_plan(self, data_path: str, split_path: str, user_request: UserRequest, dataset_profile: DatasetProfile, dataset_analysis: DatasetAnalysis, plan: Plan, max_retries=5):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = Path(temp_dir) / ".venv"
            code_path = Path(temp_dir) / "code.py"
            output_path = Path(temp_dir) / "experiment_result.json"
            requirements_path = Path(temp_dir) / "requirements.txt"

            implementation = await self.generate_implementation(data_path, split_path, user_request, dataset_profile, dataset_analysis, plan, output_path)

            for attempt in range(max_retries + 1):
                if self.verbose:
                    print(f'{datetime.now()}     running experiment...')
                try:
                    output_path.unlink(missing_ok=True)

                    with open(str(requirements_path), 'w') as f:
                        f.write('\n'.join(implementation.dependencies))

                    env_python = await create_venv(str(env_dir), str(requirements_path))

                    with open(str(code_path), 'w') as f:
                        f.write(implementation.code)

                    await asyncio.to_thread(
                        subprocess.run,
                        [str(env_python), str(code_path)],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    return Experiment(
                        plan=plan,
                        result=ExperimentResult.model_validate_json(output_path.read_text()),
                        implementation=implementation
                    )
                except (subprocess.CalledProcessError, ValidationError, FileNotFoundError) as e:
                    if attempt == max_retries:
                        raise
                    if self.verbose:
                        print(f'{datetime.now()}     experiment failed - retrying... (attempt: {attempt + 1}/{max_retries})')
                    error_message = format_subprocess_error(e)
                    implementation = await self.repair_implementation(implementation, error_message, plan, str(output_path))
