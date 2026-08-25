from google import genai
from app.graph.schemas.user_request import UserRequest, Metric
from app.graph.schemas.experiment import Experiment
from app.graph.schemas.output import OutputScripts
from app.graph.schemas.data_info import DatasetAnalysis
from app.services.tracing import traced_interactions_create
from langsmith import traceable
from pydantic import ValidationError
from datetime import datetime
import pandas as pd
import subprocess
import asyncio
import tempfile
import shutil
import json
import sys
import dotenv
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class OutputAgent():
    def __init__(self, model = "gemini-3.1-flash-lite", verbose=False):
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.model = model
            self.verbose = verbose

    def best_experiment(self, primary_metric: Metric, experiments: list[Experiment], return_index=False):
        direction = primary_metric.direction

        best_metric = None
        best_experiment = None
        best_idx = None
        for idx, experiment in enumerate(experiments):
            metric = [metric_entry.value for metric_entry in experiment.result.metrics if metric_entry.metric == primary_metric.name][0]

            if (
                (best_metric is None) or 
                (direction == 'max' and metric > best_metric) or 
                (direction == 'min' and metric < best_metric)
            ):
                best_metric = metric
                best_experiment = experiment
                best_idx = idx

        if return_index:
            return best_idx
        else:
            return best_experiment
    
    @traceable(name="OutputAgent.generate_scripts")
    async def generate_scripts(self, experiment: Experiment, dataset_analysis: DatasetAnalysis, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     generating scripts...')

        validation_error = None
        for attempt in range(max_retries + 1):
            prompt = (
                "You are generating the final deployment artifacts for an AutoML pipeline.\n\n"
                "The experiment below has already been selected as the best-performing experiment.\n\n"
                "Your task is to generate production-ready scripts that reproduce this experiment.\n\n"
                "Selected experiment:\n"
                f"{experiment.model_dump_json(indent=2)}\n\n"
                "Dataset analysis:\n"
                f"{dataset_analysis.model_dump_json(indent=2)}\n\n"
                "Requirements:\n\n"
                "- Produce output matching the Output schema exactly.\n"
                "- The training script must train the selected model on the full dataset.\n"
                "- Preserve the preprocessing, feature selection, model architecture, and hyperparameters from the selected experiment.\n"
                "- The training script must save the complete fitted pipeline as a joblib file to the output path provided as a command-line argument.\n"
                "- The prediction script must load the fitted pipeline from the model joblib path provided as a command-line argument, accept unseen CSV data, and write predictions to a CSV file.\n"
                "- The input CSV accepted by the prediction script must contain the columns listed in dataset analysis's feature_columns; drop any other columns present, if any.\n"
                "- The output CSV must be the same as the input CSV (after dropping irrelevant columns) with a single 'prediction' column appended as the very last column.\n"
                "- If the model supports probabilities, output them in one or more columns placed before the final 'prediction' column, never after it — 'prediction' must always be the last column.\n"
                "- Use only the dependencies listed in the output.\n"
                "- Do not change the selected experiment unless necessary to fix an implementation issue.\n"
                "- Both scripts must be executable directly from the command line.\n"
                "- Use argparse to parse command-line arguments.\n"
                "- The training script must support:\n"
                "    python train.py --input DATA.csv --output model.joblib\n"
                "- The prediction script must support:\n"
                "    python predict.py --model model.joblib --input DATA.csv --output predictions.csv\n"
                "- Include a standard `if __name__ == '__main__':` entry point in both scripts.\n"
                "- Do not include explanations, markdown, or code fences.\n"
                "- Return only valid JSON matching the Output schema."
            )

            if validation_error:
                prompt += (
                    "\n\nYour previous output failed schema validation:\n\n"
                    f"{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema.\n"
                )

            interaction = await traced_interactions_create(
                self.client,
                model=self.model,
                input=prompt,
                generation_config={
                    'thinking_level': 'low',
                    'temperature': 0
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': OutputScripts.model_json_schema()
                }
            )
            try:
                return OutputScripts.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     OutputScripts model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')

    @traceable(name="OutputAgent.repair_scripts")
    async def repair_scripts(self, scripts: OutputScripts, error_message: str, experiment: Experiment, dataset_analysis: DatasetAnalysis, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     repairing scripts...')

        validation_error = None
        for attempt in range(max_retries + 1):
            prompt = f"""
            Repair the previously generated deployment scripts.

            Failure:
            {error_message}

            Previous scripts:
            {scripts.model_dump_json(indent=2)}

            Selected experiment:
            {experiment.model_dump_json(indent=2)}

            Dataset analysis:
            {dataset_analysis.model_dump_json(indent=2)}

            Requirements:
            - Return the complete corrected OutputScripts.
            - Preserve the selected experiment, including preprocessing, feature selection, model architecture, and hyperparameters.
            - Fix the reported dependency, import, syntax, execution, serialization, or command-line interface error.
            - The training script must train the selected model on the full dataset.
            - The training script must support:
                python train.py --input DATA.csv --output model.joblib
            - The prediction script must support:
                python predict.py --model model.joblib --input DATA.csv --output predictions.csv
            - The prediction script must load the fitted model from the --model path.
            - The input CSV accepted by the prediction script must contain the columns listed in dataset analysis's feature_columns; drop any other columns present, if any.
            - The output CSV must be the same as the input CSV (after dropping irrelevant columns) with a single 'prediction' column appended as the very last column.
            - If the model supports probability prediction, output it in one or more columns placed before the final 'prediction' column, never after it — 'prediction' must always be the last column.
            - Ensure all required dependencies are listed.
            - Ensure both scripts are complete, directly executable, and use argparse.
            - Do not change the selected experiment unless necessary to fix an implementation issue.
            - Do not fabricate outputs, return a patch, include markdown, or add explanations.
            """

            if validation_error:
                prompt += (
                    "\n\nYour previous output failed schema validation:\n\n"
                    f"{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema.\n"
                )

            interaction = await traced_interactions_create(
                self.client,
                model=self.model,
                input=prompt,
                generation_config={
                    'thinking_level': 'low',
                    'temperature': 0
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': OutputScripts.model_json_schema()
                }
            )

            try:
                return OutputScripts.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     OutputScripts model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')

    async def validate_predict_script(self, env_python:str, predict_path: str, data_path: str, model_path: str, feature_columns: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_input = os.path.join(temp_dir, "test_input.csv")
            test_output = os.path.join(temp_dir, "test_output.csv")

            df = pd.read_csv(data_path)
            df = df.iloc[:5]
            df.to_csv(test_input, index=False)

            await asyncio.to_thread(
                subprocess.run,
                [env_python, predict_path, '--model', model_path, '--input', test_input, '--output', test_output],
                check=True,
                text=True,
                capture_output=True
            )
            df = pd.read_csv(test_output)

            output_columns = df.columns.tolist()
            assert output_columns and output_columns[-1] == 'prediction', (
                f"Expected 'prediction' to be the last output column, but the last column was "
                f"'{output_columns[-1] if output_columns else None}'. If probability columns are "
                f"included, they must come before 'prediction', not after. "
                f"Full output columns: {output_columns}"
            )

            # A probability column (or columns) is expected whenever the model supports
            # predict_proba, but must appear before the final 'prediction' column, not after.
            non_feature_columns = set(output_columns) - set(feature_columns) - {'prediction'}
            unexpected_columns = {c for c in non_feature_columns if not c.lower().startswith('prob')}
            assert not unexpected_columns, (
                f"Output CSV has unexpected columns not in feature_columns, 'prediction', or a "
                f"probability column: {sorted(unexpected_columns)}. Full output columns: {output_columns}"
            )

            missing_columns = set(feature_columns) - set(output_columns)
            assert not missing_columns, (
                f"Output CSV is missing expected feature columns: {sorted(missing_columns)}. "
                f"Full output columns: {output_columns}"
            )

    @traceable(name="OutputAgent.generate_summary")
    async def generate_summary(self, user_request: UserRequest,  experiments: list[Experiment]):
        best_experiment_idx = self.best_experiment(user_request.primary_metric, experiments, return_index=True)

        prompt = f"""
        Using the provided user_request, ordered list of experiments, and best_experiment_idx, generate a concise summary of the experimentation process.

        User request:
        {user_request}

        Experiments:
        {experiments}

        Best experiment index:
        {best_experiment_idx}

        Requirements:

        - Describe each experiment in chronological order.
        - For each experiment, briefly summarize the approach, important configuration changes, and results.
        - Explain how the experiments evolved based on earlier outcomes.
        - Assume best experiment index is zero-based, but refer to experiments using one-based numbering.
        - State why the experiment specified by best experiment index was the best based on the primary metric specified in user request.
        - Include the best experiment's primary metric value and any important tradeoffs or supporting metrics.
        - Do not invent missing information.
        - Keep the summary clear, factual, and concise.
        - Return only the final summary using markdown.
        """

        interaction = await traced_interactions_create(
            self.client,
            model=self.model,
            input=prompt,
            generation_config={
                'thinking_level': 'low',
                'temperature': 0
            }
        )

        return interaction.output_text

    @traceable(name="OutputAgent.generate_output")
    async def generate_output(self, primary_metric: Metric, dataset_analysis: DatasetAnalysis, experiments: list[Experiment], data_path: str, output_dir="out", max_retries=5):
        best_experiment = self.best_experiment(primary_metric, experiments)
        output_scripts = await self.generate_scripts(best_experiment, dataset_analysis)

        train_path = os.path.join(output_dir, 'train.py')
        predict_path = os.path.join(output_dir, 'predict.py')
        model_path = os.path.join(output_dir, 'model.joblib')
        metrics_path = os.path.join(output_dir, 'metrics.json')
        plan_path = os.path.join(output_dir, 'plan.json')
        requirements_path = os.path.join(output_dir, 'requirements.txt')
        
        for attempt in range(max_retries + 1):
            if self.verbose:
                print(f'{datetime.now()}     generating output...')
            try:
                with open(train_path, 'w') as f:
                    f.write(output_scripts.train_script)
                with open(predict_path, 'w') as f:
                    f.write(output_scripts.predict_script)
                with open(requirements_path, 'w') as f:
                    f.write('\n'.join(output_scripts.dependencies))
                with open(metrics_path, 'w') as f:
                    json.dump(best_experiment.result.model_dump(), f, indent=4)
                with open(plan_path, 'w') as f:
                    json.dump(best_experiment.plan.model_dump(), f, indent=4)

                # FOR DEBUGGING
                # with open(os.path.join(output_dir, 'experiments.json'), 'w') as f:
                #     json.dump([experiment.model_dump() for experiment in experiments], f, indent=4)
                # with open(os.path.join(output_dir, 'user_request.json'), 'w') as f:
                #     json.dump(user_request.model_dump(), f, indent=4)

                with tempfile.TemporaryDirectory() as env_dir:
                    await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, '-m', 'venv', '--clear', env_dir],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    if os.name == "nt":
                        env_python = os.path.join(env_dir, "Scripts", "python.exe")
                    else:
                        env_python = os.path.join(env_dir, "bin", "python")

                    if output_scripts.dependencies:
                        await asyncio.to_thread(
                            subprocess.run,
                            [env_python, "-m", "pip", "install", *output_scripts.dependencies],
                            check=True,
                            capture_output=True,
                            text=True
                        )

                    await asyncio.to_thread(
                        subprocess.run,
                        [env_python, train_path, '--input', data_path, '--output', model_path],
                        text=True,
                        check=True,
                        capture_output=True
                    )

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Training script did not produce {model_path}")

                    await self.validate_predict_script(env_python, predict_path, data_path, model_path, dataset_analysis.feature_columns)
                    
                    return
            except (subprocess.CalledProcessError, FileNotFoundError, AssertionError) as e:
                if attempt == max_retries:
                    shutil.rmtree(output_dir)
                    raise
                if self.verbose:
                    print(f'{datetime.now()}     output generation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
                if isinstance(e, subprocess.CalledProcessError):
                    error_message = (
                        f"Command failed: {e.cmd}\n"
                        f"Return code: {e.returncode}\n"
                        f"STDOUT:\n{e.stdout or ''}\n"
                        f"STDERR:\n{e.stderr or ''}"
                    )
                else:
                    error_message = str(e)
                output_scripts = await self.repair_scripts(output_scripts, error_message, best_experiment, dataset_analysis)

    