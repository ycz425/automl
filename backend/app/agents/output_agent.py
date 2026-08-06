from google import genai
from app.graph.schemas.user_request import UserRequest, Metric
from app.graph.schemas.experiment import Experiment
from app.graph.schemas.output import OutputScripts
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
    async def generate_scripts(self, user_request: UserRequest, experiment: Experiment, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     generating scripts...')

        validation_error = None
        for attempt in range(max_retries + 1):
            prompt = (
                "You are generating the final deployment artifacts for an AutoML pipeline.\n\n"
                "The experiment below has already been selected as the best-performing experiment.\n\n"
                "Your task is to generate production-ready scripts that reproduce this experiment.\n\n"
                "User request:\n"
                f"{user_request.model_dump_json(indent=2)}\n\n"
                "Selected experiment:\n"
                f"{experiment.model_dump_json(indent=2)}\n\n"
                "Requirements:\n\n"
                "- Produce output matching the Output schema exactly.\n"
                "- The training script must train the selected model on the full dataset.\n"
                "- Preserve the preprocessing, feature selection, model architecture, and hyperparameters from the selected experiment.\n"
                f"- The training script must save the complete fitted pipeline as a joblib file to the output path provided as a command-line argument.\n"
                f"- The prediction script must load the fitted pipeline from the model joblib path provided as a command-line argument, accept unseen CSV data, and write predictions to a CSV file.\n"
                "- If the model supports probabilities, also output predicted probabilities.\n"
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
    async def repair_scripts(self, scripts: OutputScripts, error_message: str, experiment: Experiment, max_retries=5):
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
            - If the model supports probability prediction, also output predicted probabilities.
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

    async def validate_predict_script(self, env_python:str, predict_path: str, data_path: str, model_path: str):
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
            pd.read_csv(test_output)

    @traceable(name="OutputAgent.generate_summary")
    async def generate_summary(self, user_request: UserRequest, experiments: list[Experiment]):
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
    async def generate_output(self, user_request: UserRequest, experiments: list[Experiment], data_path: str, output_dir="out", max_retries=5):
        best_experiment = self.best_experiment(user_request.primary_metric, experiments)
        output_scripts = await self.generate_scripts(user_request, best_experiment)

        train_path = os.path.join(output_dir, 'train.py')
        predict_path = os.path.join(output_dir, 'predict.py')
        model_path = os.path.join(output_dir, 'model.joblib')
        metrics_path = os.path.join(output_dir, 'metrics.json')
        plan_path = os.path.join(output_dir, 'plan.json')
        requirements_path = os.path.join(output_dir, 'requirements.txt')
        
        for attempt in range(max_retries + 1):
            if self.verbose:
                print(f'{datetime.now()}     generating output...')
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
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

                    await self.validate_predict_script(env_python, predict_path, data_path, model_path)
                    
                    return
            except subprocess.CalledProcessError as e:
                if attempt == max_retries:
                    shutil.rmtree(output_dir)
                    raise
                if self.verbose:
                    print(f'{datetime.now()}     output generation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
                error_message = (
                    f"Command failed: {e.cmd}\n"
                    f"Return code: {e.returncode}\n"
                    f"STDOUT:\n{e.stdout or ''}\n"
                    f"STDERR:\n{e.stderr or ''}"
                )
                output_scripts = await self.repair_scripts(output_scripts, error_message, best_experiment)

    