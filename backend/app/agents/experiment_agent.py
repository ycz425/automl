from google import genai
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.plan import Plan
from app.graph.schemas.experiment import ExperimentImplementation, ExperimentResult, Experiment
from pydantic import ValidationError
import tempfile
from pathlib import Path
from datetime import datetime
import asyncio
import json
import subprocess
import sys
import dotenv
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class ExperimentAgent():
    def __init__(self, model = "gemini-3.1-flash-lite", verbose=False):
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.model = model
            self.verbose = verbose

    async def generate_implementation(self, data_path, user_request: UserRequest, dataset_profile: DatasetProfile, dataset_analysis: DatasetAnalysis, plan: Plan, output_path: str, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     generating experiment implementation...')

        validation_error = None
        for attempt in range(max_retries + 1):
            prompt = f"""
            Generate executable Python code for the machine-learning experiment below.

            Requirements:
            - Follow the experiment plan exactly.
            - Use only columns listed in the dataset analysis.
            - Do not invent column names.
            - Handle missing values and feature types described in the dataset profile.
            - Read the dataset from: {data_path}
            - Save the final experiment result to:
                {output_path}
            - Make experiment_result.json conform exactly to this JSON schema:
            {json.dumps(ExperimentResult.model_json_schema(), indent=2)}
            - Use exactly the metric names specified by the user request.
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
            
            if validation_error:
                prompt += (
                    "\n\nYour previous output failed schema validation:\n\n"
                    f"{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema.\n"
                )

            interaction = await self.client.aio.interactions.create(
                model=self.model,
                input=prompt,
                generation_config={
                    'thinking_level': 'low',
                    'temperature': 0
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': ExperimentImplementation.model_json_schema()
                }
            )

            try:
                return ExperimentImplementation.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     ExperimentImplementation model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')

    
    async def repair_implementation(self, implementation: ExperimentImplementation, error_message: str, plan: Plan, output_path: str, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     repairing experiment implementation...')
        
        validation_error = None
        for attempt in range(max_retries + 1):
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
                - Do not hide errors or fabricate successful results.
                - Do not return a patch.
            """
            
            if validation_error:
                prompt += (
                    "\n\nYour previous output failed schema validation:\n\n"
                    f"{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema.\n"
                )

            interaction = await self.client.aio.interactions.create(
                model=self.model,
                input=prompt,
                generation_config={
                    'thinking_level': 'low',
                    'temperature': 0
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': ExperimentImplementation.model_json_schema()
                }
            )

            try:
                return ExperimentImplementation.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     ExperimentImplementation model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
    
    async def execute_plan(self, data_path: str, user_request: UserRequest, dataset_profile: DatasetProfile, dataset_analysis: DatasetAnalysis, plan: Plan, max_retries=5):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = Path(temp_dir) / ".venv"
            code_path = Path(temp_dir) / "code.py"
            output_path = Path(temp_dir) / "experiment_result.json"

            implementation = await self.generate_implementation(data_path, user_request, dataset_profile, dataset_analysis, plan, output_path)

            for attempt in range(max_retries + 1):
                if self.verbose:
                    print(f'{datetime.now()}     running experiment...')
                try:
                    output_path.unlink(missing_ok=True)

                    await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, "-m", "venv", "--clear", str(env_dir)],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    if os.name == "nt":
                        env_python = env_dir / "Scripts" / "python.exe"
                    else:
                        env_python = env_dir / "bin" / "python"

                    if implementation.dependencies:
                        await asyncio.to_thread(
                            subprocess.run,
                            [str(env_python), "-m", "pip", "install", *implementation.dependencies],
                            capture_output=True,
                            check=True,
                            text=True
                        )

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
                    if isinstance(e, subprocess.CalledProcessError):
                        error_message = (
                            f"Command failed: {e.cmd}\n"
                            f"Return code: {e.returncode}\n"
                            f"STDOUT:\n{e.stdout or ''}\n"
                            f"STDERR:\n{e.stderr or ''}"
                        )
                    else:
                        error_message = str(e)
                    implementation = await self.repair_implementation(implementation, error_message, plan, str(output_path))
    