from google import genai
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.plan import Plan
from app.graph.schemas.experiment import Experiment
from app.services.tracing import traced_interactions_create
from langsmith import traceable
from pydantic import ValidationError
from datetime import datetime
import json
import dotenv
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class PlanAgent():
    def __init__(self, model = "gemini-3.1-flash-lite", verbose=False):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.verbose = verbose

    @traceable(name="PlanAgent.plan")
    async def plan(self, user_request: UserRequest, dataset_profile: DatasetProfile, dataset_analysis: DatasetAnalysis, experiments: list[Experiment], max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     planning...')
        
        validation_error = None
        
        for attempt in range(max_retries + 1):
            prompt = (
                "Create one executable machine-learning experiment plan.\n\n"
                "The plan must:\n"
                "- satisfy the user's requirements and constraints;\n"
                "- use only columns identified in the dataset analysis;\n"
                "- choose appropriate preprocessing, architecture, training, and evaluation;\n"
                "- avoid inventing dataset columns;\n"
                "- use null for training settings that do not apply;\n"
                "- state any necessary assumptions.\n\n"
                "User request:\n"
                f"{user_request.model_dump_json(indent=2)}\n\n"
                "Dataset profile:\n"
                f"{dataset_profile.model_dump_json(indent=2)}\n\n"
                "Dataset analysis:\n"
                f"{dataset_analysis.model_dump_json(indent=2)}"
            )

            if experiments:
                prompt += (
                    "\n\nPrevious experiment history:\n"
                    f"{json.dumps([experiment.model_dump(mode="json", include={'plan', 'result'}) for experiment in experiments], indent=2)}\n\n"
                    "This is a plan revision, not an initial plan.\n"
                    "- Learn from the previous experiment results.\n"
                    "- Identify what likely limited performance.\n"
                    "- Preserve components that worked well.\n"
                    "- Make concrete changes intended to improve the primary metric.\n"
                    "- Do not repeat a previous plan unless there is a clear justification.\n"
                    "- Respect all original user constraints."
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
                    "thinking_level": "low",
                    "temperature": 0,
                },
                response_format={
                    "mime_type": "application/json",
                    "schema": Plan.model_json_schema(),
                },
            )

            try:
                return Plan.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     Plan model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')