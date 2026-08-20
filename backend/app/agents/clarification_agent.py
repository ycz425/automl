from google import genai
from app.services.tracing import traced_interactions_create
from app.graph.schemas.clarifiable_model import ClarifiableModel
from langsmith import traceable
from pydantic import ValidationError, BaseModel
from datetime import datetime
import dotenv
import json
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class ClarificationAgent():
    def __init__(self, model = "gemini-3.1-flash-lite", verbose=False):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.verbose = verbose

    @traceable(name="ClarificationAgent.request_clarification")
    async def request_clarification(self, data: BaseModel, problems: list[str], interaction_id: str | None = None):
        prompt = f"""
        In plain language, ask concise question(s) to clarify ambiguities with the existing data based on its JSON schema and a list of detected problems.

        Existing data:
        {data.model_dump_json(indent=2)}

        JSON schema:
        {json.dumps(data.model_json_schema(), indent=2)}

        Detected problems:
        {'\n'.join(['- ' + problem for problem in problems])}

        If this continues a prior conversation, respond to user's most recent reply before writing the question. If it didn't resolve the detected problem(s), acknowledge it and explain specifically why it was insufficient, rather than repeating the same question unchanged.
        """

        interaction = await traced_interactions_create(
            self.client,
            model=self.model,
            input=prompt,
            previous_interaction_id=interaction_id,
            generation_config={
                'thinking_level': 'low',
                'temperature': 0
            }
        )

        return interaction.output_text, interaction.id
            

    @traceable(name="ClarificationAgent.apply_clarification")
    async def apply_clarification(self, data: ClarifiableModel, question: str, problems: list[str], clarification: str, max_retries=5):
        validation_error = None
        for attempt in range(max_retries + 1):
            prompt = f"""
            Update the existing data using the user's clarification.

            Existing data:
            {data.model_dump_json(indent=2)}

            Problems detected:
            {'\n'.join(['- ' + problem for problem in problems])}

            Clarification requested:
            {question}

            User clarification:
            {clarification}

            Requirements:
            - Update only the relevant field(s).
            - Preserve all unrelated fields exactly.
            - Interpret the user's clarification in the context of the clarification question and detected problems.
            - Do not invent information not provided by the user.
            - If the clarification is insufficient, leave the field unresolved.
            - Return only complete output matching the required schema.
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
                    'schema': data.model_json_schema()
                }
            )
            try:
                return data.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')

    

