from google import genai
from app.graph.schemas.clarification_request import ClarificationRequest
from app.services.tracing import traced_interactions_create
from langsmith import traceable
from pydantic import ValidationError, BaseModel
from app.graph.schemas.llm_output import LLMOutput
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

    @traceable(name="ClarificationAgent.clarify")
    async def clarify(self, data: BaseModel, clarification_request: ClarificationRequest, user_clarification: str, output_model: type[LLMOutput], max_retries=5):
        validation_error = None
        for attempt in range(max_retries + 1):
            prompt = f"""
            Update the existing data using the user's clarification.

            Existing data:
            {data.model_dump_json(indent=2)}

            Clarification requested:
            {clarification_request.request.model_dump_json(indent=2)}

            User clarification:
            {user_clarification}

            Requirements:
            - Update only the field(s) identified by clarification_request.fields.
            - Preserve all unrelated fields exactly.
            - Interpret the user's clarification in the context of the clarification question and reason.
            - Do not invent information not provided by the user.
            - If the clarification is insufficient, leave the field unresolved and request a more specific clarification.
            - Request further clarification if any other clarifiable fields remain undetermined.
            - Judge if a field is undetermined based on existing value and field description in the following JSON schema:
            {json.dumps(data.model_json_schema(), indent=2)}
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
                    'schema': output_model.model_json_schema()
                }
            )
            try:
                return output_model.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')

    

