from google import genai
from app.graph.schemas.user_request import prompt_parse_output
from app.services.tracing import traced_interactions_create
from langsmith import traceable
from pydantic import ValidationError
from datetime import datetime
import dotenv
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class PromptAgent():
    def __init__(self, model = "gemini-3.1-flash-lite", verbose=False):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.verbose = verbose

    @traceable(name="PromptAgent.parse")
    async def parse(self, instruction, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     parsing user request...')

        validation_error = None

        for attempt in range(max_retries + 1):
            prompt = f"""
            You are the Prompt Agent in an AutoML system.

            Your job is to convert the user's request into a structured UserRequest.

            User request:
            {instruction}

            Instructions:
            - Extract only information explicitly stated by the user.
            - Do not infer dataset columns or invent missing details.
            - Leave fields null if they are unknown.
            - If required information is missing to complete the request, return clarification questions.
            - Keep assumptions to an absolute minimum.
            - Produce only valid JSON matching the provided schema.
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
                    'schema': prompt_parse_output.model_json_schema()
                }
            )
            try:
                return prompt_parse_output.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
    