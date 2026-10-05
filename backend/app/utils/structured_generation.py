from datetime import datetime
from typing import TypeVar
from google.genai import Client
from google.genai._gaos.lib.compat_errors import BadRequestError
from pydantic import BaseModel, ValidationError
from app.services.tracing import traced_interactions_create

T = TypeVar("T", bound=BaseModel)


async def generate_structured(
    client: Client,
    model: str,
    prompt: str,
    schema: type[T],
    max_retries: int = 5,
    max_output_tokens: int = 2 ** 14,
    verbose: bool = False,
    label: str = "",
) -> T:
    validation_error: str | None = None
    for attempt in range(max_retries + 1):
        full_prompt = prompt
        if validation_error:
            full_prompt += (
                "\n\nYour previous output failed schema validation:\n\n"
                f"{validation_error}\n\n"
                "Return a corrected response that strictly matches the required schema.\n"
            )

        try:
            interaction = await traced_interactions_create(
                client,
                model=model,
                input=full_prompt,
                generation_config={
                    'thinking_level': 'low',
                    'max_output_tokens': max_output_tokens
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': schema.model_json_schema(),
                },
            )
        except BadRequestError as e:
            # Gemini's copyright filter occasionally blocks a legitimate response (e.g. boilerplate code).
            # Sampling is random, so a retry usually succeeds; any other 400 is a real request error.
            if 'recitation' not in str(e).lower() or attempt == max_retries:
                raise
            if verbose:
                print(f'{datetime.now()}     {label} response blocked by recitation filter - retrying... (attempt: {attempt + 1}/{max_retries})')
            continue

        try:
            return schema.model_validate_json(interaction.output_text)
        except ValidationError as e:
            if attempt == max_retries:
                raise
            validation_error = str(e)
            if verbose:
                print(f'{datetime.now()}     {label} model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
