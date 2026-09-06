from typing import TypeVar
from google import genai
from pydantic import BaseModel
from app.utils.structured_generation import generate_structured
import dotenv
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

T = TypeVar("T", bound=BaseModel)


class LLMAgent:
    def __init__(self, model: str = "gemini-3.1-flash-lite", verbose: bool = False):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.verbose = verbose

    async def generate_structured(self, prompt: str, schema: type[T], max_retries: int = 5, label: str = "") -> T:
        return await generate_structured(
            self.client,
            self.model,
            prompt,
            schema,
            max_retries=max_retries,
            verbose=self.verbose,
            label=label,
        )
