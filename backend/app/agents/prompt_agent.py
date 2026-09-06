from app.agents.base import LLMAgent
from app.graph.schemas.user_request import UserRequest
from langsmith import traceable
from datetime import datetime


class PromptAgent(LLMAgent):
    @traceable(name="PromptAgent.parse")
    async def parse(self, instruction, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     parsing user request...')

        prompt = f"""
        You are the Prompt Agent in an AutoML system.

        Your job is to convert the user's request into a structured UserRequest.

        User request:
        {instruction}

        Instructions:
        - Extract only information explicitly stated by the user.
        - Do not infer dataset columns or invent missing details.
        - Leave fields null if they are unknown.
        - Keep assumptions to an absolute minimum.
        - Produce only valid JSON matching the provided schema.
        """

        return await self.generate_structured(prompt, UserRequest, max_retries=max_retries, label="UserRequest")
