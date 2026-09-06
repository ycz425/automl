from app.agents.base import LLMAgent
from app.services.tracing import traced_interactions_create
from app.graph.schemas.clarifiable_model import ClarifiableModel
from langsmith import traceable
from pydantic import BaseModel
import json


class ClarificationAgent(LLMAgent):
    @traceable(name="ClarificationAgent.request_clarification")
    async def request_clarification(self, data: BaseModel, problems: list[str], interaction_id: str | None = None):
        prompt = f"""
        In plain language, ask concise question(s) to clarify ambiguities with the existing data based on its JSON schema and a list of detected problems.
        If appropriate, recommend choice(s) to the user that is the most fitting for the current setup with brief explanations.

        The JSON schema and detected problems below are internal implementation details for your understanding only — the user never sees them. Never mention field names, schema terms, or the literal wording of a detected problem (e.g. "task_type", "primary_metric", "evaluation_method is unspecified") in the question. Translate each one into the plain-language concept it represents (e.g. ask what kind of prediction they want to make, or how they'd like the model's performance measured) as something a non-technical user would understand.

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

        return await self.generate_structured(prompt, type(data), max_retries=max_retries, label=type(data).__name__)
