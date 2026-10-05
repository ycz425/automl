from app.agents.base import LLMAgent
from app.services.tracing import traced_interactions_create
from app.graph.schemas.clarification import ClarifiableModel, Clarifications
from langsmith import traceable
from pydantic import BaseModel, ValidationError
from datetime import datetime
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
                'thinking_level': 'low'
            }
        )

        return interaction.output_text, interaction.id

    @traceable(name="ClarificationAgent.apply_clarification")
    async def apply_clarification[T: ClarifiableModel](self, data: T, question: str, problems: list[str], clarification: str, max_retries=5, max_merge_retries=2) -> T:
        prompt = f"""
        Determine which field(s) of the existing data the user's clarification changes, and the new value of each.

        Existing data:
        {data.model_dump_json(indent=2)}

        Problems detected:
        {'\n'.join(['- ' + problem for problem in problems])}

        Clarification requested:
        {question}

        User clarification:
        {clarification}

        Requirements:
        - Return an entry only for each field that the user's clarification changes. Fields that are not returned keep their existing values, so never return a field just to repeat its current value.
        - Each new value replaces the field's whole existing value and must have the same shape as it (e.g. a complete list for a list field).
        - Interpret the user's clarification in the context of the clarification question and detected problems.
        - Do not invent information not provided by the user.
        - If the clarification is insufficient, return no entries.
        """

        schema = Clarifications[type(data).clarifiable_fields]
        merge_error: str | None = None
        for attempt in range(max_merge_retries + 1):
            full_prompt = prompt
            if merge_error:
                full_prompt += (
                    "\nYour previous update produced invalid data when applied to the existing data:\n"
                    f"{merge_error}\n"
                    "Return corrected entries whose new values match the field types.\n"
                )

            patch = await self.generate_structured(full_prompt, schema, max_retries=max_retries, label=type(data).__name__)
            updates = {c.field: c.new_value for c in patch.clarifications}

            try:
                return type(data).model_validate({**data.model_dump(), **updates})
            except ValidationError as e:
                if attempt == max_merge_retries:
                    raise
                merge_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     clarification update invalid - retrying... (attempt: {attempt + 1}/{max_merge_retries})')
