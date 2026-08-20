from langsmith import traceable
from google.genai import Client


@traceable(run_type="llm", name="gemini.interactions.create")
async def traced_interactions_create(client: Client, **kwargs):
    return await client.aio.interactions.create(**kwargs)
