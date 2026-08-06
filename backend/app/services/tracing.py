from langsmith import traceable


@traceable(run_type="llm", name="gemini.interactions.create")
async def traced_interactions_create(client, **kwargs):
    return await client.aio.interactions.create(**kwargs)
