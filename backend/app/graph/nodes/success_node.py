from app.graph.schemas.state import AutoMLState
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext
from datetime import datetime

async def success_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [SUCCESS]')
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='completed', message=state.summary)
    return {}