from app.graph.schemas.state import AutoMLState
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext


async def clarification_router(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.problems and state.pending_clarification:
        if state.clarification_retries == 0:
            return 'block'
        else:
            return 'request_clarification'
    else:
        node = await runtime.context.status_store.get_node(runtime.execution_info.thread_id)
        if node == 'prompt_agent':
            return 'continue_data'
        elif node == 'data_agent':
            return 'continue_plan'