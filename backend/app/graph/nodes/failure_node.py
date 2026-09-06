from app.graph.schemas.state import AutoMLState
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext
from datetime import datetime


async def failure_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [FAILURE]')
    message = (
        'Could not get the information needed to continue: ' + '; '.join(state.problems)
        if state.problems else 'Failed.'
    )
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='failed', message=message)
    return {}