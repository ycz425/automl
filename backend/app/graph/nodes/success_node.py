from app.graph.schemas.state import AutoMLState
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext, report_status
from datetime import datetime

async def success_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [SUCCESS]')
    await report_status(runtime, status='completed', message=state.summary)
    return {}