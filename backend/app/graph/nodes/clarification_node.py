from app.graph.schemas.state import AutoMLState
from langgraph.types import interrupt
from datetime import datetime
from app.agents.clarification_agent import ClarificationAgent
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext


async def clarification_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(
        runtime.execution_info.thread_id,
        status='need_clarification',
        message=state.clarification_request
    )
    return {}


async def request_clarification_node(state: AutoMLState):
    print(f"{datetime.now()}     requesting clarification...")
    clarification_agent = ClarificationAgent(verbose=state.verbose)
    request, interaction_id = await clarification_agent.request_clarification(
        getattr(state, state.pending_clarification),
        state.problems,
        state.gemini_interaction_id
    )
    return {
        'clarification_request': request,
        'gemini_interaction_id': interaction_id
    }


async def apply_clarification_node(state: AutoMLState):
    user_clarification = interrupt(state.clarification_request)

    if state.verbose:
        print(f'{datetime.now()}     clarifying... (attempts left: {state.clarification_retries})')
    clarification_agent = ClarificationAgent(verbose=state.verbose)

    clarified_data = await clarification_agent.apply_clarification(
        getattr(state, state.pending_clarification),
        state.clarification_request,
        state.problems,
        user_clarification,
    )

    problems = clarified_data.problems()

    return {
        state.pending_clarification: clarified_data,
        'pending_clarification': state.pending_clarification if problems else None,
        'problems': problems,
        'clarification_request': None,
        'clarification_retries': state.clarification_retries - 1
    }
        
    