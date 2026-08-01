from app.graph.schemas.state import AutoMLState
from langgraph.types import interrupt
from datetime import datetime
from app.agents.clarification_agent import ClarificationAgent
from app.graph.schemas.user_request import prompt_parse_output
from app.graph.schemas.data_info import dataset_analysis_output
from app.graph.schemas.clarification_request import ClarificationRequest
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext


async def clarification_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(
        runtime.execution_info.thread_id,
        status='need_clarification',
        message=state.clarification_request.request.question
    )
    
    return {}


async def clarification_agent_node(state: AutoMLState):
    user_clarification = interrupt(state.clarification_request)

    if state.verbose:
        print(f'{datetime.now()}     clarifying... (attempts left: {state.clarification_retries})')
    clarification_agent = ClarificationAgent(verbose=state.verbose)

    source = state.clarification_request.source

    if  source == 'user_request':
        data = state.user_request
        output_model = prompt_parse_output
    else:
        data = state.dataset_analysis
        output_model = dataset_analysis_output

    clarification_output = await clarification_agent.clarify(
        data,
        state.clarification_request,
        user_clarification,
        output_model
    )

    if clarification_output.clarification_request:
        new_clarification_request = ClarificationRequest(source=source, request=clarification_output.clarification_request)
    else:
        new_clarification_request = None

    return {
        source: clarification_output.data,
        'clarification_request': new_clarification_request,
        'clarification_retries': state.clarification_retries - 1
    }