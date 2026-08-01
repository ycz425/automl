from app.graph.schemas.state import AutoMLState
from app.agents.prompt_agent import PromptAgent
from app.graph.schemas.clarification_request import ClarificationRequest
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext
from datetime import datetime

async def prompt_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='running', node='prompt_agent', message="Parsing request...")
    return {}


async def prompt_agent_node(state: AutoMLState):
    if state.verbose:
        print(f'{datetime.now()} [PROMPT AGENT]')
    
    prompt_agent = PromptAgent(verbose=state.verbose)
    prompt_parse_output = await prompt_agent.parse(state.user_input)

    if prompt_parse_output.clarification_request:
        clarification_request = ClarificationRequest(source='user_request', request=prompt_parse_output.clarification_request)
    else:
        clarification_request = None
    
    return {
        'user_request': prompt_parse_output.data,
        'clarification_request': clarification_request
    }
