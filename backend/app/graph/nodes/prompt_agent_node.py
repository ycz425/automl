from app.graph.schemas.state import AutoMLState
from app.agents.prompt_agent import PromptAgent
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext, report_status
from datetime import datetime

async def prompt_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await report_status(runtime, status='running', node='prompt_agent', message="Parsing request...")
    return {}


async def prompt_agent_node(state: AutoMLState):
    if state.verbose:
        print(f'{datetime.now()} [PROMPT AGENT]')
    
    prompt_agent = PromptAgent(verbose=state.verbose)
    user_request = await prompt_agent.parse(state.user_input)

    problems = user_request.problems()

    return {
        'user_request': user_request,
        'problems': problems,
        'pending_clarification': 'user_request' if problems else None
    }