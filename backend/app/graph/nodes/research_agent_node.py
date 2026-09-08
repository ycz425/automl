from langgraph.runtime import Runtime
from app.graph.schemas.state import AutoMLState
from app.graph.context import AutoMLContext, report_status
from app.agents.research_agent import ResearchAgent
from datetime import datetime


async def research_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await report_status(runtime, status='running', node='research_agent', message="Researching...")
    return {}


def research_agent_node(state: AutoMLState):
    if state.verbose:
        print(f'{datetime.now()} [RESEARCH AGENT]')

    research_agent = ResearchAgent(verbose=state.verbose)
    search_results = research_agent.search(state.user_request, state.dataset_profile, state.dataset_analysis)

    return {
        'research': search_results
    }
