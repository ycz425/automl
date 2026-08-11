from app.graph.schemas.state import AutoMLState
from app.agents.plan_agent import PlanAgent
from datetime import datetime
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext


async def plan_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='running', node='plan_agent', message="Planning...")
    return {}


async def plan_agent_node(state: AutoMLState):
    if state.verbose:
        print(f'{datetime.now()} [PLAN AGENT] (attempts left: {state.max_replans})')
    plan_agent = PlanAgent(verbose=state.verbose)

    plan = await plan_agent.plan(state.user_request, state.dataset_profile, state.dataset_analysis, state.experiments)

    return {
        'plan': plan
    }