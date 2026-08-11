from app.graph.schemas.state import AutoMLState
from app.agents.experiment_agent import ExperimentAgent
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext
from datetime import datetime


async def experiment_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='running', node='experiment_agent', message="Running experiment...")
    return {}

async def experiment_agent_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [EXPERIMENT AGENT]  (attempts left: {state.max_replans})')
    experiment_agent = ExperimentAgent(verbose=state.verbose)
    experiment = await experiment_agent.execute_plan(
        str(await runtime.context.file_storage.get_dataset_path(state.dataset_id)),
        str(state.split_path),
        state.user_request,
        state.dataset_profile,
        state.dataset_analysis,
        state.plan
    )

    return {
        'experiments': state.experiments + [experiment],
        'max_replans': state.max_replans - 1
    }

    