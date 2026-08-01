from app.graph.schemas.state import AutoMLState
from app.agents.output_agent import OutputAgent
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext
from datetime import datetime


async def output_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='running', node='output_agent', message='Generating artifacts and final summary...')
    return {}


async def output_agent_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [OUTPUT AGENT]')
    output_agent = OutputAgent(verbose=state.verbose)
    await output_agent.generate_output(
        state.user_request,
        state.experiments,
        str(await runtime.context.file_storage.get_dataset_path(state.dataset_id)),
        output_dir=str(await runtime.context.file_storage.get_run_directory(runtime.execution_info.thread_id))
    )

    return {
        'summary': await output_agent.generate_summary(state.user_request, state.experiments)
    }