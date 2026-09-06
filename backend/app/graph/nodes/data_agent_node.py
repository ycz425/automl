from app.graph.schemas.state import AutoMLState
from app.agents.data_agent import DataAgent
import pandas as pd
from datetime import datetime
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext, report_status


async def data_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await report_status(runtime, status='running', node='data_agent', message="Analyzing dataset...")
    return {}


async def data_agent_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [DATA AGENT]')

    data_agent = DataAgent(verbose=state.verbose)

    df = pd.read_csv(await runtime.context.file_storage.get_dataset_path(state.dataset_id))

    dataset_profile = data_agent.profile_dataset(df)
    dataset_analysis= await data_agent.analyse_profile(state.user_request, dataset_profile)


    problems = dataset_analysis.problems(state.user_request)

    return {
        'dataset_profile': dataset_profile,
        'dataset_analysis': dataset_analysis,
        'problems': problems,
        'pending_clarification': 'dataset_analysis' if problems else None
    }
