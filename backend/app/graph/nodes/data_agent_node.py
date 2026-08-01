from app.graph.schemas.state import AutoMLState
from app.agents.data_agent import DataAgent
from app.graph.schemas.clarification_request import ClarificationRequest
import pandas as pd
from datetime import datetime
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext


async def data_status_update_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    await runtime.context.status_store.update(runtime.execution_info.thread_id, status='running', node='data_agent', message="Analyzing dataset...")
    return {}


async def data_agent_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    if state.verbose:
        print(f'{datetime.now()} [DATA AGENT]')

    data_agent = DataAgent(verbose=state.verbose)

    df = pd.read_csv(await runtime.context.file_storage.get_dataset_path(state.dataset_id))

    dataset_profile = data_agent.profile_dataset(df)
    dataset_analysis_output = await data_agent.analyse_profile(state.user_request, dataset_profile)

    if dataset_analysis_output.clarification_request:
        stage = 'data'
        status = 'need_clarification'
        clarification_request = ClarificationRequest(source='dataset_analysis', request=dataset_analysis_output.clarification_request)
    else:
        stage = 'plan'
        status = 'running'
        clarification_request = None
        
    return {
        'status': status,
        'stage': stage,
        'dataset_profile': dataset_profile,
        'dataset_analysis': dataset_analysis_output.data,
        'clarification_request': clarification_request
    }
