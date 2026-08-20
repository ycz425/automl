from app.graph.schemas.state import AutoMLState
from app.agents.data_agent import DataAgent
import pandas as pd
from langgraph.runtime import Runtime
from app.graph.context import AutoMLContext
import json

async def data_splitting_node(state: AutoMLState, runtime: Runtime[AutoMLContext]):
    data_agent = DataAgent(verbose=state.verbose)
    df = pd.read_csv(await runtime.context.file_storage.get_dataset_path(state.dataset_id))
    splits = data_agent.create_split_index(df, state.user_request, state.dataset_analysis)
    split_path = str((await runtime.context.file_storage.get_run_directory(runtime.execution_info.thread_id)) / 'splits.json')
    with open(split_path, 'w') as f:
        json.dump(splits.model_dump(), f, indent=4)

    return {
        'split_path': split_path
    }