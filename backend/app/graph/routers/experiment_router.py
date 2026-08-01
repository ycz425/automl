from app.graph.schemas.state import AutoMLState

def experiment_router(state: AutoMLState):
    if state.max_replans == 0:
        return 'output'
    else:
        return 'replan'