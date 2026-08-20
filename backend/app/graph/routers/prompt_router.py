from app.graph.schemas.state import AutoMLState
from datetime import datetime

def prompt_router(state: AutoMLState):
    if state.problems and state.pending_clarification:
        if state.clarification_retries == 0:
            return 'block'
        else:
            return 'request_clarification'
    else:
        return 'continue'