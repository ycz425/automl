from app.graph.schemas.state import AutoMLState
from datetime import datetime

def data_router(state: AutoMLState):
    if state.clarification_request:
        if state.clarification_retries == 0:
            return 'block'
        else:
            print(f"{datetime.now()}     requesting clarification...")
            return 'request_clarification'
    else:
        return 'continue'