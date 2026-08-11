from pydantic import BaseModel
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.plan import Plan
from app.graph.schemas.experiment import Experiment
from app.graph.schemas.clarification_request import ClarificationRequest


class AutoMLState(BaseModel):
    user_input: str
    dataset_id: str
    verbose: bool = False

    user_request: UserRequest | None = None
    
    dataset_profile: DatasetProfile | None = None
    dataset_analysis: DatasetAnalysis | None = None

    split_path: str | None = None

    plan: Plan | None = None

    experiments: list[Experiment] = []

    clarification_request: ClarificationRequest | None = None
    clarification_retries: int = 10
    max_replans: int = 3

    summary: str | None = None
