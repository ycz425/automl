from pydantic import BaseModel
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from app.graph.schemas.plan import Plan
from app.graph.schemas.experiment import Experiment
from typing import Literal


type ClarifiableField = Literal[
    'user_request',
    'dataset_analysis'
]


class AutoMLState(BaseModel):
    user_input: str
    dataset_id: str
    verbose: bool = False

    gemini_interaction_id: str | None = None

    user_request: UserRequest | None = None
    
    dataset_profile: DatasetProfile | None = None
    dataset_analysis: DatasetAnalysis | None = None

    split_path: str | None = None

    plan: Plan | None = None

    experiments: list[Experiment] = []

    pending_clarification: ClarifiableField | None = None
    problems: list[str] = []
    clarification_request: str | None = None
    clarification_retries: int = 10
    max_replans: int = 3

    summary: str | None = None
