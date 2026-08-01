from pydantic import BaseModel, Field
from typing import Any, Literal
from app.graph.schemas.llm_output import LLMOutput

type ClarifiableField = Literal['target_column']

type FeatureType = Literal[
    'numerical',
    'categorical',
    'boolean',
    'datetime',
    'text',
    'identifier',
]

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    feature_type: FeatureType | None
    category_counts: dict[Any, int] | None
    mean: float| None
    std: float | None
    num_missing: int = Field(ge=0)
    missing_rate: float | None =  Field(ge=0.0, le=1.0)


class DatasetProfile(BaseModel):
    num_columns: int = Field(ge=0)
    num_rows: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)


class DatasetAnalysis(BaseModel):
    target_column: str | None = Field(default=None, description="Column that most likely represents the prediction target. Use null when it cannot be confidently determined from the column name.")
    feature_columns: list[str] = Field(default_factory=list, description="Columns considered eligible as model inputs after applying the user's inclusion and exclusion requirements.")
    excluded_columns: list[str] = Field(default_factory=list, description="Columns that should not be used as model inputs, such as identifiers, leakage variables, or user-excluded columns.")
    issues: list[str] = Field(default_factory=list, description='Potential dataset problems requiring attention, such as an unclear target, leakage risk, or unsuitable column types.')
    ambiguity_notes: list[str] = Field(default_factory=list, description='Important information that is missing, unclear, or conflicting.')


dataset_analysis_output = LLMOutput[DatasetAnalysis, ClarifiableField]