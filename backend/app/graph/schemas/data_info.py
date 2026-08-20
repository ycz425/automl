from pydantic import BaseModel, Field
from typing import Any, Literal
from app.graph.schemas.clarifiable_model import ClarifiableModel
from app.graph.schemas.user_request import UserRequest

type ClarifiableField = Literal[
    'target_column',
    'group_column'
]

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
    num_unique: int = Field(ge=0)
    num_missing: int = Field(ge=0)
    missing_rate: float | None =  Field(ge=0.0, le=1.0)


class DatasetProfile(BaseModel):
    num_columns: int = Field(ge=0)
    num_rows: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)


class DatasetAnalysis(ClarifiableModel):
    target_column: str | None = Field(default=None, description="Column that most likely represents the prediction target. Use null when it cannot be confidently determined from the column name.")
    feature_columns: list[str] = Field(default_factory=list, description="Columns considered eligible as model inputs after applying the user's inclusion and exclusion requirements.")
    excluded_columns: list[str] = Field(default_factory=list, description="Columns that should not be used as model inputs, such as identifiers, leakage variables, or user-excluded columns.")
    group_column: str | None = Field(default=None, description="Column defining groups that must remain entirely within one split, such as subject, patient, or recording ID. Return null if user request does not specify any group or it cannot be confidently determined from the column name.")

    def problems(self, user_request: UserRequest):
        problems = []
        if self.target_column is None:
            problems.append("target_column cannot be determined based on dataset column names.")

        if user_request.group_description is not None and self.group_column is None:
            problems.append("group_column cannot be determined based on dataset column names")
        return problems
            


class Split(BaseModel):
    train_idx: list[int] = Field(description="Positional row indices used for training in this fold. Use with pandas .iloc.")
    val_idx: list[int] = Field(description="Positional row indices used for validation in this fold. Use with pandas .iloc.")

class DataSplits(BaseModel):
    splits: list[Split] = Field(default_factory=list, description="Cross-validation folds. Contains only one fold if evaluation_method is train_validation_split")
