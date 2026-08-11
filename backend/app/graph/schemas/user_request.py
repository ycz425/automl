from pydantic import BaseModel, Field
from typing import Literal
from app.graph.schemas.llm_output import LLMOutput

type ClarifiableField = Literal[
    'task_type',
    'target_description',
    'include_features',
    'exclude_features',
    'primary_metric',
    'evaluation_method'
]

type TaskType = Literal[
    'binary_classification',
    'multiclass_classification',
    'regression'
]

type EvalMethod = Literal[
    'train_validation_split',
    'k_fold_cross_validation',
    'leave_one_group_out'
]


class Metric(BaseModel):
    name: str = Field(description="Name of the evaluation metric, such as 'accuracy', 'f1', 'precision', 'recall', 'auroc', 'rmse', 'mae', or 'r2'.")
    direction: Literal['max', 'min'] = Field(description="Direction of optimization for the metric.") 


class UserRequest(BaseModel):
    task_type: TaskType | None = Field(default=None, description="The machine-learning task type inferred from the request. Return null and request clarification if it cannot be determined.")
    target_description: str | None = Field(default=None, description="Plain-language description of what the model should predict. Return null and request clarification if target cannot be determined.")
    include_features: list[str] = Field(default_factory=list, description='Features the user explicitly requires. Return an empty list when none are specified. Request clarification if it overlaps with exclude_features.')
    exclude_features: list[str] = Field(default_factory=list, description='Features the user explicitly prohibits. Return an empty list when none are specified. Request clarification if it overlaps with include_features.')
    constraints: list[str] = Field(default_factory=list, description='Explicit requirements affecting data splitting, modeling, evaluation, compute, interpretability, or deployment. Return an empty list if user does not specify any constraints.')
    preferences: list[str] = Field(default_factory=list, description='Non-mandatory preferences affecting data splitting, modeling, evaluation, compute, interpretability, or deployment. Return an empty list if user does not specify any preferences.')

    primary_metric: Metric | None = Field(default=None, description="Main metric used to compare experiments, select checkpoints, and decide whether an experiment improved. Return null and request clarification if it cannot be determined.")
    secondary_metrics: list[Metric] = Field(min_length=1, description="Metrics to compute when evaluating the model, such as F1, AUROC, accuracy, RMSE, MAE, or R-squared. Must not include the primary metric.")

    evaluation_method: EvalMethod | None = Field(default=None, description="The method used to evaluate experiment results. Return null and request clarification if it cannot be determined.")
    validation_size: float | None = Field(default=None, gt=0, lt=1, description="Proportion of observations assigned to the validation set. Return null if evaluation_method is not train_validation_split.")
    num_folds: int | None = Field(default=None, ge=3, description="Number of cross-validation folds. Return null if evaluation_method is not k_fold_cross_validation.")
    stratify: bool | None = Field(default=None, description="Whether class proportions should be preserved across splits. Return false if the user does not specify a stratification. Return null if evaluation_method is leave_one_group_out or task_type is regression.")
    

prompt_parse_output = LLMOutput[UserRequest, ClarifiableField]