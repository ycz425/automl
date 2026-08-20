from pydantic import BaseModel, Field
from app.graph.schemas.clarifiable_model import ClarifiableModel
from typing import Literal

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


class UserRequest(ClarifiableModel):
    task_type: TaskType | None = Field(default=None, description="The machine-learning task type inferred from the request. Return null if it cannot be determined.")
    target_description: str | None = Field(default=None, description="Plain-language description of what the model should predict. Return null if target cannot be determined.")
    group_description: str | None = Field(default=None, description="Plain-language description of groups that must remain entirely within one split, such as subject, patient, or recording ID. Return null if the user does not specify a grouping.")
    include_features: list[str] = Field(default_factory=list, description='Features the user explicitly requires. Return an empty list when none are specified.')
    exclude_features: list[str] = Field(default_factory=list, description='Features the user explicitly prohibits. Return an empty list when none are specified.')
    constraints: list[str] = Field(default_factory=list, description='Explicit requirements affecting data splitting, modeling, evaluation, compute, interpretability, or deployment. Return an empty list if user does not specify any constraints.')
    preferences: list[str] = Field(default_factory=list, description='Non-mandatory preferences affecting data splitting, modeling, evaluation, compute, interpretability, or deployment. Return an empty list if user does not specify any preferences.')

    primary_metric: Metric | None = Field(default=None, description="Main metric used to compare experiments, select checkpoints, and decide whether an experiment improved. Return null if it cannot be determined.")
    secondary_metrics: list[Metric] = Field(min_length=1, description="Metrics to compute when evaluating the model, such as F1, AUROC, accuracy, RMSE, MAE, or R-squared. Must not include the primary metric. Select an appropriate set of metrics for the task if not provided by user.")

    evaluation_method: EvalMethod | None = Field(default=None, description="The method used to evaluate experiment results. Return null and request clarification if it cannot be determined.")
    validation_size: float | None = Field(default=None, gt=0, lt=1, description="Proportion of observations assigned to the validation set. Return null if evaluation_method is not train_validation_split.")
    num_folds: int | None = Field(default=None, ge=3, description="Number of cross-validation folds. Return null if evaluation_method is not k_fold_cross_validation.")
    stratify: bool | None = Field(default=None, description="Whether class proportions should be preserved across splits. Return false if the user does not specify a stratification. Return null if evaluation_method is leave_one_group_out or task_type is regression.")
    
    def problems(self):
        problems = []
        if self.task_type is None:
            problems.append('task_type is unspecified')

        if self.target_description is None:
            problems.append('target_description is unclear')

        if not set(self.include_features).isdisjoint(set(self.exclude_features)):
            problems.append('include_features and exclude_features overlap')

        if self.primary_metric is None:
            problems.append('primary_metric is unspecified')

        if self.evaluation_method is None:
            problems.append('evaluation_method is unspecified')

        if self.evaluation_method == 'train_validation_split' and self.num_folds is not None:
            problems.append('evaluation_method is train_validation_split but num_folds is specified')

        if self.evaluation_method == 'train_validation_split' and self.validation_size is None:
            problems.append('evaluation_method is train_validation_split but validation_size is not specified')

        if self.evaluation_method == 'k_fold_cross_validation' and self.validation_size is not None:
            problems.append('evaluation_method is k_fold_cross_validation but validation_size is specified')

        if self.evaluation_method == 'k_fold_cross_validation' and self.num_folds is None:
            problems.append('evaluation_method is k_fold_cross_validation but num_folds is not specified')

        if self.evaluation_method == 'leave_one_group_out' and self.num_folds is not None:
            problems.append('evaluation_method is leave_one_group_out but num_folds is specified')

        if self.evaluation_method == 'leave_one_group_out' and self.validation_size is not None:
            problems.append('evaluation_method is leave_one_group_out but validation_size is specified')

        return problems