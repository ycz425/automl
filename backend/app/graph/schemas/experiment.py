from pydantic import BaseModel, Field
from app.graph.schemas.plan import Plan


class MetricEntry(BaseModel):
    metric: str = Field(description="Name of the evaluation metric, such as 'accuracy', 'f1', 'precision', 'recall', 'auroc', 'rmse', 'mae', or 'r2'.")
    value: float = Field(description="Numeric value of the corresponding evaluation metric.")


class FoldMetrics(BaseModel):
    fold: int = Field(ge=0, description="Zero-based index of the cross-validation fold.")
    metrics: list[MetricEntry] = Field(default_factory=list, description="Evaluation metrics computed for this cross-validation fold.")


class ExperimentResult(BaseModel):
    metrics: list[MetricEntry] = Field(default_factory=list, description="Overall evaluation metrics for the experiment. For cross-validation, these should typically be the metrics aggregated across all folds (for example, the mean metric values). For a single train/validation/test split, these are the final metrics for that experiment.")
    fold_metrics: list[FoldMetrics] | None = Field(default=None, description="Per-fold evaluation metrics when cross-validation is used. Use null if the experiment does not perform cross-validation.")


class ExperimentImplementation(BaseModel):
    code: str = Field(description="Complete executable Python source code implementing the machine-learning experiment.")
    dependencies: list[str] = Field(default_factory=list, description="Third-party Python packages required to execute the generated code, expressed as pip package names. Include only direct dependencies that are not part of the Python standard library.")


class Experiment(BaseModel):
    plan: Plan = Field(description="Plan used to define the experiment.")
    implementation: ExperimentImplementation = Field(description="Python implementation of the experiment.")
    result: ExperimentResult = Field(description="Evaluation results produced by the experiment.")