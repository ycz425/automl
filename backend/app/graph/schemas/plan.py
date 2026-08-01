from typing import Any, Literal
from pydantic import BaseModel, Field


type SplitMethod = Literal[
    "train_validation_test",
    "k_fold_cross_validation",
    "stratified_k_fold",
    "group_k_fold",
    "time_series_split",
    "leave_one_group_out",
]


class HyperparameterEntry(BaseModel):
    name: str = Field(description="Name of the architecture-specific hyperparameter.")
    value: Any = Field(description="Proposed value of the architecture-specific hyperparameter.")


class ArchitecturePlan(BaseModel):
    architecture_name: str = Field(description="Name of the proposed model or architecture, such as 'XGBoost', 'MLP', '1D CNN-LSTM', or 'FT-Transformer'.")
    architecture_description: str = Field(description="Detailed description of the model structure, including major layers, dimensions, activations, normalization, pooling, and connections when applicable.")
    hyperparameters: list[HyperparameterEntry] = Field(default_factory=list, description="Architecture-specific tunable hyperparameters, such as hidden dimensions, number of layers, kernel sizes, dropout rates, attention heads, or tree depth.")


class TrainingStrategy(BaseModel):
    optimizer: str | None = Field(default=None, description="Optimizer used to train the model, such as AdamW, Adam, or SGD. Use null when the proposed model does not use gradient-based training.")
    loss: str | None = Field(default=None, description="Training loss function, such as cross-entropy, binary cross-entropy, mean squared error, or focal loss. Use null when the model selects its objective internally.")
    learning_rate: float | None = Field(default=None, gt=0, description="Initial optimizer learning rate. Use null when learning rate is not applicable to the proposed model.")
    weight_decay: float | None = Field(default=None, ge=0, description="Weight-decay or L2 regularization coefficient. Use null when it is not applicable.")
    batch_size: int | None = Field(default=None, ge=1, description="Number of observations processed in each training batch. Use null for models that do not train with mini-batches.")
    epochs: int | None = Field(default=None, ge=1, description="Maximum number of training epochs. Use null for models that do not use epoch-based training.")
    patience: int | None = Field(default=None, ge=1, description="Number of epochs without validation improvement allowed before early stopping. Use null when early stopping is not used.")
    scheduler: str | None = Field(default=None, description="Learning-rate scheduler, such as ReduceLROnPlateau, cosine annealing, or one-cycle scheduling. Use null when no scheduler is proposed.")
    gradient_clipping: float | None = Field(default=None, gt=0, description="Maximum gradient norm used for gradient clipping. Use null when gradient clipping is not needed.")
    additional_hyperparameters: list[HyperparameterEntry] = Field(default_factory=list, description="Additional training hyperparameters not represented by the explicit fields above.")


class SplitStrategy(BaseModel):
    method: SplitMethod = Field(description="Method used to divide the data for training and evaluation.")
    test_size: float | None = Field(default=None, gt=0, lt=1, description="Proportion of observations assigned to the test set. Use null when the selected splitting method does not use a fixed test set.")
    validation_size: float | None = Field(default=None, gt=0, lt=1, description="Proportion of observations assigned to the validation set. Use null when validation is performed through cross-validation.")
    num_folds: int | None = Field(default=None, ge=2, description="Number of cross-validation folds. Use null for methods that do not use cross-validation.")
    stratify: bool = Field(default=False, description="Whether class proportions should be preserved across splits.")
    group_column: str | None = Field(default=None, description="Column defining groups that must remain entirely within one split, such as subject, patient, or recording ID.")
    random_seed: int = Field(default=42, description="Random seed used to make data splitting reproducible.")


class Plan(BaseModel):
    preprocessing_steps: list[str] = Field(default_factory=list, description="Ordered preprocessing operations to apply before model training, such as imputation, scaling, encoding, resampling, augmentation, or feature extraction.")
    architecture_plan: ArchitecturePlan = Field(description="Proposed model architecture and its structural settings.")
    training_strategy: TrainingStrategy = Field(description="Proposed optimization, regularization, batching, and stopping strategy.")
    split_strategy: SplitStrategy = Field(description="Data-splitting and cross-validation strategy.")
    rationale: str = Field(description="Explanation of why this plan is appropriate for the user request and the observed dataset properties.")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made because required information was unavailable or ambiguous.")
    dependencies: list[str] = Field(default_factory=list, description="Python packages or external libraries required to implement the plan.")
