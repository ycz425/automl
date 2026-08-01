from pydantic import BaseModel, Field

class OutputScripts(BaseModel):
    train_script: str = Field(description="Executable Python script that trains the selected pipeline on the full dataset and saves it as model.joblib.")
    predict_script: str = Field(description="Executable Python script that loads model.joblib and generates predictions for unseen data.")
    dependencies: list[str] = Field(description="Third-party pip packages required to run the training and prediction scripts.")