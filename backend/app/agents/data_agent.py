from google import genai
import pandas as pd
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis, ColumnProfile, dataset_analysis_output, DataSplits
from app.graph.schemas.user_request import UserRequest
from app.services.tracing import traced_interactions_create
from langsmith import traceable
from pydantic import ValidationError
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, GroupShuffleSplit, GroupKFold
import numpy as np
import dotenv
import os

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class DataAgent():
    def __init__(self, model = "gemini-3.1-flash-lite", verbose=False):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.verbose = verbose

    def create_split_index(self, df: pd.DataFrame, user_request: UserRequest, dataset_analysis: DatasetAnalysis):
        X = df.drop(columns=dataset_analysis.target_column)
        y = df[dataset_analysis.target_column]
        groups = df[dataset_analysis.group_column] if dataset_analysis.group_column else np.arange(len(df))

        data_splits = {
            'splits': []
        }

        if user_request.evaluation_method == 'train_validation_split':
            if user_request.stratify:
                splits = StratifiedGroupKFold(
                    n_splits=round(1 / user_request.validation_size)
                )
            else:
                splits = GroupShuffleSplit(
                    n_splits=1,
                    test_size=user_request.validation_size,
                    random_state=42
                )
            train_idx, val_idx = next(splits.split(X, y, groups=groups))
            data_splits['splits'].append({'train_idx': train_idx, 'val_idx': val_idx})

        elif user_request.evaluation_method == 'k_fold_cross_validation':
            if user_request.stratify:
                splits = StratifiedGroupKFold(
                    n_splits=user_request.num_folds,
                )
            else:
                splits = GroupKFold(
                    n_splits=user_request.num_folds,
                )
            for train_idx, val_idx in splits.split(X, y, groups=groups):
                data_splits['splits'].append({'train_idx': train_idx, 'val_idx': val_idx})

        elif user_request.evaluation_method == 'leave_one_group_out':
            splits = LeaveOneGroupOut()
            for train_idx, val_idx in splits.split(X, y, groups=groups):
                data_splits['splits'].append({'train_idx': train_idx, 'val_idx': val_idx})

        else:
            raise RuntimeError

        return DataSplits.model_validate(data_splits)

    def identify_feature_type(self, df: pd.DataFrame, column: str):
        col_data = df[column]
        dtype = col_data.dtype

        # 1. Numerical & Boolean Check
        if pd.api.types.is_numeric_dtype(dtype):
            # Drop NaNs to safely check for boolean behavior
            non_null_values = col_data.dropna()
            
            if non_null_values.empty:
                feature_type = 'empty_numeric'
            # Check if values are strictly binary (0/1 or True/False)
            elif set(non_null_values.unique()).issubset({0, 1, True, False}):
                feature_type = 'boolean'
            else:
                feature_type = 'numerical'
                
        # 2. Categorical & String Check
        elif (isinstance(dtype, pd.CategoricalDtype) or 
            pd.api.types.is_object_dtype(dtype) or 
            pd.api.types.is_string_dtype(dtype)):
            feature_type = 'categorical'
            
        # 3. Datetime Check (Catches both timezone-naive and timezone-aware)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            feature_type = 'datetime'
            
        # 4. Fallback for unhandled types (like timedelta or period)
        else:
            feature_type = None
            
        return feature_type

    def profile_dataset(self, df: pd.DataFrame):
        if self.verbose:
            print(f'{datetime.now()}     profiling dataset...')

        columns = []
        for column in df.columns:
            feature_type = self.identify_feature_type(df, column)
            category_counts = df[column].value_counts().to_dict() if feature_type in {'categorical', 'boolean'} else None
            mean = df[column].mean() if feature_type == 'numerical' else None
            std = df[column].std() if feature_type == 'numerical' else None
            columns.append(
                ColumnProfile(
                    name=column,
                    dtype=str(df[column].dtype),
                    feature_type=feature_type,
                    category_counts=category_counts,
                    mean=mean,
                    std=std,
                    num_unique=df[column].nunique(),
                    num_missing=df[column].isna().sum(),
                    missing_rate = df[column].isna().sum() / len(df) if len(df) > 0 else None
                )
            )

        return DatasetProfile(
            num_columns = len(df.columns),
            num_rows = len(df),
            columns = columns
        )
        
    def retry_problems(self, dataset_profile: DatasetProfile, dataset_analysis: DatasetAnalysis):
        retry_problems = []

        dataset_columns = {column.name for column in dataset_profile.columns}
        feature_columns = set(dataset_analysis.feature_columns)
        excluded_columns = set(dataset_analysis.excluded_columns)

        if dataset_analysis.target_column is not None and dataset_analysis.target_column not in dataset_columns:
            retry_problems.append('target column not found in dataset profile')

        if dataset_analysis.target_column in feature_columns:
            retry_problems.append('feature_columns includes target column')

        unknown_features = feature_columns - dataset_columns
        if unknown_features:
            retry_problems.append(f'unknown columns in feature_columns: {sorted(unknown_features)}')

        unknown_excluded = excluded_columns - dataset_columns
        if unknown_excluded:
            retry_problems.append(f'unknown columns in excluded_columns: {sorted(unknown_excluded)}')

        overlap = feature_columns & excluded_columns
        if overlap:
            retry_problems.append(f'overlapping columns in feature_columns and excluded_columns: {sorted(overlap)}')

        if not (feature_columns - excluded_columns - ({dataset_analysis.target_column} if dataset_analysis.target_column else set())):
            retry_problems.append('no usable feature columns')

        return retry_problems

    @traceable(name="DataAgent.analyse_profile")
    async def analyse_profile(self, user_request: UserRequest, dataset_profile: DatasetProfile, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     analysing dataset profile...')

        validation_error = None
        retry_problems = []
        for attempt in range(max_retries + 1):
            prompt = (
                "Analyze the dataset and identify the most likely target column.\n\n"
                "User request:\n"
                f"{user_request.model_dump_json(indent=2)}\n\n"
                "Dataset profile:\n"
                f"{dataset_profile.model_dump_json(indent=2)}\n\n"
                "Only select columns that appear in the dataset profile.\n"
                "If the target column cannot be determined confidently from the column names, return null and request clarification.\n"
                "Produce only valid JSON matching the provided schema."
            )

            if validation_error:
                prompt += (
                    "\n\nYour previous output failed schema validation:\n\n"
                    f"{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema.\n"
                )
            elif retry_problems:
                prompt += (
                    "\n\nThe previous output matched the schema but failed these semantic checks:\n\n"
                    f"{'\n'.join(('- ' + problem) for problem in retry_problems)}\n\n"
                    "Correct these problems while continuing to strictly follow the schema."
                )

            interaction = await traced_interactions_create(
                self.client,
                model=self.model,
                input=prompt,
                generation_config={
                    'thinking_level': 'low',
                    'temperature': 0
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': dataset_analysis_output.model_json_schema()
                }
            )

            try:
                llm_output = dataset_analysis_output.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
                continue
            validation_error = None

            retry_problems = self.retry_problems(dataset_profile, llm_output.data)

            if not retry_problems:
                return llm_output
            
            if attempt == max_retries:
                raise RuntimeError('DataAgent failed semantic validation')
            
            if self.verbose:
                print(f'{datetime.now()}     DatasetAnalysis semantic validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
        