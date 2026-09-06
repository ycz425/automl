import pandas as pd
import numpy as np
from app.graph.schemas.data_info import DatasetAnalysis, DataSplits
from app.graph.schemas.user_request import UserRequest
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, GroupShuffleSplit, GroupKFold


class DataSplitter:
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
