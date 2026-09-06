from app.services.file_storage import FileStorage
from app.graph.schemas.data_info import DatasetAnalysis
from app.graph.schemas.user_request import UserRequest
from app.utils.model_scripts import create_venv, run_train_script, run_predict_script
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import tempfile
import json
import os
import pandas as pd
from uuid import uuid4

class RetrainService:
    def __init__(self, file_storage: FileStorage, folds: int = 5):
        self.file_storage = file_storage
        self.folds = folds

    async def create_datasets(self, thread_id: str, targets_df: pd.DataFrame):
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        data_record_path = artifacts_dir / 'data_record.csv'
        dataset_analysis_path = artifacts_dir / 'dataset_analysis.json'

        data_record_df = pd.read_csv(str(data_record_path))
        dataset_analysis = DatasetAnalysis.model_validate_json(dataset_analysis_path.read_text())

        if len(targets_df) != len(data_record_df):
            raise ValueError(
                f"Targets file has {len(targets_df)} rows, but data_record.csv has {len(data_record_df)} rows."
            )

        targets_df.columns = [dataset_analysis.target_column]
        dataset_df = pd.concat([data_record_df, targets_df], axis=1)

        dataset_id = str(uuid4())
        await self.file_storage.save_dataset(dataset_id, dataset_df)
        return dataset_id

    async def promote_challenger(self, thread_id: str, dataset_id: str):
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        train_path = str(artifacts_dir / 'train.py')
        requirements_path = str(artifacts_dir / 'requirements.txt')
        model_path = str(artifacts_dir / 'model.joblib')
        threshold_path = artifacts_dir / 'threshold.json'
        evaluate_result_path = artifacts_dir / 'evaluate_result.json'
        dataset_analysis_path = artifacts_dir / 'dataset_analysis.json'
        reference_data_path = artifacts_dir / 'reference_data.csv'
        data_record_path = artifacts_dir / 'data_record.csv'
        drift_results_path = artifacts_dir / 'drift_results.jsonl'
        data_path = str(await self.file_storage.get_dataset_path(dataset_id))

        evaluate_result = json.loads(evaluate_result_path.read_text())
        dataset_analysis = DatasetAnalysis.model_validate_json(dataset_analysis_path.read_text())

        with tempfile.TemporaryDirectory() as env_dir:
            env_python = await create_venv(env_dir, requirements_path)

            await run_train_script(env_python, train_path, data_path, model_path)

        with open(threshold_path, 'w') as f:
            json.dump({'threshold': evaluate_result['challenger_threshold']}, f, indent=4)

        # The promoted model was just trained on new data, so drift monitoring
        # needs to move with it — otherwise future predictions get compared
        # against a reference distribution the deployed model was no longer
        # actually trained on, and the rolling window would mix pre- and
        # post-promotion predictions under that now-stale baseline.
        df = pd.read_csv(data_path)
        reference_data = df[dataset_analysis.feature_columns].sample(n=min(len(df), 2000), random_state=42)
        reference_data.to_csv(reference_data_path, index=False)

        if data_record_path.exists():
            os.remove(str(data_record_path))
        if drift_results_path.exists():
            os.remove(str(drift_results_path))

        await self.delete_evaluation(thread_id)


    async def delete_evaluation(self, thread_id: str):
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        evaluate_result_path = artifacts_dir / 'evaluate_result.json'
        if evaluate_result_path.exists():
            os.remove(str(evaluate_result_path))


    async def evaluate(self, thread_id: str, dataset_id: str):
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        old_model_path = str(artifacts_dir / 'model.joblib')
        train_script_path = str(artifacts_dir / 'train.py')
        predict_script_path = str(artifacts_dir / 'predict.py')
        data_path = str(await self.file_storage.get_dataset_path(dataset_id))
        user_request_path = artifacts_dir / 'user_request.json'
        dataset_analysis_path = artifacts_dir / 'dataset_analysis.json'
        requirements_path = str(artifacts_dir / 'requirements.txt')
        threshold_path = str(artifacts_dir / 'threshold.json')

        df = pd.read_csv(data_path)
        user_request = UserRequest.model_validate_json(user_request_path.read_text())
        dataset_analysis = DatasetAnalysis.model_validate_json(dataset_analysis_path.read_text())

        # Data splitting
        data_splits = []
        groups = df[dataset_analysis.group_column] if (dataset_analysis.group_column and dataset_analysis.group_column in df.columns) else np.arange(len(df))
        splits = StratifiedGroupKFold(n_splits=self.folds) if user_request.stratify else GroupKFold(n_splits=self.folds)
        y = df[dataset_analysis.target_column]
        for train_idx, val_idx in splits.split(y, y, groups=groups):
            data_splits.append((train_idx, val_idx))

        champion = pd.DataFrame()
        challenger = pd.DataFrame()

        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = os.path.join(temp_dir, '.venv')
            env_python = await create_venv(env_dir, requirements_path)

            new_model_path = os.path.join(temp_dir, 'new_model.joblib')
            train_data_path = os.path.join(temp_dir, 'train.csv')
            val_data_path = os.path.join(temp_dir, 'val.csv')
            new_model_pred_path = os.path.join(temp_dir, 'new_preds.csv')
            old_model_pred_path = os.path.join(temp_dir, 'old_preds.csv')

            targets = df[dataset_analysis.target_column]
            challenger_thresholds = []

            # dataset_analysis.positive_class is always a JSON string, but the target
            # column keeps its native dtype (e.g. int64 for 0/1 labels) — comparing the
            # raw string against it would silently match nothing, so resolve it once to
            # the matching native value.
            positive_class = None
            if user_request.task_type == 'binary_classification':
                positive_class = self._resolve_positive_class(targets, dataset_analysis.positive_class)

            for train_idx, val_idx in data_splits:
                fold_targets = targets.iloc[val_idx]

                df.iloc[train_idx].to_csv(train_data_path, index=False)
                df.iloc[val_idx].to_csv(val_data_path, index=False)

                await run_train_script(env_python, train_script_path, train_data_path, new_model_path)

                # Challenger's own threshold isn't tuned yet, so get its raw output (a
                # probability for binary classification, otherwise the final prediction)
                # and, for binary classification, tune a threshold from this fold's own
                # held-out predictions before deriving its final labels.
                await run_predict_script(env_python, predict_script_path, new_model_path, val_data_path, new_model_pred_path)

                await run_predict_script(env_python, predict_script_path, old_model_path, val_data_path, old_model_pred_path, threshold_path)

                new_model_output = pd.read_csv(new_model_pred_path)['pred']
                old_model_preds = pd.read_csv(old_model_pred_path)['pred']

                if user_request.task_type == 'binary_classification':
                    negative_class = fold_targets[fold_targets != positive_class].iloc[0]
                    new_model_threshold = self.tune_threshold(new_model_output, fold_targets, positive_class, negative_class)
                    new_model_preds = np.where(new_model_output >= new_model_threshold, positive_class, negative_class)
                    challenger_thresholds.append(new_model_threshold)
                else:
                    new_model_preds = new_model_output

                if user_request.task_type == 'regression':
                    champion = pd.concat([champion, pd.DataFrame([self.get_regression_metrics(old_model_preds, fold_targets)])])
                    challenger = pd.concat([challenger, pd.DataFrame([self.get_regression_metrics(new_model_preds, fold_targets)])])
                else:
                    champion = pd.concat([champion, pd.DataFrame([self.get_classification_metrics(old_model_preds, fold_targets)])])
                    challenger = pd.concat([challenger, pd.DataFrame([self.get_classification_metrics(new_model_preds, fold_targets)])])

        evaluate_result = {
            'champion': champion.mean(axis=0).to_dict(),
            'challenger': challenger.mean(axis=0).to_dict(),
            'challenger_threshold': float(np.mean(challenger_thresholds)) if challenger_thresholds else None,
        }

        evaluate_result_path = artifacts_dir / 'evaluate_result.json'
        with open(evaluate_result_path, 'w') as f:
            json.dump(evaluate_result, f, indent=4)

        return evaluate_result

    @staticmethod
    def _resolve_positive_class(targets: pd.Series, positive_class: str | None):
        unique_values = list(targets.unique())

        for value in unique_values:
            if str(value) == positive_class:
                return value

        # The manual-entry labeling UI accepts free-typed text, so a label
        # that only differs from the original class name by case or stray
        # whitespace (e.g. "Yes" vs "yes") shouldn't hard-fail — fall back to
        # a looser match before giving up.
        if positive_class is not None:
            normalized = positive_class.strip().lower()
            for value in unique_values:
                if str(value).strip().lower() == normalized:
                    return value

        raise ValueError(
            f"None of this batch's labels are '{positive_class}' (found only: "
            f"{[str(v) for v in unique_values]}). If that's unexpected, double check the entered/"
            f"uploaded labels use the same class names as the original dataset. Otherwise, this "
            f"batch of rows simply happens to contain no positive examples at all — common when "
            f"the positive class is rare and the sample is small — so there's nothing to tune a "
            f"threshold against; try labeling a larger or different batch of rows."
        )

    @staticmethod
    def get_regression_metrics(preds, targets) -> dict[str, float]:
        preds = np.asarray(preds, dtype=float)
        targets = np.asarray(targets, dtype=float)
        return {
            'rmse': float(np.sqrt(mean_squared_error(targets, preds))),
            'mae': float(mean_absolute_error(targets, preds)),
            'r2': float(r2_score(targets, preds)),
        }

    @staticmethod
    def tune_threshold(probs, targets, positive_class, negative_class) -> float:
        probs = np.asarray(probs, dtype=float)
        targets = np.asarray(targets)

        best_threshold = 0.5
        best_f1 = -1.0
        for candidate in np.unique(probs):
            preds = np.where(probs >= candidate, positive_class, negative_class)
            candidate_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                best_threshold = candidate

        return float(best_threshold)

    @staticmethod
    def get_classification_metrics(preds, targets) -> dict[str, float]:
        return {
            'accuracy': float(accuracy_score(targets, preds)),
            'precision': float(precision_score(targets, preds, average='macro', zero_division=0)),
            'recall': float(recall_score(targets, preds, average='macro', zero_division=0)),
            'f1': float(f1_score(targets, preds, average='macro', zero_division=0)),
        }
