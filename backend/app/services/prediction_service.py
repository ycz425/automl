from app.services.file_storage import FileStorage
from app.utils.model_scripts import create_venv, run_predict_script
from evidently import Report
from evidently.presets import DataDriftPreset
from datetime import datetime, timezone
from app.graph.schemas.data_info import DatasetAnalysis
import pandas as pd
import tempfile
import json
import os

class PredictionService:
    def __init__(self, file_storage: FileStorage, drift_window: int = 1000, drift_floor: int = 100):
        self.file_storage = file_storage
        self.drift_window = drift_window
        self.drift_floor = drift_floor


    async def predict(self, df: pd.DataFrame, thread_id: str):
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        model_path = str(artifacts_dir / 'model.joblib')
        script_path = str(artifacts_dir / 'predict.py')
        requirements_path = str(artifacts_dir / 'requirements.txt')
        threshold_path = str(artifacts_dir / 'threshold.json')

        if not os.path.exists(model_path) or not os.path.exists(script_path) or not os.path.exists(requirements_path) or not os.path.exists(threshold_path):
            raise ArtifactNotFoundError

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'input.csv')
            env_dir = os.path.join(temp_dir, '.venv')
            output_path = os.path.join(temp_dir, 'output.csv')

            df.to_csv(input_path, index=False)

            env_python = await create_venv(env_dir, requirements_path)

            await run_predict_script(env_python, script_path, model_path, input_path, output_path, threshold_path)

            output_df = pd.read_csv(output_path)

        return output_df

    async def record_data(self, data: pd.DataFrame, thread_id: str):
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        reference_data_path = artifacts_dir / 'reference_data.csv'
        dataset_analysis_path = artifacts_dir / 'dataset_analysis.json'
        data_record_path = artifacts_dir / 'data_record.csv'

        if not reference_data_path.exists() or not dataset_analysis_path.exists():
            raise ArtifactNotFoundError

        dataset_analysis = DatasetAnalysis.model_validate_json(dataset_analysis_path.read_text())

        reference_data = pd.read_csv(str(reference_data_path))

        group_columns = (
            [dataset_analysis.group_column]
            if dataset_analysis.group_column and dataset_analysis.group_column in data.columns
            else []
        )
        columns = list(set(dataset_analysis.feature_columns + group_columns))

        if data_record_path.exists():
            existing = pd.read_csv(str(data_record_path))
        else:
            existing = pd.DataFrame(columns=columns)

        data_record = pd.concat([existing, data[columns]], ignore_index=True).tail(self.drift_window)

        # CSV has no dtype of its own — pandas re-infers it from scratch on
        # every read, so a single stray non-numeric value anywhere in the
        # rolling window can silently downgrade a whole numeric column to
        # 'object' for as long as that row stays in the window. Realigning
        # to reference_data's dtypes before writing keeps data_record.csv
        # (and every future read of it) consistent, instead of relying on
        # Evidently's best-effort dtype coercion at drift-computation time.
        for column in columns:
            if column in reference_data.columns:
                try:
                    data_record[column] = data_record[column].astype(reference_data[column].dtype)
                except (ValueError, TypeError):
                    pass

        data_record.to_csv(str(data_record_path), index=False)

        if len(data_record) < self.drift_floor:
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'rows_recorded': len(data),
                'window_size': len(data_record),
                'status': 'insufficient_data',
                'rows_needed': self.drift_floor,
            }
            await self._append_drift_entry(thread_id, entry)
            return None

        report = Report([DataDriftPreset()], include_tests=True)
        result = report.run(current_data=data_record, reference_data=reference_data)
        summary = self._summarize_drift(result.dict())

        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rows_recorded': len(data),
            'window_size': len(data_record),
            'status': 'ok',
            **summary,
        }
        await self._append_drift_entry(thread_id, entry)

        return summary

    async def _append_drift_entry(self, thread_id: str, entry: dict) -> None:
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        drift_log_path = artifacts_dir / 'drift_results.jsonl'
        with open(drift_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    async def get_drift_history(self, thread_id: str) -> list[dict]:
        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        drift_log_path = artifacts_dir / 'drift_results.jsonl'

        if not drift_log_path.exists():
            return []

        with open(drift_log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _summarize_drift(result_dict: dict) -> dict:
        metrics_by_id = {metric['id']: metric for metric in result_dict.get('metrics', [])}
        status_by_metric_id = {
            test['metric_config']['metric_id']: test['status']
            for test in result_dict.get('tests', [])
            if test.get('metric_config', {}).get('metric_id')
        }

        dataset_drift = False
        drifted_columns = []
        column_scores = {}

        for metric_id, metric in metrics_by_id.items():
            metric_type = metric.get('config', {}).get('type')
            status = status_by_metric_id.get(metric_id)

            if metric_type == 'evidently:metric_v2:DriftedColumnsCount':
                dataset_drift = status == 'FAIL'
            elif metric_type == 'evidently:metric_v2:ValueDrift':
                column = metric['config'].get('column')
                if not column:
                    continue
                drifted = status == 'FAIL'
                column_scores[column] = {
                    'method': metric['config'].get('method'),
                    'value': metric.get('value'),
                    'threshold': metric['config'].get('threshold'),
                    'drifted': drifted,
                }
                if drifted:
                    drifted_columns.append(column)

        return {'dataset_drift': dataset_drift, 'drifted_columns': drifted_columns, 'column_scores': column_scores}


class ArtifactNotFoundError(Exception):
    pass