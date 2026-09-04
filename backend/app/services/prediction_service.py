from fastapi import UploadFile
from app.services.file_storage import FileStorage
from evidently import Report
from evidently.presets import DataDriftPreset
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import tempfile
import asyncio
import subprocess
import sys
import json
import os
import io

class PredictionService:
    def __init__(self, file_storage: FileStorage = FileStorage(), drift_window: int = 1000, drift_floor: int = 100):
        self.file_storage = file_storage
        self.drift_window = drift_window
        self.drift_floor = drift_floor


    async def predict(self, file: UploadFile, thread_id: str):
        await file.seek(0)

        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        model_path = str(artifacts_dir / 'model.joblib')
        script_path = str(artifacts_dir / 'predict.py')
        requirements_path = str(artifacts_dir / 'requirements.txt')
    
        if not os.path.exists(model_path) or not os.path.exists(script_path) or not os.path.exists(requirements_path):
            raise ArtifactNotFoundError

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'input.csv')
            env_dir = os.path.join(temp_dir, '.venv')
            output_path = os.path.join(temp_dir, 'output.csv')
    
            with open(input_path, 'wb') as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
    
            if os.name == "nt":
                env_python = os.path.join(env_dir, "Scripts", "python.exe")
            else:
                env_python = os.path.join(env_dir, "bin", "python")
    
            await asyncio.to_thread(
                subprocess.run,
                [sys.executable, '-m', 'venv', '--clear', env_dir],
                check=True,
                capture_output=True,
                text=True
            )

            await asyncio.to_thread(
                subprocess.run,
                [env_python, "-m", "pip", "install", "-r", requirements_path],
                check=True,
                capture_output=True,
                text=True
            )

            await asyncio.to_thread(
                subprocess.run,
                [env_python, script_path, '--model', model_path, '--input', input_path, '--output', output_path],
                check=True,
                capture_output=True,
                text=True
            )

            output_df = pd.read_csv(output_path)

        return output_df

    async def record_data(self, file: UploadFile, thread_id: str):
        await file.seek(0)

        artifacts_dir = await self.file_storage.get_run_directory(thread_id)
        reference_data_path = artifacts_dir / 'reference_data.csv'
        data_record_path = artifacts_dir / 'data_record.csv'
        
        if not reference_data_path.exists():
            raise ArtifactNotFoundError

        raw = b""
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            raw += chunk
        data = pd.read_csv(io.BytesIO(raw))

        reference_data = pd.read_csv(str(reference_data_path))
        columns = reference_data.columns

        if data_record_path.exists():
            existing = pd.read_csv(str(data_record_path))
        else:
            existing = pd.DataFrame(columns=columns)

        data_record = pd.concat([existing, data[columns]], ignore_index=True).tail(self.drift_window)
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