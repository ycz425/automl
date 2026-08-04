from pathlib import Path
from fastapi import UploadFile
from typing import Literal
import shutil
import os
import dotenv

dotenv.load_dotenv()
STORAGE_ROOT = os.getenv('STORAGE_ROOT')


type ArtifactType = Literal[
    'metrics',
    'predict',
    'requirements',
    'train',
]


class FileStorage:
    def __init__(self, root_directory: str = STORAGE_ROOT):
        self.root_directory = Path(root_directory)

        self.datasets_directory = self.root_directory / 'datasets'
        self.datasets_directory.mkdir(parents=True, exist_ok=True)

        self.runs_directory = self.root_directory / 'runs'
        self.runs_directory.mkdir(parents=True, exist_ok=True)

    async def save_dataset(self, dataset_id: str, file: UploadFile):
        dataset_directory = self.datasets_directory / str(dataset_id)
        dataset_directory.mkdir(parents=True, exist_ok=False)

        suffix = Path(file.filename).suffix.lower()
        destination = dataset_directory / f'dataset{suffix}'

        with destination.open('wb') as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        return destination

    async def get_dataset_directory(self, dataset_id: str):
        return (self.datasets_directory / str(dataset_id)).resolve()

    async def get_dataset_path(self, dataset_id: str):
        dataset_directory = await self.get_dataset_directory(dataset_id)
        dataset_path = list(dataset_directory.glob("dataset.*"))[0]
        return dataset_path.resolve()

    async def make_run_directory(self, thread_id: str):
        run_directory = await self.get_run_directory(thread_id)
        run_directory.mkdir(parents=True, exist_ok=False)

    async def get_run_directory(self, thread_id: str):
        return (self.runs_directory / thread_id).resolve()

    async def get_run_artifacts(self, thread_id: str):
        return os.listdir(await self.get_run_directory(thread_id))

    async def delete_dataset(self, dataset_id: str):
        shutil.rmtree(str(await self.get_dataset_directory(dataset_id)))

    async def delete_run(self, thread_id: str):
        shutil.rmtree(str(await self.get_run_directory(thread_id)))

    async def dataset_exists(self, dataset_id: str):
        return (await self.get_dataset_directory(dataset_id)).exists()

    async def run_exists(self, thread_id: str):
            return (await self.get_run_directory(thread_id)).exists()

local_file_storage = FileStorage()

