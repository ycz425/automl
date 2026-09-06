from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, EventSourceResponse, Response
from pydantic import BaseModel
from app.services.automl_service import AutoMLService
from app.services.prediction_service import PredictionService, ArtifactNotFoundError
from app.services.file_storage import FileStorage
from app.services.status_store import StatusStore
from app.services.retrain_service import RetrainService
from app.utils.uploads import upload_file_to_dataframe
from app.graph.graph import build_graph
from app.services.status_store import automl_node, automl_status
import subprocess
import tempfile
import asyncio
import sys
from typing import Annotated
from uuid import uuid4
import mimetypes
import os

file_storage = FileStorage()
status_store = StatusStore()
automl_service = AutoMLService(build_graph(), file_storage=file_storage, status_store=status_store)
prediction_service = PredictionService(file_storage=file_storage)
retrain_service = RetrainService(file_storage=file_storage)

class StartAutoMLRequest(BaseModel):
    message: str
    dataset_id: str


class RunAutoMLResponse(BaseModel):
    thread_id: str


class ResumeAutoMLRequest(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: automl_status
    node: automl_node
    message: str | None


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    destination: str


class GetArtifactsResponse(BaseModel):
    artifacts: list[str]


class PredictionMetricsResponse(BaseModel):
    history: list[dict]


class LabelDataResponse(BaseModel):
    dataset_id: str


class RetrainResponse(BaseModel):
    champion: dict
    challenger: dict
    challenger_threshold: float | None


class RetrainRequest(BaseModel):
    dataset_id: str


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://automl-504220.web.app",
        "https://automl-504220.firebaseapp.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/automl", response_model=RunAutoMLResponse, status_code=202)
async def start_automl(request: StartAutoMLRequest, background_tasks: BackgroundTasks):
    thread_id = str(uuid4())
    background_tasks.add_task(
        automl_service.start,
        request.message,
        request.dataset_id,
        thread_id
    )

    return RunAutoMLResponse(thread_id=thread_id)


@app.post("/automl/{thread_id}/resume", response_model=RunAutoMLResponse, status_code=202)
async def resume_automl(thread_id: str, request: ResumeAutoMLRequest, background_tasks: BackgroundTasks):
    if not (await automl_service.exists(thread_id)):
        raise HTTPException(
            status_code=404,
            detail="AutoML thread not found"
        )
    background_tasks.add_task(
        automl_service.resume,
        request.message,
        thread_id
    )

    return RunAutoMLResponse(thread_id=thread_id)


@app.get("/automl/{thread_id}/status", response_model=StatusResponse)
async def get_automl_status(thread_id: str):
    status = await automl_service.get_status(thread_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail="AutoML thread not found"
        )
    return StatusResponse(status=status['status'], node=status['node'], message=status['message'])


@app.get("/automl/{thread_id}/stream", response_class=EventSourceResponse)
async def stream_automl_status(thread_id: str):
    event, watch = await automl_service.subscribe(thread_id)
    try:
        while True:
            await event.wait()
            event.clear()
            status = await automl_service.get_status(thread_id)
            if status is None:
                return
            yield StatusResponse(status=status['status'], node=status['node'], message=status['message'])

            if status["status"] in {
                "completed",
                "failed"
            }:
                return
    finally:
        watch.unsubscribe()

        
@app.post("/dataset/upload", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(file: Annotated[UploadFile, File()]):
    _, ext = os.path.splitext(file.filename.lower())
    if ext != '.csv':
        raise HTTPException(status_code=400, detail=f"Extension '{ext}' not allowed")
    
    df = await upload_file_to_dataframe(file)
    dataset_id = str(uuid4())
    destination = await file_storage.save_dataset(dataset_id, df)

    return DatasetUploadResponse(dataset_id=str(dataset_id), destination=str(destination))


@app.get("/artifact/{thread_id}", response_model=GetArtifactsResponse)
async def get_artifacts(thread_id: str):
    artifacts = await file_storage.get_run_artifacts(thread_id)
    return GetArtifactsResponse(artifacts=artifacts)


@app.get("/artifact/{thread_id}/download/{filename}")
async def download_artifact(thread_id: str, filename: str):
    media_type, _ = mimetypes.guess_type(filename)
    return FileResponse(
        path=os.path.join(await file_storage.get_run_directory(thread_id), filename),
        filename=filename,
        media_type=media_type or "application/octet-stream"
    )

@app.post("/predict/{thread_id}")
async def predict(thread_id: str, file: Annotated[UploadFile, File()]):
    df = await upload_file_to_dataframe(file)

    try:
        preds_df = await prediction_service.predict(df, thread_id)
    except ArtifactNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Required artifacts not found"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {e.stderr or e.stdout or str(e)}"
        )

    try:
        await prediction_service.record_data(df, thread_id)
    except Exception as e:
        print(f'[predict] record_data failed for thread {thread_id}: {e!r}')

    return Response(
        content=preds_df.to_csv(index=False).encode("utf-8"),
        media_type='text/csv'
    )


@app.get("/predict/{thread_id}/metrics", response_model=PredictionMetricsResponse)
async def get_prediction_metrics(thread_id: str):
    if not (await file_storage.run_exists(thread_id)):
        raise HTTPException(
            status_code=404,
            detail="AutoML thread not found"
        )
    history = await prediction_service.get_drift_history(thread_id)
    return PredictionMetricsResponse(history=history)


@app.post("/retrain/{thread_id}/label")
async def label_data(thread_id: str, file: Annotated[UploadFile, File()]):
    _, ext = os.path.splitext(file.filename.lower())
    if ext != '.csv':
        raise HTTPException(status_code=400, detail=f"Extension '{ext}' not allowed")

    targets_df = await upload_file_to_dataframe(file)

    try:
        dataset_id = await retrain_service.create_datasets(thread_id, targets_df)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid targets file")

    return LabelDataResponse(dataset_id=dataset_id)
    

@app.post("/retrain/{thread_id}", response_model=RetrainResponse)
async def retrain(thread_id: str, request: RetrainRequest):
    try:
        results = await retrain_service.evaluate(thread_id, request.dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return RetrainResponse(
        champion=results['champion'],
        challenger=results['challenger'],
        challenger_threshold=results['challenger_threshold']
    )


@app.post("/retrain/{thread_id}/promote")
async def prompte_challenger(thread_id: str, request: RetrainRequest):
    await retrain_service.promote_challenger(thread_id, request.dataset_id)


@app.post("/retrain/{thread_id}/clear")
async def clear_retrain(thread_id: str):
    await retrain_service.delete_evaluation(thread_id)


@app.delete("/artifact/{thread_id}/delete", status_code=204)
async def delete_artifacts(thread_id: str):
    if not (await file_storage.run_exists(thread_id)):
        raise HTTPException(
            status_code=404,
            detail="Artifacts not found"
        )
    await file_storage.delete_run(thread_id)


@app.delete("/dataset/{dataset_id}/delete", status_code=204)
async def delete_dataset(dataset_id: str):
    if not (await file_storage.dataset_exists(dataset_id)):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    await file_storage.delete_dataset(dataset_id)


@app.delete("/status/{thread_id}/delete", status_code=204)
async def delete_status(thread_id: str):
    if not (await automl_service.status_store.exists(thread_id)):
        raise HTTPException(
            status_code=404,
            detail="AutoML thread not found"
        )
    await automl_service.status_store.delete(thread_id)