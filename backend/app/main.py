from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, EventSourceResponse
from pydantic import BaseModel
from app.services.automl_service import automl_service
from app.services.status_store import automl_node, automl_status
from typing import Annotated
from uuid import uuid4, UUID
import mimetypes
import os



class StartAutoMLRequest(BaseModel):
    message: str
    dataset_id: str


class RunAutoMLResponse(BaseModel):
    thread_id: str


class ResumeAutoMLRequest(BaseModel):
    message: str


class AutoMLResponse(BaseModel):
    status: automl_status
    node: automl_node
    message: str | None


class DatasetUploadResponse(BaseModel):
    dataset_id: UUID
    destination: str


class GetArtifactsResponse(BaseModel):
    artifacts: list[str]


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/automl/", response_model=RunAutoMLResponse, status_code=202)
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


@app.get("/automl/{thread_id}/stream", response_class=EventSourceResponse)
async def stream_status(thread_id: str):
    status = await automl_service.get_status(thread_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail="AutoML thread not found"
        )

    event, watch = await automl_service.status_store.subscribe(thread_id)
    try:
        while True:
            await event.wait()
            event.clear()
            status = await automl_service.get_status(thread_id)
            if status is None:
                return
            status = await automl_service.get_status(thread_id)
            yield AutoMLResponse(status=status['status'], node=status['node'], message=status['message'])

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
    
    dataset_id = str(uuid4())
    destination = await automl_service.file_storage.save_dataset(dataset_id, file)

    return DatasetUploadResponse(dataset_id=str(dataset_id), destination=str(destination))


@app.get("/artifact/{thread_id}", response_model=GetArtifactsResponse)
async def get_artifacts(thread_id: str):
    artifacts = await automl_service.file_storage.get_run_artifacts(thread_id)
    return GetArtifactsResponse(artifacts=artifacts)


@app.get("/artifact/{thread_id}/download/{filename}")
async def download_artifact(thread_id: str, filename: str):
    media_type, _ = mimetypes.guess_type(filename)
    return FileResponse(
        path=os.path.join(await automl_service.file_storage.get_run_directory(thread_id), filename),
        filename=filename,
        media_type=media_type or "application/octet-stream"
    )


@app.delete("/artifact/{thread_id}/delete", status_code=204)
async def delete_artifacts(thread_id: str):
    if not (await automl_service.file_storage.run_exists(thread_id)):
        raise HTTPException(
            status_code=404,
            detail="Artifacts not found"
        )
    await automl_service.file_storage.delete_run(thread_id)


@app.delete("/dataset/{dataset_id}/delete", status_code=204)
async def delete_dataset(dataset_id: str):
    if not (await automl_service.file_storage.dataset_exists(dataset_id)):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    await automl_service.file_storage.delete_dataset(dataset_id)


@app.delete("/status/{thread_id}/delete", status_code=204)
async def delete_status(thread_id: str):
    if not (await automl_service.status_store.exists(thread_id)):
        raise HTTPException(
            status_code=404,
            detail="AutoML thread not found"
        )
    await automl_service.status_store.delete(thread_id)