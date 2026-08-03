import type {
  DatasetUploadResponse,
  GetArtifactsResponse,
  ResumeAutoMLRequest,
  RunAutoMLResponse,
  StartAutoMLRequest,
} from "../types/automl";
import { AppError, parseResponseError, toFriendlyMessage } from "../utils/errors";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function buildUrl(path: string): string {
  return `${API_BASE_URL.replace(/\/+$/, "")}${path}`;
}

async function requestJson<T>(
  url: string,
  init: RequestInit,
  fallbackErrorMessage: string
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    throw new AppError(toFriendlyMessage(error, fallbackErrorMessage), error);
  }

  if (!response.ok) {
    const message = await parseResponseError(response, fallbackErrorMessage);
    throw new AppError(message);
  }

  // 204 No Content has no body to parse — trying to anyway throws.
  if (response.status === 204) {
    return undefined as T;
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    throw new AppError("The server returned an unexpected response.", error);
  }
}

export async function uploadDataset(
  file: File,
  signal?: AbortSignal
): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<DatasetUploadResponse>(
    buildUrl("/dataset/upload"),
    { method: "POST", body: formData, signal },
    "Failed to upload the dataset. Please try again."
  );
}

export async function startAutoML(
  message: string,
  datasetId: string,
  signal?: AbortSignal
): Promise<RunAutoMLResponse> {
  const payload: StartAutoMLRequest = { message, dataset_id: datasetId };
  return requestJson<RunAutoMLResponse>(
    buildUrl("/automl/"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    },
    "Failed to start the AutoML run. Please try again."
  );
}

export async function resumeAutoML(
  threadId: string,
  message: string,
  signal?: AbortSignal
): Promise<RunAutoMLResponse> {
  const payload: ResumeAutoMLRequest = { message };
  return requestJson<RunAutoMLResponse>(
    buildUrl(`/automl/${encodeURIComponent(threadId)}/resume`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    },
    "Failed to resume the run. Please try again."
  );
}

export function getStreamUrl(threadId: string): string {
  return buildUrl(`/automl/${encodeURIComponent(threadId)}/stream`);
}

export async function getArtifacts(
  threadId: string,
  signal?: AbortSignal
): Promise<GetArtifactsResponse> {
  return requestJson<GetArtifactsResponse>(
    buildUrl(`/artifact/${encodeURIComponent(threadId)}`),
    { method: "GET", signal },
    "Failed to load the run's artifacts."
  );
}

export function getArtifactDownloadUrl(threadId: string, filename: string): string {
  return buildUrl(
    `/artifact/${encodeURIComponent(threadId)}/download/${encodeURIComponent(filename)}`
  );
}

export async function deleteArtifacts(threadId: string, signal?: AbortSignal): Promise<void> {
  await requestJson<unknown>(
    buildUrl(`/artifact/${encodeURIComponent(threadId)}/delete`),
    { method: "DELETE", signal },
    "Failed to delete the run's artifacts."
  );
}

export async function deleteDataset(datasetId: string, signal?: AbortSignal): Promise<void> {
  await requestJson<unknown>(
    buildUrl(`/dataset/${encodeURIComponent(datasetId)}/delete`),
    { method: "DELETE", signal },
    "Failed to delete the uploaded dataset."
  );
}

export async function deleteStatus(threadId: string, signal?: AbortSignal): Promise<void> {
  await requestJson<unknown>(
    buildUrl(`/status/${encodeURIComponent(threadId)}/delete`),
    { method: "DELETE", signal },
    "Failed to delete the run's status."
  );
}
