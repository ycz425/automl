// Network-facing DTOs matching the FastAPI backend contract.

export type DatasetUploadResponse = {
  dataset_id: string;
  destination: string;
};

export type StartAutoMLRequest = {
  message: string;
  dataset_id: string;
};

export type ResumeAutoMLRequest = {
  message: string;
};

export type RunAutoMLResponse = {
  thread_id: string;
};

export type AutoMLStatus =
  | "running"
  | "need_clarification"
  | "completed"
  | "failed";

export type AutoMLNode =
  | "prompt_agent"
  | "data_agent"
  | "plan_agent"
  | "experiment_agent"
  | "output_agent";

export type Artifact = {
  filename: string;
  label?: string;
};

// The endpoint returns bare filenames, not Artifact objects — there's no
// label metadata from this source, so callers derive a display label from
// the filename itself.
export type GetArtifactsResponse = {
  artifacts: string[];
};

export type AutoMLResponse = {
  status: AutoMLStatus;
  node: AutoMLNode;
  // Only guaranteed to be populated for "need_clarification" (the question)
  // and "completed" (a summary of the experimentation process) — null
  // otherwise.
  message: string | null;
  artifacts?: Artifact[];
};

// value/threshold can come back null — Evidently may not compute a value
// for a column at all (e.g. a dtype mismatch between reference and current
// data for that column).
export type DriftColumnScore = {
  method: string;
  value: number | null;
  threshold: number | null;
  drifted: boolean;
};

// One entry per /predict call. "insufficient_data" means the rolling window
// hasn't reached rows_needed yet, so no drift check ran for that call.
export type PredictionLogEntry = {
  timestamp: string;
  rows_recorded: number;
  window_size: number;
  status: "insufficient_data" | "ok";
  rows_needed?: number;
  dataset_drift?: boolean;
  drifted_columns?: string[];
  column_scores?: Record<string, DriftColumnScore>;
};

export type PredictionMetricsResponse = {
  history: PredictionLogEntry[];
};

export type RetrainLabelResponse = {
  dataset_id: string;
};

// Metric name -> value, e.g. { accuracy: 0.91, precision: 0.88, ... } or
// { rmse: 1.2, mae: 0.9, r2: 0.8 } — the fixed set depends on task type and
// is decided server-side, so the frontend treats it as an open record.
export type RetrainMetrics = Record<string, number>;

export type RetrainEvaluateResponse = {
  champion: RetrainMetrics;
  challenger: RetrainMetrics;
  challenger_threshold: number | null;
};

export const PIPELINE_NODES: AutoMLNode[] = [
  "prompt_agent",
  "data_agent",
  "plan_agent",
  "experiment_agent",
  "output_agent",
];

export const NODE_LABELS: Record<AutoMLNode, string> = {
  prompt_agent: "Understanding your request",
  data_agent: "Analyzing the dataset",
  plan_agent: "Designing the ML approach",
  experiment_agent: "Training and evaluating models",
  output_agent: "Preparing the final results",
};
