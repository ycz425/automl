import type { AutoMLNode, RetrainMetrics } from "./automl";

// Internal UI message model — deliberately decoupled from the network DTOs
// in automl.ts so backend shape changes don't ripple through the UI.

export type ChatRole = "user" | "assistant";

export type ChatMessageKind =
  | "text"
  | "progress"
  | "clarification"
  | "result"
  | "csv"
  | "error"
  | "retrain_result";

export type RetrainResultData = {
  datasetId: string;
  champion: RetrainMetrics;
  challenger: RetrainMetrics;
  challengerThreshold: number | null;
  /** Set once the user has acted on this comparison — hides the buttons in favor of a status line. */
  resolution?: "promoted" | "rejected";
};

export type ChatAttachment = {
  name: string;
  size?: number;
  /** Present when the CSV is held client-side (a user-selected upload). */
  file?: File;
  /** Present when the CSV came back from the backend as response text (e.g. predictions). */
  csvText?: string;
  /** Short label shown above the filename in a "csv" bubble, e.g. "Uploaded dataset". */
  caption?: string;
};

// For "progress" messages: how this pipeline stage ended, if it has.
// "active" (or undefined) shows a spinner; anything else is terminal for
// that stage's bubble — "done" only when the stage actually passed and the
// run moved on, "clarifying" when it paused for user input instead, and
// "failed" when the run failed during or after this stage.
export type ProgressStatus = "active" | "done" | "clarifying" | "failed";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  kind: ChatMessageKind;
  content: string;
  createdAt: string;
  attachment?: ChatAttachment;
  node?: AutoMLNode;
  /** Which round of the plan/experiment loop this progress update is in. */
  iteration?: number;
  progressStatus?: ProgressStatus;
  /** For "progress" messages not tied to an AutoMLNode (e.g. retraining) — overrides the node-based label. */
  label?: string;
  retrainResult?: RetrainResultData;
};

export type RunState =
  | "idle"
  | "uploading"
  | "starting"
  | "running"
  | "awaiting_clarification"
  | "resuming"
  | "completed"
  | "failed";
