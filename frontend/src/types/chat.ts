import type { Artifact, AutoMLNode } from "./automl";

// Internal UI message model — deliberately decoupled from the network DTOs
// in automl.ts so backend shape changes don't ripple through the UI.

export type ChatRole = "user" | "assistant";

export type ChatMessageKind =
  | "text"
  | "progress"
  | "clarification"
  | "result"
  | "csv"
  | "error";

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

export type ChatMessage = {
  id: string;
  role: ChatRole;
  kind: ChatMessageKind;
  content: string;
  createdAt: string;
  attachment?: ChatAttachment;
  node?: AutoMLNode;
  artifacts?: Artifact[];
  /** Which round of the plan/experiment loop this progress update is in. */
  iteration?: number;
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
