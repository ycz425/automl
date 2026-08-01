import type { Artifact, AutoMLNode } from "./automl";

// Internal UI message model — deliberately decoupled from the network DTOs
// in automl.ts so backend shape changes don't ripple through the UI.

export type ChatRole = "user" | "assistant";

export type ChatMessageKind =
  | "text"
  | "progress"
  | "clarification"
  | "result"
  | "error";

export type ChatAttachment = {
  name: string;
  size?: number;
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
