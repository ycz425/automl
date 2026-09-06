import type { AutoMLNode } from "../types/automl";
import type { ChatAttachment, ChatMessage, ProgressStatus, RetrainResultData } from "../types/chat";
import { generateId } from "./files";

// Every ChatMessage needs an id + createdAt + role, and every call site was
// hand-rolling that trio (25+ times across useAutoMLChat/useRetrain). These
// factories are the single place that scaffolding lives — callers just
// supply the fields that actually vary per message kind.

function base(role: ChatMessage["role"]): Pick<ChatMessage, "id" | "role" | "createdAt"> {
  return { id: generateId(), role, createdAt: new Date().toISOString() };
}

export function userTextMessage(content: string): ChatMessage {
  return { ...base("user"), kind: "text", content };
}

export function assistantTextMessage(content: string): ChatMessage {
  return { ...base("assistant"), kind: "text", content };
}

export function userCsvMessage(attachment: ChatAttachment): ChatMessage {
  return { ...base("user"), kind: "csv", content: "", attachment };
}

export function assistantCsvMessage(attachment: ChatAttachment): ChatMessage {
  return { ...base("assistant"), kind: "csv", content: "", attachment };
}

export function progressMessage(options: {
  content?: string;
  node?: AutoMLNode;
  label?: string;
  iteration?: number;
  progressStatus?: ProgressStatus;
}): ChatMessage {
  return {
    ...base("assistant"),
    kind: "progress",
    content: options.content ?? "",
    node: options.node,
    label: options.label,
    iteration: options.iteration,
    progressStatus: options.progressStatus ?? "active",
  };
}

export function clarificationMessage(content: string, node?: AutoMLNode): ChatMessage {
  return { ...base("assistant"), kind: "clarification", content, node };
}

export function resultMessage(content: string, node?: AutoMLNode): ChatMessage {
  return { ...base("assistant"), kind: "result", content, node };
}

export function errorMessage(content: string, node?: AutoMLNode): ChatMessage {
  return { ...base("assistant"), kind: "error", content, node };
}

export function retrainResultMessage(retrainResult: RetrainResultData): ChatMessage {
  return { ...base("assistant"), kind: "retrain_result", content: "", retrainResult };
}
