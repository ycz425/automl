import { AlertTriangle, Bot, HelpCircle, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { NODE_LABELS } from "../types/automl";
import type { ChatMessage } from "../types/chat";
import { ArtifactCard } from "./ArtifactCard";
import { AttachmentChip } from "./AttachmentChip";
import { PipelineProgress } from "./PipelineProgress";

type ChatMessageBubbleProps = {
  message: ChatMessage;
  getDownloadUrl: (filename: string) => string;
};

function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

export function ChatMessageBubble({ message, getDownloadUrl }: ChatMessageBubbleProps) {
  const isUser = message.role === "user";
  const time = new Date(message.createdAt).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });

  if (isUser) {
    return (
      <div className="flex animate-fade-in flex-col items-end gap-1.5">
        <div className="flex max-w-[85%] flex-col items-end gap-2 sm:max-w-[75%]">
          {message.attachment && (
            <AttachmentChip name={message.attachment.name} size={message.attachment.size} />
          )}
          <div className="rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white shadow-sm">
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          </div>
        </div>
        <span className="pr-1 text-[11px] text-neutral-600">{time}</span>
      </div>
    );
  }

  const isError = message.kind === "error";
  const isProgress = message.kind === "progress";
  const isClarification = message.kind === "clarification";

  return (
    <div className="flex animate-fade-in items-start gap-3">
      <div
        className={[
          "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isError
            ? "bg-red-500/10 text-red-400"
            : isClarification
            ? "bg-amber-500/10 text-amber-400"
            : "bg-indigo-500/10 text-indigo-400",
        ].join(" ")}
        aria-hidden="true"
      >
        {isError ? (
          <AlertTriangle className="h-4 w-4" />
        ) : isClarification ? (
          <HelpCircle className="h-4 w-4" />
        ) : (
          <Bot className="h-4 w-4" />
        )}
      </div>

      <div className="min-w-0 max-w-[90%] flex-1 sm:max-w-[80%]">
        <div
          className={[
            "rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm",
            isError
              ? "border border-red-900/50 bg-red-950/30"
              : "border border-neutral-800 bg-neutral-900",
          ].join(" ")}
        >
          {isProgress ? (
            <div className="flex flex-col gap-3" aria-live="polite" aria-busy="true">
              <div className="flex items-center gap-2 text-sm font-medium text-neutral-200">
                <Loader2 className="h-4 w-4 animate-spin text-indigo-400" aria-hidden="true" />
                {message.node ? NODE_LABELS[message.node] : "Working..."}
              </div>
              {message.content && (
                <p className="text-sm leading-relaxed text-neutral-400">{message.content}</p>
              )}
              <PipelineProgress currentNode={message.node} iteration={message.iteration} />
            </div>
          ) : (
            <Markdown content={message.content} />
          )}

          {message.kind === "result" && message.artifacts && message.artifacts.length > 0 && (
            <div className="mt-4 flex flex-col gap-2">
              {message.artifacts.map((artifact) => (
                <ArtifactCard
                  key={artifact.filename}
                  artifact={artifact}
                  downloadUrl={getDownloadUrl(artifact.filename)}
                />
              ))}
            </div>
          )}
        </div>
        <span className="ml-1 mt-1 inline-block text-[11px] text-neutral-600">{time}</span>
      </div>
    </div>
  );
}
