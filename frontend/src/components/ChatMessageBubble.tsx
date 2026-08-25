import { ChevronDown, FileOutput, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { NODE_LABELS } from "../types/automl";
import type { ChatMessage } from "../types/chat";
import { ArtifactCard } from "./ArtifactCard";
import { AttachmentChip } from "./AttachmentChip";
import { CsvMessageBubble } from "./CsvMessageBubble";
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

  if (message.kind === "csv" && message.attachment) {
    return (
      <div
        className={[
          "flex animate-fade-in flex-col gap-1.5",
          isUser ? "items-end" : "items-start",
        ].join(" ")}
      >
        <div className="w-full max-w-[85%] sm:max-w-[75%]">
          <CsvMessageBubble
            name={message.attachment.name}
            size={message.attachment.size}
            file={message.attachment.file}
            csvText={message.attachment.csvText}
            caption={message.attachment.caption}
            isUser={isUser}
          />
        </div>
        <span className={isUser ? "pr-1 text-[11px] text-neutral-600" : "pl-1 text-[11px] text-neutral-600"}>
          {time}
        </span>
      </div>
    );
  }

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

  return (
    <div className="flex animate-fade-in flex-col items-start gap-1.5">
      <div className="max-w-[90%] sm:max-w-[80%]">
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
              {message.node && (
                <PipelineProgress currentNode={message.node} iteration={message.iteration} />
              )}
            </div>
          ) : (
            <Markdown content={message.content} />
          )}

          {message.kind === "result" && message.artifacts && message.artifacts.length > 0 && (
            <details className="group mt-4 rounded-lg border border-neutral-800">
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium text-neutral-400 transition-colors hover:text-neutral-200">
                <FileOutput className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                Downloadable files ({message.artifacts.length})
                <ChevronDown
                  className="ml-auto h-3.5 w-3.5 shrink-0 transition-transform group-open:rotate-180"
                  aria-hidden="true"
                />
              </summary>
              <div className="flex flex-col gap-2 px-3 pb-3">
                {message.artifacts.map((artifact) => (
                  <ArtifactCard
                    key={artifact.filename}
                    artifact={artifact}
                    downloadUrl={getDownloadUrl(artifact.filename)}
                  />
                ))}
              </div>
            </details>
          )}
        </div>
      </div>
      <span className="pl-1 text-[11px] text-neutral-600">{time}</span>
    </div>
  );
}
