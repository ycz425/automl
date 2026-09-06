import { AlertCircle, CheckCircle2, HelpCircle, Loader2, Repeat } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { NODE_LABELS } from "../types/automl";
import type { ChatMessage } from "../types/chat";
import { AttachmentChip } from "./AttachmentChip";
import { CsvMessageBubble } from "./CsvMessageBubble";
import { RetrainResultMessage } from "./RetrainResultMessage";

type ChatMessageBubbleProps = {
  message: ChatMessage;
  showTimestamp: boolean;
  isRetrainResolving: boolean;
  onPromoteRetrain: (messageId: string, datasetId: string) => void;
  onRejectRetrain: (messageId: string) => void;
};

const LOOPING_NODES = new Set(["plan_agent", "experiment_agent"]);

function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

export function ChatMessageBubble({
  message,
  showTimestamp,
  isRetrainResolving,
  onPromoteRetrain,
  onRejectRetrain,
}: ChatMessageBubbleProps) {
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
        {showTimestamp && (
          <span className={isUser ? "pr-1 text-[11px] text-neutral-600" : "pl-1 text-[11px] text-neutral-600"}>
            {time}
          </span>
        )}
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
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="flex animate-fade-in flex-col items-end gap-1.5">
        {showTimestamp && <span className="pr-1 text-[11px] text-neutral-600">{time}</span>}
        <div className="flex max-w-[85%] flex-col items-end gap-2 sm:max-w-[75%]">
          {message.attachment && (
            <AttachmentChip name={message.attachment.name} size={message.attachment.size} />
          )}
          <div className="rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white shadow-sm">
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          </div>
        </div>
      </div>
    );
  }

  const isError = message.kind === "error";
  const isProgress = message.kind === "progress";
  const isRetrainResult = message.kind === "retrain_result";

  return (
    <div className="flex animate-fade-in flex-col items-start gap-1.5">
      {showTimestamp && <span className="pl-1 text-[11px] text-neutral-600">{time}</span>}
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
            <div
              className="flex flex-col gap-2"
              aria-live={message.progressStatus === "active" ? "polite" : undefined}
              aria-busy={message.progressStatus === "active" || message.progressStatus === undefined}
            >
              <div className="flex items-center gap-2 text-sm font-medium text-neutral-200">
                {message.progressStatus === "done" ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" aria-hidden="true" />
                ) : message.progressStatus === "clarifying" ? (
                  <HelpCircle className="h-4 w-4 text-amber-400" aria-hidden="true" />
                ) : message.progressStatus === "failed" ? (
                  <AlertCircle className="h-4 w-4 text-red-400" aria-hidden="true" />
                ) : (
                  <Loader2 className="h-4 w-4 animate-spin text-indigo-400" aria-hidden="true" />
                )}
                {message.label ?? (message.node ? NODE_LABELS[message.node] : "Working...")}
                {message.node && LOOPING_NODES.has(message.node) && Boolean(message.iteration) && (
                  <span
                    className="flex items-center gap-1 rounded-full bg-indigo-500/15 px-1.5 py-0.5 text-[10px] font-medium text-indigo-300"
                    title={`Round ${message.iteration}`}
                  >
                    <Repeat className="h-2.5 w-2.5" aria-hidden="true" />
                    Round {message.iteration}
                  </span>
                )}
              </div>
              {message.content && message.progressStatus !== "done" && (
                <p className="text-sm leading-relaxed text-neutral-400">{message.content}</p>
              )}
            </div>
          ) : isRetrainResult && message.retrainResult ? (
            <RetrainResultMessage
              data={message.retrainResult}
              isResolving={isRetrainResolving}
              onPromote={() => onPromoteRetrain(message.id, message.retrainResult!.datasetId)}
              onReject={() => onRejectRetrain(message.id)}
            />
          ) : (
            <Markdown content={message.content} />
          )}
        </div>
      </div>
    </div>
  );
}
