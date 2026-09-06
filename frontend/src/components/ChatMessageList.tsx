import { useMemo } from "react";
import type { ChatMessage } from "../types/chat";
import { ChatMessageBubble } from "./ChatMessageBubble";
import { EmptyState } from "./EmptyState";

type ChatMessageListProps = {
  messages: ChatMessage[];
  isRetrainResolving: boolean;
  onPromoteRetrain: (messageId: string, datasetId: string) => void;
  onRejectRetrain: (messageId: string) => void;
};

// Standard chat-app grouping (Discord, Slack, iMessage): only label a message
// with a timestamp when it starts a new group — the very first message, a
// switch between user/assistant, or enough time elapsed since the group's
// own first message (not just since the previous message — otherwise a long
// burst of messages each a minute apart would never re-show a timestamp,
// even after the group has been going for an hour).
const TIMESTAMP_GAP_MS = 5 * 60 * 1000;

function computeTimestampFlags(messages: ChatMessage[]): boolean[] {
  const flags: boolean[] = [];
  let groupStart: ChatMessage | undefined;

  for (const message of messages) {
    const isNewGroup =
      !groupStart ||
      message.role !== groupStart.role ||
      new Date(message.createdAt).getTime() - new Date(groupStart.createdAt).getTime() >=
        TIMESTAMP_GAP_MS;
    flags.push(isNewGroup);
    if (isNewGroup) groupStart = message;
  }

  return flags;
}

export function ChatMessageList({
  messages,
  isRetrainResolving,
  onPromoteRetrain,
  onRejectRetrain,
}: ChatMessageListProps) {
  const timestampFlags = useMemo(() => computeTimestampFlags(messages), [messages]);

  if (messages.length === 0) {
    return <EmptyState />;
  }

  return (
    <div
      className="flex flex-1 flex-col px-4 pb-4 pt-6 sm:px-0"
      role="log"
      aria-live="polite"
    >
      {messages.map((message, index) => {
        const showTimestamp = timestampFlags[index];
        return (
          <div key={message.id} className={index === 0 ? "" : showTimestamp ? "mt-6" : "mt-1.5"}>
            <ChatMessageBubble
              message={message}
              showTimestamp={showTimestamp}
              isRetrainResolving={isRetrainResolving}
              onPromoteRetrain={onPromoteRetrain}
              onRejectRetrain={onRejectRetrain}
            />
          </div>
        );
      })}
    </div>
  );
}
