import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types/chat";
import { ChatMessageBubble } from "./ChatMessageBubble";
import { EmptyState } from "./EmptyState";

type ChatMessageListProps = {
  messages: ChatMessage[];
  getDownloadUrl: (filename: string) => string;
};

export function ChatMessageList({ messages, getDownloadUrl }: ChatMessageListProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return <EmptyState />;
  }

  return (
    <div
      className="flex flex-1 flex-col gap-6 px-4 pb-4 pt-6 sm:px-0"
      role="log"
      aria-live="polite"
    >
      {messages.map((message) => (
        <ChatMessageBubble
          key={message.id}
          message={message}
          getDownloadUrl={getDownloadUrl}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
