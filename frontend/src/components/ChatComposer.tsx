import { Paperclip, Send } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import { isCsvFile } from "../utils/files";
import { AttachmentChip } from "./AttachmentChip";

type ChatComposerProps = {
  isEnabled: boolean;
  isSubmitting: boolean;
  canAttachFile: boolean;
  isAwaitingClarification: boolean;
  selectedFile: File | null;
  onSelectFile: (file: File | null) => void;
  onSend: (text: string) => void;
  onInvalidFile: (message: string) => void;
};

const MAX_TEXTAREA_HEIGHT = 200;

export function ChatComposer({
  isEnabled,
  isSubmitting,
  canAttachFile,
  isAwaitingClarification,
  selectedFile,
  onSelectFile,
  onSend,
  onInvalidFile,
}: ChatComposerProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    const resize = () => {
      el.style.height = "0px";
      el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
    };
    // A fresh page load can commit this effect before the surrounding flex
    // layout has settled (e.g. dvh viewport units resolving), which throws
    // off the scrollHeight measurement. Re-measuring next frame corrects it.
    resize();
    const frame = requestAnimationFrame(resize);
    return () => cancelAnimationFrame(frame);
  }, [text]);

  const requiresFile = canAttachFile;
  const canSend =
    isEnabled &&
    !isSubmitting &&
    text.trim().length > 0 &&
    (requiresFile ? Boolean(selectedFile) : true);

  function handleSend() {
    if (!canSend) return;
    onSend(text);
    setText("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!isCsvFile(file)) {
      onInvalidFile("Only .csv files are supported. Please choose a CSV dataset.");
      event.target.value = "";
      return;
    }

    onSelectFile(file);
  }

  const placeholder = !isEnabled
    ? "Please wait..."
    : isAwaitingClarification
    ? "Type your response..."
    : "Describe the target column, ML task, and any constraints...";

  return (
    <div className="border-t border-neutral-800 bg-neutral-950/95 backdrop-blur">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-2 px-4 py-4">
        {isAwaitingClarification && (
          <p className="px-1 text-xs font-medium text-amber-400">
            The agent needs more information to continue.
          </p>
        )}

        {selectedFile && (
          <AttachmentChip
            name={selectedFile.name}
            size={selectedFile.size}
            onRemove={() => onSelectFile(null)}
          />
        )}

        <div className="flex items-end gap-2 rounded-2xl border border-neutral-700 bg-neutral-900 p-2 shadow-sm transition-colors focus-within:border-indigo-500">
          {canAttachFile && (
            <>
              <label htmlFor="csv-upload-input" className="sr-only">
                Attach a CSV dataset
              </label>
              <input
                id="csv-upload-input"
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={handleFileChange}
                disabled={!isEnabled}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={!isEnabled}
                aria-label="Attach CSV dataset"
                title="Attach CSV dataset"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-neutral-400 transition-colors hover:bg-neutral-800 hover:text-neutral-100 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
              >
                <Paperclip className="h-4.5 w-4.5" aria-hidden="true" />
              </button>
            </>
          )}

          <label htmlFor="chat-composer-input" className="sr-only">
            Message
          </label>
          <textarea
            id="chat-composer-input"
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!isEnabled}
            placeholder={placeholder}
            aria-busy={isSubmitting}
            className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-1.5 text-sm text-neutral-100 placeholder-neutral-500 outline-none disabled:cursor-not-allowed disabled:opacity-60"
          />

          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send message"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
          >
            <Send className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <p className="px-1 text-[11px] text-neutral-600">
          Press Enter to send, Shift+Enter for a new line.
        </p>
      </div>
    </div>
  );
}
