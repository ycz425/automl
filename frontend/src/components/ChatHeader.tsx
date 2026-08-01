import { RotateCcw, Workflow } from "lucide-react";
import type { RunState } from "../types/chat";

type ChatHeaderProps = {
  runState: RunState;
  onClearSession: () => void;
};

const STATUS_CONFIG: Record<RunState, { label: string; dotClassName: string }> = {
  idle: { label: "Ready", dotClassName: "bg-neutral-500" },
  uploading: { label: "Uploading dataset", dotClassName: "bg-indigo-400 animate-pulse-dot" },
  starting: { label: "Starting run", dotClassName: "bg-indigo-400 animate-pulse-dot" },
  running: { label: "Running", dotClassName: "bg-indigo-400 animate-pulse-dot" },
  awaiting_clarification: { label: "Waiting for input", dotClassName: "bg-amber-400" },
  resuming: { label: "Resuming", dotClassName: "bg-indigo-400 animate-pulse-dot" },
  completed: { label: "Completed", dotClassName: "bg-emerald-400" },
  failed: { label: "Failed", dotClassName: "bg-red-400" },
};

export function ChatHeader({ runState, onClearSession }: ChatHeaderProps) {
  const status = STATUS_CONFIG[runState];

  return (
    <header className="sticky top-0 z-10 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur">
      <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400">
            <Workflow className="h-4.5 w-4.5" aria-hidden="true" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-neutral-100">AutoML Agent</span>
            <span
              className="flex items-center gap-1.5 text-xs text-neutral-500"
              aria-live="polite"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${status.dotClassName}`} aria-hidden="true" />
              {status.label}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={onClearSession}
          className="flex items-center gap-1.5 rounded-md border border-neutral-800 px-2.5 py-1.5 text-xs font-medium text-neutral-400 transition-colors hover:border-neutral-600 hover:text-neutral-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
          aria-label="Clear session and start over"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          Clear session
        </button>
      </div>
    </header>
  );
}
