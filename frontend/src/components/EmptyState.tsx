import { Sparkles } from "lucide-react";

export function EmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-16 text-center text-neutral-500">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-indigo-500/10 text-indigo-400">
        <Sparkles className="h-6 w-6" aria-hidden="true" />
      </div>
      <p className="text-sm">Starting a new conversation…</p>
    </div>
  );
}
