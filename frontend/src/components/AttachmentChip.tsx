import { FileText, X } from "lucide-react";
import { formatFileSize } from "../utils/files";

type AttachmentChipProps = {
  name: string;
  size?: number;
  onRemove?: () => void;
};

export function AttachmentChip({ name, size, onRemove }: AttachmentChipProps) {
  return (
    <div className="inline-flex max-w-full items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-800/70 px-3 py-2 text-sm">
      <FileText className="h-4 w-4 shrink-0 text-indigo-400" aria-hidden="true" />
      <span className="min-w-0 flex-1 truncate font-mono text-neutral-200" title={name}>
        {name}
      </span>
      {typeof size === "number" && (
        <span className="shrink-0 text-xs text-neutral-500">{formatFileSize(size)}</span>
      )}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove attached file ${name}`}
          className="shrink-0 rounded-md p-0.5 text-neutral-400 transition-colors hover:bg-neutral-700 hover:text-neutral-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
        >
          <X className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}
