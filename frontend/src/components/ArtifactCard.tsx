import { Download, FileOutput } from "lucide-react";
import type { Artifact } from "../types/automl";

type ArtifactCardProps = {
  artifact: Artifact;
  downloadUrl: string;
};

export function ArtifactCard({ artifact, downloadUrl }: ArtifactCardProps) {
  const displayLabel = artifact.label?.trim() || artifact.filename;

  return (
    <div className="flex items-center gap-3 rounded-lg border border-neutral-700 bg-neutral-800/60 px-4 py-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-indigo-500/10 text-indigo-400">
        <FileOutput className="h-4.5 w-4.5" aria-hidden="true" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-neutral-100">{displayLabel}</p>
        <p className="truncate font-mono text-xs text-neutral-500">{artifact.filename}</p>
      </div>
      <a
        href={downloadUrl}
        download={artifact.filename}
        aria-label={`Download ${displayLabel}`}
        className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-neutral-600 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-200 transition-colors hover:border-indigo-400 hover:text-indigo-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
      >
        <Download className="h-3.5 w-3.5" aria-hidden="true" />
        Download
      </a>
    </div>
  );
}
