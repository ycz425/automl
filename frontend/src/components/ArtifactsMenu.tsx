import { FileOutput, Loader2 } from "lucide-react";
import { useState } from "react";
import type { Artifact } from "../types/automl";
import { ArtifactCard } from "./ArtifactCard";
import { Modal } from "./Modal";

type ArtifactsMenuProps = {
  isAvailable: boolean;
  artifacts: Artifact[];
  isLoading: boolean;
  getDownloadUrl: (filename: string) => string;
  onOpen: () => void;
};

// Deliberately fetches its list fresh every time it's opened rather than
// caching what the run produced at completion — retraining/promotion can
// change model.joblib, threshold.json, etc. afterward, and this menu should
// always reflect what's actually on disk right now.
export function ArtifactsMenu({
  isAvailable,
  artifacts,
  isLoading,
  getDownloadUrl,
  onOpen,
}: ArtifactsMenuProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!isAvailable) return null;

  function open() {
    setIsOpen(true);
    onOpen();
  }

  return (
    <>
      <button
        type="button"
        onClick={open}
        aria-label="Run artifacts"
        className="flex items-center gap-1.5 rounded-md border border-neutral-800 px-2.5 py-1.5 text-xs font-medium text-neutral-400 transition-colors hover:border-neutral-600 hover:text-neutral-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
      >
        <FileOutput className="h-3.5 w-3.5" aria-hidden="true" />
        Artifacts
      </button>

      {isOpen && (
        <Modal title="Run artifacts" onClose={() => setIsOpen(false)}>
          {isLoading ? (
            <div className="flex items-center justify-center gap-2 py-10 text-sm text-neutral-500">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              Loading files...
            </div>
          ) : artifacts.length === 0 ? (
            <p className="py-10 text-center text-sm text-neutral-500">No files available yet.</p>
          ) : (
            <div className="flex flex-col gap-2">
              {artifacts.map((artifact) => (
                <ArtifactCard
                  key={artifact.filename}
                  artifact={artifact}
                  downloadUrl={getDownloadUrl(artifact.filename)}
                />
              ))}
            </div>
          )}
        </Modal>
      )}
    </>
  );
}
