import { FileOutput, Loader2, X } from "lucide-react";
import { useState } from "react";
import { createPortal } from "react-dom";
import type { Artifact } from "../types/automl";
import { ArtifactCard } from "./ArtifactCard";

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

      {isOpen &&
        createPortal(
          // Portaled to document.body: this component renders inside
          // ChatHeader's <header>, which uses backdrop-blur — a CSS
          // property that creates a new containing block for `fixed`
          // descendants, so without the portal this modal would be
          // confined to the header's own small bounding box instead of
          // the viewport.
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 shadow-xl">
              <div className="flex items-center justify-between border-b border-neutral-800 px-5 py-4">
                <h2 className="text-sm font-semibold text-neutral-100">Run artifacts</h2>
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  aria-label="Close"
                  className="rounded-md p-1 text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-300"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
                {isLoading ? (
                  <div className="flex items-center justify-center gap-2 py-10 text-sm text-neutral-500">
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Loading files...
                  </div>
                ) : artifacts.length === 0 ? (
                  <p className="py-10 text-center text-sm text-neutral-500">
                    No files available yet.
                  </p>
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
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
