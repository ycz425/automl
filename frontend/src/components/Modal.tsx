import { X } from "lucide-react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

type ModalProps = {
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** Tailwind max-width class for the panel, e.g. "max-w-lg" (default) or "max-w-2xl". */
  maxWidthClassName?: string;
};

// Always portals to document.body — a modal nested inside an ancestor with
// backdrop-filter (e.g. ChatHeader's backdrop-blur) would otherwise be
// confined to that ancestor's own bounding box instead of the viewport,
// since backdrop-filter creates a new containing block for `fixed`
// descendants. Bit ArtifactsMenu once; baking the portal in here means it
// can't happen again just because a modal gets nested somewhere new.
export function Modal({ title, onClose, children, maxWidthClassName = "max-w-lg" }: ModalProps) {
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div
        className={`flex max-h-[90vh] w-full ${maxWidthClassName} flex-col overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 shadow-xl`}
      >
        <div className="flex items-center justify-between border-b border-neutral-800 px-5 py-4">
          <h2 className="text-sm font-semibold text-neutral-100">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-300"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </div>,
    document.body
  );
}
