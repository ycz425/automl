import { Loader2, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import type { ChangeEvent } from "react";
import type { RetrainStage } from "../hooks/useRetrain";
import type { ParsedCsv } from "../utils/csv";
import { isCsvFile } from "../utils/files";

type RetrainPanelProps = {
  stage: RetrainStage;
  dataRecord: ParsedCsv | null;
  manualLabels: Record<number, string>;
  allManualLabelsFilled: boolean;
  isLoadingDataRecord: boolean;
  isSubmittingLabels: boolean;
  error: string | null;
  onSetLabel: (rowIndex: number, value: string) => void;
  onSubmitManualLabels: () => void;
  onSubmitLabelsFile: (file: File) => void;
  onClose: () => void;
};

// Once labels are submitted this panel closes — training/evaluating and
// promoting are long-running (k-fold retraining can take minutes), so the
// rest of the flow (progress, comparison, Promote/Discard) plays out as
// regular chat messages instead of holding this modal open. See useRetrain
// and RetrainResultMessage.
export function RetrainPanel({
  stage,
  dataRecord,
  manualLabels,
  allManualLabelsFilled,
  isLoadingDataRecord,
  isSubmittingLabels,
  error,
  onSetLabel,
  onSubmitManualLabels,
  onSubmitLabelsFile,
  onClose,
}: RetrainPanelProps) {
  const [mode, setMode] = useState<"manual" | "upload">("manual");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (stage === "closed") return null;

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !isCsvFile(file)) return;
    onSubmitLabelsFile(file);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 shadow-xl">
        <div className="flex items-center justify-between border-b border-neutral-800 px-5 py-4">
          <h2 className="text-sm font-semibold text-neutral-100">Retrain model</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-300"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {error && (
            <div
              role="alert"
              className="mb-4 rounded-lg border border-red-900/50 bg-red-950/30 px-3 py-2 text-xs text-red-300"
            >
              {error}
            </div>
          )}

          <div className="flex flex-col gap-4">
            <p className="text-xs text-neutral-500">
              Provide the correct label for each row below — the most recent 1,000 rows of data
              that were predicted during monitoring — then submit to train and evaluate a
              challenger model.
            </p>

            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setMode("manual")}
                className={[
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                  mode === "manual"
                    ? "bg-indigo-500/15 text-indigo-300"
                    : "text-neutral-500 hover:text-neutral-300",
                ].join(" ")}
              >
                Enter labels manually
              </button>
              <button
                type="button"
                onClick={() => setMode("upload")}
                className={[
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                  mode === "upload"
                    ? "bg-indigo-500/15 text-indigo-300"
                    : "text-neutral-500 hover:text-neutral-300",
                ].join(" ")}
              >
                Upload a CSV
              </button>
            </div>

            {isLoadingDataRecord ? (
              <div className="flex items-center gap-2 py-8 text-sm text-neutral-500">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                Loading unlabeled rows...
              </div>
            ) : mode === "manual" ? (
              dataRecord && dataRecord.rows.length > 0 ? (
                <div className="flex flex-col gap-3">
                  <div className="max-h-80 overflow-auto rounded-lg border border-neutral-800">
                    <table className="w-full min-w-max border-collapse text-left text-xs">
                      <thead className="sticky top-0 bg-neutral-900 text-neutral-400">
                        <tr>
                          {dataRecord.columns.map((column) => (
                            <th key={column} className="whitespace-nowrap px-3 py-2 font-medium">
                              {column}
                            </th>
                          ))}
                          <th className="whitespace-nowrap px-3 py-2 font-medium text-indigo-300">
                            Label
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-neutral-800">
                        {dataRecord.rows.map((row, rowIndex) => (
                          <tr key={rowIndex} className="odd:bg-neutral-900/40">
                            {dataRecord.columns.map((column) => (
                              <td
                                key={column}
                                className="whitespace-nowrap px-3 py-1.5 text-neutral-300"
                              >
                                {row[column]}
                              </td>
                            ))}
                            <td className="px-2 py-1">
                              <input
                                type="text"
                                value={manualLabels[rowIndex] ?? ""}
                                onChange={(event) => onSetLabel(rowIndex, event.target.value)}
                                className="w-24 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <button
                    type="button"
                    onClick={onSubmitManualLabels}
                    disabled={!allManualLabelsFilled || isSubmittingLabels}
                    aria-busy={isSubmittingLabels}
                    className="flex items-center justify-center gap-2 rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {isSubmittingLabels && (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    )}
                    Submit labels
                  </button>
                </div>
              ) : (
                <p className="py-8 text-center text-sm text-neutral-500">
                  No unlabeled rows are available yet.
                </p>
              )
            ) : (
              <div className="flex flex-col gap-2">
                <label htmlFor="retrain-label-upload" className="sr-only">
                  Upload a CSV of labels
                </label>
                <input
                  id="retrain-label-upload"
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileChange}
                  disabled={isSubmittingLabels}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isSubmittingLabels}
                  aria-busy={isSubmittingLabels}
                  className="flex items-center justify-center gap-2 rounded-lg border border-neutral-700 bg-neutral-900 p-3 text-sm font-medium text-neutral-300 transition-colors hover:border-indigo-500 hover:text-indigo-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSubmittingLabels ? (
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  ) : (
                    <Upload className="h-4 w-4" aria-hidden="true" />
                  )}
                  Upload a single-column CSV of labels
                </button>
                <p className="px-1 text-[11px] text-neutral-600">
                  Row count and order must match the unlabeled data — one label per row.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
