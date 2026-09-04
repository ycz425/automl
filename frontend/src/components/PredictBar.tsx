import { UploadCloud } from "lucide-react";
import { useRef } from "react";
import type { ChangeEvent } from "react";
import type { PredictionLogEntry } from "../types/automl";
import { isCsvFile } from "../utils/files";
import { PredictionMonitorDashboard } from "./PredictionMonitorDashboard";

type PredictBarProps = {
  isPredicting: boolean;
  predictionHistory: PredictionLogEntry[];
  onPredict: (file: File) => void;
  onInvalidFile: (message: string) => void;
};

export function PredictBar({
  isPredicting,
  predictionHistory,
  onPredict,
  onInvalidFile,
}: PredictBarProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    if (!isCsvFile(file)) {
      onInvalidFile("Only .csv files are supported. Please choose a CSV file to score.");
      return;
    }

    onPredict(file);
  }

  return (
    <div className="border-t border-neutral-800 bg-neutral-950/95 backdrop-blur">
      <div className="mx-auto w-full max-w-3xl">
        <PredictionMonitorDashboard history={predictionHistory} />

        <div className="flex flex-col gap-2 px-4 py-4">
          <label htmlFor="predict-upload-input" className="sr-only">
            Upload a CSV to get predictions
          </label>
          <input
            id="predict-upload-input"
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleFileChange}
            disabled={isPredicting}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isPredicting}
            aria-busy={isPredicting}
            className="flex items-center justify-center gap-2 rounded-2xl border border-neutral-700 bg-neutral-900 p-3 text-sm font-medium text-neutral-300 shadow-sm transition-colors hover:border-indigo-500 hover:text-indigo-300 disabled:cursor-not-allowed disabled:opacity-60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
          >
            <UploadCloud className="h-4 w-4" aria-hidden="true" />
            Upload a CSV to get predictions
          </button>

          <p className="px-1 text-[11px] text-neutral-600">
            Uploading a file scores it with your trained model.
          </p>
        </div>
      </div>
    </div>
  );
}
