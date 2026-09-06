import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import type { RetrainResultData } from "../types/chat";

type RetrainResultMessageProps = {
  data: RetrainResultData;
  isResolving: boolean;
  onPromote: () => void;
  onReject: () => void;
};

// Metrics computed server-side are a fixed set per task type — hardcoding
// direction here is safe since the set never grows (see RetrainService's
// get_regression_metrics / get_classification_metrics).
const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

function formatMetricName(name: string): string {
  if (name === "rmse") return "RMSE";
  if (name === "mae") return "MAE";
  if (name === "r2") return "R²";
  if (name === "f1") return "F1";
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function formatMetricValue(value: number): string {
  return Math.abs(value) >= 1000 ? value.toFixed(1) : value.toFixed(4);
}

function MetricComparisonRow({
  metric,
  championValue,
  challengerValue,
}: {
  metric: string;
  championValue: number;
  challengerValue: number;
}) {
  const lowerIsBetter = LOWER_IS_BETTER.has(metric);
  const challengerIsBetter = lowerIsBetter
    ? challengerValue < championValue
    : challengerValue > championValue;
  const championIsBetter = lowerIsBetter
    ? championValue < challengerValue
    : championValue > challengerValue;

  return (
    <tr className="border-t border-neutral-800/70">
      <td className="py-2 pl-3 pr-3 align-middle font-medium text-neutral-300">
        {formatMetricName(metric)}
      </td>
      <td
        className={[
          "py-2 pr-3 align-middle tabular-nums",
          championIsBetter ? "text-emerald-400" : "text-neutral-400",
        ].join(" ")}
      >
        {formatMetricValue(championValue)}
      </td>
      <td
        className={[
          "py-2 align-middle tabular-nums",
          challengerIsBetter ? "text-emerald-400" : "text-neutral-400",
        ].join(" ")}
      >
        {formatMetricValue(challengerValue)}
      </td>
    </tr>
  );
}

export function RetrainResultMessage({
  data,
  isResolving,
  onPromote,
  onReject,
}: RetrainResultMessageProps) {
  const metricNames = Array.from(
    new Set([...Object.keys(data.champion), ...Object.keys(data.challenger)])
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-hidden rounded-lg border border-neutral-800">
        <table className="w-full text-left text-xs">
          <thead className="bg-neutral-900 text-neutral-500">
            <tr>
              <th className="px-3 py-2 font-medium">Metric</th>
              <th className="px-3 py-2 font-medium">Current model</th>
              <th className="px-3 py-2 font-medium">New model</th>
            </tr>
          </thead>
          <tbody>
            {metricNames.map((metric) => (
              <MetricComparisonRow
                key={metric}
                metric={metric}
                championValue={data.champion[metric]}
                challengerValue={data.challenger[metric]}
              />
            ))}
          </tbody>
        </table>
      </div>

      {data.challengerThreshold !== null && (
        <p className="text-[11px] text-neutral-500">
          New model's tuned decision threshold:{" "}
          <span className="text-neutral-300">{data.challengerThreshold.toFixed(4)}</span>
        </p>
      )}

      {data.resolution ? (
        <div
          className={[
            "flex items-center gap-1.5 text-xs font-medium",
            data.resolution === "promoted" ? "text-emerald-400" : "text-neutral-500",
          ].join(" ")}
        >
          {data.resolution === "promoted" ? (
            <>
              <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
              Promoted to production
            </>
          ) : (
            <>
              <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
              Discarded
            </>
          )}
        </div>
      ) : (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onPromote}
            disabled={isResolving}
            aria-busy={isResolving}
            className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isResolving && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
            Promote new model
          </button>
          <button
            type="button"
            onClick={onReject}
            disabled={isResolving}
            aria-busy={isResolving}
            className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-neutral-700 px-4 py-2 text-sm font-medium text-neutral-300 transition-colors hover:border-red-500 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Discard
          </button>
        </div>
      )}
    </div>
  );
}
