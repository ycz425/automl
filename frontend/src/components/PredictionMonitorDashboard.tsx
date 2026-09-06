import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Database, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import type { PredictionLogEntry } from "../types/automl";

type PredictionMonitorDashboardProps = {
  history: PredictionLogEntry[];
  onOpenRetrain: () => void;
};

const SPARKLINE_WIDTH = 300;
const SPARKLINE_HEIGHT = 48;
const SPARKLINE_PAD = 4;

const ROW_TREND_WIDTH = 96;
const ROW_TREND_HEIGHT = 32;
const ROW_TREND_PAD = 3;

function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 100000) return value.toExponential(2);
  return value.toFixed(4);
}

const MAX_COLUMNS_SHOWN = 3;

function formatColumnList(columns: string[]): string {
  if (columns.length <= MAX_COLUMNS_SHOWN) return columns.join(", ");
  const shown = columns.slice(0, MAX_COLUMNS_SHOWN).join(", ");
  return `${shown} +${columns.length - MAX_COLUMNS_SHOWN} more`;
}

type SeriesPoint = { x: number; y: number };

// One column's score across every recorded prediction, plus the threshold
// it's being compared against, drawn as an axis-free mini line chart — the
// point isn't to read exact values (the table cells already show those),
// it's to see at a glance whether a score is trending toward its threshold.
function ColumnTrend({
  series,
  threshold,
  latestDrifted,
}: {
  series: SeriesPoint[];
  threshold: number | null;
  latestDrifted: boolean;
}) {
  if (series.length < 2) {
    return <span className="text-neutral-600">Not enough history yet</span>;
  }

  const values = series.map((p) => p.y);
  const minVal = threshold === null ? Math.min(...values) : Math.min(...values, threshold);
  const maxVal = threshold === null ? Math.max(...values) : Math.max(...values, threshold);
  const range = maxVal - minVal || 1;
  const maxX = series[series.length - 1].x || 1;

  const toX = (x: number) =>
    ROW_TREND_PAD + (x / maxX) * (ROW_TREND_WIDTH - 2 * ROW_TREND_PAD);
  const toY = (y: number) =>
    ROW_TREND_HEIGHT - ROW_TREND_PAD - ((y - minVal) / range) * (ROW_TREND_HEIGHT - 2 * ROW_TREND_PAD);

  const points = series.map((p) => `${toX(p.x)},${toY(p.y)}`).join(" ");
  const thresholdY = threshold === null ? null : toY(threshold);

  return (
    <svg
      width={ROW_TREND_WIDTH}
      height={ROW_TREND_HEIGHT}
      viewBox={`0 0 ${ROW_TREND_WIDTH} ${ROW_TREND_HEIGHT}`}
      role="img"
      aria-label={`Score trend, latest ${values[values.length - 1]}${threshold === null ? "" : `, threshold ${threshold}`}`}
    >
      {thresholdY !== null && (
        <line
          x1={0}
          x2={ROW_TREND_WIDTH}
          y1={thresholdY}
          y2={thresholdY}
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="2,2"
          className="text-neutral-700"
        />
      )}
      <polyline
        points={points}
        fill="none"
        strokeWidth={1.5}
        className={latestDrifted ? "text-amber-400" : "text-neutral-500"}
        stroke="currentColor"
      />
    </svg>
  );
}

// Renders inset inside PredictBar's shared box, styled as its own rounded
// card (matching CsvMessageBubble's convention) rather than a flush,
// edge-to-edge row.
export function PredictionMonitorDashboard({ history, onOpenRetrain }: PredictionMonitorDashboardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Cumulative rows processed so far, aligned to each entry — used as every
  // chart's x-axis so a large batch and a single-row batch are represented
  // proportionally, instead of every /predict call counting as one equal step.
  const { points, columnSeries } = useMemo(() => {
    let cumulativeRows = 0;
    const okPoints: SeriesPoint[] = [];
    const series: Record<string, SeriesPoint[]> = {};

    for (const entry of history) {
      cumulativeRows += entry.rows_recorded;
      if (entry.status === "ok") {
        okPoints.push({ x: cumulativeRows, y: entry.drifted_columns?.length ?? 0 });
        for (const [column, score] of Object.entries(entry.column_scores ?? {})) {
          if (score.value === null) continue;
          (series[column] ??= []).push({ x: cumulativeRows, y: score.value });
        }
      }
    }

    return { points: okPoints, columnSeries: series };
  }, [history]);

  if (history.length === 0) return null;

  const latest = history[history.length - 1];
  const maxCount = Math.max(1, ...points.map((p) => p.y));
  const maxX = points.length > 0 ? points[points.length - 1].x : 1;

  const toSvgX = (x: number) =>
    SPARKLINE_PAD + (maxX > 0 ? (x / maxX) * (SPARKLINE_WIDTH - 2 * SPARKLINE_PAD) : 0);
  const toSvgY = (y: number) =>
    SPARKLINE_HEIGHT -
    SPARKLINE_PAD -
    (y / maxCount) * (SPARKLINE_HEIGHT - 2 * SPARKLINE_PAD);

  const polylinePoints = points.map((p) => `${toSvgX(p.x)},${toSvgY(p.y)}`).join(" ");

  const columnScores = latest.status === "ok" ? latest.column_scores ?? {} : {};
  const columnEntries = Object.entries(columnScores);

  return (
    <div className="px-4 pt-3">
      <div className="overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900">
        <button
          type="button"
          onClick={() => setIsExpanded((prev) => !prev)}
          disabled={latest.status !== "ok"}
          aria-expanded={isExpanded}
          className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-neutral-800 disabled:cursor-default disabled:hover:bg-transparent"
        >
          <div
            className={[
              "flex h-9 w-9 shrink-0 items-center justify-center rounded-md",
              latest.status === "insufficient_data"
                ? "bg-neutral-500/10 text-neutral-400"
                : latest.dataset_drift
                  ? "bg-amber-500/10 text-amber-400"
                  : "bg-emerald-500/10 text-emerald-400",
            ].join(" ")}
          >
            {latest.status === "insufficient_data" ? (
              <Database className="h-4.5 w-4.5" aria-hidden="true" />
            ) : latest.dataset_drift ? (
              <AlertTriangle className="h-4.5 w-4.5" aria-hidden="true" />
            ) : (
              <CheckCircle2 className="h-4.5 w-4.5" aria-hidden="true" />
            )}
          </div>

          <div className="min-w-0 flex-1">
            <p className="truncate text-[11px] font-medium uppercase tracking-wide text-neutral-500">
              Model monitoring
            </p>
            <p className="truncate text-sm font-medium text-neutral-100">
              {latest.status === "insufficient_data"
                ? `Collecting data for drift monitoring — ${latest.window_size} of ${latest.rows_needed} rows`
                : latest.dataset_drift
                  ? `Drift detected in: ${formatColumnList(latest.drifted_columns ?? [])}`
                  : "No drift detected"}
            </p>
          </div>

          {points.length >= 2 && (
            <svg
              width={SPARKLINE_WIDTH / 3}
              height={SPARKLINE_HEIGHT / 2}
              viewBox={`0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`}
              preserveAspectRatio="none"
              className="hidden shrink-0 sm:block"
              aria-hidden="true"
            >
              <polyline
                points={polylinePoints}
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                className="text-indigo-400"
              />
            </svg>
          )}

          {latest.status === "ok" &&
            (isExpanded ? (
              <ChevronDown className="h-4 w-4 shrink-0 text-neutral-500" aria-hidden="true" />
            ) : (
              <ChevronRight className="h-4 w-4 shrink-0 text-neutral-500" aria-hidden="true" />
            ))}
        </button>

        {latest.status === "ok" && (
          <div className="border-t border-neutral-800 px-4 py-2.5">
            <button
              type="button"
              onClick={onOpenRetrain}
              className={[
                "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors",
                latest.dataset_drift
                  ? "text-amber-400 hover:bg-amber-500/10"
                  : "text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200",
              ].join(" ")}
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              {latest.dataset_drift ? "Drift detected — retrain model" : "Retrain model"}
            </button>
          </div>
        )}

        {isExpanded && latest.status === "ok" && (
          <div className="border-t border-neutral-800 bg-neutral-900 px-4 py-3">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-neutral-500">
              Per-feature drift — last {latest.window_size} rows
            </p>
            <div className="max-h-80 overflow-y-auto overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-neutral-900">
                  <tr className="text-neutral-500">
                    <th className="pb-1.5 pr-3 font-medium">Column</th>
                    <th className="pb-1.5 pr-3 font-medium">Method</th>
                    <th className="pb-1.5 pr-3 font-medium">Score</th>
                    <th className="pb-1.5 pr-3 font-medium">Threshold</th>
                    <th className="pb-1.5 font-medium">Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {columnEntries.map(([column, score]) => (
                    <tr key={column} className="border-t border-neutral-800/70">
                      <td
                        className={[
                          "py-2 pr-3 align-middle font-medium",
                          score.drifted ? "text-amber-400" : "text-neutral-300",
                        ].join(" ")}
                      >
                        {column}
                      </td>
                      <td className="py-2 pr-3 align-middle text-neutral-400">{score.method}</td>
                      <td className="py-2 pr-3 align-middle text-neutral-400">{formatScore(score.value)}</td>
                      <td className="py-2 pr-3 align-middle text-neutral-400">{formatScore(score.threshold)}</td>
                      <td className="py-2 align-middle">
                        <ColumnTrend
                          series={columnSeries[column] ?? []}
                          threshold={score.threshold}
                          latestDrifted={score.drifted}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
