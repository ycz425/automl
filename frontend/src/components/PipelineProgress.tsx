import { Check, Repeat } from "lucide-react";
import type { AutoMLNode } from "../types/automl";

type PipelineStage = {
  id: string;
  label: string;
  nodes: AutoMLNode[];
};

// plan_agent and experiment_agent loop back and forth a variable number of
// times (design a plan, try it, refine the plan, try again, ...) before
// handing off to output_agent. Tracking them as one stage means looping back
// to plan_agent after experiment_agent doesn't visually "uncomplete" a step
// that already finished — it stays a single current stage with a round
// counter instead.
const PIPELINE_STAGES: PipelineStage[] = [
  { id: "prompt", label: "Understanding your request", nodes: ["prompt_agent"] },
  { id: "data", label: "Analyzing the dataset", nodes: ["data_agent"] },
  {
    id: "plan_experiment",
    label: "Designing & training",
    nodes: ["plan_agent", "experiment_agent"],
  },
  { id: "output", label: "Preparing the final results", nodes: ["output_agent"] },
];

const LOOPING_STAGE_ID = "plan_experiment";

function findStageIndex(node?: AutoMLNode): number {
  if (!node) return -1;
  return PIPELINE_STAGES.findIndex((stage) => stage.nodes.includes(node));
}

type PipelineProgressProps = {
  currentNode?: AutoMLNode;
  /** Which round of the plan/experiment loop is active, if currently in it. */
  iteration?: number;
};

export function PipelineProgress({ currentNode, iteration }: PipelineProgressProps) {
  const currentIndex = findStageIndex(currentNode);

  return (
    <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-2" aria-label="Pipeline progress">
      {PIPELINE_STAGES.map((stage, index) => {
        const isCompleted = currentIndex >= 0 && index < currentIndex;
        const isCurrent = index === currentIndex;
        const isLooping =
          isCurrent && stage.id === LOOPING_STAGE_ID && Boolean(iteration && iteration > 1);

        return (
          <li key={stage.id} className="flex items-center gap-1.5">
            <span
              className={[
                "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors",
                isCompleted
                  ? "border-indigo-500/40 bg-indigo-500/10 text-indigo-300"
                  : isCurrent
                  ? "border-indigo-400 bg-indigo-500/20 text-indigo-200"
                  : "border-neutral-700 bg-neutral-800/40 text-neutral-500",
              ].join(" ")}
              aria-current={isCurrent ? "step" : undefined}
            >
              {isCompleted ? (
                <Check className="h-3 w-3" aria-hidden="true" />
              ) : (
                <span
                  className={[
                    "h-1.5 w-1.5 rounded-full",
                    isCurrent ? "animate-pulse-dot bg-indigo-300" : "bg-neutral-600",
                  ].join(" ")}
                  aria-hidden="true"
                />
              )}
              {stage.label}
              {isLooping && (
                <span
                  className="flex items-center gap-1 rounded-full bg-indigo-500/20 px-1.5 py-0.5 text-indigo-200"
                  title={`Round ${iteration} of designing and training`}
                >
                  <Repeat className="h-2.5 w-2.5" aria-hidden="true" />
                  {iteration}
                </span>
              )}
            </span>
            {index < PIPELINE_STAGES.length - 1 && (
              <span
                className={[
                  "h-px w-2.5 shrink-0",
                  isCompleted ? "bg-indigo-500/40" : "bg-neutral-700",
                ].join(" ")}
                aria-hidden="true"
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}
