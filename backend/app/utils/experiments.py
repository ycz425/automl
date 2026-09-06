from app.graph.schemas.user_request import MetricEntry
from app.graph.schemas.experiment import Experiment


def pick_best_experiment(primary_metric: MetricEntry, experiments: list[Experiment], return_index=False):
    direction = primary_metric.direction

    best_metric = None
    best_experiment = None
    best_idx = None
    for idx, experiment in enumerate(experiments):
        metric = [metric_entry.value for metric_entry in experiment.result.metrics if metric_entry.metric == primary_metric.name][0]

        if (
            (best_metric is None) or
            (direction == 'max' and metric > best_metric) or
            (direction == 'min' and metric < best_metric)
        ):
            best_metric = metric
            best_experiment = experiment
            best_idx = idx

    if return_index:
        return best_idx
    else:
        return best_experiment
