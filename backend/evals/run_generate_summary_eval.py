import json
from pydantic import BaseModel
from app.agents.summary_agent import SummaryAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.experiment import Experiment
from langsmith import aevaluate
from evals.utils import llm_judge


async def run_agent(inputs: dict):
    summary_agent = SummaryAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    experiments = [Experiment.model_validate(e) for e in inputs['experiments']]

    summary = await summary_agent.generate_summary(user_request, experiments)

    return {'summary': summary}


def _best_experiment(user_request: dict, experiments: list[dict]):
    primary_metric = user_request['primary_metric']
    direction = primary_metric['direction']

    best_idx = None
    best_value = None
    for idx, experiment in enumerate(experiments):
        value = next(m['value'] for m in experiment['result']['metrics'] if m['metric'] == primary_metric['name'])
        if best_value is None or (direction == 'max' and value > best_value) or (direction == 'min' and value < best_value):
            best_value = value
            best_idx = idx

    return best_idx, best_value


class SummaryAccuracyJudgment(BaseModel):
    identifies_correct_best_experiment: bool
    identifies_correct_best_experiment_reasoning: str
    states_correct_metric_value: bool
    states_correct_metric_value_reasoning: str
    no_fabrication: bool
    no_fabrication_reasoning: str
    correct_ordering_and_numbering: bool
    correct_ordering_and_numbering_reasoning: str


async def summary_accuracy(inputs: dict, outputs: dict):
    user_request = inputs['user_request']
    experiments = inputs['experiments']
    summary = outputs['summary']

    best_idx, best_value = _best_experiment(user_request, experiments)
    primary_metric = user_request['primary_metric']

    prompt = f"""
    Judge whether the following summary of a machine-learning experimentation process is
    factually accurate, given the ground truth below.

    Ground truth:
    - Primary metric: {primary_metric['name']} ({primary_metric['direction']})
    - Experiments, in chronological order (zero-based list here, but the summary should refer
      to them using one-based numbering, e.g. index 0 is "Experiment 1"):
      {json.dumps(experiments, indent=2)}
    - The actual best experiment is at zero-based index {best_idx} (i.e. "Experiment {best_idx + 1}"),
      with {primary_metric['name']} = {best_value}.

    Summary to judge:
    {summary}

    Instructions:
    - Judge whether the summary correctly identifies Experiment {best_idx + 1} as the best experiment.
    - Judge whether the summary states the correct {primary_metric['name']} value ({best_value}) for
      the best experiment, allowing for reasonable rounding.
    - Judge whether the summary avoids fabricating any approach, metric, or number not present in
      the experiments list above.
    - Judge whether experiments are described in chronological order using one-based numbering.
    """

    judgment = await llm_judge(prompt, SummaryAccuracyJudgment)

    return [
        {
            'key': 'identifies_correct_best_experiment',
            'score': judgment.identifies_correct_best_experiment,
            'comment': judgment.identifies_correct_best_experiment_reasoning
        },
        {
            'key': 'states_correct_metric_value',
            'score': judgment.states_correct_metric_value,
            'comment': judgment.states_correct_metric_value_reasoning
        },
        {
            'key': 'no_fabrication',
            'score': judgment.no_fabrication,
            'comment': judgment.no_fabrication_reasoning
        },
        {
            'key': 'correct_ordering_and_numbering',
            'score': judgment.correct_ordering_and_numbering,
            'comment': judgment.correct_ordering_and_numbering_reasoning
        },
    ]


def threshold_disclosure(inputs: dict, outputs: dict):
    """Deterministic complement to summary_accuracy: the best experiment's tuned decision
    threshold (when there is one) drives deployed predictions, so the prompt requires the
    summary to state it. Checked by direct substring search rather than an LLM judge."""
    user_request = inputs['user_request']
    experiments = inputs['experiments']
    summary = outputs['summary']

    best_idx, _ = _best_experiment(user_request, experiments)
    threshold = experiments[best_idx]['result'].get('threshold')

    if threshold is None:
        return None

    candidates = {str(threshold), f'{threshold:.2f}', f'{threshold:.4f}'}
    mentioned = any(candidate in summary for candidate in candidates)

    return {
        'key': 'threshold_disclosure',
        'score': 1.0 if mentioned else 0.0,
        'comment': 'Tuned threshold value is stated in the summary.' if mentioned else f'Tuned threshold ({threshold}) is not stated in the summary.'
    }


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_generate_summary',
        evaluators=[summary_accuracy, threshold_disclosure],
        experiment_prefix='automl_generate_summary_eval'
    ))
