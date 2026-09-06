from app.agents.base import LLMAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.experiment import Experiment
from app.services.tracing import traced_interactions_create
from app.utils.experiments import pick_best_experiment
from langsmith import traceable


class SummaryAgent(LLMAgent):
    @traceable(name="SummaryAgent.generate_summary")
    async def generate_summary(self, user_request: UserRequest, experiments: list[Experiment]):
        best_experiment_idx = pick_best_experiment(user_request.primary_metric, experiments, return_index=True)

        prompt = f"""
        Using the provided user_request, ordered list of experiments, and best_experiment_idx, generate a concise summary of the experimentation process.

        User request:
        {user_request}

        Experiments:
        {experiments}

        Best experiment index:
        {best_experiment_idx}

        Requirements:

        - Describe each experiment in chronological order.
        - For each experiment, briefly summarize the approach, important configuration changes, and results.
        - Explain how the experiments evolved based on earlier outcomes.
        - Assume best experiment index is zero-based, but refer to experiments using one-based numbering.
        - State why the experiment specified by best experiment index was the best based on the primary metric specified in user request.
        - Include the best experiment's primary metric value and any important tradeoffs or supporting metrics.
        - For any experiment whose result has a non-null `threshold` (binary classification with a tuned decision threshold), state that tuned threshold value alongside its metrics — especially for the best experiment, since that threshold is the one actually used for deployed predictions.
        - Do not invent missing information.
        - Keep the summary clear, factual, and concise.
        - Return only the final summary using markdown.
        """

        interaction = await traced_interactions_create(
            self.client,
            model=self.model,
            input=prompt,
            generation_config={
                'thinking_level': 'low',
                'temperature': 0
            }
        )

        return interaction.output_text
