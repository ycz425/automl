from app.agents.base import LLMAgent
from app.graph.schemas.user_request import UserRequest, MetricEntry
from app.graph.schemas.experiment import Experiment
from app.graph.schemas.output import OutputScripts
from app.graph.schemas.data_info import DatasetAnalysis
from app.services.tracing import traced_interactions_create
from app.utils.model_scripts import create_venv, run_train_script, run_predict_script, format_subprocess_error
from langsmith import traceable
from datetime import datetime
import pandas as pd
import subprocess
import tempfile
import shutil
import json
import os


class OutputAgent(LLMAgent):
    def best_experiment(self, primary_metric: MetricEntry, experiments: list[Experiment], return_index=False):
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

    @traceable(name="OutputAgent.generate_scripts")
    async def generate_scripts(self, experiment: Experiment, dataset_analysis: DatasetAnalysis, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     generating scripts...')

        prompt = (
            "You are generating the final deployment artifacts for an AutoML pipeline.\n\n"
            "The experiment below has already been selected as the best-performing experiment.\n\n"
            "Your task is to generate production-ready scripts that reproduce this experiment.\n\n"
            "Selected experiment:\n"
            f"{experiment.model_dump_json(indent=2)}\n\n"
            "Dataset analysis:\n"
            f"{dataset_analysis.model_dump_json(indent=2)}\n\n"
            "Requirements:\n\n"
            "- Produce output matching the Output schema exactly.\n"
            "- The training script must train the selected model on the full dataset.\n"
            "- Preserve the preprocessing, feature selection, model architecture, and hyperparameters from the selected experiment.\n"
            "- The training script must save the complete fitted pipeline as a joblib file to the output path provided as a command-line argument.\n"
            "- The input CSV accepted by the training script must contain the columns listed in dataset analysis's feature_columns plus the target column; drop any other columns present, if any (including the group column, if one is specified — it is only used for splitting, never as a model input).\n"
            "- The prediction script must load the fitted pipeline from the model joblib path provided as a command-line argument, accept unseen CSV data, and write predictions to a CSV file.\n"
            "- The input CSV accepted by the prediction script must contain the columns listed in dataset analysis's feature_columns; drop any other columns present, if any.\n"
            "- The output CSV must contain exactly one column, named 'pred', with one row per input row in the same order as the input. Do not include the input columns or an index column in the output.\n"
            "- For regression, 'pred' must contain the predicted numeric value.\n"
            "- For multiclass classification, 'pred' must contain the exact class label as it appears in the target column (e.g. the original string or value used in training) — never a numeric class index or another encoded representation.\n"
            "- For binary classification, the prediction script must accept an optional '--threshold' command-line argument giving the path to a threshold.json file; omit the flag entirely to skip thresholding. If '--threshold' is provided and the file's 'threshold' value is not null, compute the predicted probability of the class named in dataset analysis's positive_class field, then set 'pred' to positive_class when that probability is greater than or equal to the threshold and to the other class otherwise. If '--threshold' is provided but the file's 'threshold' value is null, set 'pred' to the model's default predicted class label. If '--threshold' is not provided at all, set 'pred' to the predicted probability of the class named in dataset analysis's positive_class field, as a float, instead of a class label. The '--threshold' argument has no effect for regression or multiclass classification.\n"
            "- Use only the dependencies listed in the output.\n"
            "- Do not change the selected experiment unless necessary to fix an implementation issue.\n"
            "- Both scripts must be executable directly from the command line.\n"
            "- Use argparse to parse command-line arguments.\n"
            "- The training script must support:\n"
            "    python train.py --input DATA.csv --output model.joblib\n"
            "- The prediction script must support:\n"
            "    python predict.py --model model.joblib --input DATA.csv --output predictions.csv [--threshold threshold.json]\n"
            "- Include a standard `if __name__ == '__main__':` entry point in both scripts.\n"
            "- Do not include explanations, markdown, or code fences.\n"
            "- Return only valid JSON matching the Output schema."
        )

        return await self.generate_structured(prompt, OutputScripts, max_retries=max_retries, label="OutputScripts")

    @traceable(name="OutputAgent.repair_scripts")
    async def repair_scripts(self, scripts: OutputScripts, error_message: str, experiment: Experiment, dataset_analysis: DatasetAnalysis, max_retries=5):
        if self.verbose:
            print(f'{datetime.now()}     repairing scripts...')

        prompt = f"""
            Repair the previously generated deployment scripts.

            Failure:
            {error_message}

            Previous scripts:
            {scripts.model_dump_json(indent=2)}

            Selected experiment:
            {experiment.model_dump_json(indent=2)}

            Dataset analysis:
            {dataset_analysis.model_dump_json(indent=2)}

            Requirements:
            - Return the complete corrected OutputScripts.
            - Preserve the selected experiment, including preprocessing, feature selection, model architecture, and hyperparameters.
            - Fix the reported dependency, import, syntax, execution, serialization, or command-line interface error.
            - The training script must train the selected model on the full dataset.
            - The training script must support:
                python train.py --input DATA.csv --output model.joblib
            - The prediction script must support:
                python predict.py --model model.joblib --input DATA.csv --output predictions.csv [--threshold threshold.json]
            - The prediction script must load the fitted model from the --model path.
            - The input CSV accepted by the training script must contain the columns listed in dataset analysis's feature_columns plus the target column; drop any other columns present, if any (including the group column, if one is specified — it is only used for splitting, never as a model input).
            - The input CSV accepted by the prediction script must contain the columns listed in dataset analysis's feature_columns; drop any other columns present, if any.
            - The output CSV must contain exactly one column, named 'pred', with one row per input row in the same order as the input. Do not include the input columns or an index column in the output.
            - For regression, 'pred' must contain the predicted numeric value.
            - For multiclass classification, 'pred' must contain the exact class label as it appears in the target column (e.g. the original string or value used in training) — never a numeric class index or another encoded representation.
            - For binary classification, the prediction script must accept an optional '--threshold' command-line argument giving the path to a threshold.json file; omit the flag entirely to skip thresholding. If '--threshold' is provided and the file's 'threshold' value is not null, compute the predicted probability of the class named in dataset analysis's positive_class field, then set 'pred' to positive_class when that probability is greater than or equal to the threshold and to the other class otherwise. If '--threshold' is provided but the file's 'threshold' value is null, set 'pred' to the model's default predicted class label. If '--threshold' is not provided at all, set 'pred' to the predicted probability of the class named in dataset analysis's positive_class field, as a float, instead of a class label. The '--threshold' argument has no effect for regression or multiclass classification.
            - Ensure all required dependencies are listed.
            - Ensure both scripts are complete, directly executable, and use argparse.
            - Do not change the selected experiment unless necessary to fix an implementation issue.
            - Do not fabricate outputs, return a patch, include markdown, or add explanations.
        """

        return await self.generate_structured(prompt, OutputScripts, max_retries=max_retries, label="OutputScripts")

    async def validate_predict_script(self, env_python:str, predict_path: str, data_path: str, model_path: str, threshold_path: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_input = os.path.join(temp_dir, "test_input.csv")
            test_output = os.path.join(temp_dir, "test_output.csv")

            df = pd.read_csv(data_path)
            df = df.iloc[:5]
            df.to_csv(test_input, index=False)

            await run_predict_script(env_python, predict_path, model_path, test_input, test_output, threshold_path)
            df = pd.read_csv(test_output)

            output_columns = df.columns.tolist()
            assert output_columns == ['pred'], (
                f"Expected the output CSV to contain exactly one column named 'pred', "
                f"but found: {output_columns}"
            )

            assert len(df) == len(pd.read_csv(test_input)), (
                f"Expected {len(pd.read_csv(test_input))} output rows (one per input row), got {len(df)}."
            )

    @traceable(name="OutputAgent.generate_summary")
    async def generate_summary(self, user_request: UserRequest,  experiments: list[Experiment]):
        best_experiment_idx = self.best_experiment(user_request.primary_metric, experiments, return_index=True)

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

    @traceable(name="OutputAgent.generate_output")
    async def generate_output(self, user_request: UserRequest, dataset_analysis: DatasetAnalysis, experiments: list[Experiment], data_path: str, output_dir="out", max_retries=5):
        primary_metric = user_request.primary_metric
        best_experiment = self.best_experiment(primary_metric, experiments)
        output_scripts = await self.generate_scripts(best_experiment, dataset_analysis)

        train_path = os.path.join(output_dir, 'train.py')
        predict_path = os.path.join(output_dir, 'predict.py')
        model_path = os.path.join(output_dir, 'model.joblib')
        metrics_path = os.path.join(output_dir, 'metrics.json')
        plan_path = os.path.join(output_dir, 'plan.json')
        user_request_path = os.path.join(output_dir, 'user_request.json')
        dataset_analysis_path = os.path.join(output_dir, 'dataset_analysis.json')
        threshold_path = os.path.join(output_dir, 'threshold.json')
        requirements_path = os.path.join(output_dir, 'requirements.txt')
        reference_data_path = os.path.join(output_dir, 'reference_data.csv')

        for attempt in range(max_retries + 1):
            if self.verbose:
                print(f'{datetime.now()}     generating output...')
            try:
                with open(train_path, 'w') as f:
                    f.write(output_scripts.train_script)
                with open(predict_path, 'w') as f:
                    f.write(output_scripts.predict_script)
                with open(requirements_path, 'w') as f:
                    f.write('\n'.join(output_scripts.dependencies))
                with open(metrics_path, 'w') as f:
                    json.dump(best_experiment.result.model_dump(), f, indent=4)
                with open(plan_path, 'w') as f:
                    json.dump(best_experiment.plan.model_dump(), f, indent=4)
                with open(dataset_analysis_path, 'w') as f:
                    json.dump(dataset_analysis.model_dump(), f, indent=4)
                with open(user_request_path, 'w') as f:
                    json.dump(user_request.model_dump(), f, indent=4)
                with open(threshold_path, 'w') as f:
                    json.dump({'threshold': best_experiment.result.threshold}, f, indent=4)

                df = pd.read_csv(data_path)
                df = df[dataset_analysis.feature_columns]
                df = df.sample(n=min(len(df), 2000), random_state=42)
                df.to_csv(reference_data_path, index=False)


                with tempfile.TemporaryDirectory() as env_dir:
                    env_python = await create_venv(env_dir, requirements_path)

                    await run_train_script(env_python, train_path, data_path, model_path)

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Training script did not produce {model_path}")

                    await self.validate_predict_script(env_python, predict_path, data_path, model_path, threshold_path)

                    return
            except (subprocess.CalledProcessError, FileNotFoundError, AssertionError) as e:
                if attempt == max_retries:
                    shutil.rmtree(output_dir)
                    raise
                if self.verbose:
                    print(f'{datetime.now()}     output generation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
                error_message = format_subprocess_error(e)
                output_scripts = await self.repair_scripts(output_scripts, error_message, best_experiment, dataset_analysis)
