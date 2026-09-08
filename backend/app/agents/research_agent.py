import os
from google import genai
from qdrant_client import QdrantClient
from app.agents.base import GEMINI_API_KEY
from app.utils.embeddings import embed_texts
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis, ColumnProfile
from app.graph.schemas.research import SearchResult, Point
from langsmith import traceable
from datetime import datetime

QDRANT_ENDPOINT = os.getenv('QDRANT_ENDPOINT')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '768'))

# Tokens-per-minute quota for EMBEDDING_MODEL on this API key/project — embed_texts()
# batches and paces requests to stay under this. Set to your actual quota (visible on
# the Google AI Studio / Cloud Console quota page) if it differs from the default.
EMBEDDING_TPM_LIMIT = int(os.getenv('EMBEDDING_TPM_LIMIT', '30000'))

_IMBALANCE_THRESHOLD = 0.35


class ResearchAgent:
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        tpm_limit: int = EMBEDDING_TPM_LIMIT,
        verbose: bool = False,
    ):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.dimensions = dimensions
        self.tpm_limit = tpm_limit
        self.qdrant_client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
        self.qdrant_collection = 'research'
        self.verbose = verbose

    def embed_texts(self, texts: list[str], task_type: str = 'RETRIEVAL_QUERY') -> list[list[float]]:
        if self.verbose:
            print(f'{datetime.now()}     Embedding text...')
        return embed_texts(texts, self.client, self.model, self.dimensions, self.tpm_limit, task_type=task_type)

    def _target_column_profile(self, dataset_profile: DatasetProfile, target_column: str | None) -> ColumnProfile | None:
        if target_column is None:
            return None
        return next((column for column in dataset_profile.columns if column.name == target_column), None)

    def _class_balance_description(
        self,
        user_request: UserRequest,
        dataset_analysis: DatasetAnalysis,
        dataset_profile: DatasetProfile,
    ) -> str | None:
        if user_request.task_type == 'regression':
            return None

        target_column = self._target_column_profile(dataset_profile, dataset_analysis.target_column)
        if target_column is None or not target_column.category_counts:
            return None

        total = sum(target_column.category_counts.values())
        if total == 0:
            return None

        if user_request.task_type == 'binary_classification' and dataset_analysis.positive_class is not None:
            positive_count = target_column.category_counts.get(dataset_analysis.positive_class)
            if positive_count is not None:
                pct = round(100 * positive_count / total)
                label = 'imbalanced' if pct / 100 <= _IMBALANCE_THRESHOLD or pct / 100 >= 1 - _IMBALANCE_THRESHOLD else 'roughly balanced'
                return f'target class is {label} (~{pct}% positive)'

        # Multiclass, or binary without a resolved positive_class: report the class-count spread.
        counts = sorted(target_column.category_counts.values())
        min_pct = round(100 * counts[0] / total)
        max_pct = round(100 * counts[-1] / total)
        label = ', imbalanced' if min_pct / 100 <= _IMBALANCE_THRESHOLD else ''
        return f'{len(counts)} target classes, ranging from ~{min_pct}% to ~{max_pct}% of the data{label}'

    def build_query(
        self,
        user_request: UserRequest,
        dataset_profile: DatasetProfile,
        dataset_analysis: DatasetAnalysis,
    ) -> str:
        """Render pipeline state into a short natural-language query for embedding-based
        retrieval. Deterministic string templating rather than an LLM call: dense retrieval
        embedding models are trained for short, imperfectly-phrased queries against long
        documents, so template output retrieves identically to LLM-polished prose here,
        without the added latency, cost, and run-to-run non-determinism."""
        sentences = []

        task_type = (user_request.task_type or 'machine learning').replace('_', ' ')
        target = user_request.target_description or 'an unspecified target'
        sentences.append(f'{task_type.capitalize()} task predicting {target}.')

        facts = [f'~{dataset_profile.num_rows:,} rows']

        balance = self._class_balance_description(user_request, dataset_analysis, dataset_profile)
        if balance:
            facts.append(balance)

        if dataset_analysis.group_column:
            group_label = user_request.group_description or dataset_analysis.group_column
            facts.append(f'grouped by {group_label}')

        sentences.append(', '.join(facts).capitalize() + '.')

        if user_request.constraints:
            sentences.append('Constraints: ' + '; '.join(user_request.constraints) + '.')

        if user_request.preferences:
            sentences.append('Preferences: ' + '; '.join(user_request.preferences) + '.')

        if user_request.primary_metric:
            sentences.append(f'Primary metric: {user_request.primary_metric.name}.')

        return ' '.join(sentences)

    @traceable(name="ResearchAgent.search")
    def search(
        self,
        user_request: UserRequest,
        dataset_profile: DatasetProfile,
        dataset_analysis: DatasetAnalysis
    ):
        try:
            query = self.build_query(user_request, dataset_profile, dataset_analysis)
            embedding = self.embed_texts([query])
            if self.verbose:
                print(f'{datetime.now()}     Querying collection...')
            response = self.qdrant_client.query_points(self.qdrant_collection, embedding[0])

            return SearchResult(
                points=[
                    Point(text=point.payload['text'], source=point.payload['source'], page_number= point.payload['page_number'], score=point.score)
                    for point in response.points
                ]
            )
        except:
            return None
    