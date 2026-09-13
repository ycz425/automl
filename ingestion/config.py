import os
from pathlib import Path
import dotenv

# Load this package's own .env regardless of the caller's cwd — plain load_dotenv()
# searches upward from the current working directory, which only found backend/.env
# by coincidence back when this package lived under backend/.
dotenv.load_dotenv(Path(__file__).resolve().parent / '.env')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Unset QDRANT_ENDPOINT falls back to QDRANT_PATH (on-disk local storage), and unset
# QDRANT_PATH falls back to an in-memory instance — so the pipeline runs with zero
# Qdrant setup during local development, while still pointing at a real deployment
# once QDRANT_ENDPOINT is configured.
QDRANT_ENDPOINT = os.getenv('QDRANT_ENDPOINT')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_PATH = os.getenv('QDRANT_PATH')

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '768'))

# Tokens-per-minute and requests-per-minute quotas for EMBEDDING_MODEL on this API
# key/project — embed_texts() paces batches to stay under both. Defaults match the
# Gemini API free tier (30k TPM, 15 RPM); raise these if billing is enabled and the
# account has higher limits (see https://ai.dev/rate-limit for your actual quota).
EMBEDDING_TPM_LIMIT = int(os.getenv('EMBEDDING_TPM_LIMIT', '30000'))
EMBEDDING_RPM_LIMIT = int(os.getenv('EMBEDDING_RPM_LIMIT', '15'))

DEFAULT_CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '2000'))
DEFAULT_CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '500'))
