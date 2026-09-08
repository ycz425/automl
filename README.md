# AutoML Agent

A chat-driven AutoML pipeline: describe a dataset and a prediction task in plain language,
and a LangGraph multi-agent pipeline profiles the data, plans an approach, iterates on
experiments, and produces a deployable model with training/prediction scripts. Includes
post-deployment prediction drift monitoring and a human-in-the-loop retraining flow.

## Architecture

**Backend** (`backend/`) — FastAPI + LangGraph, Python.

**Frontend** (`frontend/`) — React + TypeScript + Vite + Tailwind, a chat interface for
driving the pipeline, browsing run artifacts, monitoring predictions, and approving
retraining.

**Ingestion** (`ingestion/`) — standalone PDF-to-Qdrant pipeline (load, chunk, embed,
upsert) that populates the corpus `research_agent` queries. Decoupled from the backend
app — no shared imports — so it ships its own `requirements.txt` and `.env`.

### Pipeline

A LangGraph state machine (`backend/app/graph/graph.py`) wires together:

```
prompt_agent → data_agent → data_splitting → research_agent → plan_agent ⇄ experiment_agent → output_agent → success
                    ↑____________________________________________________↑
                          (either agent can request clarification)
```

- **`prompt_agent`** (`PromptAgent`) — parses the user's free-text request into a
  structured `UserRequest` (task type, target, metrics, evaluation method, constraints).
- **`data_agent`** (`DataAgent`) — profiles the uploaded dataset and identifies the target,
  feature, excluded, and group columns (`DatasetAnalysis`).
- **`data_splitting`** (`DataSplitter`) — deterministic (no LLM call): builds train/validation
  or cross-validation fold indices from the request's evaluation method.
- **`research_agent`** (`ResearchAgent`) — runs once per pipeline run, embedding a query
  built from the request and dataset shape (task type, target, class balance, constraints)
  and retrieving relevant excerpts from a Qdrant corpus. The result is stored in state and
  passed to `plan_agent` on every call, initial plan and every replan alike — not
  recomputed per replan. Embedding + vector search only, no LLM generation; failures
  degrade gracefully to no research context rather than failing the run.
- **`plan_agent`** (`PlanAgent`) — proposes a model architecture, preprocessing, and
  training strategy, incorporating cited research excerpts when available; revises the
  plan after each experiment.
- **`experiment_agent`** (`ExperimentAgent`) — generates and runs the experiment code for a
  plan, repairing it on failure, and loops back to `plan_agent` until the pipeline decides
  to stop.
- **`output_agent`** (`OutputAgent`) — generates the final `train.py`/`predict.py`
  deployment scripts for the best experiment and trains the deployed model.
- **`summary_agent`** (`SummaryAgent`) — writes the human-readable summary of the
  experimentation process shown to the user.
- **`clarification_agent`** (`ClarificationAgent`) — asks the user a clarifying question
  when `prompt_agent`/`data_agent` can't resolve something, and applies their answer back
  onto the structured data.

Every agent except `DataSplitter` and `ResearchAgent` (neither needs LLM generation)
inherits from `LLMAgent` (`app/agents/base.py`), which owns the Gemini client and a
shared `generate_structured(...)` retry-until-schema-valid helper
(`app/utils/structured_generation.py`).

### Post-deployment features

- **Prediction serving + drift monitoring** (`PredictionService`) — runs the deployed
  `predict.py` against new data and computes an Evidently data-drift report against a
  reference sample of the training data.
- **Retraining** (`RetrainService`) — labels the accumulated rolling window of prediction
  data, re-runs k-fold evaluation for both the currently deployed ("champion") and a
  freshly retrained ("challenger") model, and surfaces a side-by-side metrics comparison
  in the chat UI for the user to promote or reject.

### Storage

`FileStorage` (`app/services/file_storage.py`) manages uploaded datasets and per-run
artifacts (scripts, metrics, plans, drift/retrain results) under `STORAGE_ROOT`, and run
metadata is tracked in Firestore.

## Eval suite

`backend/evals/` — one LangSmith eval per LLM call (see the scripts' `aevaluate(...)`
calls), each combining LLM-as-judge and deterministic checks. `backend/evals/examples/`
holds the generator scripts and `.jsonl` datasets used to seed them. CI
(`.github/workflows/langsmith-evals.yml`) runs the full suite informationally on every PR
touching `backend/**`.

## Getting started

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # or use the repo's .venv
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Environment variables (`.env` in `backend/`):

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API access for all agents |
| `STORAGE_ROOT` | Root directory for uploaded datasets and run artifacts |
| `GOOGLE_CLOUD_PROJECT` | GCP project for Firestore run metadata |
| `LANGSMITH_API_KEY` | Tracing and eval-suite access (optional but recommended) |
| `LANGSMITH_PROJECT` | LangSmith project name for traces/evals |
| `QDRANT_ENDPOINT` / `QDRANT_API_KEY` | Qdrant cluster `research_agent` queries |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` / `EMBEDDING_TPM_LIMIT` | `research_agent`'s embedding config — defaults to `gemini-embedding-001` at 768 dims, 30k TPM |

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Ingestion

```bash
pip install -r ingestion/requirements.txt
python -m ingestion.pipeline path/to/file.pdf --collection my_collection   # run from the repo root
```

Environment variables (`.env` in `ingestion/`):

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini embeddings API access |
| `QDRANT_ENDPOINT` / `QDRANT_API_KEY` | Qdrant deployment to upsert into — omit both to fall back to `QDRANT_PATH` (on-disk) or an in-memory instance |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | Defaults to `gemini-embedding-001` at 768 dimensions |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Defaults to 1000/200 characters |

## Repo layout

```
backend/
  app/
    agents/       # one class per pipeline stage — most subclass LLMAgent; DataSplitter
                  # and ResearchAgent don't need generation
    graph/        # LangGraph state machine: graph.py, nodes/, routers/, schemas/
    services/     # FileStorage, PredictionService, RetrainService, status tracking, tracing
    utils/        # shared structured-generation + subprocess/model-script helpers
  evals/          # LangSmith eval scripts + example datasets
frontend/
  src/
    api/          # backend HTTP client
    components/   # chat UI, artifact browser, prediction/drift dashboard, retrain panel
    hooks/        # chat state, retrain flow
    types/        # shared TS types
    utils/        # chat message factories, formatting helpers
ingestion/        # standalone PDF -> chunk -> embed -> Qdrant upsert pipeline
```
