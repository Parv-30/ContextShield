# ContextShield

A context-aware multilingual (English, Hindi, Tamil) social media analytics
framework for harmful content detection, conversation escalation tracking,
and emerging toxicity trend analysis. Built as a coursework project — see
`docs/superpowers/specs/2026-09-22-contextshield-design.md` for the full
design rationale and `docs/superpowers/plans/2026-09-22-contextshield-implementation.md`
for the implementation plan.

## Components

- `backend/` — Python ML pipeline (XLM-RoBERTa multi-label classifier,
  formula-based escalation detection, BERTopic trend analysis) + FastAPI
  REST API + PostgreSQL persistence.
- `frontend/` — React + TypeScript + Tailwind CSS dashboard (Analyzer,
  Browse, Trends pages), styled per the design system in
  `design-system/contextshield/MASTER.md`.

## Setup

### 1. Backend

    cd backend
    pip install -e ".[dev]"

For GPU-accelerated training, install a CUDA-enabled PyTorch build matching
your driver *before* the line above (e.g. for CUDA 12.6:
`pip install torch --index-url https://download.pytorch.org/whl/cu126`) so
`pip install -e ".[dev]"` doesn't pull the CPU-only wheel.

Place licensed datasets in `backend/data/raw/` per `backend/data/README.md`,
or skip this to use the bundled synthetic fallback data (`backend/data/seed_fallback.jsonl`).

    python scripts/prepare_datasets.py
    python scripts/check_training_feasible.py   # optional: reports GPU/CPU device and timing estimate
    python scripts/train_classifier.py          # add --max-train-examples N to cap a large dataset
    python scripts/calibrate_thresholds.py

    psql -U postgres -c "CREATE DATABASE contextshield;"
    python scripts/init_db.py
    python scripts/seed_database.py

    uvicorn contextshield.api.main:app --reload --port 8000

Set `CONTEXTSHIELD_DATABASE_URL` (e.g.
`postgresql+psycopg2://postgres:<password>@localhost:5432/contextshield`)
if your local Postgres doesn't use the default `postgres/postgres`
credentials assumed by `contextshield/db/session.py`.

If HTTPS requests to huggingface.co fail with a certificate verification
error (common behind corporate/antivirus TLS-inspecting proxies), set
`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` to a CA bundle
that includes your proxy's root certificate merged with `certifi`'s bundle.

### 2. Frontend

    cd frontend
    npm install
    npm run dev

Open http://localhost:5173.

## Tests

    cd backend && pytest
    cd frontend && npm run test

## Known limitations

- The bundled classifier is trained on a small synthetic fallback dataset
  (`backend/data/seed_fallback.jsonl`) since the licensed HASOC and
  DravidianCodeMix/HASOC-Dravidian datasets require manual registration and
  download — see `backend/data/README.md`. Real accuracy numbers require
  those datasets.
- Escalation thresholds are calibrated from the same small dataset, so the
  stable/elevated/escalating bands are narrow; recalibrate with
  `scripts/calibrate_thresholds.py` once real data is used.
- Topic clustering (BERTopic) requires at least 10 harmful messages in the
  corpus to activate; below that, `topic_id`/`topic_terms` are `null`.
