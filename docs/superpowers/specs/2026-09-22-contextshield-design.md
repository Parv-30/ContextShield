# ContextShield — Design Spec

Date: 2026-09-22
Status: Approved for implementation planning

## 1. Overview

ContextShield is a coursework project: a context-aware multilingual (English,
Hindi, Tamil) social media analytics framework that detects harmful content in
individual messages, tracks escalation of toxicity across a conversation
thread, and surfaces emerging toxicity trends across a corpus over time. It
ships as a full web application: a React frontend, a FastAPI backend, a
PostgreSQL database, and a Python ML pipeline.

Goal class: coursework deliverable (report + working full web app demo), not
a production system. Scope has been deliberately narrowed from the original
idea to keep one genuine ML contribution (the multilingual classifier) while
keeping escalation and trend analysis lightweight, explainable, and backed by
explicit math suitable for a written report.

## 2. Architecture

```
React SPA  →  FastAPI backend  →  ML pipeline (classifier / escalation / trends)
                    ↓
              PostgreSQL
```

- **Frontend**: React single-page app. Three pages: Analyzer, Browse, Trends
  Dashboard.
- **Backend**: FastAPI REST API. Owns request validation, persistence, and
  orchestration of the ML pipeline.
- **ML pipeline**: pure Python, callable as a library from the backend (no
  separate service for coursework scope). Three components: classifier,
  escalation detector, trend analyzer.
- **Database**: PostgreSQL. Stores conversations, messages, per-message
  labels, escalation snapshots, and trend snapshots.

## 3. ML Pipeline

### 3.1 Harmful Content Classifier

- Model: `xlm-roberta-base` fine-tuned with a multi-label classification head
  (sigmoid activation, binary cross-entropy loss per label).
- Labels: `hate_speech`, `offensive`, `threat`, `harassment`, `none`.
- Training data: HASOC (English, Hindi) + DravidianCodeMix / HASOC-Dravidian
  (Tamil), remapped to the common 5-label schema. Label mapping between
  source datasets' native taxonomies and the common schema must be documented
  explicitly (this mapping is a first-class artifact for the report, not an
  implementation detail to bury).
- Output per message: probability per label, plus a derived
  **toxicity score** $s = \max(\text{prob}_{\text{hate\_speech}},
  \text{prob}_{\text{offensive}}, \text{prob}_{\text{threat}},
  \text{prob}_{\text{harassment}})$ (i.e., max of the non-"none" label
  probabilities).
- Evaluation: per-category precision/recall/F1, macro-F1, confusion analysis
  on a held-out test split.

### 3.2 Escalation Detector

Operates on a conversation's ordered messages $m_1, \dots, m_n$ with toxicity
scores $s_1, \dots, s_n \in [0,1]$ from 3.1.

- **Windowed mean** (window size $w$, default $w=3$):
  $\bar{s}_k = \frac{1}{w}\sum_{i=k-w+1}^{k} s_i$, defined for $k \ge w$.
- **Escalation slope** $\beta_k$: least-squares linear regression slope of
  $s_i$ against index $i$ over the same window of the last $w$ messages
  ending at $k$.
- **Escalation score**:
  $E_k = \alpha \cdot \bar{s}_k + (1-\alpha) \cdot \max(\beta_k, 0)$,
  default $\alpha = 0.6$. Negative slopes are clamped to 0 so only rising
  trends contribute to escalation (a falling trend should not offset a high
  absolute severity).
- **Classification bands**: thresholds $\tau_{low} < \tau_{high}$ calibrated
  as percentiles (e.g. 60th and 85th percentile) of $E_k$ computed over the
  training/validation corpus, not hardcoded magic numbers.
  - $E_k < \tau_{low}$ → `stable`
  - $\tau_{low} \le E_k < \tau_{high}$ → `elevated`
  - $E_k \ge \tau_{high}$ → `escalating`
- Output per conversation: a full timeline of $(\bar{s}_k, \beta_k, E_k,
  \text{status})$ per message index $k \ge w$, plus the conversation's
  current (latest) status.

### 3.3 Trend Analyzer

- Input: all messages classified as harmful (any non-"none" label above a
  confidence floor) across the corpus, with timestamp, language, and
  category.
- Topic clustering: BERTopic using a pretrained multilingual sentence
  embedding model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) — no custom
  training required. Produces topic clusters with representative terms and
  example messages.
- Time-series aggregation: message counts bucketed by time window (e.g.
  daily/weekly), broken down by category and language, for dashboard charts.
- Recomputed on-demand (via an explicit admin/demo trigger), not on every
  request — clustering is a batch operation over the corpus.

## 4. Data Model (PostgreSQL)

- **conversations**: `id`, `source` (`upload` | `seed`), `language`,
  `created_at`
- **messages**: `id`, `conversation_id` (FK), `seq_index`, `text`,
  `language`, `author` (optional/pseudonymous), `created_at`
- **message_labels**: `message_id` (FK), `category`, `probability`,
  `toxicity_score`
- **escalation_snapshots**: `conversation_id` (FK), `seq_index`,
  `windowed_mean`, `slope`, `escalation_score`, `status`
- **trend_snapshots**: `time_bucket`, `category`, `language`, `topic_id`,
  `topic_terms`, `message_count`

## 5. API Surface (FastAPI)

- `POST /conversations` — submit a conversation (ordered list of messages);
  runs classifier + escalation synchronously, persists results, returns full
  analysis.
- `GET /conversations/{id}` — retrieve a conversation with per-message labels
  and its escalation timeline.
- `GET /conversations` — paginated list/browse of stored conversations.
- `GET /trends` — query trend snapshots, filterable by time range, language,
  category.
- `POST /trends/recompute` — trigger topic re-clustering and aggregate
  recomputation (demo/admin action).

## 6. Frontend (React)

- **Analyzer page**: paste or upload a conversation; displays per-message
  toxicity/category badges and an escalation timeline chart (escalation
  score over message sequence, colored by status band).
- **Browse page**: list of seeded and user-submitted conversations, with
  status/category filters.
- **Trends Dashboard**: time-series charts (volume by category/language over
  time) and topic cluster cards (representative terms + example messages).

## 7. Data Ingestion

- Seeded at setup time: HASOC + DravidianCodeMix datasets run through the
  full pipeline so the Browse page and Trends Dashboard have real data
  immediately without requiring user input first.
- User-facing: paste-a-conversation or upload-a-file form on the Analyzer
  page, going through the same `POST /conversations` path as seeded data.

## 8. Error Handling

- Input validation at the API boundary: reject empty conversations, messages
  exceeding a length limit, and unsupported languages with structured 4xx
  error responses.
- Classifier/pipeline failures (e.g. model load failure, inference timeout)
  return a clear 5xx error; no silent partial results are persisted.

## 9. Testing Strategy

- **Escalation math**: unit tests for windowed mean, slope, and composite
  score computation against hand-computed expected values for known input
  sequences (including edge cases: window boundary, all-zero scores,
  monotonic increase/decrease).
- **API**: integration tests covering the submit → retrieve conversation
  flow, and trend query filtering.
- **Classifier**: evaluated via held-out test-set metrics (precision/recall/
  F1 per category, macro-F1) rather than unit tests — this is the
  report-facing evaluation artifact.
- Full exhaustive coverage is not a goal given coursework scope; the
  escalation formula and API contract are prioritized since they are the
  most bug-prone custom logic.

## 10. Out of Scope (explicit cuts from the original idea)

- No trained sequence model (e.g. RNN/transformer) for escalation — replaced
  by the explicit statistical formula in §3.2.
- No custom-trained topic model — BERTopic uses pretrained embeddings only.
- No live social media scraping/ingestion — data enters via seeded datasets
  or manual paste/upload.
- No production concerns (auth, horizontal scaling, rate limiting) beyond
  basic input validation.
