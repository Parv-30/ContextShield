# ContextShield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ContextShield end-to-end: a multilingual (EN/HI/TA) harmful-content
classifier, formula-based conversation escalation detection, BERTopic-based
trend analysis, a FastAPI backend, a PostgreSQL database, and a React
dashboard — seeded with real data and runnable locally.

**Architecture:** Python ML pipeline (classifier + escalation math + trend
clustering) used as a library by a FastAPI backend; FastAPI persists to
PostgreSQL via SQLAlchemy; React SPA (Vite) talks to FastAPI over REST.
Backend and ML pipeline live in one Python package (`backend/`) since the ML
pipeline has no independent deployment need at this scope; frontend is a
separate `frontend/` Vite project.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy, PostgreSQL, PyTorch (CUDA,
targeting an RTX 4050 laptop GPU with 6GB VRAM), HuggingFace `transformers` +
`datasets`, BERTopic, `sentence-transformers`, React + Vite, TypeScript,
Recharts (charts), pytest, vitest.

**Spec:** [docs/superpowers/specs/2026-09-22-contextshield-design.md](../specs/2026-09-22-contextshield-design.md)

## Global Constraints

- Languages: English, Hindi, Tamil only.
- Classifier labels (fixed, exact strings): `hate_speech`, `offensive`, `threat`, `harassment`, `none`.
- Toxicity score = max probability across `hate_speech`, `offensive`, `threat`, `harassment` (excludes `none`).
- Escalation default window `w=3`, default `alpha=0.6` (weight on windowed mean vs. slope).
- GPU available: RTX 4050 laptop (6GB VRAM), torch installed as `2.13.0+cu126`. Training uses the GPU (`device="cuda"`); `xlm-roberta-base`, max_seq_len ≤ 128, batch size 16 (fits comfortably in 6GB VRAM at this seq len; raise to 32 if VRAM allows — see Task 8's feasibility probe), up to 5 epochs. Task 8's feasibility probe and `--max-train-examples` cap still apply as a safety net (e.g. if the real HASOC/Dravidian datasets turn out to be very large) but are not expected to be needed for a normal-sized dataset on this GPU.
- Database: PostgreSQL (local instance, database name `contextshield`).
- No live scraping, no auth, no production hardening — coursework scope per spec §10.
- All code comments: none unless documenting a non-obvious invariant (project convention).

---

## Phase A: ML Pipeline (Classifier, Escalation, Trends)

### Task 1: Project scaffold and environment

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/contextshield/__init__.py`
- Create: `backend/README.md`
- Create: `.gitignore`

**Interfaces:**
- Produces: Python package `contextshield` importable from `backend/` (installed editable), with subpackages to be added: `contextshield.ml`, `contextshield.api`, `contextshield.db`.

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
node_modules/
dist/
build/
*.egg-info/
.env
backend/models/*.bin
backend/models/*.safetensors
backend/data/raw/
.pytest_cache/
.coverage
frontend/dist/
```

- [ ] **Step 2: Create `backend/pyproject.toml`**

```toml
[project]
name = "contextshield"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "pydantic>=2.8",
    "torch>=2.0",
    "transformers>=4.44",
    "datasets>=2.20",
    "scikit-learn>=1.5",
    "numpy>=1.26",
    "bertopic>=0.16",
    "sentence-transformers>=3.0",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27", "pytest-cov>=5.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["contextshield*"]
```

- [ ] **Step 3: Create `backend/contextshield/__init__.py`** (empty file)

- [ ] **Step 4: Create `backend/README.md`**

```markdown
# ContextShield Backend

Setup:

    cd backend
    python -m venv .venv
    .venv\Scripts\activate   # Windows
    pip install -e ".[dev]"

Run tests:

    pytest
```

- [ ] **Step 5: Install the package in editable mode and verify import**

Run: `cd backend && pip install -e ".[dev]"`
Then: `python -c "import contextshield; print('ok')"`
Expected: prints `ok`

- [ ] **Step 6: Commit**

```bash
git add .gitignore backend/pyproject.toml backend/contextshield/__init__.py backend/README.md
git commit -m "chore: scaffold backend Python package"
```

---

### Task 2: Escalation math module (pure functions, TDD)

**Files:**
- Create: `backend/contextshield/ml/__init__.py`
- Create: `backend/contextshield/ml/escalation.py`
- Test: `backend/tests/ml/test_escalation.py`
- Test: `backend/tests/__init__.py`, `backend/tests/ml/__init__.py`

**Interfaces:**
- Produces:
  - `windowed_mean(scores: list[float], k: int, w: int = 3) -> float` — mean of `scores[k-w+1:k+1]` (0-indexed, inclusive of index `k`); raises `ValueError` if `k < w - 1` or `k >= len(scores)`.
  - `escalation_slope(scores: list[float], k: int, w: int = 3) -> float` — least-squares slope of `scores[k-w+1:k+1]` vs. `[0, 1, ..., w-1]`.
  - `escalation_score(mean: float, slope: float, alpha: float = 0.6) -> float` — `alpha * mean + (1 - alpha) * max(slope, 0.0)`.
  - `classify_status(score: float, tau_low: float, tau_high: float) -> str` — returns `"stable"`, `"elevated"`, or `"escalating"`.
  - `compute_timeline(scores: list[float], w: int = 3, alpha: float = 0.6, tau_low: float = 0.0, tau_high: float = 1.0) -> list[dict]` — returns one dict per index `k >= w-1` with keys `seq_index`, `windowed_mean`, `slope`, `escalation_score`, `status`.

- [ ] **Step 1: Write failing tests for `windowed_mean` and `escalation_slope`**

Create `backend/tests/__init__.py` and `backend/tests/ml/__init__.py` (both empty).

```python
# backend/tests/ml/test_escalation.py
import pytest
from contextshield.ml.escalation import (
    windowed_mean,
    escalation_slope,
    escalation_score,
    classify_status,
    compute_timeline,
)


def test_windowed_mean_basic():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert windowed_mean(scores, k=2, w=3) == pytest.approx((0.1 + 0.2 + 0.3) / 3)


def test_windowed_mean_last_window():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert windowed_mean(scores, k=3, w=3) == pytest.approx((0.2 + 0.3 + 0.4) / 3)


def test_windowed_mean_raises_when_k_too_small():
    with pytest.raises(ValueError):
        windowed_mean([0.1, 0.2], k=0, w=3)


def test_escalation_slope_increasing():
    scores = [0.1, 0.2, 0.3]
    slope = escalation_slope(scores, k=2, w=3)
    assert slope == pytest.approx(0.1, abs=1e-9)


def test_escalation_slope_flat():
    scores = [0.5, 0.5, 0.5]
    slope = escalation_slope(scores, k=2, w=3)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_escalation_slope_decreasing():
    scores = [0.9, 0.5, 0.1]
    slope = escalation_slope(scores, k=2, w=3)
    assert slope == pytest.approx(-0.4, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_escalation.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'contextshield.ml'`)

- [ ] **Step 3: Implement `windowed_mean` and `escalation_slope`**

```python
# backend/contextshield/ml/__init__.py
```
(empty file)

```python
# backend/contextshield/ml/escalation.py
import numpy as np


def windowed_mean(scores: list[float], k: int, w: int = 3) -> float:
    if k < w - 1 or k >= len(scores):
        raise ValueError(f"k={k} out of valid range for w={w}, len={len(scores)}")
    window = scores[k - w + 1 : k + 1]
    return sum(window) / w


def escalation_slope(scores: list[float], k: int, w: int = 3) -> float:
    if k < w - 1 or k >= len(scores):
        raise ValueError(f"k={k} out of valid range for w={w}, len={len(scores)}")
    window = np.array(scores[k - w + 1 : k + 1], dtype=float)
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    y_mean = window.mean()
    numerator = ((x - x_mean) * (window - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/ml/test_escalation.py -v`
Expected: 6 passed

- [ ] **Step 5: Write failing tests for `escalation_score`, `classify_status`, `compute_timeline`**

Append to `backend/tests/ml/test_escalation.py`:

```python
def test_escalation_score_weights_mean_and_positive_slope():
    score = escalation_score(mean=0.5, slope=0.2, alpha=0.6)
    assert score == pytest.approx(0.6 * 0.5 + 0.4 * 0.2)


def test_escalation_score_clamps_negative_slope():
    score = escalation_score(mean=0.5, slope=-0.3, alpha=0.6)
    assert score == pytest.approx(0.6 * 0.5)


def test_classify_status_bands():
    assert classify_status(0.1, tau_low=0.3, tau_high=0.7) == "stable"
    assert classify_status(0.3, tau_low=0.3, tau_high=0.7) == "elevated"
    assert classify_status(0.5, tau_low=0.3, tau_high=0.7) == "elevated"
    assert classify_status(0.7, tau_low=0.3, tau_high=0.7) == "escalating"
    assert classify_status(0.9, tau_low=0.3, tau_high=0.7) == "escalating"


def test_compute_timeline_shape_and_monotonic_case():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    timeline = compute_timeline(scores, w=3, alpha=0.6, tau_low=0.2, tau_high=0.4)
    assert len(timeline) == len(scores) - (3 - 1)
    assert timeline[0]["seq_index"] == 2
    assert timeline[-1]["seq_index"] == 4
    for entry in timeline:
        assert set(entry.keys()) == {
            "seq_index", "windowed_mean", "slope", "escalation_score", "status",
        }
    assert timeline[-1]["status"] == "escalating"


def test_compute_timeline_empty_when_too_short():
    assert compute_timeline([0.1, 0.2], w=3) == []
```

- [ ] **Step 6: Run tests to verify the new ones fail**

Run: `cd backend && pytest tests/ml/test_escalation.py -v`
Expected: 5 passed (existing), 5 fail (`NameError`/`ImportError` for undefined names)

- [ ] **Step 7: Implement `escalation_score`, `classify_status`, `compute_timeline`**

Append to `backend/contextshield/ml/escalation.py`:

```python
def escalation_score(mean: float, slope: float, alpha: float = 0.6) -> float:
    return alpha * mean + (1 - alpha) * max(slope, 0.0)


def classify_status(score: float, tau_low: float, tau_high: float) -> str:
    if score >= tau_high:
        return "escalating"
    if score >= tau_low:
        return "elevated"
    return "stable"


def compute_timeline(
    scores: list[float],
    w: int = 3,
    alpha: float = 0.6,
    tau_low: float = 0.0,
    tau_high: float = 1.0,
) -> list[dict]:
    if len(scores) < w:
        return []
    timeline = []
    for k in range(w - 1, len(scores)):
        mean = windowed_mean(scores, k, w)
        slope = escalation_slope(scores, k, w)
        score = escalation_score(mean, slope, alpha)
        status = classify_status(score, tau_low, tau_high)
        timeline.append(
            {
                "seq_index": k,
                "windowed_mean": mean,
                "slope": slope,
                "escalation_score": score,
                "status": status,
            }
        )
    return timeline
```

- [ ] **Step 8: Run full test file to verify all pass**

Run: `cd backend && pytest tests/ml/test_escalation.py -v`
Expected: 11 passed

- [ ] **Step 9: Commit**

```bash
git add backend/contextshield/ml/ backend/tests/
git commit -m "feat: implement escalation math with tests"
```

---

### Task 3: Label schema mapping module

**Files:**
- Create: `backend/contextshield/ml/labels.py`
- Test: `backend/tests/ml/test_labels.py`

**Interfaces:**
- Consumes: nothing (pure data + functions).
- Produces:
  - `LABELS: list[str] = ["hate_speech", "offensive", "threat", "harassment", "none"]`
  - `toxicity_score(label_probs: dict[str, float]) -> float` — max of probs for keys in `LABELS` excluding `"none"`.
  - `map_hasoc_label(task1: str, task2: str | None) -> list[str]` — maps HASOC-style labels (`task1` in `{"HOF", "NOT"}`, `task2` in `{"HATE", "OFFN", "PRFN", None}`) to a list of applicable labels from `LABELS`.
  - `map_dravidian_label(raw_label: str) -> list[str]` — maps DravidianCodeMix/HASOC-Dravidian raw label strings to a list of applicable labels from `LABELS`. Handles labels: `"Not_offensive"`, `"Offensive_Untargeted"`, `"Offensive_Targeted_Insult_Individual"`, `"Offensive_Targeted_Insult_Group"`, `"Offensive_Targeted_Insult_Other"`, `"not-Tamil"` (treated as `["none"]`, filtered out upstream by the dataset loader).

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/ml/test_labels.py
import pytest
from contextshield.ml.labels import (
    LABELS,
    toxicity_score,
    map_hasoc_label,
    map_dravidian_label,
)


def test_labels_fixed_schema():
    assert LABELS == ["hate_speech", "offensive", "threat", "harassment", "none"]


def test_toxicity_score_picks_max_non_none():
    probs = {"hate_speech": 0.2, "offensive": 0.8, "threat": 0.1, "harassment": 0.3, "none": 0.05}
    assert toxicity_score(probs) == pytest.approx(0.8)


def test_toxicity_score_all_none():
    probs = {"hate_speech": 0.0, "offensive": 0.0, "threat": 0.0, "harassment": 0.0, "none": 0.99}
    assert toxicity_score(probs) == pytest.approx(0.0)


def test_map_hasoc_not_offensive():
    assert map_hasoc_label("NOT", None) == ["none"]


def test_map_hasoc_hate():
    assert map_hasoc_label("HOF", "HATE") == ["hate_speech"]


def test_map_hasoc_offensive():
    assert map_hasoc_label("HOF", "OFFN") == ["offensive"]


def test_map_hasoc_profane_maps_to_offensive():
    assert map_hasoc_label("HOF", "PRFN") == ["offensive"]


def test_map_dravidian_not_offensive():
    assert map_dravidian_label("Not_offensive") == ["none"]


def test_map_dravidian_untargeted():
    assert map_dravidian_label("Offensive_Untargeted") == ["offensive"]


def test_map_dravidian_targeted_individual_is_harassment():
    assert map_dravidian_label("Offensive_Targeted_Insult_Individual") == ["harassment"]


def test_map_dravidian_targeted_group_is_hate_speech():
    assert map_dravidian_label("Offensive_Targeted_Insult_Group") == ["hate_speech"]


def test_map_dravidian_targeted_other_is_offensive():
    assert map_dravidian_label("Offensive_Targeted_Insult_Other") == ["offensive"]


def test_map_dravidian_not_tamil():
    assert map_dravidian_label("not-Tamil") == ["none"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_labels.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `labels.py`**

```python
# backend/contextshield/ml/labels.py
LABELS = ["hate_speech", "offensive", "threat", "harassment", "none"]


def toxicity_score(label_probs: dict[str, float]) -> float:
    non_none = [label_probs[label] for label in LABELS if label != "none"]
    return max(non_none) if non_none else 0.0


def map_hasoc_label(task1: str, task2: str | None) -> list[str]:
    if task1 == "NOT":
        return ["none"]
    mapping = {"HATE": "hate_speech", "OFFN": "offensive", "PRFN": "offensive"}
    return [mapping.get(task2, "offensive")]


def map_dravidian_label(raw_label: str) -> list[str]:
    mapping = {
        "Not_offensive": ["none"],
        "not-Tamil": ["none"],
        "Offensive_Untargeted": ["offensive"],
        "Offensive_Targeted_Insult_Individual": ["harassment"],
        "Offensive_Targeted_Insult_Group": ["hate_speech"],
        "Offensive_Targeted_Insult_Other": ["offensive"],
    }
    return mapping.get(raw_label, ["none"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/ml/test_labels.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add backend/contextshield/ml/labels.py backend/tests/ml/test_labels.py
git commit -m "feat: add label schema mapping for HASOC and Dravidian datasets"
```

---

### Task 4: Dataset acquisition and preprocessing script

**Files:**
- Create: `backend/scripts/prepare_datasets.py`
- Create: `backend/contextshield/ml/dataset.py`
- Test: `backend/tests/ml/test_dataset.py`
- Create: `backend/data/README.md`

**Interfaces:**
- Consumes: `contextshield.ml.labels.map_hasoc_label`, `map_dravidian_label`, `LABELS`.
- Produces:
  - `to_multi_hot(labels_list: list[str]) -> list[int]` — converts a list of applicable label names into a multi-hot vector ordered per `LABELS`.
  - `build_examples(raw_rows: list[dict], text_key: str, label_fn) -> list[dict]` — takes raw rows (each a dict with at least `text_key` and whatever `label_fn` needs), returns list of `{"text": str, "language": str, "labels": list[int]}`. `label_fn(row) -> list[str]` supplies applicable label names; `language` must already be present in each raw row under key `"language"`.
  - `prepare_datasets.py` — standalone script: downloads HASOC (via HF `datasets`, dataset id `hate_speech_offensive` is NOT used — use `tasksource/hasoc-hindi` and `hasoc2019` style loading is inconsistent across HF hub; instead this script loads from local CSVs the operator places in `backend/data/raw/` per `backend/data/README.md`, since HASOC/DravidianCodeMix require manual registration/download from their official sources) and writes a unified `backend/data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` (80/10/10 split, stratified by whether any non-"none" label is present).

- [ ] **Step 1: Write failing tests for `to_multi_hot` and `build_examples`**

```python
# backend/tests/ml/test_dataset.py
from contextshield.ml.dataset import to_multi_hot, build_examples


def test_to_multi_hot_single_label():
    assert to_multi_hot(["offensive"]) == [0, 1, 0, 0, 0]


def test_to_multi_hot_none():
    assert to_multi_hot(["none"]) == [0, 0, 0, 0, 1]


def test_to_multi_hot_multiple_labels():
    result = to_multi_hot(["hate_speech", "threat"])
    assert result == [1, 0, 1, 0, 0]


def test_build_examples_basic():
    raw_rows = [
        {"text": "hello world", "language": "en", "task1": "NOT", "task2": None},
        {"text": "bad text", "language": "hi", "task1": "HOF", "task2": "HATE"},
    ]

    def label_fn(row):
        from contextshield.ml.labels import map_hasoc_label
        return map_hasoc_label(row["task1"], row["task2"])

    examples = build_examples(raw_rows, text_key="text", label_fn=label_fn)
    assert examples == [
        {"text": "hello world", "language": "en", "labels": [0, 0, 0, 0, 1]},
        {"text": "bad text", "language": "hi", "labels": [1, 0, 0, 0, 0]},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_dataset.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `dataset.py`**

```python
# backend/contextshield/ml/dataset.py
from contextshield.ml.labels import LABELS


def to_multi_hot(labels_list: list[str]) -> list[int]:
    return [1 if label in labels_list else 0 for label in LABELS]


def build_examples(raw_rows: list[dict], text_key: str, label_fn) -> list[dict]:
    examples = []
    for row in raw_rows:
        applicable = label_fn(row)
        examples.append(
            {
                "text": row[text_key],
                "language": row["language"],
                "labels": to_multi_hot(applicable),
            }
        )
    return examples
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/ml/test_dataset.py -v`
Expected: 4 passed

- [ ] **Step 5: Create `backend/data/README.md` documenting manual dataset placement**

```markdown
# Dataset Placement

HASOC and DravidianCodeMix/HASOC-Dravidian datasets require manual download
from their official distribution channels (registration required; cannot be
auto-fetched via a script due to license terms).

Place raw files here before running `scripts/prepare_datasets.py`:

- `backend/data/raw/hasoc_en.csv` — columns: `text,language,task1,task2` (language="en")
- `backend/data/raw/hasoc_hi.csv` — columns: `text,language,task1,task2` (language="hi")
- `backend/data/raw/dravidian_ta.csv` — columns: `text,language,label` (language="ta")

If these files are not found, `prepare_datasets.py` falls back to a small
bundled synthetic seed set (`backend/data/seed_fallback.jsonl`) so the
pipeline remains runnable end-to-end without the licensed datasets — this
produces a weaker classifier suitable for demoing the system's mechanics,
not for reporting real-world accuracy. Real accuracy numbers for the report
require the actual datasets to be placed as described above.
```

- [ ] **Step 6: Create `backend/scripts/prepare_datasets.py`**

```python
# backend/scripts/prepare_datasets.py
import csv
import json
import random
from pathlib import Path

from contextshield.ml.dataset import build_examples
from contextshield.ml.labels import map_hasoc_label, map_dravidian_label

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FALLBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_fallback.jsonl"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hasoc_label_fn(row: dict) -> list[str]:
    task2 = row.get("task2") or None
    return map_hasoc_label(row["task1"], task2)


def dravidian_label_fn(row: dict) -> list[str]:
    return map_dravidian_label(row["label"])


def load_all_examples() -> list[dict]:
    examples = []
    hasoc_en = RAW_DIR / "hasoc_en.csv"
    hasoc_hi = RAW_DIR / "hasoc_hi.csv"
    dravidian_ta = RAW_DIR / "dravidian_ta.csv"

    found_any = False
    if hasoc_en.exists():
        examples += build_examples(load_csv(hasoc_en), "text", hasoc_label_fn)
        found_any = True
    if hasoc_hi.exists():
        examples += build_examples(load_csv(hasoc_hi), "text", hasoc_label_fn)
        found_any = True
    if dravidian_ta.exists():
        examples += build_examples(load_csv(dravidian_ta), "text", dravidian_label_fn)
        found_any = True

    if not found_any:
        print(f"No raw datasets found in {RAW_DIR}; using fallback seed set.")
        with open(FALLBACK_PATH, encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]

    return examples


def stratified_split(examples: list[dict], seed: int = 42):
    random.Random(seed).shuffle(examples)

    def has_signal(ex):
        return ex["labels"][:-1] != [0, 0, 0, 0]

    positives = [e for e in examples if has_signal(e)]
    negatives = [e for e in examples if not has_signal(e)]

    def split_group(group):
        n = len(group)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return group[:n_train], group[n_train : n_train + n_val], group[n_train + n_val :]

    pos_train, pos_val, pos_test = split_group(positives)
    neg_train, neg_val, neg_test = split_group(negatives)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    random.Random(seed).shuffle(test)
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    examples = load_all_examples()
    train, val, test = stratified_split(examples)
    write_jsonl(PROCESSED_DIR / "train.jsonl", train)
    write_jsonl(PROCESSED_DIR / "val.jsonl", val)
    write_jsonl(PROCESSED_DIR / "test.jsonl", test)
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Create a small synthetic fallback seed set so the pipeline is runnable without licensed data**

Create `backend/data/seed_fallback.jsonl` with 60 lines (20 per language: en/hi/ta),
mixing `none` and each non-none label at least a few times each. Example content
(write all 60 lines, following this pattern — vary text and labels so every
label in `LABELS` appears at least 3 times across the file):

```jsonl
{"text": "Have a great day everyone!", "language": "en", "labels": [0, 0, 0, 0, 1]}
{"text": "You are all idiots and should leave", "language": "en", "labels": [0, 1, 0, 0, 0]}
{"text": "I will find you and hurt you", "language": "en", "labels": [0, 0, 1, 0, 0]}
{"text": "People like you don't deserve to live here", "language": "en", "labels": [1, 0, 0, 0, 0]}
{"text": "Stop bothering me, you creep", "language": "en", "labels": [0, 0, 0, 1, 0]}
```

(Continue this pattern for 60 total lines across en/hi/ta with realistic
short social-media-style sentences per language and the label distribution
described above. This is placeholder demo data only, documented as such in
`backend/data/README.md`.)

- [ ] **Step 8: Run the prepare script against the fallback path (no raw files present)**

Run: `cd backend && python scripts/prepare_datasets.py`
Expected: prints `No raw datasets found...` then `train=... val=... test=...` with all three > 0, and creates `backend/data/processed/{train,val,test}.jsonl`

- [ ] **Step 9: Commit**

```bash
git add backend/contextshield/ml/dataset.py backend/tests/ml/test_dataset.py backend/scripts/ backend/data/README.md backend/data/seed_fallback.jsonl
git commit -m "feat: add dataset preprocessing pipeline with fallback seed data"
```

---

### Task 5: Classifier training script

**Files:**
- Create: `backend/contextshield/ml/train.py`
- Create: `backend/scripts/train_classifier.py`
- Test: `backend/tests/ml/test_train.py`

**Interfaces:**
- Consumes: `backend/data/processed/{train,val,test}.jsonl` (from Task 4); `contextshield.ml.labels.LABELS`.
- Produces:
  - `load_jsonl(path: Path) -> list[dict]`
  - `MultiLabelDataset` (torch `Dataset`) — wraps tokenized examples, `__getitem__` returns `{"input_ids", "attention_mask", "labels"}` (labels as float tensor).
  - `compute_metrics(eval_pred) -> dict` — returns `{"macro_f1": float, "micro_f1": float}` using a 0.5 sigmoid threshold, plus per-label F1 as `f1_<label>` keys.
  - `train_classifier.py` — script: loads processed data, tokenizes with `xlm-roberta-base` tokenizer (max_length=128), fine-tunes via HF `Trainer` (3 epochs, batch size 16, on CPU), saves model + tokenizer to `backend/models/classifier/`, writes eval metrics to `backend/models/classifier/eval_metrics.json`.

- [ ] **Step 1: Write failing tests for `load_jsonl` and `compute_metrics`**

```python
# backend/tests/ml/test_train.py
import json
import numpy as np
from pathlib import Path
from contextshield.ml.train import load_jsonl, compute_metrics


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"text": "a", "language": "en", "labels": [0,0,0,0,1]}\n'
        '{"text": "b", "language": "hi", "labels": [1,0,0,0,0]}\n',
        encoding="utf-8",
    )
    rows = load_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["text"] == "a"
    assert rows[1]["labels"] == [1, 0, 0, 0, 0]


class FakeEvalPred:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


def test_compute_metrics_perfect_predictions():
    logits = np.array([[10, -10, -10, -10, -10], [-10, -10, -10, -10, 10]])
    labels = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
    metrics = compute_metrics(FakeEvalPred(logits, labels))
    assert metrics["macro_f1"] == 1.0
    assert metrics["micro_f1"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_train.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `backend/contextshield/ml/train.py`**

```python
# backend/contextshield/ml/train.py
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

from contextshield.ml.labels import LABELS


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MultiLabelDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["labels"], dtype=torch.float),
        }


def compute_metrics(eval_pred) -> dict:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
    }
    per_label = f1_score(labels, preds, average=None, zero_division=0)
    for label, score in zip(LABELS, per_label):
        metrics[f"f1_{label}"] = float(score)
    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/ml/test_train.py -v`
Expected: 2 passed

- [ ] **Step 5: Create `backend/scripts/train_classifier.py`**

```python
# backend/scripts/train_classifier.py
import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from contextshield.ml.labels import LABELS
from contextshield.ml.train import load_jsonl, MultiLabelDataset, compute_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "classifier"
MODEL_NAME = "xlm-roberta-base"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    test_examples = load_jsonl(DATA_DIR / "test.jsonl")

    train_dataset = MultiLabelDataset(train_examples, tokenizer)
    val_dataset = MultiLabelDataset(val_examples, tokenizer)
    test_dataset = MultiLabelDataset(test_examples, tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_dataset)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    with open(MODEL_OUT / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the training script end-to-end against the (fallback) processed data**

Run: `cd backend && python scripts/train_classifier.py`
Expected: completes without error — with the GPU this should be quick even
on the small fallback set (a smoke run to prove the pipeline works); if real
HASOC/Dravidian CSVs were placed per Task 4, expect proportionally longer but
still GPU-accelerated training — see Task 8's feasibility probe if it seems
too slow, prints `Test metrics: {...}`, and creates `backend/models/classifier/` with
`config.json`, `model.safetensors` (or `pytorch_model.bin`), tokenizer
files, and `eval_metrics.json`.

- [ ] **Step 7: Commit**

```bash
git add backend/contextshield/ml/train.py backend/scripts/train_classifier.py backend/tests/ml/test_train.py
git commit -m "feat: add classifier training script and metrics"
```

Note: do NOT commit `backend/models/` contents (already gitignored) — model
weights are a build artifact, not source.

---

### Task 6: Classifier inference module

**Files:**
- Create: `backend/contextshield/ml/classifier.py`
- Test: `backend/tests/ml/test_classifier.py`

**Interfaces:**
- Consumes: `backend/models/classifier/` (trained model from Task 5); `contextshield.ml.labels.LABELS`, `toxicity_score`.
- Produces:
  - `class Classifier` with `__init__(self, model_dir: str | Path)` and `predict(self, text: str) -> dict` returning `{"probabilities": dict[str, float], "toxicity_score": float}`; `predict_batch(self, texts: list[str]) -> list[dict]` for batched inference (later tasks use this).

- [ ] **Step 1: Write a failing test using a tiny real tokenizer/model pair (not the trained one, to keep tests fast and independent of training completing)**

```python
# backend/tests/ml/test_classifier.py
import pytest
from contextshield.ml.classifier import Classifier
from contextshield.ml.labels import LABELS


@pytest.fixture(scope="module")
def tiny_classifier(tmp_path_factory):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = tmp_path_factory.mktemp("tiny_model")
    tokenizer_name = "hf-internal-testing/tiny-random-XLMRobertaXL"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer_name, num_labels=len(LABELS), problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
    except Exception:
        pytest.skip("tiny test model unavailable offline")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return Classifier(model_dir)


def test_predict_returns_all_labels(tiny_classifier):
    result = tiny_classifier.predict("some text")
    assert set(result["probabilities"].keys()) == set(LABELS)
    assert 0.0 <= result["toxicity_score"] <= 1.0


def test_predict_batch_matches_single(tiny_classifier):
    texts = ["first", "second"]
    batch_results = tiny_classifier.predict_batch(texts)
    assert len(batch_results) == 2
    for r in batch_results:
        assert set(r["probabilities"].keys()) == set(LABELS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_classifier.py -v`
Expected: FAIL (`ModuleNotFoundError: contextshield.ml.classifier`)

- [ ] **Step 3: Implement `backend/contextshield/ml/classifier.py`**

```python
# backend/contextshield/ml/classifier.py
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from contextshield.ml.labels import LABELS, toxicity_score as compute_toxicity_score


class Classifier:
    def __init__(self, model_dir: str | Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        encodings = self.tokenizer(
            texts, truncation=True, max_length=128, padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encodings).logits
        probs = torch.sigmoid(logits).numpy()

        results = []
        for row in probs:
            label_probs = {label: float(p) for label, p in zip(LABELS, row)}
            results.append(
                {
                    "probabilities": label_probs,
                    "toxicity_score": compute_toxicity_score(label_probs),
                }
            )
        return results
```

- [ ] **Step 4: Run tests to verify they pass (or skip cleanly offline)**

Run: `cd backend && pytest tests/ml/test_classifier.py -v`
Expected: 2 passed, or 2 skipped with "tiny test model unavailable offline" if no network access — either outcome is acceptable; if skipped, proceed and revisit once network access is confirmed.

- [ ] **Step 5: Commit**

```bash
git add backend/contextshield/ml/classifier.py backend/tests/ml/test_classifier.py
git commit -m "feat: add classifier inference wrapper"
```

---

### Task 7: Threshold calibration script

**Files:**
- Create: `backend/scripts/calibrate_thresholds.py`
- Create: `backend/contextshield/ml/calibration.py`
- Test: `backend/tests/ml/test_calibration.py`

**Interfaces:**
- Consumes: `contextshield.ml.escalation.compute_timeline` internals (specifically `windowed_mean`, `escalation_slope`, `escalation_score`).
- Produces:
  - `percentile_thresholds(scores: list[float], low_pct: float = 60.0, high_pct: float = 85.0) -> tuple[float, float]` — returns `(tau_low, tau_high)` via `numpy.percentile`.
  - `calibrate_thresholds.py` — script: loads `backend/data/processed/val.jsonl`, runs the trained classifier (Task 6) on each text to get toxicity scores, groups scores by conversation is not applicable here (val set is independent messages, not conversations) — instead treats the val set's toxicity scores directly as the score distribution for calibration (documented assumption: calibration approximates the escalation-score distribution using raw toxicity scores, since no labeled conversation-level escalation dataset exists), computes `(tau_low, tau_high)`, writes them to `backend/models/classifier/thresholds.json`.

- [ ] **Step 1: Write failing test for `percentile_thresholds`**

```python
# backend/tests/ml/test_calibration.py
import pytest
from contextshield.ml.calibration import percentile_thresholds


def test_percentile_thresholds_basic():
    scores = list(range(1, 101))  # 1..100
    tau_low, tau_high = percentile_thresholds(scores, low_pct=60.0, high_pct=85.0)
    assert tau_low == pytest.approx(60.4, abs=0.5)
    assert tau_high == pytest.approx(85.15, abs=0.5)
    assert tau_low < tau_high


def test_percentile_thresholds_constant_scores():
    scores = [0.5] * 10
    tau_low, tau_high = percentile_thresholds(scores)
    assert tau_low == pytest.approx(0.5)
    assert tau_high == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_calibration.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `backend/contextshield/ml/calibration.py`**

```python
# backend/contextshield/ml/calibration.py
import numpy as np


def percentile_thresholds(
    scores: list[float], low_pct: float = 60.0, high_pct: float = 85.0
) -> tuple[float, float]:
    arr = np.array(scores, dtype=float)
    tau_low = float(np.percentile(arr, low_pct))
    tau_high = float(np.percentile(arr, high_pct))
    return tau_low, tau_high
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/ml/test_calibration.py -v`
Expected: 2 passed

- [ ] **Step 5: Create `backend/scripts/calibrate_thresholds.py`**

```python
# backend/scripts/calibrate_thresholds.py
import json
from pathlib import Path

from contextshield.ml.classifier import Classifier
from contextshield.ml.calibration import percentile_thresholds
from contextshield.ml.train import load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classifier"


def main():
    classifier = Classifier(MODEL_DIR)
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    texts = [ex["text"] for ex in val_examples]

    scores = [r["toxicity_score"] for r in classifier.predict_batch(texts)]
    tau_low, tau_high = percentile_thresholds(scores)

    output = {"tau_low": tau_low, "tau_high": tau_high, "low_pct": 60.0, "high_pct": 85.0}
    with open(MODEL_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the calibration script (requires Task 5's trained model to exist)**

Run: `cd backend && python scripts/calibrate_thresholds.py`
Expected: prints `{'tau_low': ..., 'tau_high': ..., ...}` and creates `backend/models/classifier/thresholds.json`

- [ ] **Step 7: Commit**

```bash
git add backend/contextshield/ml/calibration.py backend/scripts/calibrate_thresholds.py backend/tests/ml/test_calibration.py
git commit -m "feat: add percentile-based threshold calibration"
```

---

### Task 8: Training time-bound fallback policy (documentation + guard script)

**Files:**
- Create: `backend/scripts/check_training_feasible.py`
- Modify: `backend/scripts/train_classifier.py` (add a `--max-train-examples` CLI cap)

**Interfaces:**
- Produces: `train_classifier.py` accepts an optional `--max-train-examples N` argument (default: unlimited) that truncates `train_examples` before building the dataset — gives an escape hatch if the full/fallback dataset is too slow to train on CPU within available time.

- [ ] **Step 1: Modify `train_classifier.py` to accept `--max-train-examples`**

Add near the top of `backend/scripts/train_classifier.py`:

```python
import argparse
```

Replace `def main():` body's start with:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-examples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    test_examples = load_jsonl(DATA_DIR / "test.jsonl")
```

(Remove the now-duplicated original lines that re-declared `train_examples`,
`val_examples`, `test_examples`, `tokenizer`, `model` further down in the
function.)

- [ ] **Step 2: Create `backend/scripts/check_training_feasible.py`** — a quick timing probe

```python
# backend/scripts/check_training_feasible.py
import time
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from contextshield.ml.labels import LABELS

MODEL_NAME = "xlm-roberta-base"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification",
    )
    model.to(device)
    model.eval()

    sample_texts = ["this is a sample sentence for timing"] * 16
    encodings = tokenizer(sample_texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}

    start = time.time()
    with torch.no_grad():
        model(**encodings)
    elapsed = time.time() - start

    print(f"Device: {device}")
    print(f"One forward pass of batch=16 took {elapsed:.2f}s on this machine.")
    print("Estimated seconds per training step (fwd+bwd, ~3x forward time):", elapsed * 3)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the feasibility probe**

Run: `cd backend && python scripts/check_training_feasible.py`
Expected: prints `Device: cuda` and a timing estimate. Use this number to
decide `--max-train-examples` for the real training run in Task 5 only if
the full dataset would take too long on the GPU (e.g. if estimated per-step
time × total steps for 5 epochs exceeds ~1 hour, cap `--max-train-examples`
to fit that budget) — expected to be unnecessary for typical HASOC/Dravidian
dataset sizes on an RTX 4050.

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/train_classifier.py backend/scripts/check_training_feasible.py
git commit -m "feat: add training feasibility probe and example cap for large datasets"
```

---

### Task 9: Trend analyzer module

**Files:**
- Create: `backend/contextshield/ml/trends.py`
- Test: `backend/tests/ml/test_trends.py`

**Interfaces:**
- Consumes: list of message dicts `{"text": str, "language": str, "category": str, "timestamp": datetime}` (harmful messages only, pre-filtered by caller).
- Produces:
  - `bucket_time_series(messages: list[dict], bucket: str = "D") -> list[dict]` — groups messages by `(time_bucket, category, language)`, returns list of `{"time_bucket": str (ISO date), "category": str, "language": str, "message_count": int}`. `bucket` is a pandas-style frequency string (`"D"` = daily, `"W"` = weekly).
  - `class TopicAnalyzer` with `__init__(self, embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2")` and `fit_transform(self, texts: list[str]) -> list[dict]` returning one dict per discovered topic: `{"topic_id": int, "terms": list[str], "example_messages": list[str]}` (excludes the outlier topic `-1`).

- [ ] **Step 1: Write failing tests for `bucket_time_series`**

```python
# backend/tests/ml/test_trends.py
from datetime import datetime
from contextshield.ml.trends import bucket_time_series


def test_bucket_time_series_daily_grouping():
    messages = [
        {"text": "a", "language": "en", "category": "offensive", "timestamp": datetime(2026, 1, 1, 10, 0)},
        {"text": "b", "language": "en", "category": "offensive", "timestamp": datetime(2026, 1, 1, 15, 0)},
        {"text": "c", "language": "hi", "category": "hate_speech", "timestamp": datetime(2026, 1, 2, 9, 0)},
    ]
    result = bucket_time_series(messages, bucket="D")
    result_by_key = {(r["time_bucket"], r["category"], r["language"]): r["message_count"] for r in result}
    assert result_by_key[("2026-01-01", "offensive", "en")] == 2
    assert result_by_key[("2026-01-02", "hate_speech", "hi")] == 1


def test_bucket_time_series_empty():
    assert bucket_time_series([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/ml/test_trends.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `bucket_time_series`**

```python
# backend/contextshield/ml/trends.py
from collections import defaultdict

import pandas as pd


def bucket_time_series(messages: list[dict], bucket: str = "D") -> list[dict]:
    if not messages:
        return []

    counts = defaultdict(int)
    for msg in messages:
        period = pd.Timestamp(msg["timestamp"]).to_period(bucket)
        key = (str(period.start_time.date()), msg["category"], msg["language"])
        counts[key] += 1

    return [
        {"time_bucket": time_bucket, "category": category, "language": language, "message_count": count}
        for (time_bucket, category, language), count in counts.items()
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/ml/test_trends.py -v`
Expected: 2 passed

Note: `bucket_time_series` requires `pandas` — add `"pandas>=2.0"` to
`backend/pyproject.toml` dependencies and re-run `pip install -e ".[dev]"`
before this step if not already present as a transitive dependency of
`datasets`.

- [ ] **Step 5: Write a test for `TopicAnalyzer` (network-dependent, allow skip)**

Append to `backend/tests/ml/test_trends.py`:

```python
import pytest
from contextshield.ml.trends import TopicAnalyzer


def test_topic_analyzer_fit_transform():
    texts = [
        "I hate this group of people",
        "This group of people is terrible",
        "The weather is nice today",
        "Today the weather looks great",
    ] * 3
    try:
        analyzer = TopicAnalyzer()
        topics = analyzer.fit_transform(texts)
    except Exception:
        pytest.skip("embedding model unavailable offline")
    assert isinstance(topics, list)
    for topic in topics:
        assert set(topic.keys()) == {"topic_id", "terms", "example_messages"}
        assert topic["topic_id"] != -1
```

- [ ] **Step 6: Run to verify it fails (missing `TopicAnalyzer`)**

Run: `cd backend && pytest tests/ml/test_trends.py -v`
Expected: 2 passed (existing), 1 fails with `ImportError`

- [ ] **Step 7: Implement `TopicAnalyzer`**

Append to `backend/contextshield/ml/trends.py`:

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


class TopicAnalyzer:
    def __init__(self, embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.model = BERTopic(embedding_model=self.embedding_model, verbose=False)

    def fit_transform(self, texts: list[str]) -> list[dict]:
        topics, _ = self.model.fit_transform(texts)
        topic_info = self.model.get_topic_info()

        results = []
        for topic_id in topic_info["Topic"]:
            if topic_id == -1:
                continue
            terms = [term for term, _ in self.model.get_topic(topic_id)]
            examples = [
                text for text, t in zip(texts, topics) if t == topic_id
            ][:5]
            results.append(
                {"topic_id": int(topic_id), "terms": terms, "example_messages": examples}
            )
        return results
```

- [ ] **Step 8: Run tests to verify they pass (or skip cleanly offline)**

Run: `cd backend && pytest tests/ml/test_trends.py -v`
Expected: 3 passed, or 2 passed + 1 skipped if offline.

- [ ] **Step 9: Commit**

```bash
git add backend/contextshield/ml/trends.py backend/tests/ml/test_trends.py backend/pyproject.toml
git commit -m "feat: add trend time-series bucketing and BERTopic topic analyzer"
```

---

## Phase B: Backend API and Database

### Task 10: SQLAlchemy models and database session

**Files:**
- Create: `backend/contextshield/db/__init__.py`
- Create: `backend/contextshield/db/models.py`
- Create: `backend/contextshield/db/session.py`
- Test: `backend/tests/db/__init__.py`, `backend/tests/db/test_models.py`

**Interfaces:**
- Produces:
  - SQLAlchemy declarative models: `Conversation`, `Message`, `MessageLabel`, `EscalationSnapshot`, `TrendSnapshot` per spec §4 field lists, using `sqlalchemy.orm.DeclarativeBase`.
  - `get_engine(database_url: str)`, `get_session_factory(engine)`, `Base.metadata.create_all(engine)` usable for test setup.
  - Environment variable `CONTEXTSHIELD_DATABASE_URL` read by `session.py`'s `get_default_engine()`, defaulting to `postgresql+psycopg2://postgres:postgres@localhost:5432/contextshield`.

- [ ] **Step 1: Write failing tests using SQLite in-memory for speed (schema is portable)**

```python
# backend/tests/db/test_models.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base, Conversation, Message, MessageLabel, EscalationSnapshot, TrendSnapshot


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_create_conversation_with_messages_and_labels():
    session = make_session()
    convo = Conversation(source="upload", language="en")
    session.add(convo)
    session.flush()

    msg = Message(conversation_id=convo.id, seq_index=0, text="hello", language="en")
    session.add(msg)
    session.flush()

    label = MessageLabel(message_id=msg.id, category="offensive", probability=0.8, toxicity_score=0.8)
    session.add(label)
    session.commit()

    fetched = session.query(Conversation).first()
    assert fetched.language == "en"
    assert len(fetched.messages) == 1
    assert fetched.messages[0].labels[0].category == "offensive"


def test_escalation_snapshot_and_trend_snapshot():
    session = make_session()
    convo = Conversation(source="seed", language="hi")
    session.add(convo)
    session.flush()

    snapshot = EscalationSnapshot(
        conversation_id=convo.id, seq_index=2, windowed_mean=0.5, slope=0.1,
        escalation_score=0.4, status="elevated",
    )
    session.add(snapshot)

    trend = TrendSnapshot(
        time_bucket="2026-01-01", category="hate_speech", language="hi",
        topic_id=3, topic_terms="term1,term2", message_count=5,
    )
    session.add(trend)
    session.commit()

    assert session.query(EscalationSnapshot).count() == 1
    assert session.query(TrendSnapshot).first().message_count == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/db/test_models.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `backend/contextshield/db/models.py`**

```python
# backend/contextshield/db/models.py
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Float, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow():
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(16))
    language: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", order_by="Message.seq_index")
    escalation_snapshots: Mapped[list["EscalationSnapshot"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    seq_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(String(2000))
    language: Mapped[str] = mapped_column(String(8))
    author: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    labels: Mapped[list["MessageLabel"]] = relationship(back_populates="message")


class MessageLabel(Base):
    __tablename__ = "message_labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    category: Mapped[str] = mapped_column(String(32))
    probability: Mapped[float] = mapped_column(Float)
    toxicity_score: Mapped[float] = mapped_column(Float)

    message: Mapped["Message"] = relationship(back_populates="labels")


class EscalationSnapshot(Base):
    __tablename__ = "escalation_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    seq_index: Mapped[int] = mapped_column(Integer)
    windowed_mean: Mapped[float] = mapped_column(Float)
    slope: Mapped[float] = mapped_column(Float)
    escalation_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))

    conversation: Mapped["Conversation"] = relationship(back_populates="escalation_snapshots")


class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    time_bucket: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(8))
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    topic_terms: Mapped[str | None] = mapped_column(String(500), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer)
```

- [ ] **Step 4: Create `backend/contextshield/db/__init__.py`** (empty)

- [ ] **Step 5: Create `backend/tests/db/__init__.py`** (empty)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && pytest tests/db/test_models.py -v`
Expected: 2 passed

- [ ] **Step 7: Implement `backend/contextshield/db/session.py`**

```python
# backend/contextshield/db/session.py
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base

DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/contextshield"


def get_engine(database_url: str):
    return create_engine(database_url)


def get_default_engine():
    url = os.environ.get("CONTEXTSHIELD_DATABASE_URL", DEFAULT_DATABASE_URL)
    return get_engine(url)


def get_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine):
    Base.metadata.create_all(engine)
```

- [ ] **Step 8: Commit**

```bash
git add backend/contextshield/db/ backend/tests/db/
git commit -m "feat: add SQLAlchemy models and session management"
```

---

### Task 11: Conversation service layer (business logic connecting ML + DB)

**Files:**
- Create: `backend/contextshield/services/__init__.py`
- Create: `backend/contextshield/services/conversation_service.py`
- Test: `backend/tests/services/__init__.py`, `backend/tests/services/test_conversation_service.py`

**Interfaces:**
- Consumes: `contextshield.ml.classifier.Classifier`, `contextshield.ml.escalation.compute_timeline`, `contextshield.ml.labels.LABELS`, `contextshield.db.models.*`.
- Produces:
  - `class ConversationService` with `__init__(self, session, classifier, tau_low: float, tau_high: float)`.
  - `create_conversation(self, messages: list[dict], source: str, language: str) -> Conversation` — `messages` is `list[{"text": str, "author": str | None}]`; runs classification on each message, persists `Conversation`, `Message`, `MessageLabel` rows, computes escalation timeline via `compute_timeline`, persists `EscalationSnapshot` rows, returns the persisted `Conversation` ORM object (with relationships loaded).
  - `get_conversation(self, conversation_id: int) -> Conversation | None`.
  - `list_conversations(self, limit: int = 20, offset: int = 0) -> list[Conversation]`.

- [ ] **Step 1: Write failing tests using a fake classifier (no real model needed)**

```python
# backend/tests/services/test_conversation_service.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base
from contextshield.services.conversation_service import ConversationService


class FakeClassifier:
    def predict_batch(self, texts):
        results = []
        for i, text in enumerate(texts):
            score = min(0.1 * (i + 1), 1.0)
            probs = {"hate_speech": 0.0, "offensive": score, "threat": 0.0, "harassment": 0.0, "none": 1 - score}
            results.append({"probabilities": probs, "toxicity_score": score})
        return results


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_create_conversation_persists_messages_and_labels():
    session = make_session()
    service = ConversationService(session, FakeClassifier(), tau_low=0.2, tau_high=0.5)

    messages = [{"text": "msg1", "author": None}, {"text": "msg2", "author": "u1"}, {"text": "msg3", "author": None}]
    convo = service.create_conversation(messages, source="upload", language="en")

    assert convo.id is not None
    assert len(convo.messages) == 3
    assert convo.messages[0].seq_index == 0
    assert len(convo.messages[0].labels) == 5  # one MessageLabel row per LABELS entry


def test_create_conversation_computes_escalation_snapshots():
    session = make_session()
    service = ConversationService(session, FakeClassifier(), tau_low=0.2, tau_high=0.25)

    messages = [{"text": f"msg{i}", "author": None} for i in range(5)]
    convo = service.create_conversation(messages, source="upload", language="en")

    assert len(convo.escalation_snapshots) == 3  # w=3 default -> len(messages) - 2
    assert convo.escalation_snapshots[-1].status in {"stable", "elevated", "escalating"}


def test_get_and_list_conversations():
    session = make_session()
    service = ConversationService(session, FakeClassifier(), tau_low=0.2, tau_high=0.5)
    convo = service.create_conversation([{"text": "hi", "author": None}] * 3, source="seed", language="ta")

    fetched = service.get_conversation(convo.id)
    assert fetched is not None
    assert fetched.language == "ta"

    all_convos = service.list_conversations()
    assert len(all_convos) == 1

    assert service.get_conversation(9999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/services/test_conversation_service.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `backend/contextshield/services/conversation_service.py`**

```python
# backend/contextshield/services/__init__.py
```
(empty file)

```python
# backend/contextshield/services/conversation_service.py
from contextshield.db.models import Conversation, Message, MessageLabel, EscalationSnapshot
from contextshield.ml.escalation import compute_timeline


class ConversationService:
    def __init__(self, session, classifier, tau_low: float, tau_high: float, window: int = 3, alpha: float = 0.6):
        self.session = session
        self.classifier = classifier
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.window = window
        self.alpha = alpha

    def create_conversation(self, messages: list[dict], source: str, language: str) -> Conversation:
        texts = [m["text"] for m in messages]
        predictions = self.classifier.predict_batch(texts)

        convo = Conversation(source=source, language=language)
        self.session.add(convo)
        self.session.flush()

        toxicity_scores = []
        for idx, (msg, prediction) in enumerate(zip(messages, predictions)):
            db_message = Message(
                conversation_id=convo.id,
                seq_index=idx,
                text=msg["text"],
                language=language,
                author=msg.get("author"),
            )
            self.session.add(db_message)
            self.session.flush()

            for category, probability in prediction["probabilities"].items():
                self.session.add(
                    MessageLabel(
                        message_id=db_message.id,
                        category=category,
                        probability=probability,
                        toxicity_score=prediction["toxicity_score"],
                    )
                )
            toxicity_scores.append(prediction["toxicity_score"])

        timeline = compute_timeline(
            toxicity_scores, w=self.window, alpha=self.alpha, tau_low=self.tau_low, tau_high=self.tau_high,
        )
        for entry in timeline:
            self.session.add(
                EscalationSnapshot(
                    conversation_id=convo.id,
                    seq_index=entry["seq_index"],
                    windowed_mean=entry["windowed_mean"],
                    slope=entry["slope"],
                    escalation_score=entry["escalation_score"],
                    status=entry["status"],
                )
            )

        self.session.commit()
        self.session.refresh(convo)
        return convo

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        return self.session.get(Conversation, conversation_id)

    def list_conversations(self, limit: int = 20, offset: int = 0) -> list[Conversation]:
        return (
            self.session.query(Conversation)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
```

- [ ] **Step 4: Create `backend/tests/services/__init__.py`** (empty)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pytest tests/services/test_conversation_service.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add backend/contextshield/services/ backend/tests/services/
git commit -m "feat: add conversation service layer wiring classifier and escalation to DB"
```

---

### Task 12: FastAPI app — conversations endpoints

**Files:**
- Create: `backend/contextshield/api/__init__.py`
- Create: `backend/contextshield/api/schemas.py`
- Create: `backend/contextshield/api/deps.py`
- Create: `backend/contextshield/api/main.py`
- Create: `backend/contextshield/api/routers/__init__.py`
- Create: `backend/contextshield/api/routers/conversations.py`
- Test: `backend/tests/api/__init__.py`, `backend/tests/api/test_conversations.py`

**Interfaces:**
- Consumes: `contextshield.services.conversation_service.ConversationService`, `contextshield.db.session.*`, `contextshield.ml.classifier.Classifier`.
- Produces:
  - Pydantic schemas in `schemas.py`: `MessageIn {text: str, author: str | None = None}`, `ConversationCreateIn {messages: list[MessageIn], source: Literal["upload","seed"] = "upload", language: Literal["en","hi","ta"]}`, `MessageLabelOut {category: str, probability: float, toxicity_score: float}`, `MessageOut {id: int, seq_index: int, text: str, labels: list[MessageLabelOut]}`, `EscalationSnapshotOut {seq_index: int, windowed_mean: float, slope: float, escalation_score: float, status: str}`, `ConversationOut {id: int, source: str, language: str, messages: list[MessageOut], escalation_snapshots: list[EscalationSnapshotOut]}`.
  - `deps.py`: `get_db_session()` FastAPI dependency (yields a session, closes after), `get_classifier()` dependency (loads a module-level cached `Classifier` singleton from `CONTEXTSHIELD_MODEL_DIR` env var, default `backend/models/classifier`), `get_thresholds()` dependency (reads `thresholds.json` next to the model dir; falls back to `(0.3, 0.7)` if the file is missing, so the API remains runnable before calibration has been run).
  - `routers/conversations.py`: `POST /conversations`, `GET /conversations/{id}`, `GET /conversations` — FastAPI `APIRouter`.
  - `main.py`: `app = FastAPI()`, includes the conversations router, CORS middleware allowing `http://localhost:5173` (Vite dev server default).

- [ ] **Step 1: Write failing API tests using FastAPI's `TestClient` with a dependency override for the classifier and an in-memory SQLite DB**

```python
# backend/tests/api/test_conversations.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base
from contextshield.api.main import app
from contextshield.api import deps


class FakeClassifier:
    def predict_batch(self, texts):
        results = []
        for i, text in enumerate(texts):
            score = min(0.1 * (i + 1), 1.0)
            probs = {"hate_speech": 0.0, "offensive": score, "threat": 0.0, "harassment": 0.0, "none": 1 - score}
            results.append({"probabilities": probs, "toxicity_score": score})
        return results


@pytest.fixture()
def client():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db_session():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    def override_get_classifier():
        return FakeClassifier()

    def override_get_thresholds():
        return (0.2, 0.5)

    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    app.dependency_overrides[deps.get_classifier] = override_get_classifier
    app.dependency_overrides[deps.get_thresholds] = override_get_thresholds

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_create_and_get_conversation(client):
    payload = {
        "messages": [{"text": "hello"}, {"text": "world"}, {"text": "bad stuff"}],
        "source": "upload",
        "language": "en",
    }
    resp = client.post("/conversations", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["language"] == "en"
    assert len(body["messages"]) == 3
    assert len(body["messages"][0]["labels"]) == 5

    convo_id = body["id"]
    resp2 = client.get(f"/conversations/{convo_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == convo_id


def test_get_conversation_not_found(client):
    resp = client.get("/conversations/99999")
    assert resp.status_code == 404


def test_list_conversations(client):
    client.post("/conversations", json={"messages": [{"text": "a"}], "source": "upload", "language": "en"})
    client.post("/conversations", json={"messages": [{"text": "b"}], "source": "upload", "language": "hi"})
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_create_conversation_rejects_empty_messages(client):
    resp = client.post("/conversations", json={"messages": [], "source": "upload", "language": "en"})
    assert resp.status_code == 422


def test_create_conversation_rejects_unsupported_language(client):
    resp = client.post("/conversations", json={"messages": [{"text": "hi"}], "source": "upload", "language": "fr"})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/api/test_conversations.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `schemas.py`**

```python
# backend/contextshield/api/schemas.py
from typing import Literal
from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    author: str | None = None


class ConversationCreateIn(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    source: Literal["upload", "seed"] = "upload"
    language: Literal["en", "hi", "ta"]


class MessageLabelOut(BaseModel):
    category: str
    probability: float
    toxicity_score: float

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    seq_index: int
    text: str
    labels: list[MessageLabelOut]

    model_config = {"from_attributes": True}


class EscalationSnapshotOut(BaseModel):
    seq_index: int
    windowed_mean: float
    slope: float
    escalation_score: float
    status: str

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: int
    source: str
    language: str
    messages: list[MessageOut]
    escalation_snapshots: list[EscalationSnapshotOut]

    model_config = {"from_attributes": True}
```

- [ ] **Step 4: Implement `deps.py`**

```python
# backend/contextshield/api/deps.py
import json
import os
from functools import lru_cache
from pathlib import Path

from contextshield.db.session import get_default_engine, get_session_factory
from contextshield.ml.classifier import Classifier

_engine = get_default_engine()
_SessionFactory = get_session_factory(_engine)


def get_db_session():
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()


@lru_cache(maxsize=1)
def _load_classifier() -> Classifier:
    model_dir = os.environ.get(
        "CONTEXTSHIELD_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "classifier"),
    )
    return Classifier(model_dir)


def get_classifier() -> Classifier:
    return _load_classifier()


def get_thresholds() -> tuple[float, float]:
    model_dir = os.environ.get(
        "CONTEXTSHIELD_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "classifier"),
    )
    thresholds_path = Path(model_dir) / "thresholds.json"
    if not thresholds_path.exists():
        return (0.3, 0.7)
    with open(thresholds_path, encoding="utf-8") as f:
        data = json.load(f)
    return (data["tau_low"], data["tau_high"])
```

- [ ] **Step 5: Implement `routers/conversations.py`**

```python
# backend/contextshield/api/routers/__init__.py
```
(empty file)

```python
# backend/contextshield/api/routers/conversations.py
from fastapi import APIRouter, Depends, HTTPException

from contextshield.api import deps
from contextshield.api.schemas import ConversationCreateIn, ConversationOut
from contextshield.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_service(
    session=Depends(deps.get_db_session),
    classifier=Depends(deps.get_classifier),
    thresholds=Depends(deps.get_thresholds),
) -> ConversationService:
    tau_low, tau_high = thresholds
    return ConversationService(session, classifier, tau_low=tau_low, tau_high=tau_high)


@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(payload: ConversationCreateIn, service: ConversationService = Depends(get_service)):
    messages = [{"text": m.text, "author": m.author} for m in payload.messages]
    convo = service.create_conversation(messages, source=payload.source, language=payload.language)
    return convo


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: int, service: ConversationService = Depends(get_service)):
    convo = service.get_conversation(conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo


@router.get("", response_model=list[ConversationOut])
def list_conversations(limit: int = 20, offset: int = 0, service: ConversationService = Depends(get_service)):
    return service.list_conversations(limit=limit, offset=offset)
```

- [ ] **Step 6: Implement `main.py`**

```python
# backend/contextshield/api/__init__.py
```
(empty file)

```python
# backend/contextshield/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextshield.api.routers import conversations

app = FastAPI(title="ContextShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations.router)
```

- [ ] **Step 7: Create `backend/tests/api/__init__.py`** (empty)

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd backend && pytest tests/api/test_conversations.py -v`
Expected: 5 passed

- [ ] **Step 9: Commit**

```bash
git add backend/contextshield/api/ backend/tests/api/
git commit -m "feat: add FastAPI conversations endpoints"
```

---

### Task 13: FastAPI app — trends endpoints

**Files:**
- Create: `backend/contextshield/api/routers/trends.py`
- Modify: `backend/contextshield/api/main.py` (include the new router)
- Modify: `backend/contextshield/api/schemas.py` (add trend schemas)
- Test: `backend/tests/api/test_trends.py`

**Interfaces:**
- Consumes: `contextshield.db.models.TrendSnapshot`.
- Produces:
  - Schema `TrendSnapshotOut {time_bucket: str, category: str, language: str, topic_id: int | None, topic_terms: str | None, message_count: int}`.
  - `GET /trends?start=&end=&category=&language=` — filters `TrendSnapshot` rows by optional query params (`start`/`end` as `YYYY-MM-DD` strings compared lexicographically against `time_bucket`, `category`, `language` exact match), returns `list[TrendSnapshotOut]`.
  - `POST /trends/recompute` — for this task, implemented as a stub that returns `{"status": "not_implemented_yet"}` with 501; Task 15 replaces this with the real implementation once the trend pipeline is wired to the DB. This keeps the endpoint's contract stable for frontend integration in Phase C while the real recompute logic is built in Task 15.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/api/test_trends.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base, TrendSnapshot
from contextshield.api.main import app
from contextshield.api import deps


@pytest.fixture()
def client_and_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db_session():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    client = TestClient(app)
    session = TestSession()
    yield client, session
    app.dependency_overrides.clear()


def test_get_trends_filters_by_category_and_language(client_and_session):
    client, session = client_and_session
    session.add_all([
        TrendSnapshot(time_bucket="2026-01-01", category="offensive", language="en", topic_id=1, topic_terms="a,b", message_count=3),
        TrendSnapshot(time_bucket="2026-01-02", category="hate_speech", language="hi", topic_id=2, topic_terms="c,d", message_count=5),
    ])
    session.commit()

    resp = client.get("/trends", params={"category": "offensive"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["category"] == "offensive"

    resp2 = client.get("/trends", params={"language": "hi"})
    assert len(resp2.json()) == 1
    assert resp2.json()[0]["language"] == "hi"


def test_get_trends_date_range(client_and_session):
    client, session = client_and_session
    session.add_all([
        TrendSnapshot(time_bucket="2026-01-01", category="offensive", language="en", topic_id=None, topic_terms=None, message_count=1),
        TrendSnapshot(time_bucket="2026-02-01", category="offensive", language="en", topic_id=None, topic_terms=None, message_count=2),
    ])
    session.commit()

    resp = client.get("/trends", params={"start": "2026-01-15", "end": "2026-02-28"})
    assert len(resp.json()) == 1
    assert resp.json()[0]["time_bucket"] == "2026-02-01"


def test_recompute_stub_returns_501(client_and_session):
    client, _ = client_and_session
    resp = client.post("/trends/recompute")
    assert resp.status_code == 501
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/api/test_trends.py -v`
Expected: FAIL (404 for undefined routes, or `ModuleNotFoundError` if imports fail)

- [ ] **Step 3: Add `TrendSnapshotOut` to `schemas.py`**

Append to `backend/contextshield/api/schemas.py`:

```python
class TrendSnapshotOut(BaseModel):
    time_bucket: str
    category: str
    language: str
    topic_id: int | None
    topic_terms: str | None
    message_count: int

    model_config = {"from_attributes": True}
```

- [ ] **Step 4: Implement `routers/trends.py`**

```python
# backend/contextshield/api/routers/trends.py
from fastapi import APIRouter, Depends, HTTPException

from contextshield.api import deps
from contextshield.api.schemas import TrendSnapshotOut
from contextshield.db.models import TrendSnapshot

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("", response_model=list[TrendSnapshotOut])
def get_trends(
    start: str | None = None,
    end: str | None = None,
    category: str | None = None,
    language: str | None = None,
    session=Depends(deps.get_db_session),
):
    query = session.query(TrendSnapshot)
    if start is not None:
        query = query.filter(TrendSnapshot.time_bucket >= start)
    if end is not None:
        query = query.filter(TrendSnapshot.time_bucket <= end)
    if category is not None:
        query = query.filter(TrendSnapshot.category == category)
    if language is not None:
        query = query.filter(TrendSnapshot.language == language)
    return query.all()


@router.post("/recompute")
def recompute_trends():
    raise HTTPException(status_code=501, detail="Not implemented yet")
```

- [ ] **Step 5: Wire router into `main.py`**

```python
# backend/contextshield/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextshield.api.routers import conversations, trends

app = FastAPI(title="ContextShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations.router)
app.include_router(trends.router)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && pytest tests/api/test_trends.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add backend/contextshield/api/
git commit -m "feat: add trends query endpoint and recompute stub"
```

---

### Task 14: Database bootstrap script

**Files:**
- Create: `backend/scripts/init_db.py`

**Interfaces:**
- Consumes: `contextshield.db.session.get_default_engine`, `init_db`.
- Produces: a runnable script that creates all tables in the target PostgreSQL database (must exist already — the script does not create the database itself).

- [ ] **Step 1: Create `backend/scripts/init_db.py`**

```python
# backend/scripts/init_db.py
from contextshield.db.session import get_default_engine, init_db


def main():
    engine = get_default_engine()
    init_db(engine)
    print("Database tables created.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the PostgreSQL database (operator step, not code) and run the script**

Run (creates the DB if it doesn't exist, assuming default local postgres credentials postgres/postgres — adjust `CONTEXTSHIELD_DATABASE_URL` env var if different):

```bash
psql -U postgres -c "CREATE DATABASE contextshield;"
```

Then:

```bash
cd backend && python scripts/init_db.py
```

Expected: prints `Database tables created.` with no errors. If `psql` auth
fails (e.g. peer/password auth mismatch on this machine), set
`CONTEXTSHIELD_DATABASE_URL` to match the local Postgres setup before
re-running, e.g.:

```bash
export CONTEXTSHIELD_DATABASE_URL="postgresql+psycopg2://<user>:<password>@localhost:5432/contextshield"
```

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/init_db.py
git commit -m "feat: add database bootstrap script"
```

---

### Task 15: Seed data loader (runs full pipeline over processed datasets into the DB)

**Files:**
- Create: `backend/scripts/seed_database.py`
- Modify: `backend/contextshield/api/routers/trends.py` (replace the 501 stub with a real implementation)
- Test: `backend/tests/api/test_trends.py` (update the recompute test)

**Interfaces:**
- Consumes: `backend/data/processed/train.jsonl` (or all three splits combined) as the seed corpus; `contextshield.services.conversation_service.ConversationService`; `contextshield.ml.trends.bucket_time_series`, `TopicAnalyzer`; `contextshield.db.models.TrendSnapshot`.
- Produces:
  - `seed_database.py` — groups processed messages into synthetic "conversations" (chunks of N consecutive messages from the same language, N=6, since the processed dataset has no native conversation structure — documented as a demo-data construction choice), creates them via `ConversationService.create_conversation` with `source="seed"`, assigning synthetic sequential timestamps (one per day per conversation, starting 90 days ago) so the trends dashboard has a real time range to chart. After seeding, calls the real trend recompute logic.
  - `recompute_trends(session) -> int` function in `routers/trends.py` (or a new `contextshield/services/trend_service.py` — place it in a new `trend_service.py` for consistency with `conversation_service.py`): queries all `MessageLabel` rows joined to `Message`/`Conversation` where `category != "none"`, builds the message dict list `bucket_time_series` expects, runs `bucket_time_series` and `TopicAnalyzer.fit_transform` (only if there are >= 10 harmful messages, to avoid BERTopic failing on too little data), clears existing `TrendSnapshot` rows and inserts fresh ones, returns the count inserted.
  - `POST /trends/recompute` now calls `recompute_trends` for real and returns `{"snapshots_created": int}`.

- [ ] **Step 1: Update the trends test for real recompute behavior**

Replace `test_recompute_stub_returns_501` in `backend/tests/api/test_trends.py` with:

```python
def test_recompute_creates_snapshots_from_harmful_messages(client_and_session):
    client, session = client_and_session
    from contextshield.db.models import Conversation, Message, MessageLabel
    from datetime import datetime, timezone

    convo = Conversation(source="seed", language="en")
    session.add(convo)
    session.flush()
    for i in range(12):
        msg = Message(conversation_id=convo.id, seq_index=i, text=f"bad message {i}", language="en", created_at=datetime.now(timezone.utc))
        session.add(msg)
        session.flush()
        session.add(MessageLabel(message_id=msg.id, category="offensive", probability=0.9, toxicity_score=0.9))
    session.commit()

    resp = client.post("/trends/recompute")
    assert resp.status_code == 200
    assert resp.json()["snapshots_created"] >= 1

    trends_resp = client.get("/trends")
    assert len(trends_resp.json()) >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/api/test_trends.py -v`
Expected: the new test fails (still 501)

- [ ] **Step 3: Implement `backend/contextshield/services/trend_service.py`**

```python
# backend/contextshield/services/trend_service.py
from contextshield.db.models import Message, MessageLabel, TrendSnapshot
from contextshield.ml.trends import bucket_time_series, TopicAnalyzer

MIN_MESSAGES_FOR_TOPICS = 10


def recompute_trends(session) -> int:
    rows = (
        session.query(Message, MessageLabel)
        .join(MessageLabel, MessageLabel.message_id == Message.id)
        .filter(MessageLabel.category != "none")
        .filter(MessageLabel.probability >= 0.5)
        .all()
    )

    harmful_messages = [
        {
            "text": message.text,
            "language": message.language,
            "category": label.category,
            "timestamp": message.created_at,
        }
        for message, label in rows
    ]

    session.query(TrendSnapshot).delete()

    if not harmful_messages:
        session.commit()
        return 0

    time_series = bucket_time_series(harmful_messages)

    topic_by_bucket_key = {}
    if len(harmful_messages) >= MIN_MESSAGES_FOR_TOPICS:
        texts = [m["text"] for m in harmful_messages]
        try:
            analyzer = TopicAnalyzer()
            topics = analyzer.fit_transform(texts)
            if topics:
                topic_by_bucket_key["_top_topic"] = topics[0]
        except Exception:
            pass

    created = 0
    for entry in time_series:
        top_topic = topic_by_bucket_key.get("_top_topic")
        session.add(
            TrendSnapshot(
                time_bucket=entry["time_bucket"],
                category=entry["category"],
                language=entry["language"],
                topic_id=top_topic["topic_id"] if top_topic else None,
                topic_terms=",".join(top_topic["terms"][:5]) if top_topic else None,
                message_count=entry["message_count"],
            )
        )
        created += 1

    session.commit()
    return created
```

- [ ] **Step 4: Update `routers/trends.py`'s recompute endpoint**

Replace the `recompute_trends` function in `backend/contextshield/api/routers/trends.py` with:

```python
from contextshield.services.trend_service import recompute_trends as run_recompute


@router.post("/recompute")
def recompute_trends(session=Depends(deps.get_db_session)):
    count = run_recompute(session)
    return {"snapshots_created": count}
```

(Remove the now-unused `HTTPException` import if no longer referenced elsewhere in the file — check before removing.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pytest tests/api/test_trends.py -v`
Expected: 3 passed

- [ ] **Step 6: Create `backend/scripts/seed_database.py`**

```python
# backend/scripts/seed_database.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

from contextshield.db.session import get_default_engine, get_session_factory, init_db
from contextshield.services.conversation_service import ConversationService
from contextshield.services.trend_service import recompute_trends
from contextshield.ml.classifier import Classifier
from contextshield.ml.train import load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classifier"
CHUNK_SIZE = 6


def load_thresholds():
    import json

    thresholds_path = MODEL_DIR / "thresholds.json"
    if not thresholds_path.exists():
        return (0.3, 0.7)
    with open(thresholds_path, encoding="utf-8") as f:
        data = json.load(f)
    return (data["tau_low"], data["tau_high"])


def chunk_by_language(examples: list[dict], size: int) -> list[list[dict]]:
    by_language: dict[str, list[dict]] = {}
    for ex in examples:
        by_language.setdefault(ex["language"], []).append(ex)

    chunks = []
    for lang_examples in by_language.values():
        for i in range(0, len(lang_examples), size):
            chunk = lang_examples[i : i + size]
            if chunk:
                chunks.append(chunk)
    return chunks


def main():
    engine = get_default_engine()
    init_db(engine)
    session = get_session_factory(engine)()

    classifier = Classifier(MODEL_DIR)
    tau_low, tau_high = load_thresholds()
    service = ConversationService(session, classifier, tau_low=tau_low, tau_high=tau_high)

    all_examples = load_jsonl(DATA_DIR / "train.jsonl") + load_jsonl(DATA_DIR / "val.jsonl") + load_jsonl(DATA_DIR / "test.jsonl")
    chunks = chunk_by_language(all_examples, CHUNK_SIZE)

    base_time = datetime.now(timezone.utc) - timedelta(days=90)
    for i, chunk in enumerate(chunks):
        messages = [{"text": ex["text"], "author": None} for ex in chunk]
        language = chunk[0]["language"]
        convo = service.create_conversation(messages, source="seed", language=language)

        convo_time = base_time + timedelta(days=i)
        for message in convo.messages:
            message.created_at = convo_time
        session.commit()

    count = recompute_trends(session)
    print(f"Seeded {len(chunks)} conversations. Created {count} trend snapshots.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the seed script (requires Task 14's DB and Task 5's trained model)**

Run: `cd backend && python scripts/seed_database.py`
Expected: prints `Seeded N conversations. Created M trend snapshots.` with no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/contextshield/services/trend_service.py backend/contextshield/api/routers/trends.py backend/scripts/seed_database.py backend/tests/api/test_trends.py
git commit -m "feat: implement real trend recomputation and seed database script"
```

---

### Task 16: Upload endpoint for file-based conversation input

**Files:**
- Create: `backend/contextshield/api/routers/uploads.py`
- Modify: `backend/contextshield/api/main.py` (include router)
- Test: `backend/tests/api/test_uploads.py`

**Interfaces:**
- Consumes: `contextshield.services.conversation_service.ConversationService` (via the same `get_service` dependency factory as `conversations.py` — move `get_service` to `deps.py` if needed for reuse, or duplicate the small factory function in `uploads.py` referencing the same dependencies).
- Produces: `POST /conversations/upload` — accepts `multipart/form-data` with a `file` field (plain `.txt`, one message per line) and `language` form field; parses lines (skipping blank lines), delegates to the same conversation-creation flow as `POST /conversations` with `source="upload"`, returns `ConversationOut`.

- [ ] **Step 1: Move `get_service` into `deps.py` for reuse**

In `backend/contextshield/api/deps.py`, append:

```python
from contextshield.services.conversation_service import ConversationService


def get_conversation_service(
    session=None, classifier=None, thresholds=None,
) -> ConversationService:
    pass
```

Actually implement it properly using FastAPI's `Depends` composition — replace
the above stub with:

```python
def get_conversation_service_factory():
    def _factory(session=None, classifier=None, thresholds=None) -> ConversationService:
        tau_low, tau_high = thresholds
        return ConversationService(session, classifier, tau_low=tau_low, tau_high=tau_high)

    return _factory
```

This indirection is unnecessary complexity — instead, keep it simple: define
`get_conversation_service` directly as a FastAPI-dependency-injectable
function in `deps.py`:

```python
def get_conversation_service(
    session=Depends(get_db_session),
    classifier=Depends(get_classifier),
    thresholds=Depends(get_thresholds),
) -> ConversationService:
    tau_low, tau_high = thresholds
    return ConversationService(session, classifier, tau_low=tau_low, tau_high=tau_high)
```

Add `from fastapi import Depends` to the top of `deps.py` if not already
imported. Then update `backend/contextshield/api/routers/conversations.py`
to import and use `deps.get_conversation_service` instead of its local
`get_service` function (delete the local `get_service` definition, replace
`service: ConversationService = Depends(get_service)` with
`service: ConversationService = Depends(deps.get_conversation_service)` in
both route handlers).

- [ ] **Step 2: Write failing test for the upload endpoint**

```python
# backend/tests/api/test_uploads.py
import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base
from contextshield.api.main import app
from contextshield.api import deps


class FakeClassifier:
    def predict_batch(self, texts):
        return [
            {
                "probabilities": {"hate_speech": 0.0, "offensive": 0.1, "threat": 0.0, "harassment": 0.0, "none": 0.9},
                "toxicity_score": 0.1,
            }
            for _ in texts
        ]


@pytest.fixture()
def client():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db_session():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    app.dependency_overrides[deps.get_classifier] = lambda: FakeClassifier()
    app.dependency_overrides[deps.get_thresholds] = lambda: (0.2, 0.5)

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_upload_conversation_from_file(client):
    file_content = b"hello there\nhow are you\nsomething offensive maybe\n"
    resp = client.post(
        "/conversations/upload",
        files={"file": ("convo.txt", io.BytesIO(file_content), "text/plain")},
        data={"language": "en"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["messages"]) == 3
    assert body["source"] == "upload"


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/conversations/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        data={"language": "en"},
    )
    assert resp.status_code == 422
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && pytest tests/api/test_uploads.py -v`
Expected: FAIL (404, route doesn't exist)

- [ ] **Step 4: Implement `routers/uploads.py`**

```python
# backend/contextshield/api/routers/uploads.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, Form

from contextshield.api import deps
from contextshield.api.schemas import ConversationOut
from contextshield.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/upload", response_model=ConversationOut, status_code=201)
async def upload_conversation(
    file: UploadFile,
    language: str = Form(...),
    service: ConversationService = Depends(deps.get_conversation_service),
):
    if language not in {"en", "hi", "ta"}:
        raise HTTPException(status_code=422, detail="Unsupported language")

    raw_bytes = await file.read()
    lines = [line.strip() for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="File contains no messages")

    messages = [{"text": line, "author": None} for line in lines]
    convo = service.create_conversation(messages, source="upload", language=language)
    return convo
```

- [ ] **Step 5: Wire into `main.py`**

Add `uploads` to the import and `include_router` calls in
`backend/contextshield/api/main.py`:

```python
from contextshield.api.routers import conversations, trends, uploads
```

and add `app.include_router(uploads.router)` alongside the other two.

- [ ] **Step 6: Run tests to verify they pass, then re-run all API tests for regressions**

Run: `cd backend && pytest tests/api/ -v`
Expected: all tests pass (conversations, trends, uploads)

- [ ] **Step 7: Commit**

```bash
git add backend/contextshield/api/
git commit -m "feat: add conversation upload endpoint and refactor shared service dependency"
```

---

### Task 17: Run full backend test suite and app smoke test

**Files:** none created; verification task.

- [ ] **Step 1: Run the full backend test suite**

Run: `cd backend && pytest -v`
Expected: all tests pass (or the two documented offline-skippable tests skip cleanly)

- [ ] **Step 2: Start the API server and smoke-test it manually**

Run: `cd backend && uvicorn contextshield.api.main:app --reload --port 8000` (in background/separate terminal)

Then in another shell:

```bash
curl -X POST http://localhost:8000/conversations -H "Content-Type: application/json" -d "{\"messages\":[{\"text\":\"hello\"},{\"text\":\"you are trash\"},{\"text\":\"I will hurt you\"}],\"source\":\"upload\",\"language\":\"en\"}"
```

Expected: HTTP 201 with a JSON body containing `id`, `messages` (3 entries,
each with 5 labels), and `escalation_snapshots` (1 entry, since w=3 and
3 messages gives exactly one snapshot at seq_index=2).

- [ ] **Step 3: Stop the server**

No commit needed — this is a verification-only task.

---

## Phase C: Frontend

### Task 18: Vite React + TypeScript scaffold

**Files:**
- Create: `frontend/` (via Vite scaffolding command)
- Modify: `frontend/src/App.tsx`, `frontend/src/main.tsx`
- Create: `frontend/.env.development`

**Interfaces:**
- Produces: a running Vite dev server at `http://localhost:5173` with a placeholder page; `frontend/src/api/client.ts` will be added in Task 19 as the shared API base.

- [ ] **Step 1: Scaffold the project**

Run: `cd "C:/Users/PARV/Desktop/Social" && npm create vite@latest frontend -- --template react-ts`

- [ ] **Step 2: Install dependencies plus routing and charting libraries**

Run: `cd frontend && npm install && npm install react-router-dom recharts`

- [ ] **Step 3: Create `frontend/.env.development`**

```
VITE_API_BASE_URL=http://localhost:8000
```

- [ ] **Step 4: Replace `frontend/src/App.tsx` with a minimal placeholder**

```tsx
// frontend/src/App.tsx
function App() {
  return (
    <div>
      <h1>ContextShield</h1>
      <p>Multilingual harmful content detection and trend analysis.</p>
    </div>
  );
}

export default App;
```

- [ ] **Step 5: Run the dev server and verify it loads**

Run: `cd frontend && npm run dev`
Expected: server starts on port 5173; open `http://localhost:5173` in a
browser (or use the built-in browser tool) and confirm "ContextShield"
heading renders.

- [ ] **Step 6: Stop the dev server, commit**

```bash
git add frontend/
git commit -m "chore: scaffold React + TypeScript frontend with Vite"
```

---

### Task 19: API client module and TypeScript types

**Files:**
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/client.test.ts`
- Create: `frontend/vitest.config.ts`
- Modify: `frontend/package.json` (add vitest, add test script)

**Interfaces:**
- Produces:
  - `types.ts`: TypeScript interfaces mirroring backend schemas — `MessageLabel {category: string; probability: number; toxicity_score: number}`, `Message {id: number; seq_index: number; text: string; labels: MessageLabel[]}`, `EscalationSnapshot {seq_index: number; windowed_mean: number; slope: number; escalation_score: number; status: "stable" | "elevated" | "escalating"}`, `Conversation {id: number; source: string; language: string; messages: Message[]; escalation_snapshots: EscalationSnapshot[]}`, `TrendSnapshot {time_bucket: string; category: string; language: string; topic_id: number | null; topic_terms: string | null; message_count: number}`.
  - `client.ts`: `createConversation(payload: {messages: {text: string; author?: string}[]; source: "upload" | "seed"; language: "en" | "hi" | "ta"}): Promise<Conversation>`, `getConversation(id: number): Promise<Conversation>`, `listConversations(limit?: number, offset?: number): Promise<Conversation[]>`, `uploadConversation(file: File, language: string): Promise<Conversation>`, `getTrends(filters?: {start?: string; end?: string; category?: string; language?: string}): Promise<TrendSnapshot[]>`. All use `fetch` against `import.meta.env.VITE_API_BASE_URL`, throw an `Error` with the response body's `detail` field on non-2xx responses.

- [ ] **Step 1: Install vitest and configure it**

Run: `cd frontend && npm install -D vitest`

Create `frontend/vitest.config.ts`:

```typescript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
  },
})
```

Add to `frontend/package.json` scripts: `"test": "vitest run"`.

- [ ] **Step 2: Write failing tests for the API client using a mocked `fetch`**

```typescript
// frontend/src/api/client.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createConversation, getConversation, getTrends } from './client'

describe('api client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('createConversation posts to /conversations and returns parsed JSON', async () => {
    const mockResponse = { id: 1, source: 'upload', language: 'en', messages: [], escalation_snapshots: [] }
    ;(fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    })

    const result = await createConversation({
      messages: [{ text: 'hello' }],
      source: 'upload',
      language: 'en',
    })

    expect(result).toEqual(mockResponse)
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/conversations'),
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('getConversation throws on non-ok response with detail message', async () => {
    ;(fetch as any).mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'Conversation not found' }),
    })

    await expect(getConversation(999)).rejects.toThrow('Conversation not found')
  })

  it('getTrends builds query params from filters', async () => {
    ;(fetch as any).mockResolvedValue({ ok: true, json: async () => [] })

    await getTrends({ category: 'offensive', language: 'en' })

    const calledUrl = (fetch as any).mock.calls[0][0] as string
    expect(calledUrl).toContain('category=offensive')
    expect(calledUrl).toContain('language=en')
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd frontend && npm run test`
Expected: FAIL (module `./client` doesn't exist)

- [ ] **Step 4: Implement `types.ts`**

```typescript
// frontend/src/api/types.ts
export interface MessageLabel {
  category: string
  probability: number
  toxicity_score: number
}

export interface Message {
  id: number
  seq_index: number
  text: string
  labels: MessageLabel[]
}

export type EscalationStatus = 'stable' | 'elevated' | 'escalating'

export interface EscalationSnapshot {
  seq_index: number
  windowed_mean: number
  slope: number
  escalation_score: number
  status: EscalationStatus
}

export interface Conversation {
  id: number
  source: string
  language: string
  messages: Message[]
  escalation_snapshots: EscalationSnapshot[]
}

export interface TrendSnapshot {
  time_bucket: string
  category: string
  language: string
  topic_id: number | null
  topic_terms: string | null
  message_count: number
}
```

- [ ] **Step 5: Implement `client.ts`**

```typescript
// frontend/src/api/client.ts
import type { Conversation, TrendSnapshot } from './types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string

async function handleResponse<T>(response: Response): Promise<T> {
  const body = await response.json()
  if (!response.ok) {
    throw new Error(body.detail ?? 'Request failed')
  }
  return body as T
}

export async function createConversation(payload: {
  messages: { text: string; author?: string }[]
  source: 'upload' | 'seed'
  language: 'en' | 'hi' | 'ta'
}): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse<Conversation>(response)
}

export async function getConversation(id: number): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/conversations/${id}`)
  return handleResponse<Conversation>(response)
}

export async function listConversations(limit = 20, offset = 0): Promise<Conversation[]> {
  const response = await fetch(`${BASE_URL}/conversations?limit=${limit}&offset=${offset}`)
  return handleResponse<Conversation[]>(response)
}

export async function uploadConversation(file: File, language: string): Promise<Conversation> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('language', language)
  const response = await fetch(`${BASE_URL}/conversations/upload`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse<Conversation>(response)
}

export async function getTrends(filters?: {
  start?: string
  end?: string
  category?: string
  language?: string
}): Promise<TrendSnapshot[]> {
  const params = new URLSearchParams()
  if (filters?.start) params.set('start', filters.start)
  if (filters?.end) params.set('end', filters.end)
  if (filters?.category) params.set('category', filters.category)
  if (filters?.language) params.set('language', filters.language)
  const query = params.toString()
  const response = await fetch(`${BASE_URL}/trends${query ? `?${query}` : ''}`)
  return handleResponse<TrendSnapshot[]>(response)
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd frontend && npm run test`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: add typed API client for backend integration"
```

---

### Task 20: Analyzer page

**Files:**
- Create: `frontend/src/pages/AnalyzerPage.tsx`
- Create: `frontend/src/components/EscalationChart.tsx`
- Create: `frontend/src/components/MessageList.tsx`
- Modify: `frontend/src/App.tsx` (add routing)

**Interfaces:**
- Consumes: `createConversation`, `uploadConversation` from `api/client.ts`; `Conversation`, `Message`, `EscalationSnapshot` types.
- Produces: `AnalyzerPage` component — a textarea for pasting messages (one per line) plus a file upload input, a language selector (en/hi/ta), a submit button, and on success renders `MessageList` (per-message category badges + toxicity score) and `EscalationChart` (line chart of `escalation_score` over `seq_index`, colored/annotated by `status`).

- [ ] **Step 1: Install react-router-dom routing in `App.tsx`**

```tsx
// frontend/src/App.tsx
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import AnalyzerPage from './pages/AnalyzerPage'

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Analyzer</Link>
      </nav>
      <Routes>
        <Route path="/" element={<AnalyzerPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
```

- [ ] **Step 2: Create `components/MessageList.tsx`**

```tsx
// frontend/src/components/MessageList.tsx
import type { Message } from '../api/types'

function topCategory(message: Message): { category: string; probability: number } {
  const nonNone = message.labels.filter((l) => l.category !== 'none')
  return nonNone.reduce(
    (best, current) => (current.probability > best.probability ? current : best),
    { category: 'none', probability: 0 }
  )
}

export default function MessageList({ messages }: { messages: Message[] }) {
  return (
    <ul>
      {messages.map((message) => {
        const top = topCategory(message)
        return (
          <li key={message.id}>
            <span>{message.text}</span>
            {top.probability > 0.5 && (
              <span> [{top.category}: {top.probability.toFixed(2)}]</span>
            )}
          </li>
        )
      })}
    </ul>
  )
}
```

- [ ] **Step 3: Create `components/EscalationChart.tsx`**

```tsx
// frontend/src/components/EscalationChart.tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import type { EscalationSnapshot } from '../api/types'

export default function EscalationChart({ snapshots }: { snapshots: EscalationSnapshot[] }) {
  if (snapshots.length === 0) {
    return <p>Not enough messages for an escalation timeline (need at least 3).</p>
  }

  const data = snapshots.map((s) => ({
    seq_index: s.seq_index,
    escalation_score: s.escalation_score,
    status: s.status,
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="seq_index" label={{ value: 'Message index', position: 'insideBottom', offset: -5 }} />
        <YAxis domain={[0, 1]} />
        <Tooltip formatter={(value: number, _name, props) => [`${value.toFixed(2)} (${props.payload.status})`, 'Escalation score']} />
        <Line type="monotone" dataKey="escalation_score" stroke="#d62728" strokeWidth={2} dot />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

- [ ] **Step 4: Create `pages/AnalyzerPage.tsx`**

```tsx
// frontend/src/pages/AnalyzerPage.tsx
import { useState } from 'react'
import { createConversation, uploadConversation } from '../api/client'
import type { Conversation } from '../api/types'
import MessageList from '../components/MessageList'
import EscalationChart from '../components/EscalationChart'

export default function AnalyzerPage() {
  const [text, setText] = useState('')
  const [language, setLanguage] = useState<'en' | 'hi' | 'ta'>('en')
  const [file, setFile] = useState<File | null>(null)
  const [conversation, setConversation] = useState<Conversation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      let result: Conversation
      if (file) {
        result = await uploadConversation(file, language)
      } else {
        const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
        result = await createConversation({
          messages: lines.map((line) => ({ text: line })),
          source: 'upload',
          language,
        })
      }
      setConversation(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1>Analyze a Conversation</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste one message per line"
          rows={8}
          cols={60}
          disabled={!!file}
        />
        <div>
          <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <select value={language} onChange={(e) => setLanguage(e.target.value as 'en' | 'hi' | 'ta')}>
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="ta">Tamil</option>
        </select>
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}

      {conversation && (
        <div>
          <h2>Results</h2>
          <MessageList messages={conversation.messages} />
          <EscalationChart snapshots={conversation.escalation_snapshots} />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 5: Run the dev server and manually verify the Analyzer page works against the running backend**

Run backend (`cd backend && uvicorn contextshield.api.main:app --port 8000`)
and frontend (`cd frontend && npm run dev`) concurrently. Open
`http://localhost:5173`, paste 4+ lines of text, click Analyze, verify
message list and escalation chart render without console errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: add Analyzer page with message list and escalation chart"
```

---

### Task 21: Browse page

**Files:**
- Create: `frontend/src/pages/BrowsePage.tsx`
- Modify: `frontend/src/App.tsx` (add route + nav link)

**Interfaces:**
- Consumes: `listConversations`, `getConversation` from `api/client.ts`.
- Produces: `BrowsePage` — fetches and lists conversations (id, language, source, message count, latest escalation status), clicking one shows its full detail (reusing `MessageList` and `EscalationChart`).

- [ ] **Step 1: Create `pages/BrowsePage.tsx`**

```tsx
// frontend/src/pages/BrowsePage.tsx
import { useEffect, useState } from 'react'
import { listConversations, getConversation } from '../api/client'
import type { Conversation } from '../api/types'
import MessageList from '../components/MessageList'
import EscalationChart from '../components/EscalationChart'

export default function BrowsePage() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selected, setSelected] = useState<Conversation | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listConversations().then(setConversations).catch((err) => setError(err.message))
  }, [])

  async function handleSelect(id: number) {
    try {
      const convo = await getConversation(id)
      setSelected(convo)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load conversation')
    }
  }

  return (
    <div>
      <h1>Browse Conversations</h1>
      {error && <p role="alert">{error}</p>}
      <ul>
        {conversations.map((convo) => {
          const latestStatus = convo.escalation_snapshots.at(-1)?.status ?? 'n/a'
          return (
            <li key={convo.id}>
              <button onClick={() => handleSelect(convo.id)}>
                #{convo.id} — {convo.language} — {convo.source} — {convo.messages.length} messages — {latestStatus}
              </button>
            </li>
          )
        })}
      </ul>

      {selected && (
        <div>
          <h2>Conversation #{selected.id}</h2>
          <MessageList messages={selected.messages} />
          <EscalationChart snapshots={selected.escalation_snapshots} />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Wire the route into `App.tsx`**

```tsx
// frontend/src/App.tsx
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import AnalyzerPage from './pages/AnalyzerPage'
import BrowsePage from './pages/BrowsePage'

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Analyzer</Link> | <Link to="/browse">Browse</Link>
      </nav>
      <Routes>
        <Route path="/" element={<AnalyzerPage />} />
        <Route path="/browse" element={<BrowsePage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
```

- [ ] **Step 3: Manually verify in browser**

With backend seeded (Task 15 run) and both servers running, navigate to
`/browse`, verify the list populates and clicking an item shows its detail.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add Browse page for listing and inspecting conversations"
```

---

### Task 22: Trends Dashboard page

**Files:**
- Create: `frontend/src/pages/TrendsPage.tsx`
- Create: `frontend/src/components/TrendChart.tsx`
- Create: `frontend/src/components/TopicCards.tsx`
- Modify: `frontend/src/App.tsx` (add route + nav link)

**Interfaces:**
- Consumes: `getTrends` from `api/client.ts`; `TrendSnapshot` type.
- Produces: `TrendsPage` — filter controls (language, category dropdowns), a `TrendChart` (bar/line chart of `message_count` over `time_bucket`, grouped/colored by `category`), and `TopicCards` (cards showing distinct `topic_id`/`topic_terms` combinations found in the current filtered snapshots).

- [ ] **Step 1: Create `components/TrendChart.tsx`**

```tsx
// frontend/src/components/TrendChart.tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from 'recharts'
import type { TrendSnapshot } from '../api/types'

const COLORS: Record<string, string> = {
  hate_speech: '#d62728',
  offensive: '#ff7f0e',
  threat: '#9467bd',
  harassment: '#8c564b',
}

export default function TrendChart({ snapshots }: { snapshots: TrendSnapshot[] }) {
  const byBucket = new Map<string, Record<string, number>>()
  for (const snap of snapshots) {
    const entry = byBucket.get(snap.time_bucket) ?? {}
    entry[snap.category] = (entry[snap.category] ?? 0) + snap.message_count
    byBucket.set(snap.time_bucket, entry)
  }
  const data = Array.from(byBucket.entries())
    .map(([time_bucket, counts]) => ({ time_bucket, ...counts }))
    .sort((a, b) => a.time_bucket.localeCompare(b.time_bucket))

  const categories = Array.from(new Set(snapshots.map((s) => s.category)))

  if (data.length === 0) {
    return <p>No trend data available yet.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time_bucket" />
        <YAxis />
        <Tooltip />
        <Legend />
        {categories.map((category) => (
          <Bar key={category} dataKey={category} stackId="a" fill={COLORS[category] ?? '#7f7f7f'} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
```

- [ ] **Step 2: Create `components/TopicCards.tsx`**

```tsx
// frontend/src/components/TopicCards.tsx
import type { TrendSnapshot } from '../api/types'

export default function TopicCards({ snapshots }: { snapshots: TrendSnapshot[] }) {
  const seen = new Map<number, string>()
  for (const snap of snapshots) {
    if (snap.topic_id !== null && snap.topic_terms && !seen.has(snap.topic_id)) {
      seen.set(snap.topic_id, snap.topic_terms)
    }
  }

  if (seen.size === 0) {
    return <p>No topic clusters available yet (need more harmful messages).</p>
  }

  return (
    <div>
      {Array.from(seen.entries()).map(([topicId, terms]) => (
        <div key={topicId} style={{ border: '1px solid #ccc', padding: '0.5rem', margin: '0.5rem 0' }}>
          <strong>Topic {topicId}</strong>
          <p>{terms}</p>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Create `pages/TrendsPage.tsx`**

```tsx
// frontend/src/pages/TrendsPage.tsx
import { useEffect, useState } from 'react'
import { getTrends } from '../api/client'
import type { TrendSnapshot } from '../api/types'
import TrendChart from '../components/TrendChart'
import TopicCards from '../components/TopicCards'

export default function TrendsPage() {
  const [snapshots, setSnapshots] = useState<TrendSnapshot[]>([])
  const [language, setLanguage] = useState('')
  const [category, setCategory] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getTrends({ language: language || undefined, category: category || undefined })
      .then(setSnapshots)
      .catch((err) => setError(err.message))
  }, [language, category])

  return (
    <div>
      <h1>Trends Dashboard</h1>
      {error && <p role="alert">{error}</p>}
      <div>
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="">All languages</option>
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="ta">Tamil</option>
        </select>
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="">All categories</option>
          <option value="hate_speech">Hate speech</option>
          <option value="offensive">Offensive</option>
          <option value="threat">Threat</option>
          <option value="harassment">Harassment</option>
        </select>
      </div>
      <TrendChart snapshots={snapshots} />
      <h2>Emerging Topics</h2>
      <TopicCards snapshots={snapshots} />
    </div>
  )
}
```

- [ ] **Step 4: Wire the route into `App.tsx`**

```tsx
// frontend/src/App.tsx
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import AnalyzerPage from './pages/AnalyzerPage'
import BrowsePage from './pages/BrowsePage'
import TrendsPage from './pages/TrendsPage'

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Analyzer</Link> | <Link to="/browse">Browse</Link> | <Link to="/trends">Trends</Link>
      </nav>
      <Routes>
        <Route path="/" element={<AnalyzerPage />} />
        <Route path="/browse" element={<BrowsePage />} />
        <Route path="/trends" element={<TrendsPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
```

- [ ] **Step 5: Manually verify in browser**

With backend seeded and running, navigate to `/trends`, verify the bar chart
and topic cards render, and that language/category filters refetch data.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: add Trends Dashboard page with charts and topic cards"
```

---

### Task 23: End-to-end smoke test and README

**Files:**
- Create: `README.md` (project root)
- Modify: none (verification + documentation task)

**Interfaces:** none new.

- [ ] **Step 1: Write the root `README.md`**

```markdown
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
- `frontend/` — React + TypeScript dashboard (Analyzer, Browse, Trends
  pages).

## Setup

### 1. Backend

    cd backend
    python -m venv .venv
    .venv\Scripts\activate
    pip install -e ".[dev]"

Place licensed datasets in `backend/data/raw/` per `backend/data/README.md`,
or skip this to use the bundled synthetic fallback data.

    python scripts/prepare_datasets.py
    python scripts/check_training_feasible.py   # optional: estimate training time
    python scripts/train_classifier.py          # add --max-train-examples N if needed
    python scripts/calibrate_thresholds.py

    psql -U postgres -c "CREATE DATABASE contextshield;"
    python scripts/init_db.py
    python scripts/seed_database.py

    uvicorn contextshield.api.main:app --reload --port 8000

### 2. Frontend

    cd frontend
    npm install
    npm run dev

Open http://localhost:5173.

## Tests

    cd backend && pytest
    cd frontend && npm run test
```

- [ ] **Step 2: Run the full test suite one more time across both projects**

Run: `cd backend && pytest -v`
Run: `cd frontend && npm run test`
Expected: all pass (or documented offline skips)

- [ ] **Step 3: Full manual end-to-end walkthrough**

With PostgreSQL running, backend seeded, and both dev servers up: visit
Analyzer, submit a multi-line conversation, confirm results render; visit
Browse, confirm seeded conversations list and open; visit Trends, confirm
chart and topic cards render and filters work.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add project README with setup instructions"
```

---

## Self-Review Notes

- **Spec coverage:** §3.1 classifier → Tasks 1-8; §3.2 escalation → Task 2, 11; §3.3 trends → Task 9, 15; §4 data model → Task 10; §5 API → Tasks 12-16; §6 frontend → Tasks 18-22; §7 ingestion → Tasks 15-16; §8 error handling → Tasks 12, 16 (422/404/501→200 paths); §9 testing → present throughout; §10 out-of-scope items are not built anywhere in this plan (verified: no RNN/sequence model task, no custom topic training task, no scraping task, no auth task).
- **CPU training risk** is called out explicitly in Global Constraints and Task 8's feasibility probe + `--max-train-examples` escape hatch, since this is the single highest-risk step for an unattended overnight run.
- **Type consistency check:** `Classifier.predict_batch` return shape (`{"probabilities": dict, "toxicity_score": float}`) is used identically in Task 6, 7, 11, 12 tests. `ConversationService.create_conversation` signature (`messages, source, language`) matches its usage in Task 12 and Task 16. `compute_timeline` output keys match `EscalationSnapshot` field names via Task 11's mapping.
