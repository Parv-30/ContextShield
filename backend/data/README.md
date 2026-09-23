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

## Real dataset sources via HuggingFace Hub

As an alternative to the manual downloads above, `scripts/fetch_real_datasets.py`
fetches three real, non-synthetic datasets directly from the HuggingFace Hub
with no manual/licensed download step: Davidson (English hate speech/offensive
language) written to `backend/data/raw/davidson_en.csv`, `nikitadesai/hasoc`
(Hindi) written to `backend/data/raw/hasoc_hi.csv` (superseding the manual
`hasoc_hi.csv` above if run), and `community-datasets/offenseval_dravidian`
(Tamil) written to `backend/data/raw/dravidian_ta.csv`. These are mapped to
`contextshield.ml.labels.LABELS` via `map_davidson_label` and
`map_hasoc_hindi_label` in `contextshield/ml/labels.py`, which are already
wired into `scripts/prepare_datasets.py` and covered by
`tests/ml/test_labels.py`. **Known limitation:** none of these three real
dataset source taxonomies (Davidson, HASOC Hindi, or OffensEval Dravidian)
include an explicit "threat" category, so when training on real data alone
the `threat` label in the model's output head will be underrepresented/
effectively unreachable — only the synthetic `seed_fallback.jsonl` set below
covers `threat` examples. This is a known limitation of the available public
datasets, not a bug in the mapping code.

## About `seed_fallback.jsonl`

This file is clearly-labeled synthetic/placeholder data, not real user
content. It contains 60 short, hand-written social-media-style sentences
(20 each in English, Hindi, and Tamil) covering all five labels in
`contextshield.ml.labels.LABELS` (`hate_speech`, `offensive`, `threat`,
`harassment`, `none`), used only to exercise the preprocessing and training
pipeline mechanics when licensed datasets are unavailable.
