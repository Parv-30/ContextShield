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

## About `seed_fallback.jsonl`

This file is clearly-labeled synthetic/placeholder data, not real user
content. It contains 60 short, hand-written social-media-style sentences
(20 each in English, Hindi, and Tamil) covering all five labels in
`contextshield.ml.labels.LABELS` (`hate_speech`, `offensive`, `threat`,
`harassment`, `none`), used only to exercise the preprocessing and training
pipeline mechanics when licensed datasets are unavailable.
