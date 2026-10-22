# Log Representations and Learning Regimes for Log-based Anomaly Detection

A replication and extension of **Wu, Li & Khomh, "On the Effectiveness of Log Representation
for Log-based Anomaly Detection" (EMSE 2023)** — LOG6309E group assignment, Polytechnique Montréal.

Raw system logs are unstructured text; every machine-learning detector needs them turned into
numbers first. Wu et al. showed that this *representation* step matters as much as the model,
and that the simple Message Count Vector (MCV) is hard to beat. We replicate their RQ1 on
HDFS and BGL, then extend the study on HDFS with two models that need fewer labels, a
statistical ranking over 10 resampled train/test splits, and SHAP explanations of the winner.

## Architecture

Two stages share one representation pipeline: `part1` builds it and replicates RQ1; `part2`
reuses it unchanged — through a bridge (`part2/src/p1.py`) that loads Part 1's package under
the alias `p1` so sessions and splits are literally the same objects — and adds the extension
models, the statistical ranking, and the explanations. `notebooks/` re-implements the whole
flow cell-by-cell for Google Colab; `report/` holds the IEEE paper.

```
Loghub HDFS.log (11.2M lines) · BGL.log (4.7M lines)         data/ (gitignored)
        │
        │  PART 1 — replication of RQ1            one driver: src/run_part1.py
        │  ─────────────────────────────────────────────────────────────────
        ├─ preprocess.py          variable masking (IPs, paths, block ids, numbers)
        │                          — general framework of Qin et al., SANER 2025
        ├─ drain.py               streaming Drain parser (depth 4, similarity 0.5)
        ├─ parsing_eval.py        grouping accuracy vs Loghub-2.0 (1.000 HDFS / 0.970 BGL)
        ├─ partition.py           sessions — HDFS blocks · BGL 6-hour windows
        ├─ split.py               seeded train/test splits
        ├─ represent_mcv.py       Message Count Vector per session
        ├─ represent_bert.py      frozen BERT template embeddings, sliding windows
        ├─ feature_selection.py   Spearman correlation clustering + VIF (train data only)
        ├─ models/                rf.py — RF on MCV · lstm.py — LSTM on BERT windows
        └─ evaluate.py            F1 / AUC per run → JSON in results/ → compare with paper
        │
        │  PART 2 — extension on HDFS, seeds 0–9  one driver: src/run_resamples.py
        │  ─────────────────────────────────────────────────────────────────
        ├─ p1.py             bridge to part1 (shared sessions and splits)
        ├─ models/pca.py     PCA detector — loglizer port, tf-idf counts, SPE + Q-statistic
        ├─ models/loggpt/    LogGPT — Nokia port, GPT-2 trained on 5,000 normal sessions + RL
        ├─ scores.py         one continuous score per session → fixed decision rule
        ├─ scott_knott.R     Scott-Knott ESD ranking of the models across the 10 resamples
        └─ explain.py        SHAP on the winning RF, grouped by correlated templates
```

Three rules hold across the whole pipeline, so every number below is comparable:

- **No leakage** — vectorizers and feature selection are fit on TRAIN data only.
- **Same test set** — within a resample, every model is scored on exactly the same test
  sessions (the full 30% split, asserted by the driver), with one binary prediction per
  session (`scores.py`, fairness rules F5/F6: threshold and AUC always describe one score).
- **Everything on disk** — each run writes JSON/parquet sidecars (model, dataset, seed,
  selection, per-session scores) that `runs.csv`, the R ranking, and the report figures are
  rebuilt from.

### Method in one paragraph

Drain (depth 4, similarity 0.5) parses each log into event templates after Qin et al.'s
preprocessing masks variables — grouping accuracy 1.000 (HDFS) and 0.970 (BGL) on Loghub-2.0
ground truth. Sessions are HDFS blocks and BGL 6-hour windows. Each model receives the
representation that matches its mechanism: RF and PCA read session-level count vectors (MCV;
PCA keeps the tf-idf-weighted count matrix it was designed for), the LSTM reads frozen BERT
template embeddings event by event through sliding windows, and LogGPT reads raw template-id
sequences and learns its own embedding. All models output a continuous anomaly score per
session with a fixed decision rule (0.5, the Q-statistic, or Top-K).

### Where things live

- `part1/` — replication of RQ1: Qin-et-al. preprocessing + streaming Drain (HDFS 55 s),
  sessions, seeded splits, MCV + BERT representations, RF and LSTM (with/without
  correlation+VIF feature selection), 6 seeds, comparison with the paper. See `part1/README.md`.
- `part2/` — extension on HDFS: PCA (loglizer port) and LogGPT (Nokia port) + the two
  supervised models, 10 resamples, Scott-Knott ESD ranking (R), SHAP explanation.
  See `part2/README.md`.
- `notebooks/` — self-contained Google Colab notebooks, incl. the independent 10-seed re-run
  of the extension. See `notebooks/README.md`.
- `report/` — IEEE report (`main.tex` → `main.pdf`, 21 references, figures from `make_figures.py`).
- `DECISIONS.md` — every design decision with its rationale and its deviation from the paper.

## Results at a glance

**Part 1 — replication of RQ1** (seed 42; F1, session level, vs the paper):

| Dataset | Model | Paper | Ours |
|---|---|---|---|
| HDFS | RF + MCV | .999 | **.999** |
| HDFS | LSTM + BERT | .959 | **.987** |
| BGL | RF + MCV | .891 | **.919** |
| BGL | LSTM + BERT | .931 | **.870** |

The HDFS LSTM gap is traced to a windowing bug in the original replication package (sessions
shorter than the window were truncated); the BGL gap is consistent with a different number of
6-hour windows (826 vs the paper's 718). Three package bugs were found and measured by ablation.

**Part 2 — extension on HDFS** (median [IQR] over 10 resamples; Scott-Knott ESD puts every
model in its own group for both F1 and AUC):

| Model | Labels used | F1 | AUC |
|---|---|---|---|
| RF + MCV | all 402,542 | .998 [.9979–.9983] | .9999 |
| LSTM + BERT | all 402,542 | .979 [.9748–.9835] | .9999 |
| LogGPT (GPT-2, normal only) | 5,000 normal | .845 [.8144–.9086] | .9985 |
| PCA (no labels) | 402,542 (labels removed) | .769 [.7665–.7698] | .9589 |

Take-homes: with labels, a 10-tree Random Forest on template counts is the best and cheapest
model; without labels, the training protocol and threshold matter as much as the model; and
per-feature importance underrates correlated templates, so explanations should be grouped.

An **independent re-run of the whole pipeline on Google Colab** (different machine, Python
3.12, T4 GPU — `notebooks/03_extension_10seeds.ipynb`) reproduces the deterministic models to
four decimals (RF F1 .9982, PCA .7688) and the LSTM and LogGPT within their GPU-seed variance,
with the same Scott-Knott ordering.

## Reproducing

### The easy way — one notebook, no setup

Open `notebooks/03_extension_10seeds.ipynb` in Google Colab (GPU runtime). It downloads HDFS
itself, re-implements the pipeline cell by cell (parsing, sessions, MCV/BERT, RF, LSTM, PCA,
LogGPT, Scott-Knott), and stores everything on Google Drive so any session resumes where the
previous one stopped. First run: ~40 min without LogGPT; LogGPT adds 1–2 h per seed (it
checkpoints after every RL episode, so it can be spread over several sessions).

### The original way — part1/part2 scripts

Python 3.13 + `part1/requirements.txt` (see `part1/README.md`), then:

```bash
cd part1 && python -m src.run_part1 --dataset hdfs --seed 42 --stage all   # same for bgl
cd ../part2 && python -m src.run_resamples --seeds 0-9 --models rf,lstm,pca,loggpt
Rscript src/scott_knott.R results
```

## Data

HDFS_v1 and BGL from Loghub (Zenodo record 8196385, ~250 MB zipped): HDFS.log 11,175,629
lines / 575,061 labelled blocks (2.9% anomalous); BGL.log 4,747,963 lines / 348,460 alert
lines (7.3%). Logs are not redistributed here; every script and notebook downloads them.

## Credits and licenses

- Replication target: Wu, Li & Khomh, EMSE 2023 (arXiv:2308.08736) and their replication package (mooselab).
- Preprocessing: Qin et al., SANER 2025 — "Preprocessing is All You Need" (mooselab artifact), vendored in `part1/src/preprocess.py`.
- PCA detector: port of [logpai/loglizer](https://github.com/logpai/loglizer) (MIT) in `part2/src/models/pca.py`.
- LogGPT: port of [nokia/LogGPT](https://github.com/nokia/LogGPT) (CC-BY-NC-4.0, Han, Yuan & Trabelsi, IEEE BigData 2023) in `part2/src/models/loggpt/`.
- Datasets: Loghub (He et al., zhu2023loghub / jiang2024loghub2).

## Team

Pham Minh Dung Duong (2553085) · Thanh Nghiep Quan (2557078) · Tran Vo (2553737)
LOG6309E Intelligent DevOps, Fall 2025 — Polytechnique Montréal.
