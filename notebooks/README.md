# Notebooks — self-contained re-runs on Google Colab

These notebooks re-implement the pipeline of this repository inside notebook cells (no clone,
no venv): every algorithm is ported cell by cell from `part1/src` and `part2/src` with the
same parameters, seeds and score rules, adapted only for Colab paths and Drive caching. They
exist so that anyone — including the graders — can reproduce the results end to end without
installing anything, and so each stage can be inspected where it runs.

| Notebook | What it does | Source ported |
|---|---|---|
| `01_grab_data.ipynb` | Downloads HDFS_v1 + BGL from Loghub (Zenodo), arranges them in the repo layout, validates the counts against the paper (11,175,629 / 575,061 / 2.9% and 4,747,963 / 7.3%), stores ~2.2 GB on Drive. | `part1/README.md` data steps |
| `02_replicate_rq1.ipynb` | RQ1 replication of the *earlier* main-branch pipeline (drain3 + regexes; RF+MCV, LSTM+TF-IDF Text; with/without Spearman-clustering+VIF selection) on HDFS and BGL. Kept for comparison with the final pipeline. | former `src/` of the main branch |
| `03_extension_10seeds.ipynb` | **The final pipeline's extension, re-run independently**: downloads HDFS itself, then Qin preprocessing + streaming Drain → sessions → seeded splits → MCV + BERT embeddings → RF, LSTM, PCA, LogGPT × seeds 0–9 → Scott-Knott ESD (R). Everything persists under `MyDrive/devops6309`. | `part1/src` + `part2/src` (final pipeline) |

## Running notebook 03

1. Open it in Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU).
2. Run the cells in order. First run: download ~2 min, parsing ~10–25 min, then
   `MODELS = ['rf','lstm','pca']` ≈ 25–40 min for all 10 seeds.
3. LogGPT is the expensive part (~1–2 h per seed on a T4). Set `MODELS = ['loggpt']` and run
   it across as many Colab sessions as needed: every RL episode is checkpointed to Drive and
   training resumes automatically. The Scott-Knott cell ranks whatever models have finished
   at least 2 seeds and notes the ones it skipped.
4. All artefacts land on Google Drive (≈3 GB total):

```
MyDrive/devops6309/
├── data/        HDFS.log + anomaly_label.csv (downloaded once)
├── cache/       parsed lines/templates, sessions, splits, BERT embeddings, LogGPT checkpoints
├── results/     scores/{model}_seed{k}.parquet + .json, runs.csv, sk_f1.csv, sk_auc.csv
└── hf_cache/    bert-base-uncased + GPT-2 (downloaded once)
```

Re-running the notebook skips every finished (model, seed) pair, so it is safe to interrupt
and resume. Running it locally is also supported (everything lands in `./devops6309_cache`).

## Verification (Colab T4, 2026-10-21)

- Structure matches exactly: 47 templates; train 402,542 / test 172,519; 5,007 anomalous test sessions.
- Deterministic models match the published results to four decimals: RF F1 median .9982
  (10 seeds), PCA .7688; same Scott-Knott ordering (RF > LSTM > PCA, own group each, F1 and AUC).
- LSTM F1 median .9865 vs published .9785 — within the model's GPU-seed variance (published
  range .9637–.9895).
- LogGPT seed 0: w/o RL F1 .8334 vs published .8321; with RL the effect matches exactly
  (precision ↓, recall ↑, AUC .947 → .998); K = 7 from 15 unique keys, as documented.

Deviations from the scripts (also listed in the notebook's last cell): single-process parsing
(their 16-worker pool; Colab has 2 vCPUs), Colab's Python 3.12 stack instead of the pinned
3.13 venv, and a few side analyses that only exist in `part2/src` (package-faithful ablations,
`pca_normal5k`, the full LogGPT verification harness, SHAP).
