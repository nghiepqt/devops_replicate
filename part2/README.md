# Part 2 — Extension: PCA and LogGPT, Scott-Knott ESD ranking, SHAP explanation (HDFS)

Builds on Part 1 (`../part1/`): same parsing (Drain + Qin preprocessing, 47 templates), sessions (575,061 blocks)
and split function. Part 1's code is imported unchanged (as the package `p1`, see `src/p1.py`); no Part 1 file is
modified. Every choice is logged in `../part1/DECISIONS.md` (entries `P2-01` …).

| model | category | regime | training data per resample | score | decision |
|---|---|---|---|---|---|
| RF + MCV (no step 3a) | classical | supervised | train split, labels | `predict_proba[:, 1]` | ≥ 0.5 |
| LSTM + BERT | deep learning | supervised | train split, labels | max window P(anomaly) | ≥ 0.5 |
| PCA (Xu et al. 2009; loglizer) | classical | unsupervised | train split, **no labels** | SPE | > Q-statistic |
| LogGPT (Han et al. 2023) | LLM-based | semi-supervised | 5,000 **normal** train sessions | max rank of the actual next key | > K |

## Layout

```
part2/
  configs/part2_hdfs.yaml     main configuration (seeds 0-9, PCA and LogGPT hyper-parameters)
  configs/smoke_hdfs.yaml     smoke test (2% of each split, short training)
  src/
    p1.py                     imports Part 1's package as `p1`
    scores.py                 score definitions (F5) and session-level metrics (F6)
    models/pca.py             PCA + FeatureExtractor ported from loglizer (MIT; LICENSE_loglizer)
    models/loggpt/            LogGPT ported from github.com/nokia/LogGPT (CC-BY-NC-4.0; LICENSE, NOTICE.md)
    run_resamples.py          trains and scores all models per seed -> results/scores/, results/runs.csv
    scott_knott.R             Scott-Knott ESD (np) on F1 and AUC -> results/sk_*.csv, figures/sk_*.png
    explain.py                SHAP explanation of RF + MCV -> results/explain/
    sk_features.R             (optional) Scott-Knott ESD on per-resample cluster importance
    report.py                 builds results/RESULTS_PART2.md, descriptive_stats.csv, cost.csv, figures
    matched_size.py           control for training size, not ranked -> results/followup/matched_size.csv
    followup_stats.py         PCA stability, LogGPT spread and tokens -> results/followup/followup_stats.json
    loggpt_init_spread.py     LogGPT w/o RL, one sample, 5 init seeds -> results/followup/loggpt_init_spread.csv
  results/
    runs.csv                  one row per (seed, model): P, R, F1, AUC, TP/FP/TN/FN, times, device
    scores/{model}_seed{s}.parquet   one row per test session: session_idx, session_id, label, score, prediction
    scores/{model}_seed{s}.json      run metadata (threshold, times, LogGPT K / histories / verification, PCA components)
    sk_f1.csv, sk_auc.csv, sk_*_effect_sizes.csv, sk_input_*.csv
    descriptive_stats.csv, cost.csv, figures/, explain/, RESULTS_PART2.md
  data/                       caches: trained models, LogGPT checkpoints and training-session lists, SHAP values
  logs/                       one log per seed, explain.log
```

## Reproduce

All commands run from `part2/` with the Part 1 environment (`~/.venvs/GroupAssignment`, Python 3.13; see
`../part1/README.md`). Part 1 must have produced `part1/data/parsed/hdfs/`, `part1/data/sessions/hdfs/` and
`part1/data/features/hdfs/bert_embeddings.npy` (`python -m src.run_part1 --dataset hdfs --stage features` in `part1/`).

```bash
source ~/.venvs/GroupAssignment/bin/activate
uv pip install shap==0.54.1                       # only addition to Part 1's requirements
# R >= 4.3 with ScottKnottESD >= 3.0.0 (CRAN still serves 2.0.3):
Rscript -e "install.packages('ScottKnottESD', repos = c('https://klainfo.r-universe.dev', 'https://cloud.r-project.org'))"
```

1. Smoke test (about 2 minutes; outputs in `results/smoke/`, metrics meaningless):
   ```bash
   python -m src.run_resamples --config configs/smoke_hdfs.yaml
   ```
2. Resamples (each (model, seed) is cached; rerunning skips finished ones, `--force` redoes them; LogGPT resumes
   from its last checkpoint):
   ```bash
   python -m src.run_resamples --seeds 0-9 --models rf,lstm,pca,loggpt,pca_normal5k
   ```
   `pca_normal5k` (PCA trained on LogGPT's 5,000 normal sessions) only runs for `pca_sensitivity_seeds` (seed 0).
   `loggpt` also writes the unranked ablation `loggpt_norl` (the pre-trained model before RL fine-tuning).
3. Scott-Knott ESD:
   ```bash
   Rscript src/scott_knott.R results
   ```
4. Explanation (RF + MCV of seed 0, plus stability over all seeds):
   ```bash
   python -m src.explain --seed 0 --n-normal 5000
   ```
5. Report:
   ```bash
   python -m src.report                            # -> results/RESULTS_PART2.md
   ```

## Runtime (RTX 4060 Laptop GPU, i5-13500HX; measured, `results/cost.csv`)

Per resample, each run alone on the machine: RF ≈ 0.4 s, PCA ≈ 0.3 s, LSTM ≈ 79 s, LogGPT 2,650–3,890 s (median
≈ 3,400 s: ≈ 230 s of pre-training + 7–11 RL episodes of ≈ 330 s; RL is CPU-bound). The 10 LogGPT resamples
take ≈ 9.5 h; do not run them in parallel on one GPU (≈ 2.4× slower each, DECISIONS P2-17). `explain.py` takes
≈ 4 min, the report a few seconds.
