"""Matched training-size control for RQ3 (not ranked, DECISIONS P2-25).

  python -m src.matched_size [--config configs/part2_hdfs.yaml] [--seeds 0-9]

In RQ3, LogGPT trains on 5,000 normal sessions and the other models on the whole train split. Here RF, PCA and
the LSTM train on n sessions drawn at random from the train split of each resample (n = 500, 5,000, 50,000; class
mix as drawn, so the supervised models still see some anomaly labels). PCA also trains on LogGPT's own 5,000 normal
sessions for all seeds. Every run is scored on the same full test split as RQ3. RF and PCA are also run on the
whole train split, as a regression check against results/runs.csv.

Writes results/followup/matched_size.csv. Nothing under results/scores/ or data/models/ is touched.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from . import p1  # noqa: F401  (registers Part 1 as `p1`)
from .models.pca import PCAMCV
from .run_resamples import Ctx, _merge, load_config, p2path, parse_seeds, unlabeled
from .scores import decide, session_metrics
from p1.common import data_dir as p1_data_dir
from p1.common import set_seed
from p1.models.lstm import LSTMBert
from p1.models.rf import RFMCV

SIZES = [500, 5000, 50000]


def n_val_anomalous(sub, seed, val_ratio):
    """Anomalous sessions in the LSTM's validation set: same draw as LSTMBert.fit."""
    perm = np.random.default_rng(seed).permutation(len(sub))
    return int(sub.labels[perm[:int(round(len(sub) * val_ratio))]].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2_hdfs.yaml")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds = parse_seeds(a.seeds) if a.seeds else cfg["seeds"]
    out = p2path(cfg, "results_dir", "followup") / "matched_size.csv"
    logging.basicConfig(level=logging.WARNING)
    lg = logging.getLogger("matched_size")
    rows = []

    def record(ctx, model, n, sub, scores, pred, t, **info):
        m = session_metrics(ctx.test_s.labels.astype(np.int8), scores, pred)
        rows.append(dict(seed=ctx.seed, model=model, n_train=n, n_train_anomalous=int((sub.labels == 1).sum()),
                         precision=m["precision"], recall=m["recall"], f1=m["f1"], auc=m["auc"], tp=m["tp"],
                         fp=m["fp"], fn=m["fn"], fit_score_s=t, **info))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"seed {ctx.seed} {model:13s} n={n:6d} F1 {m['f1']:.4f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"AUC {m['auc']:.4f} ({t:.1f}s)", flush=True)

    def run_rf(ctx, n, sub):
        set_seed(ctx.seed)
        t0 = time.time()
        rf = RFMCV(ctx.templates, ctx.cfg1["rf"], ctx.seed)
        rf.fit(sub)
        sc = rf.predict_scores(ctx.test_s)
        record(ctx, "rf", n, sub, sc, decide("rf", sc, 0.5), time.time() - t0)

    def run_pca(ctx, name, n, sub):
        set_seed(ctx.seed)
        t0 = time.time()
        pca = PCAMCV(len(ctx.templates), cfg["pca"], ctx.seed, lg)
        pca.fit(unlabeled(sub))
        sc = pca.predict_scores(ctx.test_s)
        info = pca.info()
        record(ctx, name, n, sub, sc, decide("pca", sc, pca.threshold), time.time() - t0,
               n_features=info["n_features"], n_components=info["n_components"], threshold=info["threshold"])

    def run_lstm(ctx, n, sub, E):
        set_seed(ctx.seed)
        t0 = time.time()
        params = _merge(ctx.cfg1["lstm"], ctx.cfg.get("lstm_overrides", {}))
        lstm = LSTMBert(E, ctx.cfg1["window"], params, ctx.seed, lg)
        lstm.fit(sub)
        sc = lstm.predict_scores(ctx.test_s)
        record(ctx, "lstm", n, sub, sc, decide("lstm", sc, 0.5), time.time() - t0, best_epoch=lstm.best_epoch,
               n_val_anomalous=n_val_anomalous(sub, ctx.seed, params["val_ratio"]),
               best_val_f1=max(h["val_f1"] for h in lstm.history))

    for seed in seeds:
        ctx = Ctx(cfg, seed, lg)
        E = np.load(p1_data_dir(ctx.cfg1, "features") / "bert_embeddings.npy")
        rng = np.random.default_rng(10_000 + seed)
        for n in SIZES:
            sub = ctx.s.subset(np.sort(rng.choice(ctx.tr, size=n, replace=False)))
            run_rf(ctx, n, sub)
            run_pca(ctx, "pca", n, sub)
            run_lstm(ctx, n, sub, E)
        run_pca(ctx, "pca_normal5k", 5000, ctx.s.subset(ctx.normal_sample()))
        run_rf(ctx, len(ctx.train_s), ctx.train_s)
        run_pca(ctx, "pca", len(ctx.train_s), ctx.train_s)


if __name__ == "__main__":
    main()
