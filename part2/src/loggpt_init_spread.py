"""LogGPT w/o RL: spread from the random state alone, on one fixed training sample (not ranked, DECISIONS P2-27).

  python -m src.loggpt_init_spread [--config configs/part2_hdfs.yaml] [--sample-seed 0] [--init-seeds 0-4]

In the 10 resamples, a seed changes the 5,000 training sessions and the random state together. Here the training
sample and the test split stay those of `--sample-seed`, and only the seed of the initialization and dropout changes.
Pre-training only: no RL, so no sampled sequences and no temperature. Init seed = sample seed repeats the original run.
Writes results/followup/loggpt_init_spread.csv. Checkpoints under data/loggpt/ are neither read nor written.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

from . import p1  # noqa: F401  (registers Part 1 as `p1`)
from .models.loggpt import LogGPT
from .run_resamples import Ctx, load_config, p2path, parse_seeds
from .scores import decide, session_metrics
from p1.common import set_seed


class PretrainOnly(LogGPT):
    def _rl_finetune(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2_hdfs.yaml")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--init-seeds", default="0-4")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = p2path(cfg, "results_dir", "followup") / "loggpt_init_spread.csv"
    logging.basicConfig(level=logging.WARNING)
    ctx = Ctx(cfg, a.sample_seed, logging.getLogger("init_spread"))
    normal = ctx.s.subset(ctx.normal_sample())
    y = ctx.test_s.labels.astype(np.int8)
    old = pd.read_csv(out).drop_duplicates(["sample_seed", "init_seed", "K"]) if out.exists() else pd.DataFrame()
    rows = old.to_dict("records")                       # finished runs are kept, so an interrupted job can resume
    done = set(old.init_seed[old.sample_seed == a.sample_seed]) if len(old) else set()
    for init in parse_seeds(a.init_seeds):
        if init in done:
            continue
        set_seed(init)
        t0 = time.time()
        model = PretrainOnly(cfg["loggpt"], init, "cuda" if torch.cuda.is_available() else "cpu", None, None)
        model.fit(normal)
        t = time.time() - t0
        scores = model.max_rank_scores(ctx.test_s, "pre", cfg["loggpt"]["score_batch_size"])
        for K in sorted({model.K, 6, 7, 8, 9}):
            m = session_metrics(y, scores, decide("loggpt_norl", scores, K))
            rows.append(dict(sample_seed=a.sample_seed, init_seed=init, K=K, K_rule=model.K, is_rule=K == model.K,
                             precision=m["precision"], recall=m["recall"], f1=m["f1"], auc=m["auc"], train_s=t))
            if K == model.K:
                print(f"init {init}: K {K} P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} "
                      f"AUC {m['auc']:.4f} ({t:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
