"""Statistics for the instructor's follow-up questions, computed from stored results (DECISIONS P2-26).

  python -m src.followup_stats [--config configs/part2_hdfs.yaml]

(a) PCA stability: are the same anomalies missed in every resample, how far they are from the threshold, and the best
F1 any threshold could give (chosen on the test set, diagnostic only). (b) LogGPT F1 spread with the 50% rule and at
each fixed K. (c) LogGPT model size and number of tokens it processes. (d) PCA: which templates the missed anomalies
contain. (e) LogGPT: sessions with a template unseen in training, and how many of them are flagged.
Writes results/followup/followup_stats.json.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from . import p1  # noqa: F401  (registers Part 1 as `p1`)
from .run_resamples import load_config, p2path
from p1.common import load_config as load_p1_config
from p1.partition import load_sessions
from p1.split import get_split

REPLICATION_TEMPLATE = 9    # BLOCK* ask <*> to replicate <*> to datanode(s) <*>
DELETE_ERROR_TEMPLATE = 33  # Unexpected error trying to delete block <*>. BlockInfo not found in volumeMap.


def contains_template(s, template_id):
    """[n_sessions] True if the session has at least one event of this template."""
    hit = np.zeros(len(s), dtype=bool)
    hit[np.searchsorted(s.offsets, np.flatnonzero(s.events == template_id), side="right") - 1] = True
    return hit


def pca_stats(cfg):
    res = p2path(cfg, "results_dir")
    runs = pd.read_csv(res / "runs.csv")
    thr = runs[runs.model == "pca"].set_index("seed").threshold
    ds, per_seed = {}, []
    for s in cfg["seeds"]:
        d = pd.read_parquet(res / "scores" / f"pca_seed{s}.parquet")
        y, sc, pred = d.label.values, d.score.values, d.prediction.values
        p, r, _ = precision_recall_curve(y, sc)
        f = 2 * p * r / np.maximum(p + r, 1e-12)
        missed = sc[(y == 1) & (pred == 0)]
        per_seed.append(dict(seed=s, threshold=float(thr[s]), best_f1=float(f.max()), n_missed=len(missed),
                             max_missed_spe=float(missed.max()), median_missed_spe=float(np.median(missed)),
                             normal_spe_p99=float(np.quantile(sc[y == 0], .99))))
        ds[s] = d[d.label == 1].set_index("session_id").prediction
    pairs = []
    for a, b in itertools.combinations(cfg["seeds"], 2):
        shared = ds[a].index.intersection(ds[b].index)
        ma, mb = ds[a][shared] == 0, ds[b][shared] == 0
        pairs.append(dict(n_shared=len(shared), n_missed_a=int(ma.sum()), n_missed_b=int(mb.sum()),
                          n_differ=int((ma != mb).sum())))
    return dict(per_seed=per_seed, pairs=pairs)


def pca_template_stats(cfg, s):
    res = p2path(cfg, "results_dir")
    replication, delete_error = contains_template(s, REPLICATION_TEMPLATE), contains_template(s, DELETE_ERROR_TEMPLATE)
    per_seed = []
    for seed in cfg["seeds"]:
        d = pd.read_parquet(res / "scores" / f"pca_seed{seed}.parquet")
        idx, anomalous, flagged = d.session_idx.values, d.label.values == 1, d.prediction.values == 1
        per_seed.append(dict(seed=seed,
                             missed_with_replication=float(replication[idx][anomalous & ~flagged].mean()),
                             anomalous_with_delete_error=float(delete_error[idx][anomalous].mean()),
                             delete_error_detected=float(flagged[anomalous & delete_error[idx]].mean())))
    return dict(per_seed=per_seed, mean=pd.DataFrame(per_seed).drop(columns="seed").mean().to_dict())


def loggpt_unseen_stats(cfg, s):
    """Unseen = the session has a template that is not in the vocabulary built from that seed's 5,000 training sessions."""
    res = p2path(cfg, "results_dir")
    per_seed = []
    for seed in cfg["seeds"]:
        known = np.zeros(int(s.events.max()) + 1, dtype=bool)
        itos = json.load(open(res / "scores" / f"loggpt_seed{seed}.json"))["info"]["itos"]
        known[[int(t) for t in itos if str(t).isdigit()]] = True
        unseen = np.add.reduceat((~known[s.events]).astype(np.int64), s.offsets[:-1]) > 0
        d = pd.read_parquet(res / "scores" / f"loggpt_seed{seed}.parquet")
        hit, anomalous, flagged = unseen[d.session_idx.values], d.label.values == 1, d.prediction.values == 1
        per_seed.append(dict(seed=seed, n_anomalous=int(anomalous.sum()), n_anomalous_unseen=int((hit & anomalous).sum()),
                             n_normal_unseen=int((hit & ~anomalous).sum()), unseen_flagged=float(flagged[hit].mean())))
    return dict(per_seed=per_seed, mean=pd.DataFrame(per_seed).drop(columns="seed").mean().to_dict())


def loggpt_spread(cfg):
    k = pd.read_csv(p2path(cfg, "results_dir") / "loggpt_k_sensitivity.csv")
    out = {}
    for m, d in k.groupby("model"):
        rule = d[d.K == d.K_rule]
        out[m] = dict(rule=dict(f1_mean=rule.f1.mean(), f1_std=rule.f1.std(), recall_std=rule.recall.std()),
                      auc_std=rule.auc.std(), auc_min=rule.auc.min(), auc_max=rule.auc.max(),
                      fixed_k={int(K): dict(f1_mean=g.f1.mean(), f1_std=g.f1.std()) for K, g in d.groupby("K")})
    return out


def loggpt_tokens(cfg):
    """A token = one template id, <bos> or <eos>. Counted from the sessions and the run metadata, not from a profiler."""
    cfg1 = load_p1_config(cfg["part1_config"])
    s = load_sessions(cfg1)
    lens = np.diff(s.offsets)
    p = cfg["loggpt"]
    per_seed = []
    for seed in cfg["seeds"]:
        root = p2path(cfg, "data_root", "loggpt", f"seed{seed}")
        pick = np.array(json.load(open(root / "train_sessions.json"))["session_idx"])
        L = lens[pick] + 2                                   # <bos> + ids + <eos>
        n_fit = int(0.9 * len(L))
        fit, val = L[:n_fit], L[n_fit:]
        ck = torch.load(root / "pretrain.pt", map_location="cpu", weights_only=False)
        params = int(sum(v.numel() for k, v in ck["last_state"].items() if k != "lm_head.weight"))
        st = torch.load(root / "rl_state.pt", map_location="cpu", weights_only=False)
        eps = sum(1 for h in st["history"] if h["episode"] >= 0)

        def prompt(n):                                       # LogGPT._prompt
            c = np.floor(0.2 * n).astype(int)
            return np.where(c < 2, 2, np.where(c >= n - 2, n - 2, c))
        gen = int((fit - prompt(fit)).sum())                 # tokens generated per episode on the training sessions
        gen_val = int((val - prompt(val)).sum())
        beams = p["num_return_sequences"]
        tr, te = get_split(cfg1, seed, None)
        test_lens = lens[te] + 2
        seqs = {}                                            # distinct test sequences: each is scored once
        for i in te:
            seqs.setdefault(s.events[s.offsets[i]:s.offsets[i + 1]].tobytes(), int(lens[i] + 2))
        per_seed.append(dict(
            seed=seed, vocab=len(ck["itos"]), n_params=params, rl_episodes=eps,
            pretrain_tokens=int(fit.sum() * p["init_num_epochs"]),
            pretrain_val_tokens=int(val.sum() * p["init_num_epochs"]),
            rl_generated_tokens=(gen + gen_val) * eps + gen_val,
            rl_beam_token_steps=((gen + gen_val) * eps + gen_val) * beams,
            test_tokens_nominal=int(test_lens.sum()), n_distinct_test_seqs=len(seqs),
            test_tokens_forward=int(sum(seqs.values()))))
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2_hdfs.yaml")
    cfg = load_config(ap.parse_args().config)
    s = load_sessions(load_p1_config(cfg["part1_config"]))
    out = dict(pca=pca_stats(cfg), loggpt_spread=loggpt_spread(cfg), loggpt_tokens=loggpt_tokens(cfg),
               pca_templates=pca_template_stats(cfg, s), loggpt_unseen=loggpt_unseen_stats(cfg, s))
    path = p2path(cfg, "results_dir", "followup") / "followup_stats.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
