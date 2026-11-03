Every number below comes from §3–§10 above (`metrics.csv`, `seed_summary.csv`, `results/lstm_*.json`).
Rule from the brief: every |ΔF1| > 0.05 needs hypotheses. Only one required run crosses it: **BGL LSTM + BERT
(ΔF1 = −0.061)**. The other deltas are discussed briefly because they help the report.

### A. HDFS — RF + MCV: replicated (ΔF1 = −0.000)
P 0.998 / R 0.999 / F1 0.999 against the paper's 0.998 / 1.000 / 0.999. Across 6 seeds F1 = 0.998 ± 0.0003
(§7). HDFS is close to linearly separable with event counts, so no setting matters much here. Nothing to explain.

### B. HDFS — LSTM + BERT: higher than the paper (ΔF1 = +0.028; ΔR = +0.077)
Our recall is 1.000 while the paper's is 0.923. Seed mean F1 = 0.976 ± 0.008, still above the paper's 0.959.
The ablation in §10 points to the cause:
- **The package's window construction (D-17, D-18).** If we only switch to the package's `collate_fn`, the
  test recall drops to **0.938** and F1 to 0.876. That `collate_fn` keeps only the last w − L − 1 events of
  sessions with 15 ≤ L ≤ 28 (the median HDFS session has 19 events), right-pads, and reads the LSTM output at
  the last, padded, time step. The resulting recall is close to the paper's 0.923. Our pipeline feeds whole
  sessions and packs the sequences, which explains why we detect every anomalous block.
- Epoch selection is not the cause on HDFS. The best test F1 over all 30 epochs is 0.988, against 0.987 at the
  epoch selected on validation (§10).
- The paper's F1 (0.959) is above our package-windowing F1 (0.876). So the package's other differences, one
  optimiser step per session (infeasible on HDFS: about 362k steps per epoch) and window-level evaluation, may
  also have played a role. We cannot separate them on HDFS.

### C. BGL — RF + MCV: slightly above the paper (ΔF1 = +0.028), with P and R apparently swapped
We get P 0.963 / R 0.878; the paper reports P 0.830 / R 0.963. The pattern matches the package bug in D-13:
`metrics(y_test, y_)` is called with its arguments reversed, so the paper's printed "precision" is the recall
and vice versa. Read swapped, the paper's numbers are P 0.963 / R 0.830, close to ours. F1 is symmetric in P
and R, so the bug does not affect it. Across seeds F1 = 0.922 ± 0.019, about 1.6 standard deviations above the
paper's 0.891, and all six seeds are above it (lowest 0.899). The remaining differences are 826 vs 718 sessions (D-08), 10 trees with a single seed vs
an average of 5 unseeded runs (D-12), and a different parser and preprocessing (D-02).

### D. BGL — LSTM + BERT: lower than the paper (ΔF1 = −0.061; seed mean −0.103) — hypotheses required
Ours: P 0.769 / R 1.000 / F1 0.870 at seed 42, and F1 0.828 ± 0.033 over 6 seeds (min 0.785, max 0.870). The
paper: P 0.871 / R 1.000 / F1 0.931. Recall is the same; the gap is entirely precision. At seed 42 there are
27 false positives among 76 normal test sessions.

What we checked (and ruled out):
1. **Training or evaluation bugs on our side.** We ran a package-faithful ablation (§10): the package's
   windowing, padding and one optimiser step per session (gradient accumulation over chunks, mathematically
   the same step). It gives F1 0.853, and 0.846 at the window level even when the best epoch is picked on
   the test set. Our deviations therefore do not explain the gap; the package's procedure on our data
   performs the same.
2. **Model selection on validation vs test (D-19).** The best test F1 over 30 epochs is 0.870, which is
   exactly our validation-selected epoch (§10).
3. **Non-determinism.** Rerunning reproduces the identical training trajectory (D-22).

Remaining hypotheses, most likely first:
- **H1 — Different sessions (D-08).** 826 windows instead of 718 means different window boundaries and
  therefore different session contents and labels. Session-level labels on BGL are coarse: a 6-hour window
  is anomalous if *any* of its thousands of lines is (49.5% of windows here). Window boundaries change which
  normal-looking windows inherit an anomaly label. The paper's train/test split cannot be reconstructed.
- **H2 — Small, noisy test set and high variance.** With 166 test sessions, one false positive costs about
  0.004 F1 (seed 42: going from 27 to 26 false positives moves F1 from 0.870 to 0.874). Across seeds our F1 ranges over 0.085 (0.785–0.870, §7). The paper's single number, picked
  as the best epoch on the test set, is likely at the favourable end of a similar distribution.
- **H3 — Label noise × a tiny model.** Every window inherits its session's label (paper §4.4), so most windows
  of an anomalous session are labelled anomalous although they look normal. With hidden size 8, the training
  loss stays around 0.36–0.60 and never converges (`figures/lstm_training_bgl.png`), and validation precision
  oscillates between 0.40 and 0.76. The model learns "unusual templates → anomalous" with a low threshold,
  hence R = 1.000 and many false positives. Small changes in data (H1) or initialisation move the precision a
  lot.
- **H4 — Parsing and BERT input differences (D-02, D-15).** Our BGL vocabulary (421 templates; Qin
  preprocessing) differs from the paper's (hand-made regexes, count not reported). 57 templates exceed
  50 tokens, so 26 of them collapse onto the same embedding as another template (rank 405 < 421). The paper
  served TensorFlow BERT through bert-as-service; we use the HuggingFace PyTorch weights of the same checkpoint
  with the same pooling. This causes small numeric differences, plus possibly a different default
  tokenisation of `<*>`.
- **H5 — Library versions** (PyTorch 1.10 → 2.14, sklearn 0.24 → 1.9). Unlikely to be important.

The required seed-42 gap (−0.061) is below the stop threshold (0.10). The 6-seed mean gap (−0.103) is just above
it. We report it as is; nothing we can change while staying faithful to the paper's definitions closes it.

### E. Effect of step 3a (correlation + VIF) on RF + MCV
| | HDFS | BGL |
|---|---|---|
| features (seed 42) | 47 → 28 | 410 → 119 |
| F1 without → with (seed 42) | 0.999 → 0.997 | 0.919 → 0.937 |
| F1 without → with (6 seeds) | 0.998 ± 0.0003 → 0.996 ± 0.0006 | 0.922 ± 0.019 → 0.929 ± 0.041 |
| train time (seed 42) | 0.5 s → 117.7 s | 0.1 s → 73.0 s |

- **HDFS:** removing 40% of the features costs a small but systematic amount of precision (FP 9 → 21 at seed
  42; lower F1 in all 6 seeds). Many removed templates belong to the normal write pipeline (Receiving /
  Received / PacketResponder / addStoredBlock are one correlation cluster). Their counts are almost
  interchangeable, but not exactly. Random Forests are not hurt by collinearity in prediction: a split uses one
  feature at a time, and correlated duplicates only spread importance across features. So removing them can
  only lose information.
- **BGL:** the average effect is within noise (+0.007 F1) and the standard deviation doubles (one seed drops to 0.848).
  With p ≈ 410 features and n = 660 sessions, `max_features = sqrt` samples about 20 features per split, most of
  them rare. Removing redundant templates makes informative ones more likely to be sampled (seed 42: +0.018).
  However, keeping the most frequent template of a cluster sometimes discards the rarer, anomaly-specific
  member (seed 2).
- **Take-away for the report:** step 3a is valuable for *interpretation*. It yields a non-redundant set of
  templates whose importances (or SHAP values in Part 2) can be read one by one, and a list of which templates
  co-occur (`feature_selection_*.csv`, e.g. ρ = −1.000 between the two `allocateBlock` templates). For RF
  *accuracy* it brings no reliable gain, and it costs the VIF loop's run time.

### F. Split sensitivity (HDFS)
With the split the replication package actually used (loglizer "uniform": per class, the earliest 70% of blocks
are train), both models score higher than with the random split. RF reaches F1 1.000 and LSTM 0.995 (§8). On
HDFS the later blocks are not harder to classify, so the choice of split does not explain any gap with the
paper.
