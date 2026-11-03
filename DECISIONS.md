# DECISIONS — Part 1 (replication of RQ1, Wu, Li & Khomh, EMSE 2023)

Sources consulted before coding: the paper (§4 Experimental Design, §5.1 RQ1; arXiv 2308.08736), the
replication package (github.com/mooselab/suppmaterial-LogRepForAnomalyDetection: README, `logrep/`, `models/`,
`samples/`), the assignment PDF, and the brief `PART1_AGENT_BRIEF.md`. Precedence used throughout:
locked decisions (brief §2) > replication package > brief defaults. Numbers quoted below are copied from
files written by the scripts; the file is named next to each number.

---

## Data

### D-01 — Datasets and verification
- Stage: parsing
- Decision: loghub HDFS_v1 (`HDFS.log`, `preprocessed/anomaly_label.csv`) and BGL (`BGL.log`) from Zenodo record 8196385 (the link in the loghub README). 2k samples with corrected ground truth from Loghub-2.0 (`2k_dataset/{HDFS,BGL}/*_2k.log_structured_corrected.csv`).
- Rationale: locked decision D1. Verified after download: HDFS 11,175,629 lines; labels 575,061 blocks (16,838 Anomaly); BGL 4,747,963 lines, 348,460 with Label != "-" (the paper reports 4,747,963 and 348,460).
- Source: brief §4.1, paper §4.3.2
- Deviation from paper: no

## Pre-processing and parsing

### D-02 — Pre-processing: Qin et al. (SANER 2025) framework, as published
- Stage: parsing
- Decision: the artifact was found within minutes (the arXiv version of the paper, 2412.05254, links github.com/mooselab/Preprocessing-is-All-You-Need). Its `benchmark/logparser/utils/preprocessing.py` is vendored verbatim into `src/preprocess.py` (14 ordered regexes, all masks `<*>`, repeated `<*>` collapsed) together with the "estimation stage" of their `Drain.py`: the first 2000 lines are masked with the whole sequence and only the regex types that matched there are applied to the rest of the file. Their Drain ignores the dataset-specific `rex` list, so no paper/loghub regex is added.
- Rationale: locked decision D6 ("apply it following its instructions"). Regex types kept after estimation (`data/parsed/*/parse_meta.json`): HDFS `IPv4_port, package_host, path, block, numerical`; BGL `numerical` only (the first 2000 BGL lines are homogeneous). Sensitivity check `configs/bgl_qinfull.yaml` (all 14 regexes, no narrowing): also 421 templates and an identical one-to-one partition of the lines, so MCV is unaffected; only the text of a few templates differs (BERT input).
- Source: brief D6; Qin et al. artifact
- Deviation from paper: yes — the paper used Drain with hand-made extra regexes. Expected impact: small; the HDFS template count is 47 vs 46 in the paper, and parsing quality on the 2k samples is at least as good as the paper's regexes (D-04).

### D-03 — Drain implementation and parameters
- Stage: parsing
- Decision: logpai Drain tree logic (identical in logpai/logparser, Loghub-2.0 and Qin's artifact), with depth 4, st 0.5, maxChild 100, tokenisation on whitespace. Only the I/O was rewritten (`src/drain.py`, `src/parse.py`): the file is streamed, masking runs in a 16-process pool, Drain runs sequentially in the main process, and each line stores its cluster index. Final templates follow logpai's `outputResult`: a line gets its cluster's final template, and EventId = md5(template)[:8], so clusters that end with identical templates are merged.
- Rationale: the package says "loglizer with Drain parser", but does not name a version; logpai Drain is the reference implementation. The pandas-based original would hold 11M rows in RAM (machine has 15 GB, ~6 GB free). Runtime: HDFS 55 s, BGL 13 s (`parse_meta.json`).
- Source: brief §4.3; replication package README; logpai/logparser
- Deviation from paper: no (same algorithm and benchmark parameters)

### D-04 — Parsing-quality comparison on the 2k samples
- Stage: parsing
- Decision: `src/parsing_eval.py` → `results/parsing_eval.csv`, configurations none / loghub benchmark regexes / paper regexes / (BGL) paper refined regexes / Qin. GA = logpai grouping accuracy; PA = Loghub-2.0 exact-template accuracy without template correction; PA_relaxed (supplementary) collapses placeholders separated only by punctuation (`<*>:<*>` → `<*>`) before comparing.
- Rationale: GA is what matters for MCV (grouping only); PA matters for the BERT input (template text). Qin gives the best GA on both datasets (HDFS 1.000, BGL 0.970) and the best PA on BGL (0.918). On HDFS its PA is lower (0.443) for two convention reasons visible in the mismatches: the ground truth writes `IP:port` as `<*>:<*>` while Qin's default mask is `<*>`, and Qin's `numerical` regex `\b[a-fA-F\d]{4,}\b` masks words made only of hex letters, e.g. "added" (→ "is <*> to invalidSet"). For HDFS the paper's regexes are the benchmark defaults (the package lists none for HDFS), so the "paper" and "loghub" rows are identical.
- Source: brief §4.2.2
- Deviation from paper: n/a (extra analysis)

### D-05 — Identifiers extracted before masking; template ids
- Stage: parsing
- Decision: HDFS block ids with `re.findall(r'blk_-?\d+', content)` on the raw Content, de-duplicated per line (loglizer uses a set) and assigned to every block referenced. BGL keeps Label (field 1) and Timestamp (field 2). Integer template ids 1..n in order of first appearance; 0 = padding; `templates.csv` holds id, EventId, text, occurrences.
- Rationale: brief §4.2.3 and §4.3. In practice every HDFS line references exactly one block (11,175,629 events = 11,175,629 lines, `sessions_meta.json`).
- Source: brief; loglizer
- Deviation from paper: no

### D-06 — BGL lines with empty Content
- Stage: parsing
- Decision: 34,470 BGL lines have no Content (e.g. `... RAS KERNEL FATAL`). They do not match the logpai format regex, which logpai would silently drop; we keep them with an empty Content so their Label/Timestamp still count when windows are built (0 lines dropped, `parse_meta.json`). Drain gives each such line its own cluster (logpai behaviour for zero-length messages), and the md5 merge collapses them into one template with empty text (TemplateId 195). This is why BGL has 57,710 Drain clusters but 421 templates.
- Rationale: keeping every labelled line is the conservative choice for window labels.
- Source: own judgment
- Deviation from paper: unknown (the package does not show its loader); impact negligible (0.7% of lines)

### D-07 — Template count check
- Stage: parsing
- Decision: no fix was needed. HDFS: 47 templates (paper: MCV of 46 dimensions, brief range 30–60). BGL: 421 templates (paper does not report it).
- Source: brief §4.3
- Deviation from paper: yes, +1 template on HDFS; negligible impact

## Partition and split

### D-08 — BGL windows: 826 sessions instead of 718 (decided with the user)
- Stage: partition
- Decision: fixed, non-overlapping 6-hour windows on Timestamp, aligned to the first timestamp, empty windows dropped, events ordered by (Timestamp, LineId); session anomalous if any line has Label != "-". This gives **826 sessions** (409 anomalous), 33 empty windows dropped out of 859 spanned (`data/sessions/bgl/sessions_meta.json`). The paper reports 718 (a 15% difference — a stop condition). Alternatives tried (none gives 718): epoch-aligned windows 824; windows on the `Time` field 828; dropping windows with fewer than 50 / 100 lines 746 / 693. The paper's "6,565 log lines per session" implies 4,747,963 / 6,565 ≈ 723 sessions, i.e. the same lines in ~100 fewer windows; the package contains no BGL windowing code. **The user chose on 2026-10-19 to keep 826.**
- Rationale: it is the literal definition in the paper; any filter to reach 718 would be arbitrary.
- Source: paper §4.3.2; user decision
- Deviation from paper: yes — split sizes 660/166 instead of 575/143; the test anomaly ratio is similar (54.2% vs ≈56.6% implied by the paper's recall values, e.g. 0.963 = 78/81).

### D-09 — Split: random shuffle (primary); the package's own split as a sensitivity run
- Stage: split
- Decision: `numpy.random.default_rng(seed).permutation(n)`, first `int(n × ratio)` sessions = train, no stratification, chronological order kept inside sessions. HDFS 402,542 / 172,519 (exactly the paper's sizes); BGL 660 / 166. Split indices in `data/splits/{dataset}_seed{seed}.json`, with the anomaly ratios of train and test (HDFS 2.94% / 2.91%, BGL 48.33% / 54.22% for seed 42). Additionally, `--split-method loglizer_uniform` reproduces what the package's sample split actually does.
- Rationale: inspecting `samples/HDFS_100k.log_0.7_splitted.npz` shows that the package used loglizer `_split_data(split_type='uniform')`: per class, the first 70% of blocks by first appearance are train (every test block appears after every train block of the same class; 219 = int(313 × 0.7) anomalous train sessions). This contradicts the paper's "we shuffle the sessions". The paper text and the brief (and Part 2's resampling) require a seeded random split, so it is primary; the loglizer split is reported as a sensitivity run for HDFS.
- Source: paper §4.3.2; brief §5; replication package `samples/`
- Deviation from paper: no for the primary run

### D-10 — Threats to validity (to discuss in the report)
- Stage: split / parsing
- Decision: documented, not mitigated.
  1. Random shuffling of sessions can leak temporal information (future sessions in train); the paper acknowledges it. BGL windows are long (6 h) and labelled at window level, so neighbouring windows share bursts of failures.
  2. Parsing (Drain) and the BERT embeddings are computed on the full dataset before splitting: the template vocabulary contains test-only templates. No labels are used, the MCV vocabulary is fitted on train only (unseen test events are dropped, D-11), and the embeddings are computed from template text only.
  3. BGL: 826 vs 718 sessions (D-08) and a small test set (166 sessions: one session ≈ 0.6 F1 points).
  4. Single seed for the required runs; seeds 0–4 are reported as sensitivity.
- Source: brief §5
- Deviation from paper: n/a

## Representations and models

### D-11 — Message Count Vector
- Stage: representation
- Decision: per-session counts per template id; vocabulary = templates seen in training sessions; test events of unseen templates are dropped and counted (`data/features/*/mcv_random_seed42.json`: HDFS 0 events; BGL 1,889 of 646,783 test events in 6 sessions).
- Source: replication package `logrep/MCV_generator.py`
- Deviation from paper: no

### D-12 — Random Forest hyper-parameters
- Stage: model
- Decision: `RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True)`, no class weights, `random_state=seed`, `n_jobs=-1`; AUC from `predict_proba[:, 1]`. Fitted on a DataFrame whose column names are the template texts and saved with joblib (`data/models/<ds>/rf*_random_seed42.joblib`) for SHAP in Part 2.
- Rationale: the package specifies these values (`max_features='auto'` in sklearn 0.24 equals `'sqrt'` for classifiers), so they override the brief default of 100 trees. The package averages 5 unseeded runs; we use one seeded run and report seeds 0–4 separately.
- Source: replication package `models/traditional.py`
- Deviation from paper: no (same model); the averaging over 5 runs is replaced by a seed study

### D-13 — Bug in the package: precision and recall swapped for RF (and SVM)
- Stage: evaluation
- Decision: our metrics are computed with `precision_recall_fscore_support(y_true, y_pred)`. In the package, `metrics(y_pred, y_true)` is defined but called as `metrics(y_test, y_)` for SVM and RF, so the precision it prints is the recall and vice versa. F1 is unaffected.
- Rationale: explains the RF P/R pattern in the comparison. Paper BGL RF: P 0.830 / R 0.963; ours (seed 42): P 0.963 / R 0.878 — close to the paper's numbers swapped.
- Source: replication package `models/traditional.py`
- Deviation from paper: yes (we report correctly computed P and R)

### D-14 — Step 3a details (RF + MCV only)
- Stage: feature selection
- Decision: fitted on the training matrix only and applied to test as a column selection. (1) drop zero-variance features; (2) Spearman ρ, hierarchical clustering (scipy, average linkage, distance 1 − |ρ|), cut at 0.3 (|ρ| ≥ 0.7), keep per cluster the template with the most training occurrences (ties → smaller id); (3) statsmodels VIF with a constant, iteratively drop the highest VIF while max VIF > 5 (ties, e.g. several infinite VIFs → fewer occurrences, then larger id). Every removal is logged in `results/feature_selection_{dataset}.csv` with its reason; dendrograms in `results/figures/dendrogram_{dataset}.png`. Results (seed 42): HDFS 47 → 31 (correlation) → 28 (VIF); BGL 410 → 152 → 119.
- Rationale: brief §7. Example that helps explain the method: HDFS template 11 (`allocateBlock: <*>. <*>`) has ρ = −1.000 with template 2 (`allocateBlock: <*>`) because each block has exactly one allocateBlock event, of one form or the other.
- Source: brief §7
- Deviation from paper: n/a (the paper does not do step 3a)

### D-15 — BERT: checkpoint, pooling and input text (package overrides brief defaults)
- Stage: representation
- Decision: `bert-base-uncased` (the HuggingFace copy of Google's `uncased_L-12_H-768_A-12`, the model the package serves with bert-as-service), second-to-last encoder layer (`hidden_states[-2]`, 768-d), mean over all non-padding tokens *including* [CLS]/[SEP] (bert-as-service REDUCE_MEAN uses the input mask), `max_seq_len = 50` (the package's `bert-serving-start` command), input = the raw EventTemplate with its `<*>` placeholders (the package passes `event["EventTemplate"]`). Each template is embedded once → `data/features/<ds>/bert_embeddings.npy`, shape [n + 1, 768], row 0 = 0.
- Rationale: rule 3 (package beats brief defaults, which were "remove `<*>`, exclude special tokens"). Consequence on BGL: 57 templates are longer than 50 tokens and are truncated (`bert_meta.json`); 26 templates then share their vector with another template, so only 405 distinct vectors exist.
- Source: replication package `logrep/BERT_generator_server.py`; paper §4.2
- Deviation from paper: no

### D-16 — Memory layout for the LSTM
- Stage: representation
- Decision: windows stored as int32 template-id arrays; a frozen `nn.Embedding` built from `bert_embeddings.npy` (padding_idx 0) is looked up inside the model. HDFS: 403,220 training windows × 30 int32 instead of 403,220 × 30 × 768 floats (~37 GB).
- Source: brief §6.2
- Deviation from paper: no (same input values)

### D-17 — Sliding windows and the package's short-session bug
- Stage: representation
- Decision: size/stride from paper Table 1 (HDFS 30/1, BGL 50/50). A session with L ≤ w events is one window with the whole session; for L > w, windows start at 0, s, 2s, … ≤ L − w, plus one shorter window with the remaining events if any (as the package does). Each window gets its session's label.
- Rationale: for a session shorter than the window (L < w) the package's `collate_fn` keeps `fea[L − w + 1:]`. The start index is negative, so this slice keeps only the **last w − L − 1 events** whenever w − L − 1 < L. With w = 30 that is every session with 15 ≤ L ≤ 28: L = 19 keeps its last 10 events, L = 28 keeps 1. Sessions with L ≤ 14 or L = 29 are kept whole. The median HDFS session has 19 events (`sessions_meta.json`), so the package fed the LSTM roughly half of the events of most HDFS sessions. We do not reproduce this bug. For stride 1 the package also appends a redundant (L − w + 1 … L) window; we omit it because every event is already covered.
- Source: replication package `models/LSTM.py`; brief §6.2
- Deviation from paper: yes (bug not reproduced) — may explain part of the paper's lower HDFS LSTM recall (0.923)

### D-18 — Padding
- Stage: model
- Decision: right padding with id 0 + `pack_padded_sequence` with the true lengths; the classifier reads the last real hidden state `h_n`.
- Rationale: the package pads on the right and reads `out[:, -1, :]`, i.e. the hidden state after the padding.
- Source: brief §6.2
- Deviation from paper: yes (fix); impact only on short sessions/tail windows

### D-19 — LSTM architecture and training
- Stage: model
- Decision: package architecture — LSTM(768 → 8, 1 layer) → Linear(8 → 2), cross-entropy, Adam lr 1e-3, 30 epochs, ImbalancedDatasetSampler behaviour (each epoch draws as many training sessions as there are, with replacement, inverse class-frequency weights). Two changes: (a) optimisation in mini-batches of 512 windows of the drawn sessions instead of one optimiser step per session; (b) 10% of the training sessions are held out (seeded) and the epoch with the best **validation** session-level F1 is kept.
- Rationale: (a) per-session steps would mean ~362k steps per HDFS epoch, i.e. several hours for 30 epochs (stop condition > 3 h); with mini-batches HDFS trains in 76 s and BGL in 23 s on an RTX 4060. (b) the package keeps the epoch with the best F1 on the **test** set, which is optimistic. A diagnostic run (`--lstm-test-curve`, `results/lstm_test_curve_*.json`) logs test F1 per epoch without using it, to measure that optimism (BGL seed 42: best test F1 over 30 epochs = 0.870 = our selected epoch). Ablations `--lstm-package-windows` (package windowing + padding) and `--lstm-package-faithful` (also one step per session; BGL only) quantify the effect of these deviations (RESULTS.md §10). On HDFS, the package's windowing drops recall to 0.938, close to the paper's 0.923. On BGL, the package-faithful variant reaches F1 0.853, against 0.870 for ours.
- Source: replication package `models/LSTM.py`; brief §6.2
- Deviation from paper: yes; see RESULTS.md

### D-20 — Session-level merging and evaluation level
- Stage: evaluation
- Decision: session score = max window P(anomaly); anomalous if ≥ 0.5 (equivalent to the package's "any window predicted anomalous"). Primary metrics are session-level, as the paper states (§4.4). The package's `evaluation()` actually computes window-level metrics (labels broadcast to windows); we report them as supplementary (`data/predictions/<ds>/lstm_random_seed42_window_level.json`, RESULTS.md §9).
- Source: paper §4.4; replication package `models/LSTM.py`
- Deviation from paper: no for the primary metric

### D-21 — Why step 3a is not applied to LSTM + BERT
- Stage: feature selection
- Decision: not applied (locked D4); evidence in `results/bert_rank.json`. Each "feature" of the LSTM is one of 768 embedding dimensions, but these 768 columns are generated by only n templates: the n × 768 embedding matrix has rank 47 (HDFS) and 405 (BGL), and rank 46 / 404 once centred, which is what correlation and VIF see. With rank < 768, at least 721 (HDFS) / 363 (BGL) dimensions are exact linear combinations of the others, so every VIF is infinite or undefined and correlation clustering would merge arbitrary dimensions. Removing individual embedding dimensions also has no interpretation (they are not log events) and would change the pre-trained representation the paper evaluates. Collinearity also matters less for the LSTM: it does not estimate one coefficient per input dimension, it learns a projection.
- Source: brief §6.3
- Deviation from paper: n/a

## Environment

### D-22 — Environment and determinism
- Stage: evaluation
- Decision: Python 3.13.5 virtual env created with the user's `workon` + uv (`~/.venvs/GroupAssignment`); versions pinned in `requirements.txt` (torch 2.14.0+cu130, transformers 5.20.1, scikit-learn 1.9.1, statsmodels 0.15.0, numpy 2.5.3, pandas 3.0.6). Device auto-detected CUDA → MPS → CPU (here: NVIDIA RTX 4060 Laptop, CUDA 13). Seeds for `random`, `numpy`, `torch`, cuDNN deterministic. A rerun of the BGL LSTM reproduced the identical validation trajectory. HuggingFace cache in `part1/.hf_cache`.
- Rationale: Python 3.13 because the uv cache already held CUDA torch wheels for cp313 (no multi-GB download on the nearly full home partition).
- Source: brief §3
- Deviation from paper: library versions differ from the package (Python 3.9, PyTorch 1.10, sklearn 0.24); impact expected to be small

---

# DECISIONS — Part 2 (extension: PCA, LogGPT, Scott-Knott ESD, SHAP)

Sources consulted before coding: the brief `PART2_AGENT_BRIEF.md`; the LogGPT paper (arXiv 2309.14482,
`external/paper/loggpt2023.pdf`); the LogGPT code at github.com/nokia/LogGPT and github.com/hanxiao0607/LogGPT
(`external/LogGPT_nokia`, `external/LogGPT_hanxiao`); loglizer (github.com/logpai/loglizer, commit `ac67f97`,
`external/loglizer`); ScottKnottESD 3.0.0 (R sources of `sk_esd`, `scottknott`, `PartitionNonParametric`,
`checkDifference`); the Part 1 code and this file. Precedence: locked decisions and fairness protocol (brief §2–3)
> original code > paper > brief defaults. Part 2 code lives in `part2/`; numbers quoted here come from the files
named next to them.

## Organisation

### P2-01 — Output layout
- Stage: all
- Decision: the brief gives two locations for the outputs (§7 `results/part2/…`, §11 `part2/results/…`). The user
  chose §11 on 2026-10-19: everything is under `part2/` (`results/`, `data/` for caches, `logs/`). The decisions log
  stays in this file (the "existing DECISIONS.md", brief §0 rule 7).
- Source: brief §7, §11; user decision
- Deviation from paper: n/a

### P2-02 — Reuse of Part 1 code without modifying it
- Stage: all
- Decision: Part 2 imports Part 1's package under the alias `p1` (`part2/src/p1.py`; both parts are called `src`).
  Parsing, sessions, `get_split`, `MCVVectorizer`, `RFMCV`, `LSTMBert` and step 3a are used unchanged. No Part 1 file
  was edited; no Part 1 result file is written. The only Part 1 location Part 2 writes to is the split cache
  `part1/data/splits/hdfs_seed{5..9}.json`, produced by Part 1's own `get_split` (seeds 0–4 already existed and are
  reused byte for byte).
- Rationale: brief §0 rule 3 (one code version for all models of a resample).
- Source: brief §0, §6
- Deviation from paper: n/a

### P2-03 — Resampling and identical test sets (F1, F3, F8)
- Stage: split / evaluation
- Decision: seeds 0–9, Part 1's random 70/30 session-level split (402,542 / 172,519). In each seed every model is
  scored on the full test split; `run_resamples.check_same_test_sets` asserts that the `session_id` column of every
  `results/scores/*_seed{s}.parquet` equals the split's test block ids, in the same order. Every model's random seed is
  the resample seed (`set_seed(seed)` before each model, plus the model's own `seed` argument).
- Smoke test: `configs/smoke_hdfs.yaml` keeps 2% of the train and of the test split (`default_rng(seed)`), 2 LSTM
  epochs, 200 LogGPT sessions, 3 pre-training epochs and 2 RL episodes; outputs in `results/smoke/` (meaningless metrics).
- Source: brief §3 F1/F3/F8, §7
- Deviation from paper: n/a

## PCA

### P2-04 — PCA: port of loglizer, defaults verified in its source
- Stage: model
- Decision: `part2/src/models/pca.py` ports `loglizer/models/PCA.py` and `FeatureExtractor` (MIT, attribution and
  `LICENSE_loglizer` kept). Verified defaults: `n_components=0.95` (smallest k whose eigenvalues reach 95% of the
  trace), `threshold=None` → Q-statistic with `c_alpha=3.2905` (α = 0.001); `demo/PCA_demo.py` uses
  `term_weighting='tf-idf'`, `normalization='zero-mean'`. Input = Part 1's MCV with the vocabulary of the train
  split (identical to loglizer with `oov=False`), then idf = log(N / (df + 1e-8)) and the train mean, both fitted on
  train. Score = SPE = ‖(I − PPᵀ)x‖², computed for all rows at once (same value as loglizer's loop); decision
  `SPE > threshold` (strict, as loglizer).
- Rationale: brief P2 and §4. PCA receives a copy of the training sessions whose labels are all −1, so it provably
  never reads them (truly unsupervised, ≈ 2.9% contamination).
- Source: loglizer; brief §4
- Deviation from paper: no

### P2-05 — PCA sensitivity regime (not ranked)
- Stage: model
- Decision: `pca_normal5k`, seed 0 only: the same PCA fitted on the 5,000 normal sessions LogGPT uses (P2-08).
- Rationale: brief §4; the LogGPT paper's PCA (F1 = 0.087 on HDFS) follows a normal-only protocol.
- Source: brief §4
- Deviation from paper: n/a

## LogGPT

### P2-06 — Code base and port
- Stage: model
- Decision: both repositories are the same commit (`28b15f9`, 2023-12-11) and complete; ported to
  `part2/src/models/loggpt/` with the CC-BY-NC-4.0 `LICENSE`, the Nokia headers and `NOTICE.md` (mapping of every
  original function and every change). Runs on PyTorch 2.14 / transformers 5.20 in the Part 1 environment. One API
  difference had to be handled: with `output_hidden_states=True` in the config, transformers 5 `generate()` returns a
  dict; `return_dict_in_generate=False` restores the 4.18 behaviour (tensor of ids). Beam sampling
  (`num_beams=20, do_sample=True`) is still supported.
- Source: brief §5.1
- Deviation from paper: no

### P2-07 — Input and vocabulary (F2)
- Stage: representation
- Decision: LogGPT's own Drain parsing and block grouping are not used. Each session is Part 1's template-id sequence
  (same ids and order as the LSTM input); a token is the id as a string. The vocabulary is the original `WordVocab`:
  `<pad>, <unk>, <eos>, <bos>, <mask>` then the training keys by decreasing frequency, built on all 5,000 training
  sessions; a session is encoded `<bos> k1 … kL <eos>`; unseen templates → `<unk>`. Longest HDFS session: 298 events
  (`part1/data/sessions/hdfs/sessions_meta.json`) → 300 tokens < `n_positions = 512`, so **0 sessions are truncated**
  (the code has no truncation for HDFS; it would fail rather than truncate).
- Source: brief §3 F2, §5.2; LogGPT `utils/vocab.py`, `args/hdfs_config.py`
- Deviation from paper: yes, input parsing (by design of the fairness protocol)

### P2-08 — Training data: 5,000 normal sessions of the train split (F3)
- Stage: model
- Decision: `default_rng(seed).choice(normal sessions of the train split, 5000, replace=False)`, in sampled order
  (cached in `data/loggpt/seed{s}/train_sessions.json` and reused by `pca_normal5k`). As in the code, the first
  `int(0.9 · 5000)` = 4,500 are used for training and the last 500 for validation (pre-training checkpoint and RL early
  stopping). This coincides with the brief's suggested 500-session hold-out. An assertion checks that all 5,000 are
  normal, in the train split and not in the test split.
- Source: paper §V-A ("we randomly select 5000 normal log sequences"); `utils/utils.py train_test_split`;
  `model/initGPT.py` (`valid_index`)
- Deviation from paper: the test set is the 30% split instead of all remaining sessions (brief §5.6)

### P2-09 — K = int(0.5 × unique training keys)
- Stage: model
- Decision: the paper sets K to 50% of the training log keys; the HDFS code path computes
  `min(int(0.95 · n_keys), top_k=7)` (the `topk` ratio argument exists but `main.py` never passes it). We use the
  paper's ratio with the code's rounding, `int(n_keys · 0.5)`. Seed 0: 15 unique training keys (as in the paper's
  Table I) → K = 7, identical to the code's default. K and the code's value are stored per seed in
  `results/scores/loggpt_seed{s}.json` (`K`, `K_original_code_default`).
- Source: paper §V-A and Table I; `model/initGPT.py`, `model/logGPT.py`
- Deviation from paper: no

### P2-10 — Pre-training exactly as the code does it
- Stage: model
- Decision: GPT-2 from scratch (6 layers, 6 heads, n_embd 60, n_positions 512), AdamW lr 1e-4, batch 16 in a fixed
  order (`DataLoader(shuffle=False)`), 100 epochs, `labels = input_ids` (the loss also covers padding positions:
  the code does not mask them with −100). The code saves the checkpoint with the best validation loss but `main.py`
  keeps the **last-epoch** model in memory and uses it both for "LogGPT w/o RL" and as the start of RL; the port does
  the same and records the best-validation epoch (`pretrain_best_val_epoch`).
- Rationale: brief §5.1 "where the code and the paper disagree, follow the code"; the paper says nothing about these
  details. Pre-training uses only normal training sessions.
- Source: `model/initGPT.py`, `main.py`
- Deviation from paper: no

### P2-11 — RL fine-tuning exactly as the code does it (the paper calls it PPO)
- Stage: model
- Decision: for each of the 4,500 training sequences, in an order reshuffled every episode: prompt = first
  ⌊0.2·len⌋ tokens (at least 2, at most len − 2); a continuation to the full length is generated by beam sampling
  (20 beams, `do_sample=True`, model in train mode); loss = −(Σₜ log p(actual kₜ | generated prefix)) × (Σₜ ±1),
  with +1 when the actual next key is in the Top-K of the model's prediction and −1 otherwise; one Adam step
  (lr 1e-6) per sequence. Validation loss = the same quantity (no gradient) on the 500 validation sessions. Episode
  checkpoint kept if its validation loss ≤ the best so far, or if it is episode 0; a counter is incremented whenever the
  relative improvement over the previous episode is < 1% (never reset) and training stops when it exceeds 3; at most
  20 episodes. The final model is the kept checkpoint.
- Rationale: the code is a sequence-level policy gradient with one on-policy step per sample. With a single step per
  sample, PPO's ratio π_θ/π_θold equals 1 when the gradient is taken, so the clipped surrogate of the paper has the same
  gradient; the code has no ratio and no clipping. Brief: follow the code.
- Source: `model/logGPT.py`; paper §IV-C
- Deviation from paper: yes (follows the code), see rationale

### P2-12 — No test data and no anomaly labels in LogGPT's training-time decisions (F4 audit)
- Stage: model
- Decision: audit of the code: (1) pre-training checkpoint → validation loss on normal training sessions; (2) RL
  early stopping and episode selection → validation loss on the same normal sessions; (3) K → fixed ratio of the
  training keys; (4) after every RL episode the code runs `_predict_topk` on the **test set** — the output is only
  printed and never used for selection, but it is removed here (it would also cost one full test pass per episode).
  No anomalous session and no test session is used by `fit`; the stop condition of brief §10 is not triggered.
- Source: `model/initGPT.py`, `model/logGPT.py`; brief F4
- Deviation from paper: no

### P2-13 — Continuous score, `<unk>` rule and verification against the original code
- Stage: evaluation
- Decision: score = max over predicted positions of the rank of the actual next key in the softmax (1 = most
  likely), positions as in `_predict_topk`: the first key after `<bos>` is not predicted, `<eos>` is; `<unk>` gets rank
  V + 1. Decision: score > K. For speed, each distinct sequence is scored once (scores are a deterministic function of
  the sequence in eval mode; HDFS test sets contain far fewer distinct sequences than sessions) in right-padded
  batches with an attention mask. Verification on every seed (`verify` in `results/scores/loggpt*_seed{s}.json`), on
  150 anomalous + 150 normal test sessions + up to 50 sessions containing `<unk>`: (a) the batched implementation with
  the original `<unk>` handling reproduces the original `_predict_topk` (one sequence at a time, no mask,
  `torch.topk`) exactly; (b) the official score agrees on every session without `<unk>`. The `<unk>` rule differs from
  the code on purpose: in the code an unseen template maps to `<unk>`, whose probability can be in the Top-K, so such a
  session can be judged normal (observed in the smoke test with an under-trained model). The number of test decisions
  that change because of the rule is recorded per seed (`n_test_decisions_changed_by_unk_rule`).
- Source: brief §5.3–5.4; `model/initGPT.py _predict_topk`
- Deviation from paper: the `<unk>` rule (brief); the original reports a binary-score AUC, here AUC uses the rank score

### P2-14 — Seeding, checkpoints and resumability
- Stage: model
- Decision: `torch` is re-seeded with the resample seed at the start of pre-training and of RL, so a run resumed from
  the pre-training checkpoint gives the same RL run. Episode shuffles use a `numpy.random.RandomState(seed)` re-seeded
  each episode with its own `randint(0, 100)`, like the original's global `np.random`. Pre-training and every RL
  episode (model, optimiser, best state, counter, RNG states, elapsed time) are checkpointed in `data/loggpt/seed{s}/`.
  GPU kernels are not bit-for-bit deterministic, so a rerun may differ slightly.
- Source: brief §0 rule 5
- Deviation from paper: n/a

## Supervised models

### P2-15 — RF + MCV and LSTM + BERT re-run on seeds 0–9 with Part 1's code
- Stage: model
- Decision: `RFMCV(templates, cfg['rf'], seed)` (without step 3a) and `LSTMBert(E, cfg['window'], cfg['lstm'], seed)`
  exactly as in Part 1 (LSTM epoch selected on a 10% train-only validation set). Seed 0 reproduces Part 1 to every
  printed digit: RF F1 0.997904, LSTM F1 0.981265 (`part2/results/runs.csv` vs `part1/results/metrics.csv`).
- Source: brief §6
- Deviation from paper: n/a

## Cost

### P2-16 — Timing protocol (F7)
- Stage: evaluation
- Decision: wall-clock seconds measured around `fit` (training) and around scoring the full test split (inference);
  the device is recorded per run (`cpu:<model> (20 threads)` for RF/PCA, `cuda:NVIDIA GeForce RTX 4060 Laptop GPU` for
  LSTM/LogGPT). LogGPT training time = pre-training + RL; "LogGPT w/o RL" training time = pre-training only; LogGPT
  inference uses the distinct-sequence scoring of P2-13. Model loading/saving and the verification of P2-13 are not
  counted.
- Source: brief F7
- Deviation from paper: n/a

### P2-17 — All LogGPT resamples run one at a time (parallel attempt abandoned)
- Stage: evaluation (cost)
- Decision: seed 0 ran alone on the machine. Its LogGPT run took 3,892 s: 230 s of pre-training, then 11 RL
  episodes of ≈ 325 s each, early stop after episode index 10, episode 6 selected
  (`results/scores/loggpt_seed0.json`). During RL the process is bound by one CPU core (Python overhead of
  `generate`), and the GPU is ≈ 24% busy. To save time, seeds 1–9 of LogGPT were first started as 3 parallel
  processes. Measured effect: every process slowed down by ≈ 2.4× (pre-training 460–471 s instead of 230 s; initial RL
  validation 69–72 s instead of 29.5 s; `logs/seed{1,2,3}.log`), probably GPU time-slicing between three CUDA contexts.
  That is only ≈ 1.25× more throughput, timings inflated 2.4×, and > 1.5 h per resample (the brief's stop threshold)
  caused by the contention alone. The attempt was stopped after 20 minutes, before any LogGPT result was written. Its
  pre-training checkpoints were deleted, and LogGPT seeds 1–9 were rerun **sequentially**. Every timing in `runs.csv`
  therefore comes from a run that had the GPU to itself (`concurrent_processes = 1`). The SHAP explanation
  (`explain.py`, 4 CPU threads, ≈ 4 min) ran during the aborted attempt, not during any reported run.
- Source: own judgment; brief F7, §10
- Deviation from paper: n/a

### P2-24 — K sensitivity of LogGPT, post hoc (reported, never used for the ranking)
- Stage: evaluation
- Decision: with the paper's rule, `K = int(0.5 · n_keys)`, the 5,000 sampled sessions contain 15–17 unique keys
  across the seeds. That gives K = 7 in 3 seeds and K = 8 in 7 seeds, whereas the code's default is always 7.
  `report.loggpt_k_sensitivity` re-thresholds the stored scores at K ∈ {6, 7, 8, 9} for every seed
  (`results/loggpt_k_sensitivity.csv`). This is a sensitivity analysis only: the ranked LogGPT keeps the locked
  50% rule. The RL reward of each seed used that seed's own K, so the re-thresholding changes only the decision rule.
- Rationale: brief §12.8 ("K fixed from the original paper rather than tuned"); explains part of the spread of
  LogGPT's F1 across resamples.
- Source: own analysis
- Deviation from paper: n/a

## Statistical ranking

### P2-18 — Scott-Knott ESD 3.0.0, non-parametric version: what it actually computes
- Stage: ranking
- Decision: `sk_esd(x, version = "np", alpha = 0.05)` on a 10 × 4 data frame (rows = seeds, columns = RF_MCV,
  LSTM_BERT, PCA, LogGPT), once for F1 and once for AUC (`part2/src/scott_knott.R`; ScottKnottESD 3.0.0 was already
  installed from r-universe; the script asserts ≥ 3.0.0). Reading the package source
  (`PartitionNonParametric`): models are ordered by **median**; the split point is the one with the largest
  Kruskal–Wallis statistic; a candidate group is kept together when **Cliff's δ** between its first and last model
  is "negligible" (|δ| < 0.147). The `alpha` argument is passed through but is **not used** by the np partition.
  `checkDifference()` reports Cohen's d (a parametric effect size); the script adds Cliff's δ for every pair, since that
  is what the np version uses. Outputs: `sk_{f1,auc}.csv`, `sk_{f1,auc}_effect_sizes.csv`, `sk_input_*.csv`,
  `figures/sk_*.png` (the package plot shows the mean ± 1 sd of each model, because `sk_esd` calls `scottknott(..., dispersion = "s")`; the np groups themselves are ordered by medians).
- Source: brief P6, §8; ScottKnottESD 3.0.0 source
- Deviation from paper: n/a

## Explanation (E3)

### P2-19 — SHAP set-up
- Stage: explanation
- Decision: RF + MCV without step 3a of seed 0 (`part2/data/models/rf_seed0.joblib`, the model ranked in E2),
  `shap.TreeExplainer(clf)` (shap 0.54.1, default tree-path-dependent algorithm, additivity checked), explaining
  `predict_proba[:, 1]` (probability of the anomaly class). Sample: every anomalous test session of seed 0 (5,007)
  + 5,000 normal test sessions drawn with `numpy.random.default_rng(0)` (`results/explain/sample_seed0.csv`). Features
  are labelled `T<id>: <template text truncated>`.
- Source: brief §9
- Deviation from paper: n/a

### P2-20 — Grouped attribution by the step-3a correlation clusters
- Stage: explanation
- Decision: clusters recomputed on seed 0's train split with steps 1–2 of Part 1's step 3a (`explain.correlation_clusters`
  reuses `p1.feature_selection._spearman` and the same linkage/cut/representative rule). Check: the recomputed clusters
  reproduce exactly the 16 correlation removals of `part1/results/feature_selection_hdfs_random_seed0.csv`. Seed 0
  has 30 clusters (11 with more than one member). Grouped SHAP of a session = sum of its members' SHAP values
  (additivity); importance = mean |grouped SHAP|. `shap_grouped.csv` also gives the sum of the members' mean |SHAP|,
  which exceeds the grouped value when members push in opposite directions.
- Source: brief §9.2
- Deviation from paper: n/a

### P2-21 — Cross-checks
- Stage: explanation
- Decision: (1) RF + MCV with step 3a, trained with Part 1's code on seed 0 (27 features), SHAP on the same sample. Its
  features are the representatives that survived the VIF step, so the comparison with the grouped ranking uses the 27
  clusters that keep a feature. (2) sklearn `permutation_importance` of the main model, scoring F1, 10 repeats,
  `random_state = 0`, same sample. (3) The same permutation done per cluster: all member columns are permuted with
  the same row permutation (own implementation, 10 repeats). Agreement is measured with Kendall's τ-b (scipy) on the
  importance values and with the overlap of the top-5 sets (`shap_crosscheck.csv`, detail in
  `shap_crosscheck_detail.csv`). The sample is 50% anomalous, so its F1 (0.999) is not the test F1.
- Source: brief §9.3
- Deviation from paper: n/a

### P2-22 — Local cases
- Stage: explanation
- Decision: on the full seed-0 test set (TP 4,998, FN 9, FP 12): 2 TP (one drawn at random with `default_rng(0)`,
  and the TP with the lowest score), every FN (9), and the 5 FP with the highest scores. One waterfall plot per case
  (`explain/local/`). `local_cases.md` gives the template sequence (consecutive repeats collapsed), the 5 largest
  |SHAP| features with the count in the session, the median count and the share of normal **training** sessions
  that contain the template, and a one-line reading **generated from those numbers** (no hand-written interpretation).
- Source: brief §9.4
- Deviation from paper: n/a

### P2-23 — Stability over the 10 resamples
- Stage: explanation
- Decision: for every seed, that seed's RF model and the same sample design (all anomalous + 5,000 normal test
  sessions, `default_rng(seed)`). Main table: grouping fixed to **seed 0's clusters**, so that groups mean the same thing
  in every resample. The clusters recomputed on each seed's own train split are never identical to seed 0's (adjusted
  Rand index 0.61–0.93, `stability_info.json`). The main difference: in 7 of 9 seeds the write-pipeline templates
  1, 3, 4, 5 form one cluster, which seed 0 splits into {1, 5} and {3, 4, 16}. So the same importance is also reported
  with each seed's own clusters (`stability_own_clusters.csv`). Optional step done: ScottKnottESD (np) on the 10 ×
  n_clusters matrix of per-resample importance (`sk_feature_importance.csv`, `src/sk_features.R`).
- Source: brief §9.5
- Deviation from paper: n/a

### P2-25 — Control for training size (reported, never used for the ranking)
- Stage: evaluation (added after the presentation, following the instructor's comment on the first threat of slide 12)
- Decision: `src/matched_size.py` retrains RF, LSTM and PCA on n = 500, 5,000 and 50,000 sessions drawn at random from each train
  split (`default_rng(10000 + seed)`, class mix as drawn), plus PCA on LogGPT's 5,000 normal sessions for all 10 seeds. All runs use
  the same full test splits as the ranking (`results/followup/matched_size.csv`). RF and PCA on the whole train split are also rerun as a
  regression check: F1 is identical to `results/runs.csv`, as is PCA (5,000 normal) at seed 0. LSTM at n = 500 is not reliable: 3 of 10
  validation sets (50 sessions) contain no anomaly, so epoch 1 is kept. LogGPT is not retrained on other sizes (57 min per run).
- Source: instructor feedback, 2026-10-24
- Deviation from paper: n/a

### P2-26 — Statistics for the instructor's questions on PCA, LogGPT spread and tokens
- Stage: evaluation
- Decision: `src/followup_stats.py` computes from stored results (`results/followup/followup_stats.json`): the anomalies PCA misses per
  pair of resamples, their SPE, and the best F1 over all thresholds (chosen on the test set, diagnostic only); LogGPT F1 std under the
  50% rule and at fixed K; LogGPT size and token counts. A token is a template id, `<bos>` or `<eos>`. Counts come from session lengths
  and the run logs, not from a profiler. Beam steps = generated tokens x 20 and are an estimate. Test scoring runs each distinct sequence once.
  Temperature: `generate(num_beams=20, do_sample=True)` sets none, so the library default (1.0) applies; scoring does not sample.
  Added while reviewing the report (2026-11-02), from `results/scores/*.parquet` and the sessions: the share of PCA's missed anomalies with the
  replication template (T9) and of anomalies with the deletion error (T33, and how many of them PCA detects), plus the LogGPT sessions with a
  template unseen in training and how many of them are flagged (keys `pca_templates`, `loggpt_unseen`).
- Source: instructor feedback, 2026-10-24
- Deviation from paper: n/a

### P2-27 — LogGPT w/o RL: spread from the random state alone
- Stage: evaluation (added after the presentation, following the instructor's comment on LogGPT's spread)
- Decision: `src/loggpt_init_spread.py` repeats pre-training with init seeds 0-4 on the 5,000 sessions and the test split of seed 0, changing
  only the seed of the initialization and dropout (`results/followup/loggpt_init_spread.csv`). K = 7 in all runs. F1 0.843 ± 0.010 (0.830-0.857),
  against 0.848 ± 0.039 over the 10 resamples. RL is not repeated (about 53 min per run), so the effect of its sampling and of the temperature is not
  measured. One sample and five runs: an indication only. Init seed 0 repeats the original run, with close but not identical results
  (F1 0.8305 vs 0.8321, AUC 0.9468 vs 0.9595); we did not look for the cause. The first job was stopped after four runs when the session ended;
  the fifth run was added by resuming, and the CSV keeps all five.
- Source: instructor feedback, 2026-10-24
- Deviation from paper: n/a
