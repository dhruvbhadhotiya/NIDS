## 2026-07-22 — XAI Complete Rewrite (Research-Backed)

All XAI bugs fixed after 3 rounds of Kaggle errors. Web-verified against official docs.

### Bugs Found & Fixed (Round 2)

| # | Bug | Root cause (web-verified) | Fix |
|---|-----|--------------------------|-----|
| 1 | `name 'torch' is not defined` in GNNExplainer | `torch.tensor()` used without local import | Added `import torch` in `explain_gnn()` |
| 2 | Same in IntegratedGradients | `torch.zeros_like()` without local import | Added `import torch` in `explain_integrated_gradients()` |
| 3 | GNNExplainer for HeteroGNN would crash | **GNNExplainer does NOT support HeteroData** in any stable PyG ([GitHub #9973](https://github.com/pyg-team/pytorch_geometric/discussions/9973)) | Early bail-out in `explain_gnn()` with clear message |
| 4 | SHAP crash on multi-class XGBoost | **SHAP ≥0.45.0 changed return format** from `list[array]` to single 3D `[N,F,C]` array ([PR #3318](https://github.com/shap/shap/pull/3318)) | Unified internal format: always `shap_stacked [C,N,F]`, handles both old and new SHAP |
| 5 | `.mode().item()` crash | PyTorch `Tensor.mode()` returns named tuple `(values, indices)` ([docs](https://pytorch.com.tw/docs/2.5/generated/torch.mode.html)) | `.mode().item()` → `.mode().values.item()` (4 sites) |
| 6 | SHAP bar chart: wrong shape | `np.abs(list).mean(0).mean(1)` → `[N]` not `[F]` | Unified `np.abs(shap_stacked).mean(axis=(0,1))` → correct `[F]` |
| 7 | GNNExplainer random windows | Flow-level prediction index used as window index | Scan first 200 test windows for 3 diverse majority classes |
| 8 | IntegratedGradients only worked for homo graphs | Hardcoded to `edge_attr` | Now handles both: homo (edge_attr) and hetero (flow.x) |

### XAI Routing (final)

| Model | SHAP | LIME | GNNExplainer | IntegratedGradients |
|-------|------|------|-------------|-------------------|
| XGBoost | ✅ | ✅ | — | — |
| E-GraphSAGE | — | — | ✅ | ✅ |
| EGAT | — | — | ✅ | ✅ |
| HeteroGNN | — | — | ⛔ (not supported by PyG) | ✅ (Captum, supports hetero) |

### Files changed
- `nids/eval/xai.py` — complete rewrite of SHAP, GNNExplainer, IntegratedGradients, and XAI routing
- Re-upload to Kaggle required

---

## 2026-07-21 — Code Audit Fixes Applied (Round 1)

10 fixes applied across 8 files. See `CODE_AUDIT.md` for full details.

### Fixed

| ID | Severity | What | Files |
|----|----------|------|-------|
| C3 | 🔴 | Edge embedding now includes `edge_attr`: `[src∥dst∥src*dst∥edge_attr]`. Head input dim: `hidden*3` → `hidden*3 + edge_dim` | `egraphsage.py`, `egat.py` |
| C1 | 🔴 | `scheduler.step(val_f1)` → `scheduler.step()` — LR now properly cosine-anneals | `egraphsage.py`, `hetero_gnn.py` |
| C6 | 🔴 | Class weight label collection capped at 100 batches (was full dataset: ~106K iters on CICIDS) | `egraphsage.py`, `hetero_gnn.py`, `egat.py` |
| C4 | 🔴 | GNNExplainer now detects HeteroData — uses `x_dict`/`edge_index_dict`/`task_level='node'` | `xai.py` |
| C5 | 🔴 | `graph_root` default path fixed to correct Kaggle dataset path | `xai.py` |
| H5 | 🟠 | EGAT: `F.elu` → `F.relu` — consistent with other models | `egat.py` |
| H3 | 🟠 | LIME warns when used with GNN models (self-loop graph = misleading) | `xai.py` |
| H4 | 🟠 | `!pip install shap lime captum -q` in all 4 notebook setup cells | 4 notebooks |
| H6 | 🟠 | `embedding_quality()` now prints exception type+message on failure | `metrics.py` |
| M2 | 🟡 | ModelConfig defaults updated: `hidden=256, hops=3, fanout=(25,15,10), batch_size=128` | `base.py` |

### Not yet fixed (needs user discussion or Kaggle-side only)

| ID | Reason |
|----|--------|
| C2 (EGAT OOM CICIDS) | Architecture fixes may reduce memory; reassess after C3+C1 re-run |
| H2 (Dead classes) | Needs anomaly detection / binary classifiers — design discussion, not a code fix |
| H1 (GNNs < XGBoost) | C3 + C1 expected to close most of this gap |

### Next: Re-upload to Kaggle
- `nids/` package (7 files changed)
- 4 notebooks from `notebooks/`
- Run order: XGBoost → E-GraphSAGE → HeteroGNN → EGAT

---

## 2026-07-21 — Code Audit Complete (CODE_AUDIT.md)

26 findings across the codebase. Top findings:

| ID | Severity | Finding |
|----|----------|---------|
| C3 | 🔴 CRITICAL | **Edge features missing from final edge embedding** — E-GraphSAGE & EGAT classify edges using `[src_emb∥dst_emb∥src*dst]` WITHOUT the edge's own flow features. After mean aggregation, the edge's features are diluted ~1/in_degree. This is likely why XGBoost (flat features) beats all GNNs. |
| C1 | 🔴 CRITICAL | **Scheduler bug in E-GraphSAGE & HeteroGNN** — `scheduler.step(val_f1)` passes F1 score (float ~0.5) as epoch number to CosineAnnealingWarmRestarts, keeping LR stuck at ~99% max. Cosine annealing never activates. |
| C2 | 🔴 CRITICAL | **EGAT OOM on CICIDS** — GATv2Conv edge-attention ~14.2 GiB on T4 (14.56 GiB total) |
| C4 | 🔴 CRITICAL | **XAI broken for HeteroGNN** — GNNExplainer assumes homogeneous Data, crashes on HeteroData |
| C6 | 🔴 CRITICAL | **_build_criterion wastes full train pass** — ~106K batch iterations just for class weights on CICIDS |

Full document: `CODE_AUDIT.md` — no fixes applied yet.

---

## 2026-07-21 — v2 Full Results: 4 Models × 2 Datasets Complete

### Summary Table

| Dataset | Model        | Val F1 | Test Macro-F1 | Weighted-F1 | Accuracy | MCC    | Kappa  | ROC-AUC(macro) | PR-AUC(macro) | Train Time | Stop      | Status |
|---------|--------------|--------|----------------|-------------|----------|--------|--------|----------------|----------------|------------|-----------|--------|
| UNSW    | XGBoost      | 0.5647 | 0.5866         | 0.9836      | 0.9846   | 0.8538 | 0.8537 | 0.9939         | 0.6172         | 18.0 min   | Round 273 | OK |
| UNSW    | E-GraphSAGE  | 0.5097 | 0.5022         | 0.9815      | 0.9819   | 0.8279 | 0.8276 | 0.9539         | 0.4758         | 9.2 min    | Epoch 129 | OK |
| UNSW    | HeteroGNN    | 0.5013 | 0.5074         | 0.9811      | 0.9820   | 0.8286 | 0.8284 | 0.9065         | 0.5298         | 9.1 min    | Epoch 67  | OK |
| UNSW    | EGAT         | 0.3997 | 0.3734         | 0.9669      | 0.9697   | 0.7153 | 0.7133 | 0.9915         | 0.4856         | 7.6 min    | Epoch 57  | OK |
| CICIDS  | XGBoost      | 0.7000 | 0.6331         | 0.9863      | 0.9895   | 0.9467 | 0.9463 | 0.9862         | 0.6443         | 129.2 min  | Round 220 | OK |
| CICIDS  | E-GraphSAGE  | 0.6965 | 0.6380         | 0.9835      | 0.9864   | 0.9309 | 0.9305 | 0.9840         | 0.6568         | 79.5 min   | Epoch 151 | OK |
| CICIDS  | HeteroGNN    | 0.7041 | 0.6244         | 0.9804      | 0.9807   | 0.9057 | 0.9053 | 0.9809         | 0.6847         | 76.8 min   | Epoch 70  | OK |
| CICIDS  | EGAT         | —      | —              | —           | —        | —      | —      | —              | —              | —          | —         | **FAILED — CUDA OOM at GATv2Conv edge-attention update (~14.2GiB used / 14.56GiB GPU)** |

### GNN Structural Metrics

| Dataset | Model       | Avg Nodes | Avg Edges | Neighbor Coverage | Silhouette | Davies-Bouldin |
|---------|-------------|-----------|-----------|--------------------|------------|----------------|
| UNSW    | E-GraphSAGE | 1918      | 1671      | 71.95%             | 0.5363     | 2.9495         |
| UNSW    | HeteroGNN   | 1703      | 6684      | 100.00%            | 0.7398     | N/A            |
| UNSW    | EGAT        | 1918      | 1671      | 71.95%             | 0.5195     | N/A            |
| CICIDS  | E-GraphSAGE | 3389      | 3051      | 93.69%             | 0.1686     | 1.4938         |
| CICIDS  | HeteroGNN   | 3566      | 12205     | 100.00%            | 0.3714     | N/A            |

### UNSW Per-Class F1

| Class          | XGBoost | E-GraphSAGE | HeteroGNN | EGAT   |
|----------------|---------|-------------|-----------|--------|
| Analysis       | 0.2408  | 0.4080      | 0.3526    | 0.5330 |
| Backdoor       | 0.0054  | 0.0052      | 0.0000    | 0.0000 |
| Benign         | 1.0000  | 0.9999      | 1.0000    | 0.9997 |
| DoS            | 0.3820  | 0.2505      | 0.2163    | 0.0000 |
| Exploits       | 0.7376  | 0.6990      | 0.6890    | 0.6816 |
| Fuzzers        | 0.7483  | 0.7511      | 0.7338    | 0.0462 |
| Generic        | 0.8364  | 0.7815      | 0.7893    | 0.3727 |
| Reconnaissance | 0.6905  | 0.6387      | 0.6438    | 0.6317 |
| Shellcode      | 0.6179  | 0.4026      | 0.3903    | 0.4692 |
| Worms          | 0.6071  | 0.0851      | 0.2588    | 0.0000 |

### CICIDS Per-Class F1

| Class                     | XGBoost | E-GraphSAGE | HeteroGNN | EGAT |
|---------------------------|---------|-------------|-----------|------|
| Benign                    | 0.9973  | 0.9956      | 0.9924    | —    |
| Bot                       | 0.9978  | 0.9988      | 0.9985    | —    |
| Brute_Force_-Web          | 0.0148  | 0.2389      | 0.1600    | —    |
| Brute_Force_-XSS          | 0.0000  | 0.0000      | 0.0408    | —    |
| DDOS_attack-HOIC          | 0.9999  | 1.0000      | 1.0000    | —    |
| DDOS_attack-LOIC-UDP      | 0.3907  | 0.4220      | 0.3983    | —    |
| DDoS_attacks-LOIC-HTTP    | 0.9893  | 0.9737      | 0.9737    | —    |
| DoS_attacks-GoldenEye     | 0.9862  | 0.8730      | 0.8568    | —    |
| DoS_attacks-Hulk          | 0.9983  | 0.9114      | 0.9114    | —    |
| DoS_attacks-SlowHTTPTest  | 0.0000  | 0.0000      | 0.0000    | —    |
| DoS_attacks-Slowloris     | 0.8918  | 0.9072      | 0.9072    | —    |
| FTP-BruteForce            | 0.7856  | 0.7856      | 0.7856    | —    |
| Infilteration             | 0.4453  | 0.3750      | 0.3417    | —    |
| SQL_Injection             | 0.0000  | 0.0893      | 0.0000    | —    |
| SSH-Bruteforce            | 0.9996  | 1.0000      | 1.0000    | —    |

### Key Findings

1. **XGBoost still wins on macro-F1** — 0.587 (UNSW) and 0.633 (CICIDS). All GNNs trail. Graph topology as constructed (IP:port, 60s windows) doesn't add discriminative power beyond flat features. GNNs' only advantage is on a few specific classes (E-GraphSAGE: Analysis=0.408 vs XGBoost=0.241; Brute_Force_-Web=0.239 vs 0.015).

2. **v2 improvements barely moved the needle** — E-GraphSAGE UNSW macro-F1: 0.488 (v1) → 0.502 (v2), gain of only +0.014. Focal loss + class weights + 3-hop + 256 hidden gave nowhere near the projected +0.20 gain. The bottleneck is NOT the loss function — it's that the graph topology simply doesn't encode these attacks differently from Benign.

3. **EGAT is broken** — UNSW macro-F1=0.3734 (worst by far). Fuzzers collapsed from 0.75→0.05, Generic from 0.84→0.37. Probable scheduler bug: `scheduler.step()` called without `val_f1` argument (other models pass `scheduler.step(val_f1)`). Also uses ELU instead of ReLU.

4. **EGAT OOM on CICIDS** — GATv2Conv edge attention needs ~14.2 GiB for CICIDS windows (avg 3,389 nodes). T4 has 14.56 GiB total. Even with Layer-1 EdgeSAGEConv fallback, later GATv2Conv layers explode on large windows.

5. **Dead classes persist across ALL models** — Backdoor (UNSW, 700 samples), SlowHTTPTest (CICIDS, 15,833 samples), Brute_Force-XSS (CICIDS, 69 samples), SQL_Injection (CICIDS, 66 samples). These are undetectable from features alone — need different approach (anomaly detection, oversampling, or specialized detectors).

6. **No XAI run yet** — All 4 notebooks have XAI cells but they haven't been executed due to errors. XAI needs debugging before next run.

### Regression vs v1 (E-GraphSAGE UNSW)

| Class     | v1 F1 | v2 F1 | Delta  |
|-----------|-------|-------|--------|
| Worms     | 0.000 | 0.085 | +0.085 |
| DoS       | 0.018 | 0.251 | +0.233 |
| Shellcode | 0.557 | 0.403 | -0.154 |
| Analysis  | 0.408 | 0.408 | 0.000  |
| **Macro** | 0.488 | 0.502 | +0.014 |

### Target vs Actual

| Metric                | Target  | Actual (best) | Delta   |
|-----------------------|---------|---------------|---------|
| UNSW Macro-F1         | >0.70   | 0.587 (XGB)   | -0.113  |
| CICIDS Macro-F1       | >0.78   | 0.638 (E-SAGE) | -0.142  |
| Paper weighted-F1     | 0.907   | 0.986 (XGB)   | +0.079  |

### XAI Status
- **All 4 notebooks have XAI cells but XAI execution FAILED** — `run_xai_for_model` has bugs:
  - Default `graph_root` points to old path `/kaggle/input/nf-v3-windowed-graphs` (missing `datasets/dhruvbhadhotiya/`)
  - GNNExplainer hardcoded for edge-level explanation — won't work for HeteroGNN (node-level)
  - HeteroData not handled (GNNExplainer assumes `graph_data.x`/`graph_data.edge_index`, not `graph_data['flow'].x`)
  - LIME for GNNs creates self-loop graph losing all topology
  - Missing `captum` in pip install cell (Integrated Gradients requires it)

### Next Steps

1. **Fix EGAT scheduler bug** — `scheduler.step()` → `scheduler.step(val_f1)` in `nids/models/egat.py` line 202
2. **Fix XAI for all models** — Multiple bugs in `nids/eval/xai.py` (GNNExplainer for HeteroGNN, graph_root default path, LIME self-loop issue)
3. **Add `captum` to notebooks** — `!pip install shap lime captum -q` in Cell 1
4. **Re-run EGAT on UNSW** after scheduler fix to see if performance recovers
5. **EGAT CICIDS memory** — Need gradient accumulation or window-size capping for CICIDS
6. **Investigate why GNNs don't beat XGBoost** — Possible root causes: (a) graph topology doesn't encode attack signatures, (b) window scheme loses temporal ordering, (c) mean aggregation washes out rare-class signals
7. **Dead class strategy** — Separate anomaly detectors or specialized binary classifiers for Backdoor, SlowHTTPTest, Brute_Force-XSS, SQL_Injection
8. **Run XAI after fixes** — Validating what features/models attend to will inform next architecture changes

---

## 2026-07-20 — XAI (Explainable AI) Module Added

### New file: `nids/eval/xai.py`
5 explainer functions + 1 all-in-one runner:

| Function | Target Model | Method | Output |
|----------|-------------|--------|--------|
| `explain_xgboost_shap()` | XGBoost | SHAP TreeExplainer | Summary plot, per-class SHAP, feature importance bar chart |
| `explain_gnn()` | E-GraphSAGE / HeteroGNN / E-GAT | PyG GNNExplainer | Edge importance histogram, node feature importance bar chart |
| `explain_integrated_gradients()` | Any differentiable GNN | Captum IntegratedGradients | Per-feature attribution bar chart (requires `pip install captum`) |
| `explain_lime()` | Any (model-agnostic) | LIME TabularExplainer | Per-sample LIME HTML reports |
| `run_xai_for_model()` | Auto-detect | Dispatches to correct explainer based on `model_name` | All of the above |

### Notebooks updated (all 4, now 10 cells each)
Each notebook has 2 new cells:
- **Cell 5:** XAI after UNSW training — `run_xai_for_model(result_unsw)`
- **Cell 8:** XAI after CICIDS training — `run_xai_for_model(result_cicids)`

XAI outputs saved to `/kaggle/working/viz/{unsw,cicids}/xai/`
  - XGBoost: `shap/` (SHAP plots) + `lime/` (LIME HTML)
  - GNNs: `gnnexplainer/` (GNNExplainer edge + feature mask plots)
  - All: `integrated_gradients/` (Captum IG, if installed)

### SHAP feature names note
SHAP uses numeric feature indices (`feat_0`, `feat_1`, ...). The actual feature names from `nids.config.NUMERIC_COLS` are not mapped yet — feature names from the categorical vocabulary are lost during one-hot encoding. A future improvement would map SHAP indices back to original feature names.

---

## 2026-07-20 — Bug Fixes for v2 Model Run

### XGBoost v2 Results (UNSW — SUCCESS)
- **Macro-F1: 0.5868** (best so far! E-GraphSAGE v1 was 0.488)
- **Weighted-F1: 0.9836**
- **Worms F1: 0.630** (was 0.000 in E-GraphSAGE v1!)
- **Shellcode F1: 0.633** (was 0.557 in v1)
- **Backdoor F1: 0.005** — still terrible (700 samples, near-zero detection)
- **Analysis F1: 0.222** — improved from 0.408 (wait, actually slightly worse than v1 but better than dead)
- **DoS F1: 0.367** — massive improvement from 0.018 in E-GraphSAGE v1

### Bugs Fixed (2026-07-20 session)

| # | Model | Error | Root Cause | Fix |
|---|-------|-------|-----------|-----|
| 1 | XGBoost | `ValueError: x and y must have same first dimension, shapes (0,) and (242,)` | `TrainHistory.train_loss` empty (XGBoost doesn't compute per-iteration train loss). `plot_training_history` assumed matching lengths. | Populated `train_loss` with `val_mlogloss` as proxy. Made `plot_training_history` handle mismatched array lengths. |
| 2 | E-GraphSAGE | `AcceleratorError: CUDA error: invalid argument` on CICIDS | CUDA fragmentation from UNSW training not freed before CICIDS model allocation. 3-hop/256-hidden model too large for T4 GPU with fragmented memory. | Added `torch.cuda.empty_cache()` before `model.to(device)` in all GNN `fit()` methods. Reduced batch_size from 256→128. |
| 3 | HeteroGNN | `ValueError: need at least one array to concatenate` in `_build_criterion` | Label extraction checked `hasattr(batch, 'edge_label')`/`hasattr(batch, 'y')` before checking `batch['flow'].y` — HeteroData `hasattr` behavior inconsistent. | Reordered checks: check `batch.node_types` and `batch['flow'].y` FIRST. Added explicit error if no labels found. |
| 4 | E-GAT | `OutOfMemoryError: CUDA out of memory` on both datasets | Custom `EdgeGATConv` created massive intermediate tensors: `lin_src(x).view(-1, H, C)` = `[N, 4, 64]` × float32 for heads=4. For large windows this exceeded 14.5 GB T4. | Replaced custom `EdgeGATConv` with: Layer 1 = `EdgeSAGEConv` (mean agg, memory-efficient). Layers 2+ = PyG native `GATv2Conv(edge_dim=...)` (optimized C++ scatter). Heads reduced 4→2. Batch_size 256→128. Added `torch.cuda.empty_cache()`. |

### Code Changes (2026-07-20)
| File | Change |
|------|--------|
| `nids/models/xgb.py` | `fit()` populates `history.train_loss` (fixes plot crash) |
| `nids/models/egraphsage.py` | `fit()` calls `torch.cuda.empty_cache()` before `model.to(device)` |
| `nids/models/hetero_gnn.py` | `_build_criterion()` checks HeteroData labels first; `fit()` calls `empty_cache()` |
| `nids/models/egat.py` | Fully rewritten — Layer 1 uses `EdgeSAGEConv`, Layers 2+ use `GATv2Conv`. Heads=2. Memory-efficient. |
| `nids/eval/viz.py` | `plot_training_history()` handles mismatched train/val loss lengths (XGBoost) |
| `notebooks/*_v2.ipynb` + `train_egat.ipynb` | batch_size reduced 256→128 (CICIDS OOM fix) |

### XGBoost Per-Class vs E-GraphSAGE v1
| Class | E-GraphSAGE v1 F1 | XGBoost v2 F1 | Delta |
|-------|-------------------|---------------|-------|
| Worms | 0.000 | **0.630** | +0.630 |
| Shellcode | 0.557 | **0.633** | +0.076 |
| DoS | 0.018 | **0.367** | +0.349 |
| Backdoor | 0.000 | **0.005** | +0.005 |
| Generic | 0.784 | **0.836** | +0.052 |
| Fuzzers | 0.734 | **0.749** | +0.015 |
| Exploits | 0.718 | **0.735** | +0.017 |
| Recon | 0.663 | **0.691** | +0.028 |
| Analysis | 0.408 | **0.222** | -0.186 |
| Macro-F1 | 0.488 | **0.587** | +0.099 |

**Key insight:** XGBoost on flat features beats E-GraphSAGE v1 (0.587 vs 0.488 macro-F1). Rare classes (Worms=25, Shellcode=358) are better detected by flat features than graph topology alone. This means either: (a) these attacks don't manifest in graph topology, or (b) class imbalance was the dominant problem and XGBoost handles it better natively. The v2 GNNs with focal loss + class weights should close this gap.

---

## 2026-07-20 — Improvement Roadmap + E-GAT + Focal Loss + Deeper Models

### Analysis of First Training Run

**E-GraphSAGE v1 Results:**
| Dataset | Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---------|----------|----------|-------------|-----|
| UNSW | 98.32% | 0.4882 | 0.9814 | 0.8398 |
| CICIDS | 98.72% | 0.6203 | 0.9840 | 0.9350 |

**HeteroGNN v1 Results:**
| Dataset | Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---------|----------|----------|-------------|-----|
| UNSW | 98.32% | 0.4841 | 0.9818 | 0.8403 |
| CICIDS | 98.78% | 0.6180 | 0.9842 | 0.9378 |

**3 killers dragging macro-F1 down:**
1. **Dead classes (F1=0.00):** Worms(25), Backdoor(700), Brute_Force-XSS(69), SQL_Injection(66), DoS_attacks-SlowHTTPTest(15,833!). SlowHTTPTest is most concerning — 15k samples but zero detection.
2. **DoS blindness (F1=0.018):** UNSW DoS(897). IP:port topology alone can't distinguish DoS from Benign bursts.
3. **Rare but recoverable (F1=0.33-0.56):** Analysis(185), Shellcode(358), Infilteration(17k).

**Paper comparison:** Paper reports 90.7% weighted F1 (multi-class, TON_IoT). Our weighted F1 is 98.1% (actually higher!), but weighted F1 is dominated by Benign (95% of data). Macro-F1 is the real metric for NIDS.

### v2 Improvements Implemented

1. **Focal Loss** (`nids/models/heads.py`): `FocalLoss(gamma=2.0)` class added. Downweights easy Benign examples.
2. **Class Weights** (`nids/models/heads.py`): `compute_class_weights()` with inverse/inverse_sqrt/balanced schemes.
3. **ModelConfig expanded** (`nids/models/base.py`): Added `focal_gamma` and `class_weight` fields.
4. **Cosine Annealing Scheduler**: Replaced ReduceLROnPlateau with `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` in both GNN models.
5. **Deeper Architecture**: Default changed to `hidden=256, hops=3, fanout=(25, 15, 10)`.
6. **E-GAT Model** (`nids/models/egat.py`): New model with custom `EdgeGATConv` — multi-head attention instead of mean aggregation. Registered as `"egat"`.
7. **4 new notebooks** in `notebooks/`: `train_egraphsage_v2.ipynb`, `train_hetero_gnn_v2.ipynb`, `train_egat.ipynb`, `train_xgboost_v2.ipynb`.

### Expected improvements v1→v2
| Change | Expected macro-F1 gain |
|---|---|
| Focal loss (gamma=2) | +0.08–0.12 |
| Class weights (inverse_sqrt) | +0.10–0.15 |
| 3-hop + hidden=256 | +0.03–0.05 |
| Cosine warm restarts | +0.02–0.03 |
| **Target:** UNSW macro-F1: 0.488→>0.70, CICIDS: 0.620→>0.78 |

---

## 2026-07-20 — Per-Model Notebooks + XGBoost Fix

### Issues Fixed
1. **XGBoost `gpu_hist` crash**: Kaggle's xgboost lacks GPU support. Changed to `tree_method='hist'` in `nids/models/xgb.py`.
2. **RAM exhaustion**: Training 6 models in one notebook exhausts Kaggle's ~30GB. Split into per-model notebooks.
3. **`'NoneType' has no attribute 'stored_arrays'`**: XGBoost notebook had `SCHEME = 'None'` (string) instead of `SCHEME = None` (Python None). Fixed in all notebooks.
4. **Model routing confusion**: Moved all model imports to Cell 1 (global) so registrations fire before training.

### Created
- `notebooks/` folder with 3 notebooks: `train_egraphsage.ipynb`, `train_hetero_gnn.ipynb`, `train_xgboost.ipynb`
- Each trains one model on both UNSW + CICIDS sequentially with memory cleanup between

---

## 2026-07-19/20 — Phase 2 Initial Training + Dimension Mismatch Bug

### Features Built
1. **`nids/models/` package** (6 files):
   - `base.py`: `Model` ABC + `ModelConfig` + `TrainHistory` dataclasses
   - `heads.py`: `MLPHead`, `EdgeMLPHead`, `ProjectionHead`
   - `egraphsage.py`: E-GraphSAGE with custom `EdgeSAGEConv` (edge features in message function)
   - `hetero_gnn.py`: HeteroGNN with `HeteroConv` (SAGEConv per relation type)
   - `xgb.py`: XGBoost baseline (`requires_scheme=None`, flat features only)
   - `__init__.py`

2. **`nids/eval/` expanded**:
   - `metrics.py`: 20+ metrics — confusion matrix, per-class precision/recall/F1, ROC-AUC (macro/micro), PR-AUC, MCC, Cohen's Kappa, silhouette score, Davies-Bouldin, neighbor coverage, graph topology stats, `print_report()`
   - `viz.py`: 9 plot functions + `generate_all_eval_plots()`

### Critical Bug: Dimension Mismatch
**Root cause:** `manifest.feat_dim` computed via formula (`37 + sum(vocab_sizes)`) but actual `flow_features.npy` has different column count due to Spearman pruning reducing numeric cols.

**Example:** UNSW manifest said 178, actual file had 165 → `mat1 (327964×166) and mat2 (179×128)` error.

**Fix:**
1. `nids/graphs/dataset.py`: `feat_dim` property now reads from `self.flow_features.shape[1]` (actual file)
2. `nids/graphs/builder.py`: `_build_feature_matrix()` returns actual dimensions; manifest uses captured values

---

## 2026-07-19 — Phase 1 Complete: Graph Artifacts Built

### What was built
- **`nids/` package** — 17 Python files implementing the full data pipeline
- **`laptop_phase1.ipynb`** — 10-cell Jupyter notebook running Stages 1-4 on laptop

### Pipeline Stages
1. **Stage 1: Load & Clean** — Raw CSV → clean parquet. Chunked loading for CICIDS (20M rows). Inf→NaN→median fill. Drop NaN IDs + duplicates.
2. **Stage 2: Chronological Split** — Per-class time-sorted, 70/15/15. Ensures every class appears in every split.
3. **Stage 3: Feature Engineering** — Log-transform skewed features → Spearman pruning (|r|>0.95) → StandardScaler → categorical top-15+OTHER → attack2id. ALL fitting on TRAIN only.
4. **Stage 4: Graph Construction** — Time-sort within split → build global IP:port + host vocabs → 60s windows → float16 feature matrix → save 8 `.npy` arrays + 1 `.parquet` per split.

### Graph Artifacts Produced
| Dataset | N flows | Windows | Feat Dim | IP:port Nodes | Host Nodes | Disk |
|---------|---------|---------|----------|---------------|------------|------|
| UNSW | 2,350,609 | 1,747 | 165-178 | 1,092,975 | 44 | ~0.8 GB |
| CICIDS | 19,486,987 | 7,546 | 143 | ... | ... | ~6.2 GB |
| **Total** | | | | | | **~7.0 GB** |

### Architecture Decisions
- **float16** features → halves storage
- **CSR window_ptr** → zero-copy per-window slicing
- **Global node IDs** → per-window local reindexing at load time (cheap np.unique)
- **Features stored ONCE** → both graph schemes reconstruct lazily
- **One (dataset, split) at a time** → bounded RAM
- **Chunked CSV loading** for CICIDS → stays under ~6 GB RAM

### Key Bug Fixes During Build
1. **pyarrow append not supported**: Changed chunked loading to write individual temp parquets, then concat.
2. **Config variable name mismatch**: `categorical_cols` (lowercase, not imported) → `CATEGORICAL_COLS` (uppercase, from config).
3. **DATASET_KEYS updated** to match actual CSV filenames (e.g., `NF-UNSW-NB15-v3`).
4. **NUMERIC_COLS expanded** from 36 to 38 (added `DNS_QUERY_ID`, `DNS_TTL_ANSWER`).
5. **Line counter optimized**: Buffered 8MB reads instead of per-byte iteration.

### How Window Graphs Work
- Flows are time-sorted within each split
- 60-second non-overlapping windows: `window_id = (timestamp - t0) // 60000`
- Sparse windows (<5 flows) merge forward into next window
- Each window = one PyG `Data` or `HeteroData` reconstructed at `__getitem__` time
- Windows mix ALL attack types that co-occur in that time slice (not per-attack windows)
- Per-class chronological split ensures: test = latest 15% of each class (future relative to train) + every attack type appears in every split

### Files Created (laptop)
```
nids/
├── __init__.py, config.py, registry.py, paths.py, contracts.py
├── data/
│   ├── __init__.py, base.py (DatasetAdapter ABC), unsw.py, cicids.py
├── graphs/
│   ├── __init__.py, base.py (GraphScheme ABC), windowing.py
│   ├── ipport.py (IP:port scheme), hetero.py (host↔flow scheme)
│   ├── builder.py (Stage 4 driver), dataset.py (WindowGraphDataset)
├── eval/
│   ├── __init__.py, metrics.py, viz.py
├── models/
│   ├── __init__.py, base.py, heads.py
│   ├── egraphsage.py, hetero_gnn.py, xgb.py, egat.py
├── notebooks/
│   ├── train_egraphsage.ipynb, train_egraphsage_v2.ipynb
│   ├── train_hetero_gnn.ipynb, train_hetero_gnn_v2.ipynb
│   ├── train_xgboost.ipynb, train_xgboost_v2.ipynb
│   └── train_egat.ipynb
laptop_phase1.ipynb
kaggle_phase2.ipynb (legacy — use per-model notebooks instead)
requirements.txt
```

### Environment
- **Laptop:** Ryzen 5 7520U, 16GB RAM, Windows 11, Python 3.11.9 venv
- **Kaggle:** T4 GPU, ~30GB RAM, Python 3.12
- **Key dependencies:** numpy 1.26.4, pandas 2.2.2, scipy 1.14.1, scikit-learn 1.5.2, torch + torch_geometric

### Kaggle Datasets Created
| Dataset | Contents | Size |
|---------|----------|------|
| `nf-v3-windowed-graphs` | `graphs/v=fv1/w=60/` | ~7 GB |
| `nids-package` | `nids/` folder | <1 MB |

### Reference Paper
"Graph-based Intrusion Detection System Using General Behavior Learning" (GBA-IDS)
- Binary: 99.6% Acc, 99.1% F1 (UNSW)
- Multi-class: 90.7% Weighted F1 (TON_IoT)
- Key insight: attention-based models pay more attention to minor classes; mean aggregation "can conceal the anomaly in a single flow"

---

*— Project started 2026-07-18. Laptop environment: Windows 11, Python 3.11.9, Ryzen 5 7520U 16GB RAM.*
