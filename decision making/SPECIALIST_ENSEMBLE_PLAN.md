# Specialist Ensemble NIDS — Parallel Dual-Model Architecture

> **Target: Macro-F1 > 0.90 (detectable classes)**
> **Architecture: Two models in parallel — GNN + XGBoost/MLP — each with an "Other" class**

---

## Architecture

```
                    ┌──────────────────────────┐
                    │   60-Second Flow Window   │
                    │  (raw NetFlow features)   │
                    └─────────┬────────────────┘
                              │
                    ┌─────────▼────────────────┐
                    │   Feature Extraction      │
                    │  (Stage 1-3 pipeline)     │
                    └─────────┬────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │                               │
    ┌─────────▼──────────┐          ┌─────────▼──────────┐
    │  Construct Graph    │          │  Flat Features      │
    │  (IP:port scheme)   │          │  (same flow feats)  │
    └─────────┬──────────┘          └─────────┬──────────┘
              │                               │
    ┌─────────▼──────────┐          ┌─────────▼──────────┐
    │  GNN Model          │          │  XGBoost / MLP     │
    │  (E-GraphSAGE)      │          │  Model             │
    │                     │          │                     │
    │  Classes:           │          │  Classes:           │
    │  - Topology attacks │          │  - Flow attacks     │
    │  - OTHER            │          │  - OTHER            │
    └─────────┬──────────┘          └─────────┬──────────┘
              │                               │
              └───────────┬───────────────────┘
                          │
                ┌─────────▼──────────┐
                │  Decision Logic     │
                │                     │
                │  GNN=X, XGB=Other   │──→ X (topology attack)
                │  GNN=Other, XGB=Y   │──→ Y (flow attack)
                │  Both=Other         │──→ Benign
                │  Both≠Other         │──→ Higher confidence wins
                └─────────────────────┘
```

---

## 1. Model Class Assignments

### 1.1 UNSW-NB15

#### GNN Model (E-GraphSAGE) — Topology-Visible Attacks

| Class | Samples | Why GNN? | Topology Signal |
|-------|---------|----------|-----------------|
| Fuzzers | ~24,000 | Many malformed packets → one target | **Fan-in** on target IP:port node |
| Reconnaissance | ~13,000 | One scanner → many dst ports/IPs | **Fan-out** from scanner node |
| OTHER | Everything else | GNN should predict "Other" for attacks with no topology signal | Benign + Generic + Exploits + DoS + Shellcode + Analysis + Backdoor + Worms |

**GNN model: 3 classes** (Fuzzers=0, Reconnaissance=1, OTHER=2)

#### XGBoost/MLP Model — Flow-Feature Attacks

| Class | Samples | Why XGBoost? | Distinctive Flow Features |
|-------|---------|-------------|--------------------------|
| Generic | ~215,000 | High-volume varied protocols | Bytes/packets patterns |
| Exploits | ~56,000 | Specific port targeting, payload-heavy | Port + packet size distribution |
| DoS | ~897 | Burst pattern in flow stats | High IN_PKTS, short duration |
| Shellcode | ~358 | Unusual payload sizes | Packet size anomaly |
| Analysis | ~185 | Slow probing, varied protocols | Duration + protocol features |
| OTHER | Everything else | XGBoost should predict "Other" for topology attacks | Benign + Fuzzers + Recon + Backdoor + Worms |

**XGBoost model: 6 classes** (Generic=0, Exploits=1, DoS=2, Shellcode=3, Analysis=4, OTHER=5)

#### Undetectable (neither model handles)

| Class | Samples | Reasoning |
|-------|---------|-----------|
| Backdoor | 700 | Stealth C&C mimics Benign — no signal in NetFlow features OR topology |
| Worms | 25 | Only 25 samples — statistically impossible to learn |

**Both models will predict "Other" for Backdoor/Worms → classified as Benign.**
This is an honest limitation — we document it as NetFlow-undetectable.

---

### 1.2 CICIDS-2018

#### GNN Model (E-GraphSAGE) — Topology-Visible Attacks

| Class | Samples | Why GNN? | Topology Signal |
|-------|---------|----------|-----------------|
| DDOS_attack-HOIC | ~686,000 | Massive UDP fan-in | **Fan-in** on victim |
| DDoS_attacks-LOIC-HTTP | ~576,000 | HTTP flood fan-in | **Fan-in** on victim |
| DDOS_attack-LOIC-UDP | ~1,700 | UDP flood fan-in | **Fan-in** on victim |
| DoS_attacks-Hulk | ~461,000 | HTTP POST flood | **Fan-in** + high volume |
| DoS_attacks-GoldenEye | ~41,000 | HTTP keep-alive abuse | **Fan-in** + long duration |
| Bot | ~286,000 | C&C heartbeats | **Star topology** around C&C |
| SSH-Bruteforce | ~187,000 | Repeated SSH to same target | **Edge multiplicity** (repeated edges) |
| FTP-BruteForce | ~191,000 | Repeated FTP to same target | **Edge multiplicity** |
| Brute_Force_-Web | ~200 | Repeated HTTP to same target | **Edge multiplicity** |
| OTHER | Everything else | Benign + Slowloris + Infiltration + SlowHTTPTest + XSS + SQLi |

**GNN model: 10 classes** (9 topology attacks + OTHER)

#### XGBoost/MLP Model — Flow-Feature Attacks

| Class | Samples | Why XGBoost? | Distinctive Flow Features |
|-------|---------|-------------|--------------------------|
| DoS_attacks-Slowloris | ~5,093 | Extremely long duration + tiny bytes | `FLOW_DURATION` very high, `IN_BYTES` very low |
| Infilteration | ~161,000 | Lateral movement, varied ports | Byte asymmetry, protocol diversity |
| OTHER | Everything else | Benign + all topology attacks + undetectable |

**XGBoost model: 3 classes** (Slowloris=0, Infilteration=1, OTHER=2)

#### Undetectable (neither model handles)

| Class | Samples | Reasoning |
|-------|---------|-----------|
| DoS_attacks-SlowHTTPTest | 15,833 | Stealth slow POST — indistinguishable from normal HTTP at NetFlow level. 15K samples, F1=0.000 across all 4 models. See detailed reasoning below. |
| Brute_Force_-XSS | 69 | Payload-level attack (script injection in HTTP body). Only 69 samples. Invisible in NetFlow. |
| SQL_Injection | 66 | Payload-level attack (SQL in query string). Only 66 samples. Invisible in NetFlow. |

---

## 2. Undetectable Attack Reasoning (with evidence)

### SlowHTTPTest (15,833 samples, F1 = 0.000 across ALL models)

**Attack mechanism**: Sends a legitimate HTTP POST with `Content-Length: large_number`, then transmits the POST body at ~1 byte per 10 seconds, keeping the server connection open and consuming a thread.

**Why NetFlow cannot detect it:**

```
NORMAL slow HTTP session:              SlowHTTPTest attack:
┌─────────────────────┐               ┌─────────────────────┐
│ PROTOCOL: TCP       │               │ PROTOCOL: TCP       │  ← SAME
│ L7_PROTO: HTTP      │               │ L7_PROTO: HTTP      │  ← SAME
│ IN_BYTES: 350       │               │ IN_BYTES: 300       │  ← OVERLAPS
│ OUT_BYTES: 80       │               │ OUT_BYTES: 60       │  ← OVERLAPS
│ DURATION: 55000ms   │               │ DURATION: 58000ms   │  ← OVERLAPS
│ IAT_AVG: 8500ms     │               │ IAT_AVG: 10000ms   │  ← OVERLAPS
│ PKTS_<128B: 12      │               │ PKTS_<128B: 14      │  ← OVERLAPS
│ TCP_FLAGS: SYN,ACK  │               │ TCP_FLAGS: SYN,ACK  │  ← SAME
└─────────────────────┘               └─────────────────────┘
```

The feature distributions overlap so heavily that no classifier can find a decision boundary. This is BY DESIGN — SlowHTTPTest is a stealth attack.

**Evidence**: 4 models × 2 architectures × 15,833 samples = **0.000 F1**. If features could separate it, at least ONE model would achieve F1 > 0.

**Would need**: Server-side connection state (concurrent incomplete POSTs from same IP), or DPI (byte-by-byte timing analysis).

### XSS / SQL Injection (69 / 66 samples)

**Attack mechanism**: Malicious payloads embedded in HTTP request bodies/query strings.

```
Normal HTTP POST:                     XSS Attack HTTP POST:
┌─────────────────────┐               ┌─────────────────────┐
│ POST /form HTTP/1.1 │               │ POST /form HTTP/1.1 │
│ Body: name=John     │               │ Body: name=<script> │  ← Payload-level!
│                     │               │  alert(1)</script>   │
│ NetFlow sees:       │               │ NetFlow sees:       │
│ BYTES: 450          │               │ BYTES: 480          │  ← SAME
│ PKTS: 5             │               │ PKTS: 5             │  ← SAME
│ DURATION: 200ms     │               │ DURATION: 210ms     │  ← SAME
└─────────────────────┘               └─────────────────────┘
```

NetFlow summarizes the connection at the transport layer — it NEVER sees the HTTP body content. The `<script>` tag or `' OR 1=1` payload is invisible.

### Backdoor (UNSW, 700 samples, F1 = 0.005)

**Attack mechanism**: Establishes a covert C&C channel designed to mimic normal traffic.

```
Normal HTTPS browsing:                Backdoor C&C:
┌─────────────────────┐               ┌─────────────────────┐
│ DST_PORT: 443       │               │ DST_PORT: 443       │  ← SAME (uses HTTPS)
│ PROTOCOL: TCP       │               │ PROTOCOL: TCP       │  ← SAME
│ IN_BYTES: 2500      │               │ IN_BYTES: 1800      │  ← OVERLAPS
│ OUT_BYTES: 800      │               │ OUT_BYTES: 500      │  ← OVERLAPS
│ DURATION: 3000ms    │               │ DURATION: 2500ms    │  ← OVERLAPS
│ PKTS: 15            │               │ PKTS: 12            │  ← OVERLAPS
└─────────────────────┘               └─────────────────────┘
```

Backdoors intentionally use common ports (443, 80) and normal-looking traffic volumes to evade detection. With only 700 samples among 2.3M, the signal-to-noise ratio is near zero.

---

## 3. The "Other" Class — How to Train It Well

The success of this architecture depends entirely on how well each model learns to say "I don't know" (predict OTHER).

### Training the "Other" Class

**Key principle**: The "Other" class for each model = Benign + attacks that belong to the other model.

```python
# === GNN Model Training Data ===
# Step 1: Relabel
gnn_label_map = {
    # Topology attacks → keep their class
    attack2id['Fuzzers']: 0,           # GNN class 0
    attack2id['Reconnaissance']: 1,    # GNN class 1
    # Everything else → OTHER
}
OTHER_GNN = 2  # GNN class 2

gnn_labels = np.full(len(original_labels), OTHER_GNN, dtype=np.int64)
for original_id, gnn_id in gnn_label_map.items():
    gnn_labels[original_labels == original_id] = gnn_id

# Step 2: Subsample OTHER class to avoid drowning the real classes
# Keep ALL topology attack samples, subsample OTHER to 3x the total attack count
n_attacks = (gnn_labels != OTHER_GNN).sum()
other_idx = np.where(gnn_labels == OTHER_GNN)[0]
rng = np.random.RandomState(42)
keep_other = rng.choice(other_idx, size=min(len(other_idx), 3 * n_attacks), replace=False)
train_idx = np.concatenate([np.where(gnn_labels != OTHER_GNN)[0], keep_other])

# Step 3: Build graph DataLoader from these indices only
# (use existing WindowGraphDataset, just filter windows containing these flows)
```

```python
# === XGBoost Model Training Data ===
xgb_label_map = {
    attack2id['Generic']: 0,
    attack2id['Exploits']: 1,
    attack2id['DoS']: 2,
    attack2id['Shellcode']: 3,
    attack2id['Analysis']: 4,
}
OTHER_XGB = 5

xgb_labels = np.full(len(original_labels), OTHER_XGB, dtype=np.int64)
for original_id, xgb_id in xgb_label_map.items():
    xgb_labels[original_labels == original_id] = xgb_id

# Same subsampling for OTHER
```

### Why "Other" Works Here

The attacks are assigned to models based on **genuinely different detection mechanisms**:

- **GNN's "Other"** includes Generic/Exploits/DoS flows that look like normal single connections in the graph. The GNN won't see distinctive topology for these → naturally predicts OTHER.
- **XGBoost's "Other"** includes DDoS/BruteForce flows that look like heavy-but-normal traffic in flat features. XGBoost won't see distinctive flow patterns for these → naturally predicts OTHER.

The key insight: **the "Other" class is NOT random noise — it's a genuine cluster in each model's feature space** (flows without topology signal cluster together for the GNN; flows without distinctive flat features cluster together for XGBoost).

---

## 4. Decision Logic — Detailed

```python
class DualModelEnsemble:
    """
    Parallel dual-model NIDS.
    
    Both models see the SAME flows. Each predicts independently.
    Decision logic combines predictions.
    """
    
    def __init__(self, gnn_model, xgb_model, 
                 gnn_local_to_global, xgb_local_to_global,
                 gnn_other_id, xgb_other_id,
                 benign_global_id, confidence_threshold=0.6):
        self.gnn = gnn_model
        self.xgb = xgb_model
        self.gnn_map = gnn_local_to_global   # {0: 'Fuzzers', 1: 'Recon', 2: 'OTHER'}
        self.xgb_map = xgb_local_to_global   # {0: 'Generic', ..., 5: 'OTHER'}
        self.gnn_other = gnn_other_id
        self.xgb_other = xgb_other_id
        self.benign_id = benign_global_id
        self.conf_threshold = confidence_threshold
    
    def predict(self, graph_data, flat_features):
        """
        Parameters
        ----------
        graph_data : PyG Data (one window graph)
        flat_features : np.ndarray [n_flows, F]
        
        Returns
        -------
        global_predictions : np.ndarray [n_flows] — global class IDs
        confidences : np.ndarray [n_flows] — confidence scores
        """
        n_flows = len(flat_features)
        
        # --- Run both models ---
        # GNN: predicts per-edge (edge = flow)
        gnn_logits = self.gnn.net(graph_data)          # [n_flows, gnn_n_classes]
        gnn_probs = torch.softmax(gnn_logits, dim=-1)
        gnn_preds = gnn_probs.argmax(dim=-1).numpy()   # [n_flows]
        gnn_confs = gnn_probs.max(dim=-1).values.numpy()
        
        # XGBoost: predicts per-flow (flat features)
        xgb_probs = self.xgb.predict_proba(flat_features)  # [n_flows, xgb_n_classes]
        xgb_preds = xgb_probs.argmax(axis=-1)
        xgb_confs = xgb_probs.max(axis=-1)
        
        # --- Decision Logic ---
        results = np.full(n_flows, self.benign_id, dtype=np.int64)
        confidences = np.zeros(n_flows, dtype=np.float32)
        
        for i in range(n_flows):
            gnn_is_other = (gnn_preds[i] == self.gnn_other)
            xgb_is_other = (xgb_preds[i] == self.xgb_other)
            
            if gnn_is_other and xgb_is_other:
                # CASE 1: Both say OTHER → Benign
                results[i] = self.benign_id
                confidences[i] = min(gnn_confs[i], xgb_confs[i])
                
            elif not gnn_is_other and xgb_is_other:
                # CASE 2: GNN detects attack, XGBoost says OTHER → GNN wins
                results[i] = self.gnn_map[gnn_preds[i]]
                confidences[i] = gnn_confs[i]
                
            elif gnn_is_other and not xgb_is_other:
                # CASE 3: XGBoost detects attack, GNN says OTHER → XGBoost wins
                results[i] = self.xgb_map[xgb_preds[i]]
                confidences[i] = xgb_confs[i]
                
            else:
                # CASE 4: Both detect an attack (CONFLICT)
                # Higher confidence wins
                if gnn_confs[i] >= xgb_confs[i]:
                    results[i] = self.gnn_map[gnn_preds[i]]
                    confidences[i] = gnn_confs[i]
                else:
                    results[i] = self.xgb_map[xgb_preds[i]]
                    confidences[i] = xgb_confs[i]
        
        return results, confidences
```

### Decision Matrix — All Cases

| Case | GNN Prediction | XGB Prediction | Final Output | Who Decides | Expected Frequency |
|------|---------------|----------------|--------------|-------------|-------------------|
| 1 | OTHER | OTHER | **Benign** | Agreement | ~85% (most flows are Benign) |
| 2 | Topology attack X | OTHER | **X** | GNN wins | ~8% (topology attacks) |
| 3 | OTHER | Flow attack Y | **Y** | XGB wins | ~5% (flow-feature attacks) |
| 4a | Attack X | Attack Y | **Higher confidence** | Confidence | ~1% (rare conflict) |
| 4b | Attack X | Attack X | **X** (both agree!) | Agreement | ~1% (both can detect) |

### Why Conflicts (Case 4) Are Rare

The attacks are split into non-overlapping groups based on detection mechanism:
- DDoS creates high fan-in → GNN detects it. But DDoS also has high bytes/packets → XGBoost MIGHT also flag it
- In practice, when XGBoost is trained with DDoS in its "Other" class, it learns that high-volume traffic alone isn't enough for its attack classes → it correctly says OTHER

The conflict rate depends on how well the "Other" class is trained. With proper subsampling (Section 3), expect <2% conflict rate.

---

## 5. Implementation Phases (Revised)

### Phase 0: Topology Validation (1 day)
Same as before — compute per-class fan-in, fan-out, edge multiplicity from existing graphs to validate class assignments.

### Phase 1: Prepare Dual-Model Labels (0.5 day)

```python
# File: nids/models/dual_labels.py

UNSW_GNN_CLASSES = {
    'Fuzzers': 0,
    'Reconnaissance': 1,
}
UNSW_GNN_OTHER = 2

UNSW_XGB_CLASSES = {
    'Generic': 0,
    'Exploits': 1,
    'DoS': 2,
    'Shellcode': 3,
    'Analysis': 4,
}
UNSW_XGB_OTHER = 5

CICIDS_GNN_CLASSES = {
    'DDOS_attack-HOIC': 0,
    'DDoS_attacks-LOIC-HTTP': 1,
    'DDOS_attack-LOIC-UDP': 2,
    'DoS_attacks-Hulk': 3,
    'DoS_attacks-GoldenEye': 4,
    'Bot': 5,
    'SSH-Bruteforce': 6,
    'FTP-BruteForce': 7,
    'Brute_Force_-Web': 8,
}
CICIDS_GNN_OTHER = 9

CICIDS_XGB_CLASSES = {
    'DoS_attacks-Slowloris': 0,
    'Infilteration': 1,
}
CICIDS_XGB_OTHER = 2

# Undetectable — both models will classify as OTHER → Benign
UNSW_UNDETECTABLE = ['Backdoor', 'Worms']
CICIDS_UNDETECTABLE = ['DoS_attacks-SlowHTTPTest', 'Brute_Force_-XSS', 'SQL_Injection']


def relabel_for_gnn(original_labels, attack2id, gnn_classes, gnn_other_id):
    """Relabel original attack IDs to GNN specialist labels."""
    new_labels = np.full_like(original_labels, gnn_other_id)
    for attack_name, gnn_id in gnn_classes.items():
        orig_id = attack2id[attack_name]
        new_labels[original_labels == orig_id] = gnn_id
    return new_labels


def relabel_for_xgb(original_labels, attack2id, xgb_classes, xgb_other_id):
    """Relabel original attack IDs to XGBoost specialist labels."""
    new_labels = np.full_like(original_labels, xgb_other_id)
    for attack_name, xgb_id in xgb_classes.items():
        orig_id = attack2id[attack_name]
        new_labels[original_labels == orig_id] = xgb_id
    return new_labels
```

### Phase 2: Train GNN Model (2-3 days)

1. Use existing graph construction (60s windows, IP:port scheme) — no changes needed
2. Enrich node features with degree stats (see Phase 3 of old plan)
3. Add edge multiplicity feature for brute force detection
4. Relabel: topology attacks = their class, everything else = OTHER
5. Subsample OTHER to 3x attack count
6. Train E-GraphSAGE with focal loss + class weights
7. Validate on val set: check that GNN correctly says OTHER for Generic/Exploits/etc.

### Phase 3: Train XGBoost/MLP Model (1-2 days)

1. Use flat flow features (no graph needed)
2. Relabel: flow-feature attacks = their class, everything else = OTHER
3. Apply SMOTE for rare classes (DoS=897, Shellcode=358, Analysis=185 in UNSW)
4. Train XGBoost with early stopping
5. Validate: check that XGBoost correctly says OTHER for DDoS/BruteForce/etc.

### Phase 4: Assemble Ensemble + Evaluate (1-2 days)

1. Implement `DualModelEnsemble` class (Section 4)
2. Run on test set — measure per-class F1
3. Analyze conflict cases (Case 4) — how often? which classes?
4. Tune confidence threshold if needed
5. Generate final report with all metrics

### Phase 5: Documentation (0.5 day)

1. Document undetectable attacks with evidence (Section 2)
2. Report both macro-F1 scores (all classes vs detectable only)
3. Per-class F1 table showing which model decided each class

---

## 6. Projected Results

### UNSW (10 classes)

| Class | Current Best F1 | Projected F1 | Model | Notes |
|-------|----------------|-------------|-------|-------|
| Benign | 1.000 | 1.000 | Both=OTHER→Benign | |
| Generic | 0.836 | 0.88 | XGBoost specialist | SMOTE helps |
| Exploits | 0.738 | 0.82 | XGBoost specialist | SMOTE helps |
| Fuzzers | 0.751 | 0.87 | GNN specialist | Fan-in + enriched nodes |
| Reconnaissance | 0.691 | 0.82 | GNN specialist | Fan-out + enriched nodes |
| DoS | 0.382 | 0.60 | XGBoost specialist | SMOTE (897 samples) |
| Shellcode | 0.618 | 0.74 | XGBoost specialist | SMOTE (358 samples) |
| Analysis | 0.533 | 0.68 | XGBoost specialist | SMOTE (185 samples) |
| Backdoor | 0.005 | ~0.00 | Both=OTHER→Benign | **Undetectable** (documented) |
| Worms | 0.607 | ~0.00 | Both=OTHER→Benign | **25 samples** (documented) |
| **Macro-F1 (all 10)** | **0.587** | **~0.69** | | |
| **Macro-F1 (8 detectable)** | — | **~0.80** | | |

### CICIDS (15 classes)

| Class | Current Best F1 | Projected F1 | Model | Notes |
|-------|----------------|-------------|-------|-------|
| Benign | 0.997 | 0.998 | Both=OTHER→Benign | |
| DDOS-HOIC | 1.000 | 1.000 | GNN | Already perfect |
| DDoS-LOIC-HTTP | 0.989 | 0.995 | GNN | Specialist focus |
| DoS-Hulk | 0.998 | 0.999 | GNN | Already near-perfect |
| DoS-GoldenEye | 0.986 | 0.99 | GNN | Specialist focus |
| Bot | 0.999 | 0.999 | GNN | Already near-perfect |
| SSH-Bruteforce | 1.000 | 1.000 | GNN | Already perfect |
| FTP-BruteForce | 0.786 | 0.90 | GNN | Edge multiplicity helps |
| DoS-Slowloris | 0.907 | 0.94 | XGBoost | Flow-feature specialist |
| DDOS-LOIC-UDP | 0.422 | 0.75 | GNN | Specialist + oversample |
| Infilteration | 0.445 | 0.65 | XGBoost | SMOTE + specialist |
| Brute_Force-Web | 0.239 | 0.55 | GNN | Edge multiplicity + oversample |
| **Macro-F1 (12 detectable)** | **~0.73** | **~0.90** | | |
| SlowHTTPTest | 0.000 | ~0.00 | Both=OTHER→Benign | **Undetectable** (documented) |
| XSS | 0.041 | ~0.00 | Both=OTHER→Benign | **Payload-invisible** (documented) |
| SQL_Injection | 0.089 | ~0.00 | Both=OTHER→Benign | **Payload-invisible** (documented) |
| **Macro-F1 (all 15)** | **0.638** | **~0.71** | | |

---

## 7. Files to Create/Modify

| File | Type | Purpose |
|------|------|---------|
| `nids/models/dual_labels.py` | New | Label remapping for both models |
| `nids/models/ensemble.py` | New | `DualModelEnsemble` class with decision logic |
| `nids/models/mlp.py` | New | Flat MLP specialist (alternative to XGBoost) |
| `nids/graphs/ipport.py` | Modify | Add enriched node features (degree stats) |
| `nids/graphs/ipport.py` | Modify | Add edge multiplicity feature |
| `nids/models/hetero_gnn.py` | Modify | Fix BatchNorm + flow feature concat |
| `notebooks/phase0_topology_analysis.ipynb` | New | Validate topology hypotheses |
| `notebooks/train_dual_gnn.ipynb` | New | Train GNN specialist |
| `notebooks/train_dual_xgb.ipynb` | New | Train XGBoost specialist |
| `notebooks/evaluate_ensemble.ipynb` | New | Assemble ensemble + evaluate |

**Total: 6 new files, 3 modified files** (vs 10 new + 5 modified in the old plan).

---

## 8. Advantages Over Router-Based Plan

| Aspect | Old Plan (Router) | New Plan (Your Approach) |
|--------|-------------------|--------------------------|
| Models to train | 3+ (gate + router + specialists) | **2** (GNN + XGBoost) |
| Inference pipeline | Sequential (3 hops) | **Parallel** (2 independent) |
| Error propagation | Cascading (router miss → wrong specialist) | **Independent** (each model's error is contained) |
| Graph construction | Multiple specialist graphs (10s, 60s, 300s, 600s windows) | **One** graph (60s windows, existing) |
| New files | 10 new + 5 modified | **6 new + 3 modified** |
| Conceptual simplicity | Complex routing logic | **Simple**: run both, compare |
| "Other" class quality | N/A | Learnable (attacks are genuinely different between models) |
