# HATS: Hierarchical Graph Attention Network for Stock Movement Prediction -- Curated Technical Extraction

Source: Kim et al., Korea University, arXiv:1908.07999v3, 2019

---

## 1. PROBLEM FORMULATION

**Task:** Predict directional movement of individual stocks (node classification) and market indices (graph classification) using historical price data and corporate relational graphs.

**Input:** Sequence of historical price change rates for company i over a lookback window of length l:
```
[R^{t-l}_i, R^{t-l+1}_i, R^{t-l+2}_i, ..., R^{t-1}_i]
```

**Price change rate:**
```
R^t_i = (P^t_i - P^{t-1}_i) / P^{t-1}_i
```
where P^t_i is closing price of company i at time t.

**Output:** Three-class classification: [up, neutral, down], determined by two threshold values on price change ratio.

---

## 2. GRAPH CONSTRUCTION

### 2.1 Data Source

Corporate relational data collected from **Wikidata** (free collaborative knowledge base). Entities = companies. Properties = edges/relations.

- 431 companies from S&P 500 (after filtering those with no relations)
- 75 relation types total (direct relations + meta-paths)

### 2.2 Meta-Path Construction

The knowledge graph is heterogeneous (multiple node types: persons, organizations, countries, etc.). Only company-to-company edges are relevant. Since direct company-company edges are sparse, **meta-paths** (max 2 hops) are used to connect companies through intermediate entities.

**Example:**
- [Apple, Founded by, Steve Jobs] + [Steve Jobs, Board Member, Google]
- Meta-path: [Founded by, Board member] connects Apple to Google through shared node Steve Jobs

This converts a heterogeneous graph into a homogeneous graph with only company nodes.

### 2.3 Graph Notation

- V: set of vertices (companies)
- E: set of edges
- A: adjacency matrix, n x n, with A_ij = w_ij > 0
- D: degree matrix, d_ii = sum_j(a_ij)
- X in R^{n x f}: node feature matrix
- For spatial-temporal graph: X in R^{t x n x f}

### 2.4 Relation Types (Appendix -- 34 Direct/Meta-Path Property Codes)

| Code | Relation Name |
|------|--------------|
| P17 | Country |
| P31 | Instance of |
| P112 | Founded by |
| P121 | Item operated |
| P127 | Owned by |
| P131 | Located in administrative territorial entity |
| P138 | Named after |
| P155 | Follows |
| P156 | Followed by |
| P159 | Headquarters location |
| P166 | Award received |
| P169 | Chief executive officer |
| P176 | Manufacturer |
| P355 | Subsidiary |
| P361 | Part of |
| P400 | Platform |
| P414 | Stock Exchange |
| P452 | Industry |
| P463 | Member of |
| P488 | Chairperson |
| P495 | Country of origin |
| P625 | Coordinate location |
| P740 | Location of formation |
| P749 | Parent organization |
| P793 | Significant event |
| P1056 | Product or material produced |
| P1343 | Described by source |
| P1344 | Participant of |
| P1454 | Legal form |
| P1552 | Has quality |
| P1830 | Owner of |
| P1889 | Different from |
| P3320 | Board member |
| P5009 | Complies with |
| P6379 | Has works in the collection |

72 total relation combinations (direct edges + meta-paths) are enumerated in Appendix Table 6 (e.g., P452-P452 = Industry-Industry, P749-P127 = Parent organization-Owned by, etc.).

### 2.5 Relation Quality Findings

**Best 10 relations (by F1 on Phase 4, GCN single-relation test):**

| Relation Type | F1 |
|--------------|-----|
| Industry-Legal form | 0.3276 |
| Industry-Product or material produced | 0.3251 |
| Parent organization-Owner of | 0.3250 |
| Owned by-Subsidiary | 0.3247 |
| Parent organization | 0.3247 |
| Founded by-Founded by | 0.3245 |
| Follows | 0.3244 |
| Complies with-Complies with | 0.3242 |
| Owner of-Parent organization | 0.3241 |
| Subsidiary-Owner of | 0.3241 |

**Worst 10 relations:**

| Relation Type | F1 |
|--------------|-----|
| Legal form-Instance of | 0.3110 |
| Instance of-Legal form | 0.3082 |
| Location of formation-Country | 0.3070 |
| Country-Location of formation | 0.3053 |
| Stock Exchange | 0.2952 |
| Country of origin-Country | 0.2948 |
| Country-Country of origin | 0.2886 |
| Country-Country of origin | 0.2851 |
| Instance of-Instance of | 0.2748 |
| Stock Exchange-Stock Exchange | 0.2665 |

**Key insight:** Best-worst F1 gap is ~6%. Densely connected semantically weak relations (country, stock exchange) add noise. Dominant-subordinate relations (parent-subsidiary, ownership) and industry-product relations are most informative.

---

## 3. ARCHITECTURE: THREE-MODULE FRAMEWORK

```
Raw Price Data --> [Feature Extraction Module] --> e^t_i
                          |
                          v
e^t_i + Graph  --> [Relational Modeling Module (HATS)] --> e_bar_i
                          |
                          v
e_bar_i        --> [Task-Specific Module] --> Y_hat
```

### 3.1 Feature Extraction Module

Converts raw price change rate sequences into node feature vectors.

- **LSTM:** Used for individual stock prediction. 2 layers, hidden size 128. Trained with RMSProp optimizer.
- **GRU:** Used for index movement prediction (simpler, easier to train with deeper architecture, more consistent results).
- Lookback period: 50 days
- Input feature: price change rate (scalar per day), so input vector length = 50
- Output: e^t_i in R^f (f-dimensional feature vector for company i at time t)

### 3.2 Relational Modeling Module: HATS (Hierarchical Attention Network)

Two-level attention mechanism operating over a multi-relational graph.

**Notation:**
- e^t_i in R^f: feature vector of company i at time t (from feature extraction)
- N^{r_m}_i: set of neighboring nodes of i for relation type m
- e_{r_m} in R^d: learnable embedding vector for relation type m
- d: dimension of relation type embedding

#### Level 1: State (Node-Level) Attention

Selectively aggregates information from neighbors connected by the SAME relation type.

**Step 1 -- Concatenate** relation embedding, target node, and neighbor node representations:
```
x^{r_m}_{ij} in R^{2f+d} = CONCAT(e_{r_m}, e_i, e_j)    where j in N^{r_m}_i
```

**Step 2 -- Compute raw attention score:**
```
v_{ij} = x^{r_m}_{ij} * W_s + b_s                         (Eq. 3.1)
```
where W_s in R^{2f+d}, b_s in R are learnable parameters.

**Step 3 -- Softmax normalization over neighbors of same relation type:**
```
alpha^{r_m}_{ij} = exp(v_{ij}) / sum_{k in N^{r_m}_i} exp(v_{ik})    (Eq. 3.2)
```

**Step 4 -- Weighted aggregation to produce per-relation summary vector:**
```
s^{r_m}_i = sum_{j in N^{r_m}_i} alpha^{r_m}_{ij} * e_j              (Eq. 3.3)
```

s^{r_m}_i = summarized information from relation m for company i.

#### Level 2: Relation (Graph-Level) Attention

Selectively weights the summarized information ACROSS different relation types.

**Step 1 -- Concatenate** summarized relation vector, target node representation, and relation embedding:
```
x_tilde^{r_m}_i in R^{2f+d} = CONCAT(s^{r_m}_i, e_i, e_{r_m})
```

**Step 2 -- Compute raw relation attention score:**
```
v_tilde^{r_m}_i = x_tilde^{r_m}_i * W_r + b_r             (Eq. 3.4)
```
where W_r in R^{2d+f}, b_r in R are learnable parameters.

**Step 3 -- Softmax normalization across all non-empty relation types:**
```
alpha_tilde^{r_m}_i = exp(v_tilde^{r_m}_i) / sum_{k: |N^{r_k}_i| != 0} exp(v_tilde^{r_k}_i)    (Eq. 3.5)
```

Note: only relation types where node i has at least one neighbor participate.

**Step 4 -- Weighted aggregation across relations:**
```
e^r_i = sum_k alpha_tilde^{r_k}_i * s^{r_k}_i              (Eq. 3.6)
```

#### Residual Addition

Final updated node representation:
```
e_bar_i = e^r_i + e_i                                       (Eq. 3.7)
```

This is a **residual connection**: original node features + aggregated relational information.

### 3.3 Task-Specific Module: Individual Stock Prediction (Node Classification)

Linear transformation + softmax over updated node representation:
```
Y_hat_i = softmax(e_bar_i * W^n_p + b^n_p)                  (Eq. 3.8)
```
where W^n_p in R^{d x l}, b^n_p in R^l, l = number of movement classes (3).

**Loss function -- Cross-entropy over all companies:**
```
Loss_node = - sum_{i in Z_u} sum_{c=1}^{l} Y_{ic} * ln(Y_hat_{ic})    (Eq. 3.9)
```
where Z_u = all companies in dataset, Y_{ic} = ground truth one-hot label.

### 3.4 Task-Specific Module: Market Index Prediction (Graph Classification)

**Step 1 -- Mean pooling** over constituent company representations:
```
g^k_p = (1/n_k) * sum_{i in V_k} e_bar_i                   (Eq. 3.9b)
```
where V_k = set of constituent companies of index k, n_k = count.

**Step 2 -- Combine** pooled graph representation with index's own feature vector:
```
g^k = g^k_p + g^k_e                                         (Eq. 3.10)
```
where g^k_e = feature vector extracted from index's own historical price data using the feature extraction module (GRU).

**Step 3 -- Linear prediction:**
```
Y_hat = softmax(g^k * W^g_p + b^g_p)                        (Eq. 3.11)
```
where W^g_p in R^{d x l}, b^g_p in R^l.

**Loss function:**
```
Loss_graph = - sum_{c=1}^{l} Y_c * ln(Y_hat_c)              (Eq. 3.12)
```

---

## 4. GCN BASELINE (for comparison)

Two convolution layers + one prediction layer:
```
Y_GCN = softmax(A_hat * ReLU(A_hat * ReLU(A_hat * X' * W^(0)) * W^(1)) * W^(2))    (Eq. 5.6)
```
where:
```
A_hat = D_tilde^{-1/2} * A_tilde * D_tilde^{-1/2}
A_tilde = A + I  (adjacency with self-connections)
D_tilde = degree matrix of A_tilde
```

---

## 5. SPECTRAL GRAPH CONVOLUTION (Preliminary)

Spectral convolution filter:
```
f_theta(M, x) = U * M * U^T * x                             (Eq. 2.1)
```
where x in R^n, U = eigenvector matrix of graph Laplacian, M = diagonal matrix.

Chebyshev approximation:
```
M approx sum_{k=0}^{K} theta_k * T_k(Lambda_tilde)          (Eq. 2.2)
```
where Lambda_tilde = (2/lambda_max) * Lambda - I, lambda_max = largest eigenvalue of L.

Chebyshev recurrence: T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x), T_1(x) = x, T_0(x) = 1.

GCN uses K=1 simplification.

---

## 6. TRAINING PROCEDURE

### Hyperparameters
- Lookback period: 50 days
- Learning rate: 1e-3 to 1e-5
- Weight decay: 1e-4 to 1e-5
- Dropout: 0.1 to 0.9
- Activation: ReLU
- Optimizer: Adam (for HATS and most models), RMSProp (for standalone LSTM)
- Early stopping: based on F1 score on evaluation set
- All experiments repeated 5 times, results averaged

### Data Split
- 431 S&P 500 companies
- Period: 2013/02/08 to 2019/06/17 (1174 trading days)
- 12 phases, each: 250 days training / 50 days evaluation / 100 days testing
- Three-class labeling: two thresholds on price change ratio -> [up, neutral, down]

### Trading Strategy (for profitability evaluation)
- Neutralized portfolio: long top-15 companies by predicted up-probability, short top-15 by predicted down-probability

### Baseline Model Architectures
| Model | Architecture |
|-------|-------------|
| MLP | 2 hidden layers (16, 8 dims) + 1 prediction layer |
| CNN | 4 layers: 2 conv (filter sizes 32, 8; kernel 5) + 2 pooling |
| LSTM | 2 layers, hidden size 128, RMSProp optimizer |
| GCN | 2 conv layers + 1 prediction layer (Eq. 5.6), all relation types |
| GCN-Top20 | Same as GCN, adjacency from top-20 best-performing relation types only |
| TGC | Temporal Graph Convolution (Feng et al.); assigns weights to neighbors based on current state |

### Implementation
- Framework: TensorFlow
- Code: https://github.com/dmis-lab/hats

---

## 7. EVALUATION METRICS

### Classification
**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)                  (Eq. 5.3)
```

**Precision and Recall:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)                                      (Eq. 5.4)
```

**F1 (Macro):**
```
F1 = 2 * Recall * Precision / (Recall + Precision)           (Eq. 5.5)
```
Averaged across all classes for macro F1.

### Profitability
**Portfolio Return:**
```
Return^t_i = sum_{i in F^{t-1}} ((p^t_i - p^{t-1}_i) / p^{t-1}_i) * (-1)^{Action^{t-1}_i}    (Eq. 5.1)
```
where F^{t-1} = portfolio at t-1, Action = 0 for long, 1 for short.

**Annualized Sharpe Ratio:**
```
Sharpe_a = E[R_a - R_f] / std[R_a - R_f]                    (Eq. 5.2)
```
where R_f = 13-week Treasury bill rate.

---

## 8. BENCHMARK RESULTS

### 8.1 Individual Stock Prediction -- F1 Score (12 Phases)

| Phase | MLP | CNN | LSTM | GCN | GCN-Top20 | TGC | HATS |
|-------|------|------|------|------|-----------|------|------|
| 1 | 0.2876 | 0.3111 | 0.3173 | 0.2874 | 0.3161 | 0.3110 | **0.3314** |
| 2 | 0.2862 | 0.3208 | 0.3228 | 0.3068 | 0.3339 | 0.3088 | **0.3347** |
| 3 | 0.2763 | 0.2938 | 0.3064 | 0.2692 | **0.3113** | 0.2237 | 0.3100 |
| 4 | 0.2810 | 0.3176 | 0.3030 | 0.2940 | 0.3240 | 0.2970 | **0.3267** |
| 5 | 0.2873 | 0.3354 | 0.3333 | 0.3116 | 0.3450 | 0.3329 | **0.3496** |
| 6 | 0.2855 | 0.3265 | 0.3229 | 0.2914 | 0.3140 | 0.2798 | **0.3394** |
| 7 | 0.2876 | 0.3111 | 0.3173 | 0.2874 | 0.3161 | 0.3110 | **0.3314** |
| 8 | 0.2862 | 0.3208 | 0.3228 | 0.3068 | 0.3339 | 0.3088 | **0.3347** |
| 9 | 0.2741 | 0.2390 | 0.2793 | 0.2980 | 0.3160 | 0.2851 | **0.3219** |
| 10 | 0.2529 | 0.2128 | 0.3134 | 0.3002 | 0.3272 | 0.2951 | **0.3243** |
| 11 | 0.2500 | 0.2270 | 0.2997 | 0.2714 | 0.3031 | 0.2577 | **0.3091** |
| 12 | 0.2678 | 0.2921 | 0.2968 | 0.2956 | 0.3299 | 0.3270 | **0.3396** |
| **Avg** | 0.2769 | 0.2923 | 0.3113 | 0.2933 | 0.3225 | 0.2948 | **0.3294** |

### 8.2 Individual Stock Prediction -- Accuracy (12 Phases)

| Phase | MLP | CNN | LSTM | GCN | GCN-Top20 | TGC | HATS |
|-------|------|------|------|------|-----------|------|------|
| **Avg** | 0.3505 | 0.3762 | 0.3732 | 0.3877 | 0.3880 | 0.3910 | **0.3931** |

### 8.3 Individual Stock Prediction -- Avg Daily Return (%) and Sharpe Ratio

| Metric | MLP | CNN | LSTM | GCN | GCN-Top20 | TGC | HATS |
|--------|------|------|------|------|-----------|------|------|
| Avg Daily Return (%) | 0.0142 | -0.0147 | 0.0432 | 0.0285 | 0.0529 | 0.0376 | **0.0961** |
| Avg Sharpe (Annualized) | 0.4050 | -0.4821 | 1.1523 | 0.7803 | 0.9589 | 1.0026 | **1.9914** |

**HATS Sharpe ratio is 19.8% higher than next best baseline (TGC: 1.0026).**

### 8.4 Market Index Prediction (5 S&P Sector Indices, Averaged over 12 Phases)

**F1 Score:**

| Index | MLP | CNN | LSTM | GCN | TGC | HATS |
|-------|------|------|------|------|------|------|
| S5CONS | 0.2986 | 0.3013 | 0.3405 | 0.3410 | 0.3322 | **0.3758** |
| S5FINL | 0.3002 | 0.3157 | 0.2859 | 0.3040 | 0.3051 | **0.3148** |
| S5INFT | 0.2867 | 0.3036 | 0.3454 | 0.3423 | 0.3391 | **0.3518** |
| S5ENRS | 0.2785 | 0.3011 | 0.3109 | 0.2848 | 0.2736 | **0.3267** |
| S5UTIL | 0.2928 | 0.3025 | 0.2942 | 0.3111 | 0.2911 | **0.3256** |
| **Avg** | 0.2913 | 0.3049 | 0.3154 | 0.3166 | 0.3082 | **0.3389** |

**Accuracy:**

| Index | MLP | CNN | LSTM | GCN | TGC | HATS |
|-------|------|------|------|------|------|------|
| **Avg** | 0.3313 | 0.3537 | 0.3718 | 0.3771 | 0.3819 | **0.3948** |

---

## 9. KEY FINDINGS ON RELATION ATTENTION SCORES

From case study visualization of learned attention weights:

**Highest attention relations:** Dominant-subordinate relationships (parent organization-subsidiary, ownership chains), industrial dependency relations.

**Lowest attention relations:** Geographical features (country, location, headquarters location).

The model learns to suppress noisy, dense, semantically weak relations and amplify sparse, semantically meaningful corporate structural relations -- without manual selection.

---

## 10. COMPLETE EQUATION REFERENCE

| Eq. | Description |
|-----|-------------|
| 2.1 | Spectral convolution filter: f_theta(M,x) = U M U^T x |
| 2.2 | Chebyshev polynomial approximation of M |
| 2.4 | GraphSAGE neighborhood aggregation |
| 2.5 | GraphSAGE node update with concatenation |
| 3.1 | State attention raw score: v_{ij} = x^{r_m}_{ij} W_s + b_s |
| 3.2 | State attention softmax: alpha^{r_m}_{ij} |
| 3.3 | Per-relation summary vector: s^{r_m}_i = weighted sum of neighbor embeddings |
| 3.4 | Relation attention raw score: v_tilde^{r_m}_i = x_tilde^{r_m}_i W_r + b_r |
| 3.5 | Relation attention softmax: alpha_tilde^{r_m}_i |
| 3.6 | Aggregated relation representation: e^r_i = weighted sum of relation summaries |
| 3.7 | Residual update: e_bar_i = e^r_i + e_i |
| 3.8 | Individual stock prediction: Y_hat_i = softmax(e_bar_i W^n_p + b^n_p) |
| 3.9 | Node classification loss: cross-entropy over all companies |
| 3.9b | Mean graph pooling: g^k_p = mean of e_bar_i over constituent companies |
| 3.10 | Graph representation: g^k = g^k_p + g^k_e |
| 3.11 | Index prediction: Y_hat = softmax(g^k W^g_p + b^g_p) |
| 3.12 | Graph classification loss: cross-entropy |
| 5.1 | Portfolio return calculation |
| 5.2 | Annualized Sharpe ratio |
| 5.3 | Accuracy |
| 5.4 | Precision and Recall |
| 5.5 | F1 score |
| 5.6 | GCN baseline: Y_GCN = softmax(A_hat ReLU(A_hat ReLU(A_hat X' W^0) W^1) W^2) |

---

## 11. ARCHITECTURAL DIMENSIONS SUMMARY

| Component | Dimension |
|-----------|-----------|
| Input per company per day | Scalar (price change rate) |
| Lookback window | 50 days |
| LSTM layers / hidden | 2 / 128 |
| Feature vector e_i | R^f |
| Relation embedding e_{r_m} | R^d |
| State attention input x^{r_m}_{ij} | R^{2f+d} |
| State attention params W_s | R^{2f+d}, b_s in R |
| Relation attention input x_tilde^{r_m}_i | R^{2f+d} |
| Relation attention params W_r | R^{2d+f}, b_r in R |
| Prediction layer W^n_p | R^{d x l} (l=3 classes) |
| Number of relation types | 75 (72 enumerated in appendix) |
| Number of companies | 431 |
| Number of market indices | 5 (S5CONS, S5FINL, S5INFT, S5ENRS, S5UTIL) |
| Training days per phase | 250 |
| Eval days per phase | 50 |
| Test days per phase | 100 |
| Number of phases | 12 |
| Portfolio size | 15 long + 15 short |
