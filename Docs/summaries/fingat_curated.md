# FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks

Paper: IEEE TKDE 2021 | Authors: Yi-Ling Hsu, Yu-Che Tsai, Cheng-Te Li
Code: https://github.com/Roytsai27/Financial-GraphAttention

---

## 1. PROBLEM FORMULATION

**Task:** Given historical stock price time series for n stocks, recommend the top-K most profitable stocks (highest return ratio) for the next trading day.

**Return Ratio** of stock s_q at day j of week i:

```
R^{s_q}_{ij} = (p^{s_q}_{ij} - p^{s_q}_{i(j-1)}) / p^{s_q}_{i(j-1)}       [Eq. 1]
```

where p^{s_q}_{ij} is the stock price of stock s_q at day j of week i.

**Notation:**
- S = {s_1, s_2, ..., s_n}: universe of n stocks
- v^{s_q}_{ij}: feature vector of stock s_q at day j of week i
- pi_c: sector (industry category) that stock s_q belongs to
- D^{s_q}_i = {v^{s_q}_{i(j-d)}, ..., v^{s_q}_{i(j-1)}}: feature matrix from d past days
- d = 5 (one trading week)
- Goal: predict R^{s_q}_{(i+1)1} for every stock, then rank to produce top-K list

---

## 2. ARCHITECTURE OVERVIEW

FinGAT has three components executed sequentially:

```
[Stock-Level Modeling] --> [Sector-Level Modeling] --> [Multi-Task Learning]
```

### Stock-Level Modeling:
1. Feature Extraction --> per-stock daily feature vectors
2. Short-term Sequential Learning (Attentive GRU) --> weekly embedding a_i
3. Intra-sector Relation Modeling (GAT on fully-connected same-sector graph) --> g_i
4. Long-term Sequential Learning (Attentive GRU over past t weeks) --> tau^A_i(s_q), tau^G_i(s_q)

### Sector-Level Modeling:
5. Intra-sector Graph Pooling (MaxPool) --> sector embedding z_{pi_c}
6. Inter-sector Relation Modeling (GAT on fully-connected sector graph) --> tau_i(pi_c)

### Model Learning:
7. Embedding Fusion (concatenation + linear + ReLU) --> tau^F_i(s_q)
8. Multi-Task Prediction: return ratio ranking + movement classification

---

## 3. FEATURE ENGINEERING

### Basic Daily Features (per stock, per day j):
- open_j: opening price
- close_j: closing price
- high_j: highest price
- low_j: lowest price
- adjclose_j: adjusted closing price
- Return ratio (Eq. 1)

### Hand-Crafted Features:

**Price-ratio features:**
```
F_mu = (mu_j / close_j) - 1                                                [Eq. 2]
```
where mu in {open, high, low}. Produces 3 features.

**Moving-average features:**
```
F_phi = ( sum_{j=0}^{phi-1} adjclose_j / phi ) / adjclose_{j-1}            [Eq. 3]
```
where phi in {5, 10, 15, 20, 25, 30}. Produces 6 features.

**Total feature vector v^{s_q}_{ij}:** concatenation of all basic + hand-crafted features.

---

## 4. SHORT-TERM SEQUENTIAL LEARNING

Processes one week (d=5 trading days) of feature vectors for a single stock.

### GRU Encoding:
```
h^{s_q}_{ij} = GRU(v^{s_q}_{ij}, h^{s_q}_{i(j-1)})                       [Eq. 4]
```

### Attention over days within a week:
Given H_i = {h^{s_q}_{i1}, h^{s_q}_{i2}, ..., h^{s_q}_{id}}:

```
a^{s_q}_i = Attention(H_i) = sum_j alpha^{s_q}_j * h^{s_q}_{ij}           [Eq. 5]

alpha^{s_q}_j = softmax(tanh(W_0 * h^{s_q}_{ij}))                         [Eq. 6]
```

where W_0 is a learnable parameter matrix.

**Output:** a^{s_q}_i -- attentive short-term embedding for stock s_q at week i.

---

## 5. GRAPH CONSTRUCTION

### Intra-Sector Graph (one per sector):

For each sector pi_c, construct fully-connected graph:
```
G_{pi_c} = (M_{pi_c}, E_{pi_c})
```
- M_{pi_c}: set of stocks belonging to sector pi_c (nodes)
- E_{pi_c}: edges connecting every pair of stocks within the sector
- No self-loops (s_q != s_r)
- Node features initialized with a^{s_q}_i (short-term attentive embeddings)
- No pre-defined relationships required -- all edges exist, attention learns weights

### Inter-Sector Graph (single global graph):

Fully-connected graph over all sectors:
```
G_pi = (Z_pi, E_pi)
```
- Z_pi = {z_{pi_1}, z_{pi_2}, ..., z_{pi_c}}: all sector nodes with pooled embeddings
- E_pi: edges connecting every pair of sectors
- Node features initialized with z_{pi_c} (pooled from intra-sector graph)

---

## 6. GRAPH ATTENTION NETWORK (GAT) MECHANISM

### Intra-Sector GAT:

Applied to each intra-sector graph G_{pi_c}:

```
GAT(G_{pi_c}; s_q) = ReLU( sum_{s_n in Gamma(s_q)} beta_{qn} * W_1 * a^{s_n}_i )    [Eq. 7]
```

where:
- Gamma(s_q): neighbors of s_q in G_{pi_c} (all other same-sector stocks)
- W_1: learnable weight matrix
- beta_{qn}: attention weight from stock s_n to stock s_q

### Attention Weight Computation:

```
beta_{qn} = exp(LeakyReLU(r^T [W_2 a^{s_q}_i || W_2 a^{s_n}_i]))
             / sum_{s_n in Gamma(s_q)} exp(LeakyReLU(r^T [W_2 a^{s_q}_i || W_2 a^{s_n}_i]))
                                                                            [Eq. 8]
```

where:
- r: learnable vector (projects concatenation to scalar)
- ||: concatenation operator
- W_2: learnable weight matrix
- LeakyReLU activation before softmax normalization

**Output:** g^{s_q}_i = GAT(G_{pi_c}; s_q) -- graph-based embedding encoding intra-sector relations for stock s_q at week i.

### Inter-Sector GAT:

Same GAT mechanism applied to G_pi:
```
tau_i(pi_c) = GAT(G_pi, pi_c)                                              [Eq. 12]
```
Produces sector-level embeddings capturing sector-sector interactions.

---

## 7. LONG-TERM SEQUENTIAL LEARNING

Aggregates embeddings across the past t weeks (week i-t to week i-1).

### Two input sequences:

```
U^G_i(s_q) = {g^{s_q}_{i-t}, g^{s_q}_{i-t+1}, ..., g^{s_q}_{i-1}}   (graph-based)
U^A_i(s_q) = {a^{s_q}_{i-t}, a^{s_q}_{i-t+1}, ..., a^{s_q}_{i-1}}   (primitive)
                                                                            [Eq. 9]
```

### Attentive GRU applied separately to each:

```
tau^G_i(s_q) = Attention(U^G_i(s_q))    (long-term intra-sector embedding)
tau^A_i(s_q) = Attention(U^A_i(s_q))    (long-term primitive embedding)
                                                                            [Eq. 10]
```

Same Attentive GRU mechanism as in short-term learning (Eqs. 5-6), but operating over weeks instead of days.

---

## 8. INTRA-SECTOR GRAPH POOLING

Generates sector-level embedding from stock-level long-term embeddings.

```
z_{pi_c} = MaxPool({tau^G_i(s_q) | forall s_q in M_{pi_c}})               [Eq. 11]
```

**Element-wise max pooling:**
```
MaxPool(X) = [max({x_1 | forall x in X}), max({x_2 | forall x in X}), ..., max({x_epsilon | forall x in X})]
```
- x: epsilon-dimensional vector
- Takes element-wise maximum across all stock embeddings in the sector
- No learnable parameters

Output: Z_pi = {z_{pi_1}, z_{pi_2}, ..., z_{pi_c}} -- set of all sector embeddings.

---

## 9. EMBEDDING FUSION

Concatenate all three embedding streams and project:

```
tau^F_i(s_q) = ReLU([tau^G_i(s_q) || tau^A_i(s_q) || tau_i(pi_c)] * W_f)   [Eq. 13]
```

where:
- tau^G_i(s_q): long-term intra-sector embedding (stock-level, graph-aware)
- tau^A_i(s_q): long-term primitive embedding (stock-level, no graph)
- tau_i(pi_c): inter-sector embedding (sector-level, from sector GAT)
- pi_c is the sector that s_q belongs to
- W_f: learnable weight matrix
- ||: concatenation

---

## 10. MULTI-TASK PREDICTION HEADS

### Return Ratio Prediction (regression):
```
y_hat^{return}_i(s_q) = e_1^T * tau^F_i(s_q) + b_1                         [Eq. 14a]
```

### Movement Prediction (binary classification):
```
y_hat^{move}_i(s_q) = sigmoid(e_2^T * tau^F_i(s_q) + b_2)                  [Eq. 14b]
```

where e_1, e_2 in R^d are task-specific hidden vectors, b_1, b_2 are bias terms.

---

## 11. LOSS FUNCTIONS

### Combined Loss:
```
L_{FinGAT} = (1 - delta) * L_{rank} + delta * L_{move} + lambda * ||Theta||^2    [Eq. 15]
```

### Pairwise Ranking Loss (return ratio ranking):
```
L_{rank} = sum_i sum_{s_q} sum_{s_k} max(0, -Delta_hat * Delta)            [Eq. 16a]

where:
  Delta_hat = y_hat^{return}_i(s_q) - y_hat^{return}_i(s_k)   (predicted difference)
  Delta     = y^{return}_i(s_q)     - y^{return}_i(s_k)        (ground-truth difference)
```

Penalizes when predicted pairwise ordering disagrees with ground-truth ordering.

### Binary Cross-Entropy Loss (movement prediction):
```
L_{move} = -sum_i sum_{s_q} [ y^{move}_i * log(y_hat^{move}_i(s_q))
           + (1 - y^{move}_i) * log(1 - y_hat^{move}_i(s_q)) ]             [Eq. 16b]
```

where y^{move}_i = 1 if ground-truth return ratio is positive, 0 otherwise.

### Regularization:
- L2 regularization on all learnable parameters Theta
- lambda = 0.0001

---

## 12. FinGAT-NT VARIANT (No Sector Information)

When sector information is unavailable:

1. Remove sector-level modeling entirely (Section 4.2)
2. Remove intra-sector graph construction
3. Create single fully-connected graph G_T over ALL stocks
4. Apply GAT to G_T with node features a^{s_q}_i
5. Modified fusion (no inter-sector embedding):

```
tau^F_i(s_q) = ReLU([tau^G_i(s_q) || tau^A_i(s_q)] * W_f)                  [Eq. 19]
```

Caveat: O(n^2) complexity since all stocks are connected. Authors suggest selecting stock subsets.

---

## 13. TRAINING PROCEDURE

### Data Instance Construction:
- Every consecutive 16 trading days forms one instance
- 3 weeks (15 days) as input features, 16th day as prediction target
- Sliding window with stride 1 for daily predictions
- t = 3 weeks (default for long-term learning)

### Data Splits:

| Market | Stocks | Sectors | Train Days | Val Days | Test Days |
|--------|--------|---------|------------|----------|-----------|
| Taiwan Stock | 100 | 5 | 579 | 193 | 193 |
| S&P 500 | 424 | 9 | 579 | 193 | 193 |
| NASDAQ | 1026 | 112 | 756 | 252 | 237 |

Split ratio: 60% train / 20% validation / 20% test (Taiwan, S&P 500).
NASDAQ follows RankLSTM setting.

### Hyperparameters:
- GRU hidden dimension: 16
- GAT hidden dimension: 16
- Learning rate: searched in {0.0005, 0.001, 0.005}
- Batch size: 128
- delta (task balance): 0.01
- lambda (L2 regularization): 0.0001
- Optimizer: Adam
- Framework: PyTorch + PyTorch Geometric
- GPU: NVIDIA GeForce GTX 1080 Ti
- Results: average of 10 runs on test data

---

## 14. EVALUATION METRICS

### Mean Reciprocal Rank (MRR@K):
```
MRR@K = (1/K) * sum_{s_q in L@K(R_hat)} 1 / rank(R^{s_q}_{ij})           [Eq. 17]
```
where rank() returns the ground-truth rank of stock s_q.

### Precision@K:
```
Precision@K = |L@K(R_hat) intersect L@K(R)| / K                           [Eq. 18]
```

### Accuracy (ACC):
Correct binary movement predictions / total test instances.

---

## 15. BENCHMARK RESULTS

### Main Results (K=5, best gains over RankLSTM):

| Dataset | Metric | MLP | GRU | GRU+Att | FineNet | RankLSTM | FinGAT | Improvement |
|---------|--------|-----|-----|---------|---------|----------|--------|-------------|
| Taiwan Stock | MRR@5 | 0.2842 | 0.3115 | 0.3435 | 0.3742 | 0.3962 | **0.4391** | +10.83% |
| Taiwan Stock | Prec@5 | 0.0500 | 0.0622 | 0.0811 | 0.0867 | 0.1011 | **0.1133** | +12.08% |
| Taiwan Stock | ACC | 0.4514 | 0.4812 | 0.4948 | 0.5295 | 0.5539 | **0.5682** | +2.58% |
| S&P 500 | MRR@5 | 0.0844 | 0.1158 | 0.1321 | 0.1502 | 0.1736 | **0.1974** | +13.71% |
| S&P 500 | Prec@5 | 0.0172 | 0.0266 | 0.0301 | 0.0387 | 0.0398 | **0.0419** | +5.28% |
| S&P 500 | ACC | 0.535 | 0.5342 | 0.5353 | 0.539 | 0.5411 | **0.5425** | +0.26% |
| NASDAQ | MRR@5 | 6.13e-3 | 9.31e-3 | 9.28e-3 | 1.34e-2 | 1.81e-2 | **2.03e-2** | +12.15% |
| NASDAQ | Prec@5 | 1.12e-3 | 3.85e-3 | 3.94e-3 | 4.18e-3 | 4.35e-3 | **4.63e-3** | +6.43% |
| NASDAQ | ACC | 0.1374 | 0.1758 | 0.1539 | 0.1905 | 0.2318 | **0.2579** | +11.25% |

### Taiwan Stock K=10:
| Model | MRR@10 | Prec@10 |
|-------|--------|---------|
| RankLSTM | 0.7838 | 0.1717 |
| FinGAT | **0.8479** | **0.2022** |
| Improvement | +8.18% | +17.76% |

### Average K=5 improvement over RankLSTM across all 3 datasets:
- MRR: +12.23%
- Precision: +7.93%

### NASDAQ Weakness (K=10):
FinGAT underperforms RankLSTM at K=10 on NASDAQ (MRR: -12.22%, Prec: -7.96%).

---

## 16. ABLATION STUDY

### Taiwan Stock Dataset:

| Variant | MRR@5 | Prec@5 | MRR@10 | Prec@10 | ACC |
|---------|-------|--------|--------|---------|-----|
| Full FinGAT | **0.4391** | **0.1133** | **0.8479** | **0.2022** | **0.5682** |
| w/o intra-sector GAT | 0.3576 | 0.1033 | 0.7128 | 0.1317 | 0.5412 |
| w/o inter-sector GAT | 0.3950 | 0.1122 | 0.7464 | 0.1511 | 0.5509 |
| w/o multi-task learning | 0.4215 | 0.1127 | 0.8023 | 0.1856 | 0.5342 |
| w/ MSE (replace BCE) | 0.3486 | 0.0744 | 0.6867 | 0.1228 | 0.5078 |

### S&P 500 Dataset:

| Variant | MRR@5 | Prec@5 | MRR@10 | Prec@10 | ACC |
|---------|-------|--------|--------|---------|-----|
| Full FinGAT | **0.1974** | **0.0419** | **0.3357** | **0.0677** | **0.5425** |
| w/o intra-sector GAT | 0.1432 | 0.0355 | 0.2391 | 0.0500 | 0.5284 |
| w/o inter-sector GAT | 0.1369 | 0.0301 | 0.2382 | 0.0398 | 0.5371 |
| w/o multi-task learning | 0.1773 | 0.0409 | 0.2904 | 0.0581 | 0.5411 |
| w/ MSE (replace BCE) | 0.1072 | 0.0172 | 0.2203 | 0.0387 | 0.5177 |

### Key Ablation Findings:
- Removing intra-sector GAT causes the largest performance drop
- Removing multi-task learning causes the smallest drop
- Replacing BCE movement loss with MSE regression drastically hurts performance (flat loss curve hinders optimization)
- All three components contribute positively

---

## 17. HYPERPARAMETER SENSITIVITY

### Number of Training Weeks (t):
- t >= 3 weeks needed for good performance
- t = 1 or 2 leads to worse results
- Both short-term and long-term patterns are essential

### Embedding Dimension:
- Optimal: 16
- Too small (8): underfitting
- Too large (32, 64): overfitting

### Balancing Parameter delta:
- Optimal: delta = 0.01
- delta = 0: only ranking loss --> poor performance
- delta = 1: only movement loss --> poor performance
- Small delta favoring ranking loss works best
- Joint optimization outperforms either single task

---

## 18. COMPLETE FORWARD PASS (PSEUDOCODE)

```
Input: Stock price time series for all stocks over past t weeks

FOR each stock s_q:
    FOR each week i in {i-t, ..., i-1}:
        1. Extract daily features v_{ij} for days j in week i
        2. Run GRU over daily features: h_{ij} = GRU(v_{ij}, h_{i(j-1)})
        3. Attention aggregate: a_i = sum_j alpha_j * h_{ij}           # short-term

    FOR each week i in {i-t, ..., i-1}:
        4. Build intra-sector fully-connected graph G_{pi_c}
        5. Run GAT on G_{pi_c} with node features a_i:
           g_i = GAT(G_{pi_c}; s_q)                                    # intra-sector

    6. Long-term GRU+Attention over {a_{i-t},...,a_{i-1}} --> tau^A_i(s_q)
    7. Long-term GRU+Attention over {g_{i-t},...,g_{i-1}} --> tau^G_i(s_q)

FOR each sector pi_c:
    8. MaxPool over {tau^G_i(s_q) | s_q in pi_c} --> z_{pi_c}

9. Build inter-sector fully-connected graph G_pi
10. Run GAT on G_pi with node features z_{pi_c}:
    tau_i(pi_c) = GAT(G_pi, pi_c)                                      # inter-sector

FOR each stock s_q (belonging to sector pi_c):
    11. Fuse: tau^F_i = ReLU([tau^G_i || tau^A_i || tau_i(pi_c)] W_f)
    12. Predict: y_hat_return = e_1^T tau^F + b_1
    13. Predict: y_hat_move = sigmoid(e_2^T tau^F + b_2)

14. Rank stocks by y_hat_return, output top-K
15. Loss = (1-delta)*L_rank + delta*L_move + lambda*||Theta||^2
```

---

## 19. KEY DESIGN DECISIONS AND RATIONALE

| Decision | Rationale |
|----------|-----------|
| Fully-connected graphs (no pre-defined edges) | Real stock relationships are hidden/confidential; let attention learn which connections matter |
| Hierarchical: stock-level then sector-level | Captures both fine-grained (stock-stock) and coarse-grained (sector-sector) interactions |
| Two parallel embedding streams (a_i and g_i) | Disentangles pure sequential patterns from graph-augmented patterns |
| MaxPool for graph pooling | Simple, parameter-free aggregation from stock to sector level |
| Pairwise ranking loss (not MSE) | Directly optimizes relative ordering, which is what top-K recommendation needs |
| Multi-task with BCE (not MSE as auxiliary) | BCE has steeper loss curve than MSE, easier to optimize |
| delta = 0.01 (ranking dominant) | Primary task is ranking; movement prediction is auxiliary regularizer |
| GRU (not LSTM) | Compact; fewer parameters with comparable performance for financial time series |
| Separate short-term and long-term GRUs | Different temporal granularities require different learned dynamics |

---

## 20. ATTENTION WEIGHT ANALYSIS (EMPIRICAL FINDINGS)

### Intra-Sector Attention Patterns:
- "Consumer & Goods" sector (Taiwan): attention clusters into subgroups (supply chain correlation)
- "Energy" sector (S&P 500): attention weights nearly uniform (intense competition, high mutual influence)

### Inter-Sector Attention Patterns:
- Taiwan Stock: "Semiconductors" and "Construction" have highest cross-sector attention (economic fundamentals of Taiwan)
- S&P 500: "Energy" sector has highest correlation with all other sectors (oil price impact on entire market)

### Distribution Statistics:
- Inter-sector attention: right-skewed distribution (few dominant sectors influence many)
- Intra-sector attention: concentrated in (0.2, 0.4) range (no single stock dominates within a sector)
- Inter-sector variance > intra-sector variance (sector-level influence is more dynamic than stock-level)
