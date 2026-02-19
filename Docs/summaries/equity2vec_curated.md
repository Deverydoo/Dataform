# Equity2Vec: Technical Extraction

## Problem Formulation

Universe of n stocks: s_1, s_2, ..., s_n

**Target variable (log return):**

```
r_i_t = log(p_i_t / p_i_{t-1})
```

where p_i_t is the open price of stock i on day t.

**Regression objective:** Learn f such that r_i_{t+1} = f(x_i_{<=t}), where x_i_{<=t} is the historical feature vector for stock i up to time t.

---

## Architecture Overview

Three-component pipeline:

1. **Equity2Vec** -- graph-based cross-sectional embedding (static + dynamic)
2. **Heterogeneous data source integration** -- technical factors, news signals, cross-sectional embeddings concatenated
3. **Sequential model** -- LSTM with temporal attention producing return forecasts

---

## Component 1: Static Embedding via Matrix Factorization

### Co-occurrence Matrix

X in R^{n x n}, where X_{i,j} counts the number of news articles that mention both s_i and s_j within the global observation window (all news before testing phase).

### Objective Function (with bias and regularization)

Each stock i is assigned a latent vector e_i in R^d and scalar bias b_i.

```
J_s = sum_{i,j=1}^{n} (e_i^T * e_j + b_i + b_j - X_{i,j})^2 + beta * ||theta||^2
```

where:

```
||theta|| = (||e_i||^2 + ||e_j||^2) + b_i^2 + b_j^2
```

beta controls L2 regularization strength. This is analogous to GloVe-style factorization applied to stock co-occurrence rather than word co-occurrence.

**Output:** Static embedding e_i in R^d for each stock i.

---

## Component 2: Dynamic Embedding via Temporal Graph + Attention

### Temporal Stock Graph Construction

**Step 1 -- Daily graph:** G_tilde_t = {V, E_t} where V = {s_1, ..., s_n}. Edge (s_i, s_j) in E_t iff s_i and s_j are co-mentioned in news on date t. Edge weight = number of co-occurrences across all news on date t.

**Step 2 -- Smoothed graph:** Maintain sliding window of size w. Construct G_t by exponential moving average over G_tilde_{t-w}, ..., G_tilde_t, assigning larger weight to more recent graphs.

### k-Nearest Neighbor Filtering

For stock s_i, define S_t(i) as the set of k nearest neighbors in G_t, ranked by edge weight (co-occurrence magnitude). k is a hyperparameter.

### Attention-Based Neighbor Aggregation

**Final dynamic representation:**

```
c_i_t = sum_{j in S_t(i)} alpha_{ij} * e_j
```

**Constraint:**

```
sum_{j in S_t(i)} alpha_{ij} = 1
```

**Attention weights (softmax):**

```
alpha_{ij} = exp(f(e_i, e_j)) / sum_{l in S_t(i)} exp(f(e_i, e_l))
```

**Compatibility function (single-hidden-layer feedforward):**

```
f(e_i, e_j) = v_a^T * tanh(W_a * [e_i ; e_j] + b_a)
```

where v_a and W_a are weight matrices, b_a is bias; [;] denotes concatenation. All parameters learned end-to-end via backpropagation.

---

## Component 3: Heterogeneous Feature Integration + Sequential Model

### Feature Concatenation

```
h_i_t = [c_i_t, g_i_t]
```

where g_i_t is the dynamic input vector from technical factors and news signals, and [.,. ] denotes direct concatenation.

### LSTM Sequence Model

```
v_i_{t-T}, ..., v_i_{t-1}, v_i_t = LSTM(h_i_1, h_i_2, ..., h_i_T ; theta_l)
```

where theta_l denotes LSTM parameters.

### Temporal Attention Layer

```
r_i_{t+1} = sum_p beta_p * v_i_p
```

```
beta_p = exp(f(v_i_p, v_i_q)) / sum_q exp(f(v_i_p, v_i_q))
```

beta_p assigns importance weight to each historical time step p.

Empirical temporal attention distribution (5-day lookback):

| Lag     | -5 day | -4 day | -3 day | -2 day | -1 day |
|---------|--------|--------|--------|--------|--------|
| Weight  | 0.0055 | 0.0265 | 0.1662 | 0.3064 | 0.4954 |

---

## Loss Function

Mean squared error over m trading days and n stocks:

```
J = (1 / (m * n)) * sum_{i=1}^{n} sum_{t=1}^{m} (r_i_{t+1} - r_hat_i_{t+1})^2
```

Optimized via gradient descent (Adam optimizer).

---

## Training Pipeline (Algorithm 1)

```
Input:  Online news corpus, technical factors
Output: Predictions r_hat_i_{t+1}

Variables: Stock embedding matrix E, attention params alpha_{i,j},
           LSTM params theta_l, temporal attention params beta_i

1. Build global stock co-occurrence matrix X
2. Matrix factorization on X with loss J_s (Eq. 2) -> static embeddings
3. REPEAT until convergence:
     For each stock s_i in universe:
       For each timestamp t:
         a. Build temporal graph G_t
         b. For each neighbor j in S_t(i):
              Compute alpha_{i,j} via Eq. 5
         c. Compute c_i_t (final cross-sectional representation)
         d. Concatenate with g_i_t -> h_i_t (Eq. 7)
         e. Forecast r_hat_i_{t+1} via LSTM + temporal attention (Eq. 9)
       Compute prediction loss J (Eq. 10)
       Update all parameters via gradient of J
```

---

## Feature Descriptions

### Technical Factors (337 total)

All derived from price and dollar volume via mathematical calculation.

| Factor      | Description                                                |
|-------------|------------------------------------------------------------|
| EMA         | Exponential moving average over price or dollar volume     |
| RSI         | Magnitude of recent price changes (relative strength)      |
| ROC         | Price variation from one period to the next (rate of change)|
| Volume Std  | Standard deviation of volume                               |
| VCR         | Volume cumulative return                                   |

### News Signals

- Pre-trained Word2Vec on training news corpus to produce word embeddings
- News vector = average of all word vectors in the article
- Daily news vector for stock i on date t = average of all news vectors referencing stock i on date t
- Timestamps: use news signals on the next trading day or later (avoid look-ahead bias)

### Cross-Sectional Signals (Equity2Vec output)

- Static embedding e_i from global co-occurrence matrix factorization
- Dynamic representation c_i_t from temporal graph attention aggregation

---

## Data Specifications

- Universe: ~3,600 stocks, Chinese equity market, 2009--2018
- News corpus: 2.6 million articles (Sina Finance), 2009/01/01 to 2018/08/30
- Average stocks per news article: 2.94
- Prediction target: next 5-day returns (open-to-open)
- Out-of-sample: last 3 years (2016, 2017, 2018)

### Train/Validation/Test Split

- Training: 3 years
- Validation: 1 year
- Testing: 1 year
- Model retrained each testing year (rolling window)
- Gap of 10 trading days between train/val and val/test to prevent look-ahead

Example split: Train 2012-01-01 to 2014-12-31, Validate 2015-01-15 to 2015-12-16, Test 2016-01-01 to 2016-12-31.

---

## Hyperparameter Search Space

| Parameter                        | Search Range                  |
|----------------------------------|-------------------------------|
| Embedding dimension d            | {32, 64, 128, 256}            |
| Number of LSTM cells             | {2, 5, 10, 20}               |
| Number of neighbors k            | greedy search: 0 to all       |
| Sliding window w (temporal graph)| {2, 5, 10, 20, 60}           |
| Learning rate (Adam)             | {0.001, 0.01}                |
| Batch size                       | {128, 256}                   |

Selection method: standard grid search on validation set.

---

## Evaluation Metrics

### 1. Pearson Correlation

Between predicted returns r_hat_t and actual returns r_t. Direction matters more than magnitude for return forecasting.

### 2. t-Statistic (Newey-West adjusted)

For each day t, run cross-sectional regression:

```
r_t = beta_t * r_hat_t + epsilon
```

Test null hypothesis: beta_t = 0 for all t. Newey-West estimator adjusts for serial correlation in residuals.

### 3. Profit and Loss (PnL)

```
PnL_t = (1/n) * sum_i sign(r_hat_i_t) * r_i_t,    t = 1, ..., m
```

Portfolio construction:
- Top 20% strongest forecast signals selected
- Dollar position of stock i proportional to forecast r_hat_i_t
- Holding period: 5 days
- Evaluated on both long-short and long-index (long-only minus market index) portfolios

---

## Ablation Results Summary (2018 out-of-sample correlation)

Tested combinations: Tech only, News only, Tech+News, Tech+Graph, Graph+News, full Equity2Vec. Each component contributes; graph component yields largest marginal gain. Full model (all three signal types) achieves highest correlation.

---

## Embedding Variants Tested

| Variant     | Description                                              |
|-------------|----------------------------------------------------------|
| NoEmd       | Remove Equity2Vec module entirely                        |
| PoorMan     | Remove neighbor influence (static embedding only)        |
| SectEmd     | Replace with aggregated same-sector embedding            |
| OneEmd      | Replace with embedding from all neighbors (unfiltered)   |
| RadiusEmd   | Use radius-based neighbor selection instead of k-NN      |

Full Equity2Vec (k-NN + attention) outperforms all variants across all test years.
