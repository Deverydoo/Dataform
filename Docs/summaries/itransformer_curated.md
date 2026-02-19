# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
**Source:** ICLR 2024 (Liu et al., Tsinghua University / Ant Group)

---

## Core Insight

**What "inverted" means:** In vanilla Transformers for time series, each token is formed by multiple variates at the same timestamp, and attention is applied across temporal tokens (time steps). iTransformer *inverts* this: each token is the entire time series of a single variate, and attention is applied across variate tokens (channels).

**Why this works:**
- **Attention on variates** captures multivariate correlations explicitly. Each variate token contains a full series representation, so the attention score map A (N x N) directly reveals pairwise variate dependencies.
- **FFN on temporal dimension** learns nonlinear temporal representations per-variate. By the universal approximation theorem, FFN neurons act as filters that portray intrinsic time series properties (amplitude, periodicity, frequency spectrums). This is shared identically across variates, enabling transferability.
- **Why vanilla Transformer fails:** Embedding multivariate points of the same timestamp into one token fuses physically distinct measurements (e.g., temperature and rainfall) into indistinguishable channels. Time-aligned embedding fails when variates have systematic time lags. Permutation-invariant attention is improperly applied on the temporal dimension where sequence order matters.

**Key consequence:** iTransformer uses NO modified Transformer components. Only the architecture (which dimensions are fed to which components) is changed.

---

## iTransformer Architecture

### Problem Formulation

Given historical observations:
- **X** = {x_1, ..., x_T} in R^{T x N} with T time steps and N variates
- Predict future S time steps: **Y** = {x_{T+1}, ..., x_{T+S}} in R^{S x N}

Notation:
- X_{t,:} = simultaneously recorded time points at step t (all variates at time t)
- X_{:,n} = whole time series of variate indexed by n

### Architecture: Encoder-Only

No encoder-decoder structure. No position embedding needed (sequence order is implicitly stored in neuron permutation of FFN).

### Algorithm (Pseudocode)

```
Input:  X in R^{T x N}  (T = lookback length, N = number of variates)
Output: Y_hat in R^{S x N}  (S = prediction length)
Params: D = token dimension, L = number of iTransformer blocks

1.  X = X.transpose()                    # X becomes R^{N x T}
2.  H^0 = MLP(X)                         # Embedding: R^{N x T} -> R^{N x D}
3.  For l in {1, ..., L}:
4.      H^{l-1} = LayerNorm(H^{l-1} + SelfAttn(H^{l-1}))   # H in R^{N x D}
5.      H^l     = LayerNorm(H^{l-1} + FeedForward(H^{l-1})) # H in R^{N x D}
6.  End For
7.  Y_hat = MLP(H^L)                     # Projection: R^{N x D} -> R^{N x S}
8.  Y_hat = Y_hat.transpose()            # Y_hat becomes R^{S x N}
9.  Return Y_hat
```

### Component Details

**1. Input Representation (Embedding)**

Each variate's full lookback series X_{:,n} in R^T is independently embedded into a token of dimension D:

```
h^0_n = Embedding(X_{:,n})
```

- Embedding: R^T -> R^D implemented by MLP (multi-layer perceptron)
- This is the extreme case of Patching (patch size = entire series length)
- Result: H = {h_1, ..., h_N} in R^{N x D}, containing N variate tokens

**2. Layer Normalization (applied per-variate along feature dimension)**

```
LayerNorm(H) = { (h_n - Mean(h_n)) / sqrt(Var(h_n))  |  n = 1, ..., N }
```

- h_n in R^D is the token (series representation) of variate n
- Mean and Var computed over the D-dimensional feature vector of each variate
- Effect: normalizes each variate to Gaussian, reducing discrepancies from inconsistent measurements
- Acts as implicit non-stationarity handling (similar to RevIN)
- Contrast: vanilla Transformer normalizes across variates at same timestamp, causing oversmoothing

**3. Inverted Self-Attention (across variates)**

Given H = {h_1, ..., h_N} in R^{N x D}:

Linear projections produce:
```
Q, K, V in R^{N x d_k}
```
where d_k is the projected dimension.

Pre-Softmax attention score between variate tokens i and j:

```
A_{i,j} = (Q K^T / sqrt(d_k))_{i,j}  proportional to  q_i^T k_j
```

- q_i, k_j in R^{d_k} are query and key of variate tokens
- Since each token is normalized on its feature dimension, entries reveal variate-wise correlation
- Full score map A in R^{N x N} exhibits multivariate correlations
- Highly correlated variates are weighted more in the value aggregation

The attention output follows standard scaled dot-product attention:
```
Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
```

**4. Feed-Forward Network (applied per-variate for temporal patterns)**

- FFN is identically applied to each variate token independently
- Acts on the D-dimensional series representation
- Learns temporal patterns: amplitude, periodicity, frequency spectrums
- Neurons act as filters for intrinsic time series properties
- Equivalent to combining Channel Independence + linear forecasting in one operation
- Shared weights across variates enable transferability to unseen variates

**5. Output Projection**

```
Y_hat_{:,n} = Projection(h^L_n)
```

- Projection: R^D -> R^S implemented by MLP
- Maps each variate's final representation to its predicted future series

### iTransformers (Plug-in Framework)

The inverted architecture works with any attention variant:
- Vanilla Transformer -> iTransformer
- Reformer -> iReformer
- Informer -> iInformer
- Flowformer -> iFlowformer
- FlashAttention -> iFlashformer

Efficient attention mechanisms reduce complexity when variate count N is large (attention is O(N^2) on variates).

---

## Key Equations

### Equation 1: Overall iTransformer Formulation

```
h^0_n = Embedding(X_{:,n})
H^{l+1} = TrmBlock(H^l),          l = 0, ..., L-1
Y_hat_{:,n} = Projection(h^L_n)
```

Where:
- X_{:,n} in R^T = lookback series of variate n
- H = {h_1, ..., h_N} in R^{N x D} = N variate tokens of dimension D
- Superscript = layer index
- Embedding: R^T -> R^D (MLP)
- Projection: R^D -> R^S (MLP)
- TrmBlock = LayerNorm + Self-Attention + LayerNorm + FFN (with residual connections)

### Equation 2: Layer Normalization

```
LayerNorm(H) = { (h_n - Mean(h_n)) / sqrt(Var(h_n))  |  n = 1, ..., N }
```

Where:
- h_n in R^D = feature vector of variate token n
- Mean(h_n) = mean over D dimensions
- Var(h_n) = variance over D dimensions

### Attention Score Map

```
A_{i,j} = (Q K^T / sqrt(d_k))_{i,j}  proportional to  q_i^T k_j
```

Where:
- Q, K, V in R^{N x d_k} = projected queries, keys, values
- q_i, k_j in R^{d_k} = query/key of individual variate tokens
- A in R^{N x N} = multivariate correlation map
- d_k = projected dimension per head

### Pearson Correlation (for interpretability analysis)

```
rho_{xy} = sum_i (x_i - x_bar)(y_i - y_bar) / (sqrt(sum_i (x_i - x_bar)^2) * sqrt(sum_i (y_i - y_bar)^2))
```

Where:
- x_i, y_i in R = time points of paired variates
- Used to compare learned attention maps against ground-truth variate correlations

### Loss Function

- L2 loss (MSE) for model optimization

### Evaluation Metrics

- **MSE** (Mean Squared Error): lower is better
- **MAE** (Mean Absolute Error): lower is better

---

## Experimental Results

### Benchmark Datasets

| Dataset | Variates (N) | Prediction Lengths (S) | Dataset Size (Train, Val, Test) | Frequency | Domain |
|---------|-------------|----------------------|-------------------------------|-----------|--------|
| ETTh1, ETTh2 | 7 | {96, 192, 336, 720} | (8545, 2881, 2881) | Hourly | Electricity |
| ETTm1, ETTm2 | 7 | {96, 192, 336, 720} | (34465, 11521, 11521) | 15min | Electricity |
| Exchange | 8 | {96, 192, 336, 720} | (5120, 665, 1422) | Daily | Economy |
| Weather | 21 | {96, 192, 336, 720} | (36792, 5271, 10540) | 10min | Weather |
| ECL | 321 | {96, 192, 336, 720} | (18317, 2633, 5261) | Hourly | Electricity |
| Traffic | 862 | {96, 192, 336, 720} | (12185, 1757, 3509) | Hourly | Transportation |
| Solar-Energy | 137 | {96, 192, 336, 720} | (36601, 5161, 10417) | 10min | Energy |
| PEMS03 | 358 | {12, 24, 48, 96} | (15617, 5135, 5135) | 5min | Transportation |
| PEMS04 | 307 | {12, 24, 48, 96} | (10172, 3375, 3375) | 5min | Transportation |
| PEMS07 | 883 | {12, 24, 48, 96} | (16911, 5622, 5622) | 5min | Transportation |
| PEMS08 | 170 | {12, 24, 48, 96} | (10690, 3548, 3548) | 5min | Transportation |

Fixed lookback length T = 96 for all datasets.

### Main Results (Averaged MSE/MAE Across All Prediction Lengths, T=96)

| Dataset | iTransformer | PatchTST | DLinear | TimesNet | Crossformer | FEDformer | Autoformer | Stationary |
|---------|-------------|----------|---------|----------|-------------|-----------|------------|------------|
| ECL | **0.178/0.270** | 0.205/0.290 | 0.212/0.300 | 0.192/0.295 | 0.244/0.334 | 0.214/0.327 | 0.227/0.338 | 0.193/0.296 |
| ETT (Avg) | 0.383/0.399 | 0.381/0.397 | 0.442/0.444 | 0.391/0.404 | 0.685/0.578 | 0.408/0.428 | 0.465/0.459 | 0.471/0.464 |
| Exchange | 0.360/0.403 | 0.367/0.404 | **0.354/0.414** | 0.416/0.443 | 0.940/0.707 | 0.519/0.429 | 0.613/0.539 | 0.461/0.454 |
| Traffic | **0.428/0.282** | 0.481/0.304 | 0.625/0.383 | 0.620/0.336 | 0.550/0.304 | 0.610/0.376 | 0.628/0.379 | 0.624/0.340 |
| Weather | **0.258/0.278** | 0.259/0.281 | 0.265/0.317 | 0.259/0.287 | 0.259/0.315 | 0.309/0.360 | 0.338/0.382 | 0.288/0.314 |
| Solar-Energy | **0.233/0.262** | 0.270/0.307 | 0.330/0.401 | 0.301/0.319 | 0.641/0.639 | 0.291/0.381 | 0.885/0.711 | 0.261/0.381 |
| PEMS (Avg) | **0.119/0.218** | 0.217/0.305 | 0.320/0.394 | 0.148/0.246 | 0.220/0.304 | 0.224/0.327 | 0.614/0.575 | 0.151/0.249 |

iTransformer is particularly strong on high-dimensional datasets (ECL: 321 variates, Traffic: 862, PEMS: 170-883). On low-variate datasets like Exchange (8 variates), DLinear can be competitive.

### Per-Horizon Results (Selected Datasets, MSE)

**ECL (321 variates):**
| Horizon | iTransformer | PatchTST | TimesNet | DLinear |
|---------|-------------|----------|----------|---------|
| 96 | **0.148** | 0.181 | 0.168 | 0.197 |
| 192 | **0.162** | 0.188 | 0.184 | 0.196 |
| 336 | **0.178** | 0.204 | 0.198 | 0.209 |
| 720 | **0.225** | 0.246 | 0.220 | 0.245 |

**Traffic (862 variates):**
| Horizon | iTransformer | PatchTST | TimesNet | DLinear |
|---------|-------------|----------|----------|---------|
| 96 | **0.395** | 0.462 | 0.593 | 0.650 |
| 192 | **0.417** | 0.466 | 0.617 | 0.598 |
| 336 | **0.433** | 0.482 | 0.629 | 0.605 |
| 720 | **0.467** | 0.514 | 0.640 | 0.645 |

**Weather (21 variates):**
| Horizon | iTransformer | PatchTST | TimesNet | Crossformer |
|---------|-------------|----------|----------|-------------|
| 96 | 0.174 | 0.177 | 0.172 | **0.158** |
| 192 | 0.221 | 0.225 | 0.219 | **0.206** |
| 336 | 0.278 | 0.278 | 0.280 | **0.272** |
| 720 | 0.358 | 0.354 | 0.365 | **0.398** |

### Framework Generality: Performance Promotion via Inverting

Average MSE reduction when applying the inverted framework to existing Transformers:

| Model | Transformer | Reformer | Informer | Flowformer | Flashformer |
|-------|------------|----------|----------|------------|-------------|
| **Avg Promotion** | **38.9%** | **36.1%** | **28.5%** | **16.8%** | **32.2%** |

Per-dataset promotion (iTransformer vs vanilla Transformer, MSE reduction):

| Dataset | ETT | ECL | PEMS | Solar | Traffic | Weather |
|---------|-----|-----|------|-------|---------|---------|
| Promotion | 86.1% | 35.6% | 28.0% | 9.0% | 35.6% | 60.2% |

### Ablation Studies

Component arrangement experiments (averaged MSE across all horizons):

| Design | Variate Dim | Temporal Dim | ECL MSE | Traffic MSE | Weather MSE | Solar MSE |
|--------|------------|-------------|---------|-------------|-------------|-----------|
| **iTransformer** | **Attention** | **FFN** | **0.178** | **0.428** | 0.258 | **0.233** |
| Replace | Attention | Attention | 0.193 | 0.913 | 0.255 | 0.261 |
| Vanilla Transformer | FFN | Attention | 0.202 | 0.863 | 0.258 | 0.285 |
| Replace | FFN | FFN | 0.182 | 0.599 | **0.248** | 0.269 |
| w/o Attention | Attention | w/o | 0.189 | 0.456 | 0.261 | 0.258 |
| w/o FFN | w/o | FFN | 0.193 | 0.461 | 0.265 | 0.261 |

Key findings:
- Attention on temporal tokens (rows 2,3) causes severe degradation on Traffic (0.913, 0.863 vs 0.428)
- FFN-only (row 4) is competitive on low-variate Weather (0.248) but degrades on high-variate datasets
- Both attention and FFN contribute: removing either hurts performance

### Lookback Window Utilization

iTransformers benefit from increased lookback length T in {48, 96, 192, 336, 720}, while vanilla Transformers degrade. This validates that FFN on temporal dimension leverages extended history effectively, similar to linear forecasters.

### Variate Generalization

Training on 20% of variates (one of five random partitions), then forecasting all variates without fine-tuning:
- iTransformers show smaller performance increases (degradation from partial training) compared to CI-Transformers
- iTransformers predict all variates simultaneously; CI-Transformers must predict one-by-one (slow inference)
- FFN learns transferable temporal representations shared across variates

### Efficient Training Strategy

Randomly sample a subset of variates per batch during training; predict all variates at inference:
- 20% sampling ratio: performance remains comparable to full-variate training
- Memory footprint reduced significantly (e.g., Traffic from ~7.5GB to much lower)

### Efficiency Comparison (Input-96-Predict-96)

**Traffic (862 variates):**
| Model | Memory | Speed (ms/iter) | MSE |
|-------|--------|-----------------|-----|
| iTransformer | 7.50 GB | 265 ms | 0.428 |
| iTransformer (Efficient) | reduced | reduced | comparable |
| iFlowformer | 1.66 GB | 91 ms | ~0.524 |
| PatchTST | 8.58 GB | 635 ms | 0.481 |
| Crossformer | 9.74 GB | 702 ms | 0.550 |
| Transformer | 1.16 GB | 145 ms | 0.665 |
| DLinear | 0.91 GB | 60 ms | 0.625 |
| TiDE | 2.72 GB | 130 ms | 0.760 |

**Weather (21 variates):**
| Model | Memory | Speed (ms/iter) |
|-------|--------|-----------------|
| iTransformer | 0.88 GB | 30 ms |
| iTransformer (Efficient) | 0.87 GB | 29 ms |
| iFlowformer | 0.89 GB | 30 ms |
| PatchTST | 1.09 GB | 31 ms |
| Crossformer | 1.18 GB | 110 ms |
| Transformer | 1.09 GB | 85 ms |
| DLinear | 0.83 GB | 28 ms |

Attention complexity is O(N^2) where N = number of tokens. For iTransformer N = variates; for vanilla Transformer N = time steps. When variates >> time steps (Traffic: 862 vs 96), iTransformer is less efficient unless using linear attention or efficient training strategy.

### Robustness (5 Random Seeds)

| Dataset | Horizon | MSE (mean +/- std) |
|---------|---------|-------------------|
| ECL | 96 | 0.148 +/- 0.000 |
| ECL | 192 | 0.162 +/- 0.002 |
| ECL | 336 | 0.178 +/- 0.000 |
| ECL | 720 | 0.225 +/- 0.006 |
| Traffic | 96 | 0.395 +/- 0.001 |
| Traffic | 192 | 0.417 +/- 0.002 |
| Traffic | 336 | 0.433 +/- 0.004 |
| Traffic | 720 | 0.467 +/- 0.003 |
| Weather | 96 | 0.174 +/- 0.000 |
| Weather | 192 | 0.221 +/- 0.002 |
| Weather | 336 | 0.278 +/- 0.002 |
| Weather | 720 | 0.358 +/- 0.000 |
| Solar-Energy | 96 | 0.203 +/- 0.002 |

Standard deviations are very small, indicating stable performance.

### CKA Similarity Analysis

Centered Kernel Alignment (CKA) between first and last block output features:
- iTransformers consistently achieve higher CKA similarity than vanilla Transformers
- Higher CKA similarity correlates with better forecasting performance for time series (a low-level generative task)
- Clear division line: all iTransformers cluster at high CKA / low MSE; all vanilla Transformers at low CKA / high MSE

### Attention Map Interpretability

On Solar-Energy dataset:
- Shallow layer (Layer 1): learned attention map resembles Pearson correlations of lookback (input) series
- Deep layer (Layer L): learned attention map resembles Pearson correlations of future (target) series
- Demonstrates encoding of past and decoding for future happens progressively through layers

---

## Hyperparameters

### Model Configuration

| Parameter | Value / Range |
|-----------|-------------|
| Number of iTransformer blocks (L) | {2, 3, 4} |
| Token / hidden dimension (D) | {256, 512} |
| Optimizer | Adam |
| Learning rate | {1e-3, 5e-4, 1e-4} |
| Loss function | L2 (MSE) |
| Batch size | 32 |
| Training epochs | 10 |
| Lookback length (T) | 96 (fixed for main experiments) |
| Prediction lengths (S) | {96, 192, 336, 720} for most; {12, 24, 36, 48} for PEMS |
| Position embedding | None (not needed) |
| Architecture | Encoder-only |
| Hardware | Single NVIDIA P100 16GB GPU |
| Framework | PyTorch |

### Hyperparameter Sensitivity Findings

- **Learning rate:** Most impactful; requires careful selection when variate count is large (ECL, Traffic). Values of 1e-3 can cause instability on high-dimensional datasets.
- **Block number (L):** Not necessarily better when larger. L=2 or L=3 often sufficient. L=4 can overfit on smaller datasets.
- **Hidden dimension (D):** Not necessarily better when larger. D=256 or D=512 typical. D=1024 can overfit.
- Sensitivity tests performed at T=96, S=96.

### Data Processing

- Train/validation/test split: chronological order (no data leakage)
- Same data processing protocol as TimesNet benchmark
- All baseline models reproduced using TimesNet benchmark repository with original configurations

---

## Architectural Properties

1. **No position embedding needed:** Sequence order is implicitly captured by neuron ordering in the FFN, since FFN operates on the temporal (series) dimension.

2. **Variable variate count:** Attention accepts variable numbers of tokens, so the model can train on N variates and infer on M != N variates.

3. **Efficient training via variate sampling:** Random subset of variates per batch; full variate prediction at inference. Memory scales with sampled count, not total variate count.

4. **Univariate degeneration:** When N=1, iTransformer reduces to a stackable linear forecaster (attention degenerates with a single token).

5. **Complexity:** Attention is O(N^2) on variates (vs O(T^2) on time steps for vanilla). Beneficial when N < T; requires efficient attention when N >> T.
