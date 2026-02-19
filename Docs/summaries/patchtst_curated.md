# PatchTST: A Time Series is Worth 64 Words (ICLR 2023) -- Technical Extraction

**Source:** Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR 2023.

---

## 1. PatchTST Architecture

### 1.1 Problem Formulation

Given a multivariate time series with look-back window L: `(x_1, ..., x_L)` where each `x_t` is a vector of dimension M (number of variates/channels), forecast T future values `(x_{L+1}, ..., x_{L+T})`.

### 1.2 Channel-Independence

- The multivariate input `x in R^{M x L}` is split into M univariate series: `x^(i) in R^{1 x L}` for `i = 1, ..., M`.
- Each univariate series is fed **independently** into the same shared Transformer backbone.
- All channels share the same embedding weights and Transformer weights.
- The Transformer produces per-channel predictions: `x_hat^(i) = (x_hat^(i)_{L+1}, ..., x_hat^(i)_{L+T}) in R^{1 x T}`.
- Final output: concatenate all M channel predictions into `x_hat in R^{M x T}`.

**Why channel-independence works better than channel-mixing:**
1. **Adaptability:** Each series generates its own attention maps; different series can learn different attention patterns. Channel-mixing forces all series to share the same attention pattern.
2. **Data efficiency:** Channel-mixing models require substantially more training data to learn cross-channel + temporal correlations jointly. On typical benchmark dataset sizes, channel-independent models converge faster.
3. **Reduced overfitting:** Channel-mixing models overfit after a few epochs on test loss; channel-independent models continue to optimize.
4. **Noise robustness:** Noise in one channel does not propagate to other channels through the embedding space.

### 1.3 Patching

Each univariate series `x^(i)` is divided into patches (overlapped or non-overlapped).

**Variables:**
- `P` = patch length (number of time steps per patch)
- `S` = stride (non-overlapping region between consecutive patches)
- `L` = look-back window length
- `N` = number of patches

**Patching formula:**

```
N = floor((L - P) / S) + 2
```

**Note:** The paper pads S repeated copies of the last value `x^(i)_L` to the end of the original sequence before patching (hence +2 instead of +1).

**Output of patching:**

```
x^(i)_p in R^{P x N}
```

where each column is one patch of length P.

**Complexity reduction:** Number of input tokens reduces from L to approximately L/S. Attention map complexity reduces **quadratically** from O(L^2) to O((L/S)^2).

**Three benefits of patching:**
1. Local semantic information is retained in the embedding (unlike point-wise tokens).
2. Computation and memory usage of attention maps are quadratically reduced.
3. The model can attend to longer history within the same computational budget.

### 1.4 Patch Embedding (Linear Projection)

Patches are mapped to the Transformer latent space of dimension D via a trainable linear projection:

```
W_p in R^{D x P}     (projection matrix)
W_pos in R^{D x N}   (learnable additive positional encoding)

x^(i)_d = W_p * x^(i)_p + W_pos
```

where `x^(i)_d in R^{D x N}` is the embedded input fed to the Transformer encoder.

**Positional encoding:** Learnable additive position encoding (not sinusoidal).

### 1.5 Transformer Encoder

Uses a **vanilla Transformer encoder** (no custom attention modifications).

**Multi-head attention:** For each head `h = 1, ..., H`:

```
Q^(i)_h = (x^(i)_d)^T * W^Q_h    (query)
K^(i)_h = (x^(i)_d)^T * W^K_h    (key)
V^(i)_h = (x^(i)_d)^T * W^V_h    (value)
```

where:
- `W^Q_h, W^K_h in R^{D x d_k}` (query and key projection matrices)
- `W^V_h in R^{D x D}` (value projection matrix)
- `d_k` = dimension per head

**Scaled dot-product attention:**

```
(O^(i)_h)^T = Attention(Q^(i)_h, K^(i)_h, V^(i)_h) = Softmax(Q^(i)_h * (K^(i)_h)^T / sqrt(d_k)) * V^(i)_h
```

where `O^(i)_h in R^{D x N}` is the attention output for head h.

**Key architectural details:**
- Uses **BatchNorm** (not LayerNorm). BatchNorm outperforms LayerNorm in time series Transformer (per Zerveas et al., 2021).
- Feed forward network with residual connections.
- Output representation: `z^(i) in R^{D x N}`.

### 1.6 Prediction Head (Supervised)

After the Transformer encoder produces `z^(i) in R^{D x N}`:

```
Flatten layer -> Linear head -> x_hat^(i) in R^{1 x T}
```

The flatten layer concatenates the D-dimensional representations of all N patches into a single vector of size D*N, then a linear layer maps this to T output values.

### 1.7 Efficient Implementation

The batch of samples `x in R^{M x L}` with batch size B produces a 4D tensor of size `B x M x P x N`. By reshaping to `(B * M) x P x N`, any standard Transformer implementation can consume this batch directly. No special operators needed.

For channel-mixing variant (ablation): reshape to `B x (M * P) x N` instead.

---

## 2. Key Equations

### 2.1 Patching Formula

```
N = floor((L - P) / S) + 2
```
- L = look-back window length
- P = patch length
- S = stride
- N = number of patches
- Padding: S copies of x^(i)_L appended to sequence end before patching

### 2.2 Patch Embedding

```
x^(i)_d = W_p * x^(i)_p + W_pos
```
- W_p in R^{D x P}: trainable linear projection
- W_pos in R^{D x N}: learnable additive positional encoding
- x^(i)_p in R^{P x N}: patched input
- x^(i)_d in R^{D x N}: embedded input to Transformer

### 2.3 Multi-Head Attention

```
Q^(i)_h = (x^(i)_d)^T * W^Q_h
K^(i)_h = (x^(i)_d)^T * W^K_h
V^(i)_h = (x^(i)_d)^T * W^V_h

(O^(i)_h)^T = Softmax(Q^(i)_h * (K^(i)_h)^T / sqrt(d_k)) * V^(i)_h
```

### 2.4 Loss Function (Supervised)

MSE loss averaged over all M channels:

```
L = E_x [ (1/M) * sum_{i=1}^{M} || x_hat^(i)_{L+1:L+T} - x^(i)_{L+1:L+T} ||_2^2 ]
```

- x_hat^(i)_{L+1:L+T}: predicted values for channel i
- x^(i)_{L+1:L+T}: ground truth values for channel i
- M: number of channels/variates

### 2.5 Instance Normalization (RevIN)

Normalizes each time series instance `x^(i)` to zero mean and unit standard deviation **before patching**. The mean and standard deviation are added back to the output prediction.

```
x^(i)_norm = (x^(i) - mean(x^(i))) / std(x^(i))    [applied before patching]
x_hat^(i)_final = x_hat^(i) * std(x^(i)) + mean(x^(i))    [reversed at output]
```

Purpose: Mitigate distribution shift between training and testing data (Kim et al., 2022).

---

## 3. Self-Supervised Pre-training

### 3.1 Masked Patch Prediction Objective

Uses masked autoencoding at the **patch level** (not time-step level).

**Key design choices:**
- Patches are **non-overlapping** for self-supervised learning (S = P), so that observed patches do not contain information about masked patches.
- A subset of patch indices is selected uniformly at random.
- Selected patches are masked by setting their values to **zero**.
- The model is trained to **reconstruct the masked patches**.

**Why patch-level masking is better than point-level masking:**
- Point-level masked values can be easily inferred by interpolating with immediate neighboring time values, which does not require high-level understanding.
- Patch-level masking forces the model to learn abstract representations of the entire signal.

### 3.2 Masking Strategy

- **Masking ratio:** 40% of patches are masked with zero values.
- Selection: Uniform random selection of patch indices.
- Each time series has its own latent representation, cross-learned via shared weights.

### 3.3 Pre-training Architecture

Same Transformer encoder as supervised settings, but:
- The supervised prediction head is **removed**.
- A `D x P` linear layer is attached to reconstruct masked patches.

### 3.4 Pre-training Loss

MSE loss between reconstructed masked patches and original values of those patches.

### 3.5 Fine-tuning Procedure

**Pre-training:** 100 epochs on unlabeled data.

**Option (a) -- Linear probing:**
- Freeze the entire network except the model head.
- Train the model head only for 20 epochs.

**Option (b) -- End-to-end fine-tuning (two-step strategy):**
1. Linear probing for 10 epochs (update model head only, freeze rest).
2. End-to-end fine-tuning of the entire network for 20 epochs.

This two-step strategy (linear probing then fine-tuning) outperforms fine-tuning directly (per Kumar et al., 2022).

### 3.6 Transfer Learning Capability

- Pre-training data can contain a **different number of time series** than downstream data (because each series has its own latent representation via shared weights).
- Example: Pre-train on Electricity (321 series), transfer to Weather (21 series) or ETTh1 (7 series).

---

## 4. Hyperparameters

### 4.1 Default Model Parameters

| Parameter | Default Value | Small Dataset Value |
|-----------|---------------|---------------------|
| Encoder layers | 3 | 3 |
| Number of heads (H) | 16 | 4 |
| Latent dimension (D) | 128 | 16 |
| Feed-forward dimension (F) | 256 | 128 |
| FF activation | GELU | GELU |
| Dropout probability | 0.2 | 0.2 |
| Normalization | BatchNorm | BatchNorm |

**Small datasets** = ILI, ETTh1, ETTh2 (reduced parameters to mitigate overfitting).

**Feed-forward network:** 2 linear layers: D -> F -> D (i.e., 128 -> 256 -> 128).

### 4.2 Patch and Input Configuration

| Variant | Look-back (L) | Patch length (P) | Stride (S) | Number of patches (N) |
|---------|---------------|-------------------|------------|----------------------|
| PatchTST/64 | 512 | 16 | 8 | 64 |
| PatchTST/42 | 336 | 16 | 8 | 42 |

### 4.3 Self-Supervised Pre-training Parameters

| Parameter | Value |
|-----------|-------|
| Input sequence length | 512 |
| Patch size | 12 |
| Stride | 12 (non-overlapping) |
| Number of patches | 42 |
| Masking ratio | 40% |
| Pre-training epochs | 100 |
| Linear probing epochs | 20 (option a) or 10 (option b step 1) |
| Fine-tuning epochs | 20 (option b step 2) |
| Reconstruction head | D x P linear layer |

### 4.4 Prediction Lengths

- **ILI dataset:** T in {24, 36, 48, 60}
- **All other datasets:** T in {96, 192, 336, 720}

### 4.5 Look-back Windows for Baselines

- Transformer-based models default: L = 96
- DLinear default: L = 336
- Baselines re-run with L in {24, 48, 96, 192, 336, 720}, best selected
- ILI baselines: L in {24, 36, 48, 60, 104, 144}

### 4.6 Robustness to Model Parameters

Tested 6 combinations: (layers, D) = (3,128), (3,256), (4,128), (4,256), (5,128), (5,256) with F = 2D. Results show minimal variance across combinations (except ILI which shows higher sensitivity).

### 4.7 Random Seed

Default seed: 2021. Robustness validated with seeds {2019, 2020, 2021, 2022, 2023}. Standard deviations are small across all datasets.

---

## 5. Experimental Results

### 5.1 Benchmark Datasets

| Dataset | Features (M) | Timesteps | Frequency |
|---------|-------------|-----------|-----------|
| Weather | 21 | 52,696 | - |
| Traffic | 862 | 17,544 | - |
| Electricity | 321 | 26,304 | Hourly |
| ILI | 7 | 966 | Weekly |
| ETTh1 | 7 | 17,420 | Hourly |
| ETTh2 | 7 | 17,420 | Hourly |
| ETTm1 | 7 | 69,680 | 15-min |
| ETTm2 | 7 | 69,680 | 15-min |

**Large datasets** (Weather, Traffic, Electricity): More stable results, less susceptible to overfitting.

### 5.2 Multivariate Forecasting Results (Supervised, Table 3)

**Overall improvement of PatchTST/64 vs best Transformer baseline:**
- MSE reduction: **21.0%**
- MAE reduction: **16.7%**

**Overall improvement of PatchTST/42 vs best Transformer baseline:**
- MSE reduction: **20.2%**
- MAE reduction: **16.4%**

#### Weather Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer | Informer |
|---|-------------|-------------|---------|-----------|------------|----------|
| 96 | 0.149/0.198 | 0.152/0.199 | 0.176/0.237 | 0.238/0.314 | 0.249/0.329 | 0.354/0.405 |
| 192 | 0.194/0.241 | 0.197/0.243 | 0.220/0.282 | 0.275/0.329 | 0.325/0.370 | 0.419/0.434 |
| 336 | 0.245/0.282 | 0.249/0.283 | 0.265/0.319 | 0.339/0.377 | 0.351/0.391 | 0.583/0.543 |
| 720 | 0.314/0.334 | 0.320/0.335 | 0.323/0.362 | 0.389/0.409 | 0.415/0.426 | 0.916/0.705 |

#### Traffic Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer | Informer |
|---|-------------|-------------|---------|-----------|------------|----------|
| 96 | 0.360/0.249 | 0.367/0.251 | 0.410/0.282 | 0.576/0.359 | 0.597/0.371 | 0.733/0.410 |
| 192 | 0.379/0.256 | 0.385/0.259 | 0.423/0.287 | 0.610/0.380 | 0.607/0.382 | 0.777/0.435 |
| 336 | 0.392/0.264 | 0.398/0.265 | 0.436/0.296 | 0.608/0.375 | 0.623/0.387 | 0.776/0.434 |
| 720 | 0.432/0.286 | 0.434/0.287 | 0.466/0.315 | 0.621/0.375 | 0.639/0.395 | 0.827/0.466 |

#### Electricity Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer | Informer |
|---|-------------|-------------|---------|-----------|------------|----------|
| 96 | 0.129/0.222 | 0.130/0.222 | 0.140/0.237 | 0.186/0.302 | 0.196/0.313 | 0.304/0.393 |
| 192 | 0.147/0.240 | 0.148/0.240 | 0.153/0.249 | 0.197/0.311 | 0.211/0.324 | 0.327/0.417 |
| 336 | 0.163/0.259 | 0.167/0.261 | 0.169/0.267 | 0.213/0.328 | 0.214/0.327 | 0.333/0.422 |
| 720 | 0.197/0.290 | 0.202/0.291 | 0.203/0.301 | 0.233/0.344 | 0.236/0.342 | 0.351/0.427 |

#### ILI Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer | Informer |
|---|-------------|-------------|---------|-----------|------------|----------|
| 24 | 1.319/0.754 | 1.522/0.814 | 2.215/1.081 | 2.624/1.095 | 2.906/1.182 | 4.657/1.449 |
| 36 | 1.579/0.870 | 1.430/0.834 | 1.963/0.963 | 2.516/1.021 | 2.585/1.038 | 4.650/1.463 |
| 48 | 1.553/0.815 | 1.673/0.854 | 2.130/1.024 | 2.505/1.041 | 3.024/1.145 | 5.004/1.542 |
| 60 | 1.470/0.788 | 1.529/0.862 | 2.368/1.096 | 2.742/1.122 | 2.761/1.114 | 5.071/1.543 |

#### ETTh1 Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer |
|---|-------------|-------------|---------|-----------|
| 96 | 0.370/0.400 | 0.375/0.399 | 0.375/0.399 | 0.376/0.415 |
| 192 | 0.413/0.429 | 0.414/0.421 | 0.405/0.416 | 0.423/0.446 |
| 336 | 0.422/0.440 | 0.431/0.436 | 0.439/0.443 | 0.444/0.462 |
| 720 | 0.447/0.468 | 0.449/0.466 | 0.472/0.490 | 0.469/0.492 |

#### ETTh2 Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer |
|---|-------------|-------------|---------|-----------|
| 96 | 0.274/0.337 | 0.274/0.336 | 0.289/0.353 | 0.332/0.374 |
| 192 | 0.341/0.382 | 0.339/0.379 | 0.383/0.418 | 0.407/0.446 |
| 336 | 0.329/0.384 | 0.331/0.380 | 0.448/0.465 | 0.400/0.447 |
| 720 | 0.379/0.422 | 0.379/0.422 | 0.605/0.551 | 0.412/0.469 |

#### ETTm1 Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer |
|---|-------------|-------------|---------|-----------|
| 96 | 0.293/0.346 | 0.290/0.342 | 0.299/0.343 | 0.326/0.390 |
| 192 | 0.333/0.370 | 0.332/0.369 | 0.335/0.365 | 0.365/0.415 |
| 336 | 0.369/0.392 | 0.366/0.392 | 0.369/0.386 | 0.392/0.425 |
| 720 | 0.416/0.420 | 0.420/0.424 | 0.425/0.421 | 0.446/0.458 |

#### ETTm2 Dataset (MSE / MAE)

| T | PatchTST/64 | PatchTST/42 | DLinear | FEDformer |
|---|-------------|-------------|---------|-----------|
| 96 | 0.166/0.256 | 0.165/0.255 | 0.167/0.260 | 0.180/0.271 |
| 192 | 0.223/0.296 | 0.220/0.292 | 0.224/0.303 | 0.252/0.318 |
| 336 | 0.274/0.329 | 0.278/0.329 | 0.281/0.342 | 0.324/0.364 |
| 720 | 0.362/0.385 | 0.367/0.385 | 0.397/0.421 | 0.410/0.420 |

### 5.3 Self-Supervised Results (Table 4, Selected)

Self-supervised pre-training with fine-tuning vs supervised from scratch on large datasets:

#### Weather (Self-supervised Fine-tuning / Linear Probing / Supervised)

| T | Fine-tune MSE | Lin. Prob. MSE | Supervised MSE |
|---|---------------|----------------|----------------|
| 96 | **0.144** | 0.158 | 0.152 |
| 192 | **0.190** | 0.203 | 0.197 |
| 336 | **0.244** | 0.251 | 0.249 |
| 720 | **0.320** | 0.321 | 0.320 |

#### Electricity (Self-supervised Fine-tuning / Linear Probing / Supervised)

| T | Fine-tune MSE | Lin. Prob. MSE | Supervised MSE |
|---|---------------|----------------|----------------|
| 96 | **0.126** | 0.138 | 0.130 |
| 192 | **0.145** | 0.156 | 0.148 |
| 336 | **0.164** | 0.170 | 0.167 |
| 720 | **0.193** | 0.208 | 0.202 |

#### Traffic (Self-supervised Fine-tuning / Linear Probing / Supervised)

| T | Fine-tune MSE | Lin. Prob. MSE | Supervised MSE |
|---|---------------|----------------|----------------|
| 96 | **0.352** | 0.399 | 0.367 |
| 192 | **0.371** | 0.412 | 0.385 |
| 336 | **0.381** | 0.425 | 0.398 |
| 720 | **0.425** | 0.460 | 0.434 |

**Key finding:** On large datasets, self-supervised pre-training + fine-tuning outperforms supervised training from scratch.

### 5.4 Transfer Learning Results (Table 5)

Pre-trained on Electricity, fine-tuned on other datasets. Fine-tuning MSE is slightly worse than same-dataset pre-training but still competitive with or better than other Transformer baselines. Only the linear head or entire model is retrained for fewer epochs, yielding significant computational savings.

### 5.5 Comparison with Self-Supervised Baselines (Table 6)

On ETTh1 dataset (linear probing only), PatchTST vs contrastive methods:

| T | PatchTST (transferred) | PatchTST (self-sup) | BTSF | TS2Vec | TNC | TS-TCC |
|---|------------------------|---------------------|------|--------|-----|--------|
| 24 | **0.312** MSE | 0.362 | 0.541 | 0.599 | 0.632 | 0.653 |
| 48 | **0.339** MSE | 0.354 | 0.613 | 0.629 | 0.705 | 0.720 |
| 168 | 0.424 MSE | **0.419** | 0.640 | 0.755 | 1.097 | 1.129 |
| 336 | 0.472 MSE | **0.445** | 0.864 | 0.907 | 1.454 | 1.492 |
| 720 | 0.508 MSE | **0.487** | 0.993 | 1.048 | 1.604 | 1.603 |

Improvement range: **34.5% to 48.8%** over best baseline across prediction lengths.

---

## 6. Ablation Studies

### 6.1 Patching and Channel-Independence (Table 7 / Table 10)

Four ablation variants tested with PatchTST/42 (L=336):
- **P+CI:** Both patching and channel-independence (full PatchTST)
- **CI:** Only channel-independence (P=1, S=1)
- **P:** Only patching (channel-mixing: reshape to B x (M*P) x N)
- **Original:** Neither (original TST model, Zerveas et al., 2021)

#### Weather Dataset (MSE), T=96

| P+CI | CI only | P only | Original (TST) | FEDformer |
|------|---------|--------|-----------------|-----------|
| **0.152** | 0.164 | 0.168 | 0.177 | 0.238 |

#### Electricity Dataset (MSE), T=96

| P+CI | CI only | P only | Original (TST) | FEDformer |
|------|---------|--------|-----------------|-----------|
| **0.130** | 0.136 | 0.196 | 0.205 | 0.186 |

#### Traffic Dataset (MSE), T=96

| P+CI | CI only | P only | Original (TST) | FEDformer |
|------|---------|--------|-----------------|-----------|
| **0.367** | 0.397 | 0.595 | OOM | 0.576 |

**Findings:**
- Both patching and channel-independence improve performance.
- Combined (P+CI) is always best.
- Without patching, large datasets (Traffic, Electricity) run out of GPU memory (NVIDIA A40 48GB) even at batch size 1.

### 6.2 Varying Patch Length

Fixed L=336, stride S=P (non-overlapping), prediction T=96. Tested P = {2, 4, 8, 12, 16, 24, 32, 40}.

**Findings:**
- MSE scores do not vary significantly with different P choices (model is robust to patch length).
- Overall, PatchTST benefits from increased patch length in both forecasting and computation.
- Ideal patch length depends on dataset; P in {8, 16} are generally good choices.

### 6.3 Varying Look-back Window (Table 9)

Tested L = {24, 48, 96, 192, 336, 720} with PatchTST/42. (For ILI: L = {24, 48, 96, 192, 336} mapped to {24, 36, 48, 60, 104, 144}).

**Key finding:** PatchTST **consistently reduces MSE** as L increases, unlike other Transformer baselines (FEDformer, Autoformer, Informer) which do NOT benefit from longer look-back windows.

Example -- Weather, T=96 (MSE):

| L=24 | L=48 | L=96 | L=192 | L=336 | L=720 |
|------|------|------|-------|-------|-------|
| 0.222 | 0.212 | 0.178 | 0.160 | 0.152 | 0.147 |

Example -- Electricity, T=96 (MSE):

| L=24 | L=48 | L=96 | L=192 | L=336 | L=720 |
|------|------|------|-------|-------|-------|
| 0.268 | 0.225 | 0.174 | 0.138 | 0.130 | 0.130 |

### 6.4 Instance Normalization Ablation (Table 11)

PatchTST with (+in) and without (-in) instance normalization:

| Dataset | PatchTST/64 +in | PatchTST/64 -in | PatchTST/42 +in | PatchTST/42 -in |
|---------|-----------------|-----------------|-----------------|-----------------|
| Weather T=96 | **0.149** | 0.161 | **0.152** | 0.156 |
| Traffic T=96 | **0.360** | 0.413 | **0.367** | 0.425 |
| Electricity T=96 | **0.129** | 0.133 | **0.130** | 0.131 |
| ILI T=24 | **1.319** | 3.563 | **1.522** | 3.489 |
| ETTh1 T=96 | **0.370** | 0.385 | **0.375** | 0.388 |

**Finding:** Instance normalization provides slight improvement. But even without it, PatchTST outperforms other Transformer methods on most datasets. The primary improvement comes from patching and channel-independence.

### 6.5 Channel-Independence Applied to Other Models (Table 15)

Channel-independence (CI) is a general technique. When applied to existing models:

| Model | Weather T=96 MSE (original -> CI) | ETTh1 T=96 MSE (original -> CI) |
|-------|-----------------------------------|----------------------------------|
| Informer | 0.300 -> **0.174** | 0.865 -> **0.590** |
| Autoformer | 0.266 -> **0.227** | 0.449 -> **0.414** |
| FEDformer | 0.217 -> **0.214** | 0.376 -> **0.387** |

CI generally improves forecasting for all tested models, though they still do not match PatchTST with vanilla attention.

### 6.6 Computational Speedup from Patching (Table 1)

Training time with L=336:

| Dataset | With patch (s) | Without patch (s) | Speedup |
|---------|---------------|-------------------|---------|
| Traffic | 464 | 10,040 | **x22** |
| Electricity | 300 | 5,730 | **x19** |
| Weather | 156 | 680 | **x4** |

---

## 7. Robustness Analysis

### 7.1 Random Seed Variance (Table 14)

5 seeds: {2019, 2020, 2021, 2022, 2023}. Selected examples (supervised PatchTST/42):

| Dataset | T | MSE (mean +/- std) |
|---------|---|-------------------|
| Weather | 96 | 0.1525 +/- 0.0024 |
| Traffic | 96 | 0.3669 +/- 0.0006 |
| Electricity | 96 | 0.1304 +/- 0.0006 |
| ETTh1 | 96 | 0.3752 +/- 0.0008 |
| ETTm2 | 96 | 0.1647 +/- 0.0011 |

Standard deviations are consistently small, confirming robustness.

### 7.2 Model Parameter Sensitivity

6 parameter combinations tested: (layers L, dim D) = (3,128), (3,256), (4,128), (4,256), (5,128), (5,256) with F=2D. Results for T=96 forecasting show minimal variance across all datasets except ILI (which shows higher variance due to small dataset size).

---

## 8. Architecture Summary (Pseudocode)

```
INPUT: x in R^{M x L}  (M channels, L timesteps)

FOR each channel i = 1 to M (in parallel via reshaping):
    1. Instance Normalize: x^(i) = (x^(i) - mean) / std
    2. Pad: append S copies of last value
    3. Patch: split into N = floor((L-P)/S) + 2 patches of length P
       -> x^(i)_p in R^{P x N}
    4. Embed: x^(i)_d = W_p * x^(i)_p + W_pos   (R^{D x N})
    5. Transformer Encoder (n layers):
       FOR each layer:
           Multi-Head Attention (H heads, BatchNorm, residual)
           Feed-Forward (D -> F -> D, GELU, residual)
       -> z^(i) in R^{D x N}
    6. Supervised head: Flatten(z^(i)) -> Linear -> x_hat^(i) in R^{1 x T}
    7. Reverse normalization: x_hat^(i) = x_hat^(i) * std + mean

OUTPUT: x_hat in R^{M x T}  (concatenate all channels)
```

**Self-supervised variant:** Replace step 6 with D x P linear layer to reconstruct masked patches. Mask 40% of patches with zeros before step 4.
