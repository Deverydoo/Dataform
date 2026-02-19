# Transformers in Time-Series Analysis -- Curated Technical Extraction

> Source: Ahmed et al., "Transformers in Time-Series Analysis: A Tutorial," arXiv:2205.01138v2, July 2023

---

## 1. Time-Series Fundamentals

### Problem Formulations

**Time-series data definition:** Ordered samples, observations, or features recorded sequentially over time at a fixed sampling interval.

**Forecasting (Regression):**
- Given a historical sequence of M time steps, predict H future time steps.
- Output: real-valued numbers.
- Auto-regressive property: information must propagate back to the beginning of the sequence during generation.

**Classification:**
- Categorize a given time-series into one or more target classes.
- Final layer: linear + softmax/sigmoid.
- Identical backbone to forecasting; only the output head and loss function change.

**Anomaly Detection:**
- Identify whether an observed point is an anomaly, a changing point, or part of a normal pattern.
- Anomaly Transformer uses association discrepancy as a detection signal.

### Classical Baselines

| Model | Type |
|---|---|
| AR (Autoregressive) | Statistical |
| MA (Moving Average) | Statistical |
| ARMA | Statistical |
| ARIMA | Statistical |
| Spectral Analysis | Statistical |
| RNN | Neural sequential |
| LSTM | Neural sequential (gated) |
| GRU | Neural sequential (gated) |

**RNN/LSTM/GRU limitations relevant to transformer motivation:**
- Sequential processing: each cell depends on the previous cell's output.
- Back-propagation through time (BPTT) suffers from vanishing/exploding gradients on long sequences.
- Cannot fully exploit GPU/TPU parallelism.

---

## 2. Attention for Time-Series

### Self-Attention as a Two-Step Process

**Step 1 -- Normalized pairwise correlation:**

For input vectors {x_i}, i=1..n, where x in R^d:

```
w_ij = softmax(x_i^T x_j) = exp(x_i^T x_j) / sum_k exp(x_i^T x_k)
```

Constraint: sum_j(w_ij) = 1 for each i.

**Step 2 -- Weighted combination:**

```
z_i = sum_j (w_ij * x_j),   for all 1 <= i <= n
```

Result: z_i is a new representation of x_i that is most similar to the x_j having the largest attention weight w_ij.

### Query-Key-Value Formulation

For each input x_i, compute:

```
q_i = W_q * x_i       (query, dimension s1)
k_i = W_k * x_i       (key,   dimension s1)
v_i = W_v * x_i       (value, dimension s)
```

Where W_q, W_k in R^{s1 x d}, W_v in R^{s x d} are learnable weight matrices.

**Output (unscaled):**

```
z_i = sum_j softmax(q_i^T k_j) * v_j
```

### Scaled Dot-Product Attention

The dot product grows with vector dimension, making softmax sensitive to large values. Scaling:

```
z_i = sum_j softmax(q_i^T k_j / sqrt(d_q)) * v_j
```

**Matrix form (the canonical attention equation):**

```
Z = softmax(Q K^T / sqrt(d_k)) * V
```

Where Q, K in R^{s1 x n}, V in R^{s x n}, Z in R^{s x n}.

### Why Self-Attention Works for Time-Series

- Correlates ALL elements in a sequence to each other in parallel (no sequential bottleneck).
- No vanishing gradients from recurrence.
- Learns pairwise temporal dependencies directly.
- Captures long-range dependencies that RNNs struggle with.

### Modifications Needed for Time-Series

1. **Positional encoding** must encode temporal position since attention is permutation-invariant.
2. **Input embedding** changes from word embeddings to 1-D convolution projections of scalar/vector signals.
3. **Causal masking** in decoders prevents attending to future time steps.
4. **Complexity reduction** needed because O(L^2) attention is prohibitive for long sequences.

---

## 3. Positional Encoding for Time

### Sinusoidal PE (Original Vaswani et al.)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Variables:
- `pos`: position (time step) in the input sequence.
- `i`: position along the embedding vector dimension, ranging from 0 to d/2 - 1.
- `d`: dimension of the embedding vectors.
- The constant 10,000 can differ depending on the input sequence's length.

**Key property:** Encodes RELATIVE position, not just absolute. For any fixed offset k, PE(pos+k) is a linear transformation of PE(pos).

### Rotation Matrix Proof of Relative Encoding

For PE vector p_i in R^2, the transformation T that maps position a to position a+k:

```
T [sin(f_i * a), cos(f_i * a)]^T = [sin(f_i * (a+k)), cos(f_i * (a+k))]^T
```

Expanding via angle-addition identities:

```
T = [ cos(f_i * k)   sin(f_i * k) ]
    [-sin(f_i * k)   cos(f_i * k) ]
```

This is a rotation matrix that depends only on the relative offset k, NOT on the absolute position a.

### Analogy with Binary Encoding

- Sinusoidal PE is analogous to alternating bits in a binary number.
- The least significant bit oscillates at the highest frequency.
- Moving to higher-order bits, oscillation frequency decreases.
- Similarly, moving along the PE vector dimension, the sinusoidal frequency decreases.

### Combining PE with Input Embeddings

PE vectors are ADDED (element-wise sum) to the input embeddings. This works because:
- PE encodes positional information.
- Embeddings encode semantic/feature information.
- They reside in approximately orthogonal subspaces.
- Analogy: frequency modulation in digital communication -- a low-frequency signal rides over a high-frequency carrier without interference.

### Time-Series-Specific Positional Encodings

**Timestamp Embedding:** Embeds calendar features (seconds, minutes, hours, weeks, years) as learned vectors.

**Stamp Embedding (SE) -- used in Informer:**
Captures hierarchical time information (week, month, year) and occasional events (holidays). Added to input along with fixed sinusoidal PE.

**Traffic Transformer -- four temporal encoding strategies:**

**(a) Continuity of time-series data:**
1. **Relative position encoding:** Encodes relative position of a time step within the source-target window (not global). Same time step gets different encoding depending on its window position. Uses sine/cosine.
2. **Global position encoding:** Encodes using sine/cosine over the entire sequence to capture both local and global positions.

**(b) Periodicity of time-series data:**
3. **Periodic position encoding:** Encodes daily periodicity (288 positions = 24*60/5 for 5-min intervals) and weekly periodicity (7 positions).
4. **Time-series segments:** Concatenates daily and weekly data segments to the M most recent time steps.

---

## 4. Architecture Adaptations

### Standard Transformer Encoder Block

```
Input -> Multi-Head Self-Attention -> Add & LayerNorm -> Feed-Forward -> Add & LayerNorm -> Output
```

- Feed-forward: two linear layers with ReLU activation.
- Residual connections around both the attention and feed-forward sublayers.
- Input and output of multi-head self-attention have the SAME dimension: dim(X) = dim(Z).

### Standard Transformer Decoder Block

```
Input -> Masked Multi-Head Self-Attention -> Add & LayerNorm
      -> Encoder-Decoder Attention -> Add & LayerNorm
      -> Feed-Forward -> Add & LayerNorm -> Output
```

- Encoder-decoder attention: keys and values from the last encoder; queries from the preceding masked self-attention.
- Masking ensures the decoder cannot attend to future positions during training.
- During training: ground-truth target is fed to decoder (teacher forcing).
- During inference: predicted outputs are fed back auto-regressively.

### Multi-Head Self-Attention Mechanism

**Step 1:** For r heads, generate r sets of weight matrices {W_q^(l), W_k^(l), W_v^(l)} for l=1..r.

**Step 2:** Compute scaled dot-product attention in parallel for each head:
```
Z^(l) = softmax(Q^(l) K^(l)T / sqrt(d_k)) * V^(l)
```

**Step 3:** Concatenate and project:
```
Z = W_o * [Z^(1); Z^(2); ...; Z^(r)]
```
Where W_o in R^{d x (r*s)} is a learnable weight matrix.

**Analogy:** Multiple heads are analogous to multiple convolutional kernels -- each head learns distinct correlation patterns.

### Key Adaptations for Time-Series

| Component | NLP Transformer | Time-Series Adaptation |
|---|---|---|
| Input embedding | Word embedding lookup | 1-D convolution to project scalar/vector inputs to d-dimensional vectors |
| Positional encoding | Sinusoidal or learned | Sinusoidal + stamp embedding (SE) for calendar features; periodic encodings |
| Decoder input | Word embeddings of previous tokens | Start token + zero-padded placeholder for target sequence |
| Output layer | Softmax over vocabulary | Linear regression head (forecasting) or softmax/sigmoid (classification) |
| Attention | Standard O(L^2) | ProbSparse, LogSparse, convolutional, or other efficient variants |
| Preprocessing | Tokenization | Masking, noise injection, variable selection |

### Post-LN vs. Pre-LN Transformer

**Post-LN (original):**
- Layer normalization OUTSIDE the residual block.
- Slower convergence; requires learning rate warm-up.

**Pre-LN:**
- Layer normalization INSIDE the residual block (before sublayer).
- Faster convergence; no warm-up required.
- Controls gradient magnitudes and balances residual dependencies.
- Inferior empirical performance compared to post-LN.

### Alternatives to Layer Normalization

**Admin (Adaptive Model Initialization):**
- Combines ease of training from pre-LN with performance of post-LN.
- Better initialization captures input-output dependencies.

**ReZero:**
- Replaces layer normalization with a trainable scalar alpha, initialized to 0.
- Network initially computes the identity function.
- Contributions from self-attention and MLP layers are introduced gradually.
- Residual formula: x + alpha * F(x), with alpha initially 0.

### Dense Interpolation (SAnD)

Replaces the decoder block entirely with a dense interpolation layer.
- Incorporates temporal order into processing.
- Eliminates output embedding.
- Reduces total number of layers.
- Has tunable hyperparameters for performance.

### Gating Operations

**Gated Transformer Networks (GTN):**
- Merges output of two encoder towers via concatenation and gating.

**Temporal Fusion Transformers (TFT):**
- Uses Gated Linear Units (GLUs) to selectively emphasize or suppress parts of the network based on the dataset.

---

## 5. Specific Models and Their Key Innovations

### Informer (Zhou et al., 2021)

**Problem solved:** Standard self-attention has O(L^2) complexity, prohibitive for long sequences.

**Key innovation 1 -- ProbSparse Self-Attention:**
- Selects only the top log(L) dominant query vectors based on dot-product sparsity.
- Remaining query vectors are set to zero.
- Complexity reduction: O(L * log(L)) per layer.
- With J stacked encoders: O(J * L * log(L)) total memory.

**Key innovation 2 -- Self-Attention Distilling:**
- Multi-head attention output -> 1-D convolution (kernel size 3) -> ELU activation -> max-pooling (stride 2).
- Halves the representation at each layer, forming a pyramid structure.
- Removes redundant value vector combinations.
- Inspired by dilated convolution.

**Key innovation 3 -- Stacked Replicas:**
- Additional stacks with input length halved from the previous stack.
- Outputs concatenated with the main stack for robustness.

**Key innovation 4 -- Generative Decoder:**
- Input = start token concatenated with zero-padded target placeholder.
- Predicts ALL outputs in one forward pass (non-autoregressive).
- Considerably reduces inference time.

**Input representation:**
- Scalar input -> d-dimensional vectors via 1-D convolution.
- Encoder input = scalar projection (u_i) + PE_i + stamp embedding (SE).

**Benchmarks:** Outperformed ARIMA, Prophet, LSTMa, LSTnet, DeepAR on electricity consumption and weather datasets.

### LogSparse Transformer (Li et al., 2019)

**Problem solved:** O(L^2) memory per self-attention layer.

**Key innovation -- LogSparse Attention:**
- Each time step attends to previous time steps selected with exponential step size.
- Memory reduction: O(L^2) -> O(L * log_2(L)) per self-attention layer.

**Attention patterns:**
1. LogSparse for distant time steps + canonical attention for nearby time steps.
2. Restart LogSparse step size after a particular range.

**Causal Convolutional Self-Attention:**
- Standard attention = convolution with kernel size 1 (point-wise similarity only).
- Causal convolutional attention uses kernel size > 1 for query and key generation.
- Value vectors still use kernel size 1.
- Captures local context/shape information.
- Ensures current position has no access to future information (causal constraint).

### Simply Attend and Diagnose -- SAnD (Song et al., 2018)

**Domain:** Clinical time-series (ICU data, MIMIC-III).

**Architecture:**
1. Input embedding: 1-D convolution mapping inputs to d-dimensional vectors (d > number of variables).
2. Hard-coded positional encoding added.
3. Multi-head attention with restricted (causal) self-attention.
4. Stacked attention modules.
5. Dense interpolation layer (replaces decoder) with positional encoding for temporal structure.
6. Linear + softmax/sigmoid for classification.

**Result:** Outperformed RNNs on all MIMIC-III benchmark tasks.

### Traffic Transformer (Cai et al., 2020)

**Domain:** Traffic forecasting (speed, density, volume from road sensors).

**Architecture:**
- Graph Neural Network (spatial dependencies) -> Transformer (temporal dependencies).
- GNN output forms the Transformer input after PE addition.

**Two methods for incorporating position information:**
1. Addition: PE vectors added directly to input vectors.
2. Similarity-based: Attention weights for PE computed via dot product and used to tweak input attention weights.

**Four temporal encoding strategies:** (See Section 3 above for details on relative, global, periodic, and segment-based encodings.)

**Benchmarks:** Tested on METR-LA and PeMS datasets.

### Tightly-Coupled Convolutional Transformer -- TCCT (Shen & Wang, 2022)

**Three attention improvements:**

1. **CSPAttention (Cross Stage Partial Attention):**
   - Combines CSPNet with self-attention.
   - Incorporates the input feature map into the output stage.
   - Reduces time complexity and memory.

2. **Dilated Causal Convolution in distilling:**
   - Replaces canonical convolution.
   - Provides exponentially growing receptive field.
   - Enhances locality.

3. **Pass-through mechanism:**
   - Concatenates feature maps from multiple scales within self-attention.
   - Enables stacking of self-attention blocks.
   - Improves fine-scale feature detection.

### NAST -- Non-Autoregressive Spatial-Temporal Transformer (Chen et al., 2021)

**Key innovation -- Spatial-Temporal Attention Block:**
- Spatial attention block (predictions across space) combined with temporal attention block (predictions across time).
- Improves learning in both spatial and temporal domains simultaneously.

### Temporal Fusion Transformer -- TFT (Lim et al., 2021)

**Key features:**
- Temporal self-attention decoder for long-term dependencies.
- Gated Linear Units (GLUs) for dataset-adaptive emphasis/suppression.
- Variable selection for input preprocessing.
- Inherently interpretable architecture.

### YFormer (Madhusudhanan et al., 2021)

**Key innovations:**
- U-Net inspired architecture.
- Sparse attention mechanism.
- Downsampling decoder.
- Better detection of long-range effects.

### Gated Transformer Networks -- GTN (Liu et al., 2021)

**Key innovation:**
- Two Transformer encoder towers.
- Gating mechanism concatenates and combines outputs.
- Designed for multivariate time-series classification.
- Inherently interpretable.

### Anomaly Transformer (Xu et al., 2021)

- BERT-inspired encoder-only architecture.
- Self-supervised / unsupervised learning.
- Designed for time-series anomaly detection.
- Uses association discrepancy as the detection mechanism.

### SITS-BERT (Yuan & Lin, 2020)

- BERT-inspired self-supervised pretraining.
- Learns from unlabeled satellite imagery data.
- Applied to region classification from satellite image time series.

### TabAConvBERT (Shankaranarayana & Runje, 2021)

- BERT-inspired architecture with 1-D convolutions.
- Attention augmented convolutional transformer.
- Designed for tabular time-series data.
- Uses masking and timestamp embedding in preprocessing.

### Adversarial Sparse Transformer (Wu et al., 2020)

- GAN-inspired: Transformer as generator, separate discriminator network.
- Generator creates forecasts; discriminator classifies them as real/fake.
- Adversarial training improves forecast realism.

### Deep Transformer for Influenza Forecasting (Wu et al., 2020)

**Setup:** Predict 1 week of flu cases from 10 previous weeks.

**Time-Delay Embedding (TDE):**
```
TDE_{d,tau}(x(t)) = [x_t, x_{t-1}, ..., x_{t-(d-1)*tau}]
```
- Embeds each scalar input x_t into a d-dimensional time-delay space.
- Optimal dimension d=8 with tau=1 gave minimum RMSE.
- Multivariate features tested: week number, first-order differences, second-order differences.

**Result:** Outperformed ARIMA, LSTMs, and seq2seq with attention in RMSE.

### Self-Supervised Transformer for Time-Series -- STraTS (Tipirneni & Reddy, 2021)

- Self-supervised learning for multivariate clinical time-series.
- Handles missing values.
- Inherently interpretable.

---

## 6. Input Representations

### Word Embedding to Time-Series Embedding

| NLP | Time-Series |
|---|---|
| Discrete tokens (words) | Continuous scalar or vector signals |
| Embedding lookup table | 1-D convolution filters |
| Vocabulary-sized embedding matrix | Projection to d-dimensional space |

### 1-D Convolution Projection

- Maps scalar input to d-dimensional vectors u_i.
- Retains local context through convolution kernel.
- Used in Informer, SAnD, and others.

### Multivariate Handling

- For multivariate data with M variables, 1-D convolution maps the M-dimensional input to d-dimensional vectors (d > M).
- Variable selection mechanisms (e.g., TFT) can preprocess to select relevant input variables.

### Time-Delay Embedding (TDE)

```
TDE_{d,tau}(x(t)) = [x_t, x_{t-1}, ..., x_{t-(d-1)*tau}]
```

- Converts a scalar time series into a d-dimensional vector representation.
- Parameters: d = embedding dimension, tau = delay parameter.
- Derived from Takens' embedding theorem in dynamical systems theory.

### Combined Input for Encoder

```
encoder_input_i = u_i + PE_i + SE_i
```

Where:
- u_i = scalar/vector projection via 1-D convolution
- PE_i = fixed sinusoidal positional encoding
- SE_i = stamp embedding for calendar/hierarchical time features

### Data Preprocessing Techniques for Time-Series Transformers

| Technique | Effect |
|---|---|
| Masking | Removes input features; trains model to predict missing features |
| Noise injection | Adds noise to input; improves robustness |
| Variable selection | Selects relevant input variables; reduces dimensionality |

---

## 7. Training Objectives

### Forecasting Losses

**Root Mean Square Error (RMSE):** Used as evaluation metric in influenza forecasting experiments.

**General regression losses:** MSE and MAE are standard for time-series forecasting (implicit throughout).

### Classification Losses

- Softmax output + cross-entropy loss for multi-class classification.
- Sigmoid output + binary cross-entropy for binary/multi-label classification.

### GAN-Based Training

- Generator loss: produce forecasts classified as "real" by the discriminator.
- Discriminator loss: distinguish real time-series from generated forecasts.
- Adversarial training: alternating optimization of generator and discriminator.

### Self-Supervised Pretraining Objectives

- Masked feature prediction (analogous to masked language modeling in BERT).
- The model learns to reconstruct masked portions of the input time series.

---

## 8. All Equations

### Eq. 1 -- Normalized Dot-Product (Basic Self-Attention Weight)
```
w_ij = softmax(x_i^T x_j) = exp(x_i^T x_j) / sum_k exp(x_i^T x_k)
```
- w_ij: attention weight from position i to position j
- x_i, x_j: input vectors in R^d
- sum_j(w_ij) = 1

### Eq. 2 -- Self-Attention Output
```
z_i = sum_{j=1}^{n} w_ij * x_j,   for all 1 <= i <= n
```
- z_i: new representation of position i
- Weighted sum of all input vectors

### Eq. 3 -- Query, Key, Value Projection
```
q_i = W_q * x_i
k_i = W_k * x_i
v_i = W_v * x_i
```
- W_q, W_k in R^{s1 x d}: learnable weight matrices for query and key
- W_v in R^{s x d}: learnable weight matrix for value
- q_i in R^{s1}, k_i in R^{s1}, v_i in R^{s}

### Eq. 4 -- QKV Self-Attention (Unscaled)
```
z_i = sum_j softmax(q_i^T k_j) * v_j
```

### Eq. 5 -- Scaled Dot-Product Attention (Element-wise)
```
z_i = sum_j softmax(q_i^T k_j / sqrt(d_q)) * v_j
```
- d_q: dimension of query/key vectors
- Scaling prevents softmax saturation for large dimensions

### Eq. 6 / Eq. 7 -- Scaled Dot-Product Attention (Matrix Form)
```
Z = softmax(Q K^T / sqrt(d_k)) * V
```
- Q, K in R^{s1 x n}: query and key matrices for n positions
- V in R^{s x n}: value matrix
- Z in R^{s x n}: output matrix
- d_k: dimension of key vectors

### Eq. 8 -- Sinusoidal PE (Even Dimensions)
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
```

### Eq. 9 -- Sinusoidal PE (Odd Dimensions)
```
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- pos: position (time step) in the sequence
- i: dimension index, 0 <= i <= d/2 - 1
- d: embedding dimension
- 10,000 can be adjusted based on sequence length

### Eq. 10-12 -- Relative Position via Rotation Matrix
Given absolute position a and relative offset k:
```
T [sin(f_i * a), cos(f_i * a)]^T = [sin(f_i * (a+k)), cos(f_i * (a+k))]^T
```

Expanding:
```
[x1*sin(f_i*a) + y1*cos(f_i*a)]   [sin(f_i*a)cos(f_i*k) + sin(f_i*k)cos(f_i*a)]
[x2*sin(f_i*a) + y2*cos(f_i*a)] = [cos(f_i*a)cos(f_i*k) - sin(f_i*a)sin(f_i*k)]
```

### Eq. 13 -- Rotation Matrix for Relative Positional Encoding
```
T = [ cos(f_i * k)   sin(f_i * k)]
    [-sin(f_i * k)   cos(f_i * k)]
```
- f_i: frequency at dimension i
- k: relative offset between positions
- T depends only on k (relative), not on absolute position a

### Eq. 14 -- Time-Delay Embedding
```
TDE_{d,tau}(x(t)) = [x_t, x_{t-1}, ..., x_{t-(d-1)*tau}]
```
- d: embedding dimension
- tau: delay parameter
- x_t: scalar observation at time t

### Multi-Head Output Projection (Implicit)
```
Z = W_o * Concat(Z^(1), Z^(2), ..., Z^(r))
```
- W_o in R^{d x (r*s)}: output projection matrix
- r: number of attention heads

### Learning Rate Schedule (Original Transformer)
```
lr increases linearly for first N steps, then decreases proportionally to 1/sqrt(step_number)
```

### Complexity Equations

| Model | Self-Attention Complexity | Memory |
|---|---|---|
| Standard Transformer | O(L^2) | O(L^2) |
| Informer (ProbSparse) | O(L log L) | O(J * L * log L) with J encoders |
| LogSparse | O(L log_2 L) | O(L log_2 L) per layer |

---

## 9. Comparison Tables

### Informer Benchmark Results

| Model | Type | Beaten by Informer |
|---|---|---|
| ARIMA | Statistical | Yes |
| Prophet | Statistical | Yes |
| LSTMa (LSTM + Attention) | Neural | Yes |
| LSTnet | Neural | Yes |
| DeepAR | Neural (probabilistic) | Yes |

Datasets: Electricity consumption, weather.

### SAnD (Clinical Time-Series) Results

| Model | Domain | Beaten by SAnD |
|---|---|---|
| RNNs | Clinical (MIMIC-III) | Yes on all benchmark tasks |

### Satellite Image Classification Results

| Model | Performance vs. Transformer |
|---|---|
| LSTM-RNN | Worse on raw data |
| MS-ResNet | Worse on raw data |
| DuPLO | Worse on raw data |
| TempCNN | Worse on raw data |
| Random Forest | Worse on raw data |

Note: On pre-processed data, all methods performed similarly.

### Influenza Forecasting Results

| Model | Metric | Comparison |
|---|---|---|
| ARIMA | RMSE | Worse than Transformer |
| LSTM | RMSE | Worse than Transformer |
| Seq2Seq + Attention | RMSE | Worse than Transformer |
| Transformer + TDE (d=8, tau=1) | RMSE | Best result |

### Time-Series Transformer Taxonomy

| Model | Task | Key Mechanism | Year |
|---|---|---|---|
| SAnD | Classification | Dense interpolation, restricted attention | 2018 |
| LogSparse | Forecasting | LogSparse attention, causal conv self-attention | 2019 |
| Traffic Transformer | Forecasting | GNN + Transformer, 4 temporal encoding strategies | 2020 |
| SITS-BERT | Classification | Self-supervised pretraining, satellite imagery | 2020 |
| Adversarial Sparse Transformer | Forecasting | GAN-inspired adversarial training | 2020 |
| Deep Transformer (Influenza) | Forecasting | Time-delay embedding | 2020 |
| Informer | Forecasting | ProbSparse attention, distilling, generative decoder | 2021 |
| TCCT | Forecasting | CSPAttention, dilated causal conv, pass-through | 2022 |
| NAST | Forecasting | Spatial-temporal attention block | 2021 |
| TFT | Forecasting | GLU gating, variable selection, interpretability | 2021 |
| YFormer | Forecasting | U-Net inspired, sparse attention, downsampling | 2021 |
| GTN | Classification | Dual tower gating, interpretability | 2021 |
| Anomaly Transformer | Anomaly detection | Association discrepancy, self-supervised | 2021 |
| TabAConvBERT | Classification | BERT + 1-D convolution, tabular time-series | 2021 |
| STraTS | Classification | Self-supervised, missing value handling | 2021 |

### Attention Complexity Comparison

| Attention Type | Time Complexity | Space Complexity |
|---|---|---|
| Canonical (full) | O(L^2) | O(L^2) |
| ProbSparse (Informer) | O(L log L) | O(L log L) |
| LogSparse | O(L log_2 L) | O(L log_2 L) |
| CSPAttention (TCCT) | Reduced via CSPNet | Reduced via CSPNet |
| Sparse (YFormer) | Sub-quadratic | Sub-quadratic |

---

## 10. Limitations and Alternatives

### When Transformers Struggle on Time-Series

1. **Quadratic complexity:** Standard self-attention scales O(L^2) in time and memory, making long sequence modeling prohibitive without architectural modifications.

2. **Small datasets:** Deep Transformers require large datasets for training from scratch. Small datasets lead to overfitting and poor generalization. Mitigations:
   - Pre-trained models.
   - Small batch sizes (but increases update variance).
   - Better initialization (Xavier, T-Fixup, DT-Fixup).

3. **Training instability:** Transformers are difficult to train from scratch. Issues:
   - Sensitive to learning rate.
   - Performance drop in early epochs before convergence.
   - Requires warm-up schedule for post-LN architecture.
   - Vanishing gradients through layer normalization.

4. **Task-dependent pretraining:** Unlike NLP where one pretrained model serves many tasks, time-series data varies enormously (electricity, temperature, traffic, health, satellite). Pretraining must be task/domain specific, limiting transfer learning.

5. **Pre-processed data equalizer:** On pre-processed satellite data, the Transformer's advantage over simpler models (LSTM, CNN, random forest) largely disappears. The Transformer's advantage is most pronounced on raw data.

6. **Large model size:** Trade-offs between model capacity and overfitting risk. Strategies:
   - Train large, then compress (pruning, quantization).
   - Lottery ticket hypothesis for finding sparse sub-networks.

7. **Interpretability:** Black-box nature; post hoc explanations may not accurately reflect internal processing. Inherently interpretable architectures (TFT, GTN, STraTS) address this but are model-specific.

### Training Best Practices Summary

| Practice | Recommendation |
|---|---|
| Model size | Start large, then compress (not small then grow) |
| Batch size | Use a large minimum; optimal depends on model complexity |
| Learning rate | Too small = slow convergence; too large = non-convergence |
| Learning rate schedule | Linear warm-up then 1/sqrt(step) decay |
| Gradient clipping | Use standard clipping; avoid step-size proportional to batch size |
| Initialization | Xavier (common), T-Fixup (removes warm-up + LayerNorm need), DT-Fixup (for small datasets + deep models) |
| Layer normalization | Pre-LN for easier training; post-LN for better final performance; Admin or ReZero for best of both |
| Optimizer | Adam with adaptive per-parameter learning rates |
| Residual connections | Essential for training deeper networks |
| Frameworks | HuggingFace Transformers, PyTorch Lightning, TensorFlow, DeepSpeed, MosaicML Composer |

### Alternatives and Future Directions

- Incorporating recurrent components back into Transformers (R-Transformer).
- Uncertainty estimation in time-series prediction.
- Multimodal foundation models combining images, video, text, and time-series.
- Robustness and failure detection for trustworthy deployment.
