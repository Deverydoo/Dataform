# DeepLOB: Deep Convolutional Neural Networks for Limit Order Books -- Technical Extraction

**Source:** Zhang, Zohren, Roberts, "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books," arXiv:1808.03668v6, 2020.
**Code:** https://github.com/zcakhaa

---

## 1. Limit Order Book (LOB) Data Representation

### 1.1 Raw LOB Structure

Each LOB state contains bid and ask sides with multiple price levels.

**Per-level data:**
- `p_a^(i)(t)` = ask price at level i at time t
- `v_a^(i)(t)` = ask volume at level i at time t
- `p_b^(i)(t)` = bid price at level i at time t
- `v_b^(i)(t)` = bid volume at level i at time t

**Number of levels:** 10 on each side (bid and ask).

**Features per timestamp:** 40 total (10 levels x 2 sides x 2 features per level [price, volume]).

### 1.2 Feature Ordering Within Each Timestamp

The 40 features are ordered as:

```
x_t = { p_a^(i)(t), v_a^(i)(t), p_b^(i)(t), v_b^(i)(t) }  for i = 1, ..., 10
```

Concretely: `[ask_price_L1, ask_vol_L1, bid_price_L1, bid_vol_L1, ask_price_L2, ask_vol_L2, ..., ask_price_L10, ask_vol_L10, bid_price_L10, bid_vol_L10]`

### 1.3 Input Tensor

A single input uses the 100 most recent LOB states:

```
X = [x_1, x_2, ..., x_t, ..., x_100]^T  in  R^{100 x 40}
```

- **Time dimension:** 100 (rows)
- **Feature dimension:** 40 (columns)

---

## 2. Data Normalization

### 2.1 FI-2010 Dataset

Z-score normalization provided by the dataset. Three schemes available (z-score, min-max, decimal precision); z-score used with minimal difference between schemes.

### 2.2 LSE Dataset (Rolling Z-Score)

Dynamic z-score normalization:

```
x_normalized(day_t) = (x(day_t) - mean(previous 5 days)) / std(previous 5 days)
```

- Separate normalization per instrument.
- Rolling window of 5 previous trading days for mean and standard deviation.
- Avoids static normalization problems caused by regime shifts in financial time-series.

---

## 3. Labelling / Prediction Targets

### 3.1 Mid-Price Definition

```
p_t = (p_a^(1)(t) + p_b^(1)(t)) / 2                        ... (Eq. 1)
```

### 3.2 Smoothed Price Averages

```
m^-(t) = (1/k) * sum_{i=0}^{k} p_{t-i}     (past mean)     ... (Eq. 2a)
m^+(t) = (1/k) * sum_{i=1}^{k} p_{t+i}     (future mean)   ... (Eq. 2b)
```

Where `k` = prediction horizon.

### 3.3 Percentage Change Methods

**Method 1 (FI-2010, from Ntakaris et al.):**

```
l_t = (m^+(t) - p_t) / p_t                                   ... (Eq. 3)
```

Smoothing applied only to future prices. Produces less consistent labels.

**Method 2 (LSE, from Tsantekidis et al.):**

```
l_t = (m^+(t) - m^-(t)) / m^-(t)                             ... (Eq. 4)
```

Smoothing applied to both past and future prices. Produces more consistent trading signals.

### 3.4 Label Assignment (3-Class Classification)

```
label(t) =  +1 (up)         if l_t >  alpha
             0 (stationary)  if -alpha <= l_t <= alpha
            -1 (down)        if l_t < -alpha
```

Where `alpha` is the threshold for percentage change.

### 3.5 Prediction Horizons Tested

- FI-2010 Setup 1: k = {10, 50, 100}
- FI-2010 Setup 2: k = {10, 20, 50}
- LSE Dataset: k = {20, 50, 100}

---

## 4. Model Architecture (DeepLOB)

Three sequential building blocks: (1) Convolutional layers, (2) Inception Module, (3) LSTM + Output.

### 4.1 Convolutional Block (Feature Extraction from LOB Structure)

**Layer 1: Price-Volume Pairing**
- Filter size: `(1 x 2)` with 16 filters
- Stride: `(1 x 2)`
- Activation: Leaky-ReLU (negative slope = 0.01)
- Zero padding (preserves time dimension)
- Purpose: Summarizes information between price and volume `{p^(i), v^(i)}` at each order book level
- Input: `(100, 40)` --> Output: `(100, 20)`
- Stride is critical: without it, the same parameters would be shared between `{p^(i), v^(i)}` and `{v^(i), p^(i+1)}`, incorrectly coupling price and volume across levels

**Layer 2: Cross-Level Pairing (Micro-Price Formation)**
- Filter size: `(4 x 1)` with 16 filters
- Activation: Leaky-ReLU (negative slope = 0.01)
- Zero padding
- Purpose: Temporal convolution over 4 timesteps within each feature channel
- Input: `(100, 20)` --> Output: `(100, 20)` (same spatial dims due to padding)

**Layer 3: Cross-Level Aggregation**
- Filter size: `(4 x 1)` with 16 filters
- Activation: Leaky-ReLU (negative slope = 0.01)
- Zero padding
- Input: `(100, 20)` --> Output: `(100, 20)`

**Layer 4: Bid-Ask Pairing**
- Filter size: `(1 x 2)` with 16 filters
- Stride: `(1 x 2)`
- Activation: Leaky-ReLU (negative slope = 0.01)
- Zero padding
- Purpose: Pairs bid and ask information at each level, forming imbalance/micro-price features
- Input: `(100, 20)` --> Output: `(100, 10)`

**Layer 5: Temporal Convolution**
- Filter size: `(4 x 1)` with 16 filters
- Activation: Leaky-ReLU (negative slope = 0.01)
- Zero padding
- Input: `(100, 10)` --> Output: `(100, 10)`

**Layer 6: Temporal Convolution**
- Filter size: `(4 x 1)` with 16 filters
- Activation: Leaky-ReLU (negative slope = 0.01)
- Zero padding
- Input: `(100, 10)` --> Output: `(100, 10)`

**Layer 7: Full Cross-Level Integration**
- Filter size: `(1 x 10)` with 16 filters
- Activation: Leaky-ReLU (negative slope = 0.01)
- Purpose: Integrates all 10 levels into a single representation
- Input: `(100, 10)` --> Output: `(100, 1)`

**Key design note:** The convolutional block operates 1x2 filters with stride 2 to pair {price, volume} then {bid, ask}, followed by temporal 4x1 filters, and a final 1x10 filter to collapse all levels. No pooling layers are used in the convolutional block (pooling causes underfitting for time-series LOB data).

### 4.2 Micro-Price Analogy

The second convolutional layer with `(1 x 2)` stride `(1 x 2)` effectively learns the micro-price:

```
p_micro = I * p_a^(1) + (1 - I) * p_b^(1)                   ... (Eq. 7)

I = v_b^(1) / (v_a^(1) + v_b^(1))                            (imbalance)
```

Unlike the classical micro-price (L1 only), the convolutions form micro-prices for ALL levels of the LOB.

### 4.3 Inception Module

Applied after the convolutional block. Input feature maps have shape `(100, 1)` per filter (100 timesteps, 1 spatial dim).

**Architecture ("Inception@32"):**

Three parallel branches plus max-pooling:

**Branch 1:**
```
Input --> Conv 1x1@32 --> Conv 3x1@32
```

**Branch 2:**
```
Input --> Conv 1x1@32 --> Conv 5x1@32
```

**Branch 3:**
```
Input --> MaxPool 3x1 (stride=1, zero padding) --> Conv 1x1@32
```

**Output:** Concatenation of all three branch outputs along the filter dimension.

- All convolutions use 32 filters.
- `1x1` convolutions implement the Network-in-Network approach for dimensionality reduction and nonlinear transformation.
- `3x1` and `5x1` capture local interactions over 3 and 5 timesteps respectively (multi-scale temporal features).
- Max-pooling branch uses stride=1 and zero padding to preserve temporal dimension.
- Purpose: Different filter sizes capture dynamics at multiple timescales (analogous to moving averages with different decay weights in technical analysis).

### 4.4 LSTM Module

- **Units:** 64 LSTM units
- Replaces fully connected layers (a single FC layer with 64 units would require 630,000+ parameters; LSTM requires only ~60,000)
- Captures temporal dependencies among extracted features from the Inception Module output
- Takes the full temporal sequence of Inception features as input

### 4.5 Output Layer

- Dense layer with softmax activation
- 3 output neurons (down, stationary, up)
- Output represents probability of each price movement class

### 4.6 Total Parameter Count

```
DeepLOB: ~60,000 parameters
```

Comparison:
- CNN-I: 768,000 parameters
- BoF: 86,000 parameters
- N-BoF: 12,000 parameters

### 4.7 Summary of Layer Sequence

```
Input: (100 x 40)
  |
Conv 1x2@16, stride=(1,2), Leaky-ReLU, zero-pad   --> (100, 20, 16)
Conv 4x1@16, Leaky-ReLU, zero-pad                  --> (100, 20, 16)
Conv 4x1@16, Leaky-ReLU, zero-pad                  --> (100, 20, 16)
  |
Conv 1x2@16, stride=(1,2), Leaky-ReLU, zero-pad   --> (100, 10, 16)
Conv 4x1@16, Leaky-ReLU, zero-pad                  --> (100, 10, 16)
Conv 4x1@16, Leaky-ReLU, zero-pad                  --> (100, 10, 16)
  |
Conv 1x10@16, Leaky-ReLU                           --> (100, 1, 16)
  |
Inception Module @32 (3 branches, concat)           --> (100, 1, 96)
  |
LSTM (64 units)                                     --> (64,)
  |
Dense + Softmax (3 classes)                         --> (3,)
```

---

## 5. FIR Filter Interpretation

Convolutional filters are discrete convolutions / finite impulse response (FIR) filters:

```
y(n) = sum_{k=0}^{M} b_k * x(n - k)                         ... (Eq. 5)
```

Where:
- `y(n)` = output signal at time n
- `x(n)` = input signal at time n
- `M` = filter order
- `b_k` = filter coefficients (learned, not designed from signal processing theory)

**Translation equivariance:** If a feature pattern appears at time t in one sample and time t' in another, the same convolutional filter will detect it at both locations.

---

## 6. Training Procedure

### 6.1 Loss Function

Categorical cross-entropy (3-class classification):

```
L = -sum_{c=1}^{3} y_c * log(p_c)
```

Where `y_c` is the one-hot ground truth and `p_c` is the softmax output probability for class c.

### 6.2 Optimizer

ADAM optimizer with:
- Learning rate: 0.01
- Epsilon: 1 (non-default; standard default is 1e-8)
- Other ADAM parameters: defaults (beta_1=0.9, beta_2=0.999)

### 6.3 Mini-Batch Size

32 (deliberately small; large batches converge to narrow sharp minima while small batches converge to broad shallow minima with better generalization).

### 6.4 Early Stopping

- Stop training when validation accuracy does not improve for 20 consecutive epochs.
- FI-2010: approximately 100 epochs total.
- LSE: approximately 40 epochs total.

### 6.5 Activation Functions

- All convolutional layers: Leaky-ReLU with negative slope = 0.01 (selected via grid search on validation set).
- Output layer: Softmax.

### 6.6 Hardware

Single NVIDIA Tesla P100 GPU. Framework: Keras on TensorFlow backend.

### 6.7 Inference Speed

```
DeepLOB forward pass: 0.253 ms per sample
```

Comparison:
| Model     | Forward (ms) |
|-----------|-------------|
| CNN-I     | 0.025       |
| LSTM      | 0.061       |
| C(TABL)   | 0.229       |
| DeepLOB   | 0.253       |
| N-BoF     | 0.524       |
| BoF       | 0.972       |

---

## 7. Experimental Setup

### 7.1 FI-2010 Dataset

- **Source:** Nasdaq Nordic stock market, 5 stocks, 10 consecutive days.
- **Normalization:** z-score (pre-provided).
- **Downsampled:** Non-overlapping blocks of 10 events.
- **Setup 1 (Anchored Forward Validation):** 9-fold. In fold i, train on first i days, test on day (i+1), for i = 1, ..., 9. Results averaged over all folds.
- **Setup 2 (Fixed Train/Test Split):** First 7 days = training, last 3 days = testing. Better for deep learning (more training data).

### 7.2 LSE Dataset

- **Source:** London Stock Exchange, full LOB updates.
- **Instruments (training):** Lloyds (LLOY), Barclays (BARC), Tesco (TSCO), BT, Vodafone (VOD) -- among most liquid LSE stocks.
- **Instruments (transfer learning):** HSBC, Glencore (GLEN), Centrica (CNA), BP, ITV -- also liquid, NOT in training set.
- **Period:** 3 Jan 2017 to 24 Dec 2017 (12 months).
- **Trading hours:** 08:30:00 to 16:00:00 (no auctions).
- **Total samples:** 134+ million.
- **Average events per day per stock:** 150,000.
- **Average inter-event time:** 0.192 seconds (irregular spacing).
- **Split:** First 6 months = training, next 3 months = validation, last 3 months = testing.
- **LOB levels:** 10 per side (40 features per timestamp).
- **Raw data:** No preprocessing beyond normalization (no downsampling, no feature engineering).

---

## 8. Benchmark Results

### 8.1 FI-2010 Setup 1 (Anchored Forward, Averaged Over 9 Folds)

| Model     | k   | Accuracy % | Precision % | Recall % | F1 %  |
|-----------|-----|-----------|------------|---------|-------|
| C(TABL)   | 10  | 78.01     | 72.03      | 74.04   | 72.84 |
| **DeepLOB** | **10** | **78.91** | **78.47** | **78.91** | **77.66** |
| C(TABL)   | 50  | 74.81     | 74.58      | 74.27   | 74.32 |
| **DeepLOB** | **50** | **75.01** | **75.10** | **75.01** | **74.96** |
| C(TABL)   | 100 | 74.07     | 73.51      | 73.80   | 73.52 |
| **DeepLOB** | **100** | **76.66** | **76.77** | **76.66** | **76.58** |

### 8.2 FI-2010 Setup 2 (7-Day Train / 3-Day Test)

| Model     | k   | Accuracy % | Precision % | Recall % | F1 %  |
|-----------|-----|-----------|------------|---------|-------|
| C(TABL)   | 10  | 84.70     | 76.95      | 78.44   | 77.63 |
| **DeepLOB** | **10** | **84.47** | **84.00** | **84.47** | **83.40** |
| C(TABL)   | 20  | 73.74     | 67.18      | 66.94   | 66.93 |
| **DeepLOB** | **20** | **74.85** | **74.06** | **74.85** | **72.82** |
| C(TABL)   | 50  | 79.87     | 79.05      | 77.04   | 78.44 |
| **DeepLOB** | **50** | **80.51** | **80.38** | **80.51** | **80.35** |

### 8.3 LSE Dataset Results

**In-sample stocks (LLOY, BARC, TSCO, BT, VOD):**

| k   | Accuracy % | Precision % | Recall % | F1 %  |
|-----|-----------|------------|---------|-------|
| 20  | 70.17     | 70.17      | 70.17   | 70.15 |
| 50  | 63.93     | 63.43      | 63.93   | 63.49 |
| 100 | 61.52     | 60.73      | 61.52   | 60.65 |

**Transfer learning stocks (GLEN, HSBC, CNA, BP, ITV) -- NOT in training set:**

| k   | Accuracy % | Precision % | Recall % | F1 %  |
|-----|-----------|------------|---------|-------|
| 20  | 68.62     | 68.64      | 68.63   | 68.48 |
| 50  | 63.44     | 62.81      | 63.45   | 62.84 |
| 100 | 61.46     | 60.68      | 61.46   | 60.77 |

Transfer learning accuracy is within ~1-2% of in-sample accuracy, indicating universal feature extraction.

---

## 9. Key Design Decisions and Rationale

### 9.1 No Pooling in Convolutional Layers

- Pooling causes underfitting for LOB time-series data.
- Pooling is designed for images where feature existence matters more than location.
- In time-series, feature location (timestamp) is critical.
- Pooling is used ONLY inside the Inception Module (max-pool 3x1 with stride=1, zero-padding).

### 9.2 Stride Usage in First Convolution

- Stride (1,2) ensures parameter sharing does not conflate price and volume across adjacent levels.
- Without stride, the same filter would be applied to `{p^(i), v^(i)}` AND `{v^(i), p^(i+1)}`, which is semantically incorrect.

### 9.3 LSTM vs Fully Connected

- A FC layer with 64 units after Inception would require 630,000+ parameters.
- LSTM with 64 units: ~60,000 parameters (10x fewer).
- LSTM additionally models temporal dependencies in extracted features.

### 9.4 Inception Module as Learned Moving Averages

- `3x1` filters = short-term patterns (3 timesteps).
- `5x1` filters = medium-term patterns (5 timesteps).
- Analogous to technical analysis moving averages with different periods, but with learned weights.
- Avoids manual selection of decay parameters.

---

## 10. Trading Simulation

### 10.1 Rules

- Trade size: mu = 1 share per trade.
- Signal at time t: {-1, 0, +1} from network output.
- On +1 signal: buy mu shares at time t+5 (slippage buffer), hold until -1 signal.
- On -1 signal: sell all mu shares.
- On 0 signal: do nothing.
- Same logic for short selling.
- All positions closed at end of day (no overnight holding).
- No trades during auction periods.
- Mid-price execution assumed (no transaction costs).

### 10.2 Results

- Statistically significant positive returns across all stocks and prediction horizons.
- T-statistics used as evaluation metric (equivalent to Sharpe ratios for HFT).
- Longer prediction horizons yield lower accuracy but higher cumulative profits (more robust signals, fewer trades).

---

## 11. Sensitivity Analysis (LIME)

- LIME (Local Interpretable Model-agnostic Explanations) used for input importance.
- Locally perturbs input and observes prediction variation.
- Key finding: DeepLOB uses information across many levels and timestamps, while CNN-I concentrates on few regions due to max-pooling collapsing spatial information.
- Price and volume information at different levels and time horizons contribute to predictions, consistent with econometric understanding.

---

## 12. Hyperparameter Summary Table

| Hyperparameter                | Value                          |
|-------------------------------|--------------------------------|
| Input size                    | 100 x 40                       |
| LOB levels                    | 10 per side                    |
| Conv filters (all conv layers)| 16                             |
| Inception filters             | 32                             |
| LSTM units                    | 64                             |
| Activation (conv)             | Leaky-ReLU (slope=0.01)        |
| Activation (output)           | Softmax                        |
| Loss function                 | Categorical cross-entropy      |
| Optimizer                     | ADAM                           |
| Learning rate                 | 0.01                           |
| ADAM epsilon                  | 1                              |
| Batch size                    | 32                             |
| Early stopping patience       | 20 epochs                      |
| Epochs (FI-2010)              | ~100                           |
| Epochs (LSE)                  | ~40                            |
| Normalization window (LSE)    | 5 previous days                |
| Total parameters              | ~60,000                        |
| Number of output classes      | 3 (down, stationary, up)       |
