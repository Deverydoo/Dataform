# TimeGPT-1 -- Technical Extraction

Source: "TimeGPT-1" by Azul Garza, Cristian Challu, Max Mergenthaler-Canseco (Nixtla), arXiv:2310.03589v3, May 2024.

---

## TimeGPT Architecture

### Foundation Model Concept

TimeGPT is a pre-trained foundation model for time series forecasting. It is NOT based on an existing large language model (LLM). It is a standalone transformer architecture trained from scratch specifically for time series data, optimized to minimize forecasting error.

Core property: the model produces accurate predictions for diverse datasets not seen during training (zero-shot inference), without re-training parameters.

### Base Architecture: Encoder-Decoder Transformer

TimeGPT is a Transformer-based time series model with self-attention mechanisms, following the architecture from Vaswani et al. (2017). The disclosed structural components are:

**Encoder side:**
- Input Embedding layer
- Positional Encoding (local positional encoding to enrich input)
- Multi-Head Attention
- CNN (convolutional) block
- Add & Norm (residual connections + layer normalization)

**Decoder side:**
- Output Embedding (outputs shifted right, autoregressive)
- Positional Encoding
- Masked Multi-Head Attention (causal masking)
- Multi-Head Attention (cross-attention to encoder)
- CNN block
- Add & Norm (residual connections + layer normalization)

**Output head:**
- Linear layer mapping decoder output to the forecasting window dimension

The architecture consists of multiple layers, each with residual connections and layer normalization. CNN blocks replace the standard feed-forward layers found in the vanilla transformer.

### Input Representation

- The model takes a window of historical values as input.
- Local positional encoding is added to the input to capture temporal position within the window.
- Inputs consist of: target variable historical values `y[0:t]`, exogenous covariates `x[0:t+h]`, and events/calendar variables.
- The model handles time series of varied frequencies and characteristics, accommodating different input sizes and forecasting horizons.
- Minimal preprocessing: most time series included in raw form; processing limited to format standardization and filling missing values.

### Zero-Shot Inference

Zero-shot inference means the pre-trained model is directly applied to a new forecasting task without re-training any parameters theta on the new dataset. The model uses previous historical values as inputs and produces forecasts directly.

### Conformal Prediction for Uncertainty Quantification

Conformal prediction is a non-parametric framework for generating prediction intervals with a pre-specified level of coverage accuracy. Key properties:
- Does not require strict distributional assumptions
- Model-agnostic and domain-agnostic
- During inference on a new time series, rolling forecasts are performed on the latest available data to estimate the model's errors for that particular target series
- These historical errors are then used to construct prediction intervals

Referenced: Shafer and Vovk (2008), Stankeviciute et al. (2021).

---

## Training

### Pre-Training Dataset

- Over **100 billion data points** from the largest collection of publicly available time series at time of publication.
- Domains covered: finance, economics, demographics, healthcare, weather, IoT sensor data, energy, web traffic, sales, transport, banking.
- Characteristics: multiple seasonalities, cycles of different lengths, various trend types, varying noise levels, outliers, clean patterns, significant noise, unexpected events.
- Frequencies: monthly, weekly, daily, hourly (and potentially others).
- Preprocessing: format standardization and filling missing values. Most series included in raw form.

### Training Objective / Loss Function

The forecasting task objective is to estimate the conditional distribution (Equation 1 below). The specific loss function used for training is not explicitly named in the paper, but the model is described as "trained to minimize the forecasting error."

### Training Procedure

- Multi-day training period on a cluster of **NVIDIA A10G GPUs**.
- Implemented in **PyTorch**.
- Optimizer: **Adam** with a learning rate decay strategy that reduced the rate to **12% of its initial value**.
- Extensive hyperparameter exploration to optimize learning rates, batch sizes, and other parameters.
- Consistent with Brown et al. (2020) findings: **larger batch size** and **smaller learning rate** proved beneficial.
- Scaling laws on dataset size and model size were leveraged (consistent with findings from Zalando, OpenAI, Alibaba, Amazon).

### Fine-Tuning Procedure

Fine-tuning adjusts the pre-trained model parameters on a task-specific dataset, starting from the pre-trained weights. The paper shows that fine-tuning improves performance (measured by rMAE) over approximately 0 to 500 fine-tuning steps, with a diminishing-returns curve. The rMAE drops from approximately 0.820 (zero-shot) to approximately 0.780 after ~200-300 steps on the tested subset.

---

## Key Equations

### Equation 1: Forecasting Task Formulation

```
P(y[t+1:t+h] | y[0:t], x[0:t+h]) = f_theta(y[0:t], x[0:t+h])
```

**Variable definitions:**
- `f_theta` : forecasting model parameterized by theta, mapping X -> Y
- `X` : feature space, defined as X = {y[0:t], x[0:t+h]}
- `Y` : dependent variable space, defined as Y = {y[t+1:t+h]}
- `h` : forecast horizon
- `y` : target time series
- `y[0:t]` : historical values of the target series from time 0 to t
- `y[t+1:t+h]` : future values to be predicted
- `x` : exogenous covariates
- `x[0:t+h]` : exogenous covariates from time 0 to t+h (includes future known covariates)
- `theta` : model parameters

### Equation 2: Evaluation Metrics

**Relative Mean Absolute Error (rMAE):**

```
rMAE = [ sum_{i=1}^{n} sum_{t=1}^{h} |y_{i,t} - y_hat_{i,t}| ] / [ sum_{i=1}^{n} sum_{t=1}^{h} |y_{i,t} - y_hat_base_{i,t}| ]
```

**Relative Root Mean Square Error (rRMSE):**

```
rRMSE = [ sum_{i=1}^{n} sqrt( sum_{t=1}^{h} (y_{i,t} - y_hat_{i,t})^2 ) ] / [ sum_{i=1}^{n} sqrt( sum_{t=1}^{h} (y_{i,t} - y_hat_base_{i,t})^2 ) ]
```

**Variable definitions:**
- `n` : number of time series in the dataset
- `h` : forecast horizon
- `y_{i,t}` : actual value of time series i at time t
- `y_hat_{i,t}` : predicted value of time series i at time t
- `y_hat_base_{i,t}` : prediction from the baseline model (Seasonal Naive) for time series i at time t
- Both metrics are normalized against Seasonal Naive performance (Seasonal Naive = 1.000 by definition)
- Scale-independent, enabling cross-frequency and cross-dataset comparisons
- Normalization applied at a global scale for each comprehensive dataset

### Transfer Learning Formulation

- Source dataset: `D_s = {(X, y) | X in X, y in Y}` (the large pre-training corpus)
- Target dataset: `D_t` (the new unseen dataset)
- Zero-shot: pre-trained theta applied directly to D_t without parameter updates
- Fine-tuning: theta is further optimized on D_t starting from pre-trained values

---

## Experimental Results

### Test Set

- Over **300,000 time series** never seen during training.
- Domains: finance, web traffic, IoT, weather, demand, electricity.
- Evaluation on the last forecasting window of each time series.

### Forecasting Horizons by Frequency

| Frequency | Horizon (h) |
|-----------|-------------|
| Monthly   | 12          |
| Weekly    | 1           |
| Daily     | 7           |
| Hourly    | 24          |

### Zero-Shot Performance: Full Benchmark Table (rMAE / rRMSE)

| Model            | Monthly rMAE | Monthly rRMSE | Weekly rMAE | Weekly rRMSE | Daily rMAE | Daily rRMSE | Hourly rMAE | Hourly rRMSE |
|------------------|-------------|---------------|------------|-------------|-----------|------------|------------|-------------|
| ZeroModel        | 2.045       | 1.568         | 6.075      | 6.075       | 2.989     | 2.395      | 10.255     | 8.183       |
| HistoricAverage  | 1.349       | 1.106         | 4.188      | 4.188       | 2.509     | 2.057      | 2.216      | 1.964       |
| SeasonalNaive    | 1.000       | 1.000         | 1.000      | 1.000       | 1.000     | 1.000      | 1.000      | 1.000       |
| Theta            | 0.839       | 0.764         | 1.061      | 1.061       | 0.841     | 0.811      | 1.163      | 1.175       |
| DOTheta          | 0.799       | 0.734         | 1.056      | 1.056       | 0.837     | 0.806      | 1.157      | 1.169       |
| ETS              | 0.942       | 0.960         | 1.079      | 1.079       | 0.944     | 0.970      | 0.998      | 1.009       |
| CES              | 1.024       | 0.946         | 1.002      | 1.002       | 0.919     | 0.899      | 0.878      | 0.896       |
| ADIDA            | 0.852       | 0.769         | 1.364      | 1.364       | 0.908     | 0.868      | 2.307      | 2.207       |
| IMAPA            | 0.852       | 0.769         | 1.364      | 1.364       | 0.908     | 0.868      | 2.307      | 2.207       |
| CrostonClassic   | 0.989       | 0.857         | 1.805      | 1.805       | 0.995     | 0.933      | 2.157      | 2.043       |
| LGBM             | 1.050       | 0.913         | 0.993      | 0.993       | 2.506     | 2.054      | 0.733      | 0.709       |
| LSTM             | 0.836       | 0.778         | 1.002      | 1.002       | 0.852     | 0.832      | 0.974      | 0.955       |
| DeepAR           | 0.988       | 0.878         | 0.987      | 0.987       | 0.853     | 0.826      | 1.028      | 1.028       |
| TFT              | 0.752       | 0.700         | 0.954      | 0.954       | 0.817     | 0.791      | 1.120      | 1.112       |
| NHITS            | 0.738       | 0.694         | 0.883      | 0.883       | 0.788     | 0.771      | 0.829      | 0.860       |
| **TimeGPT**      | **0.727**   | **0.685**     | **0.878**  | **0.878**   | **0.804** | **0.780**  | **0.852**  | **0.878**   |

### Key Observations from Results

- TimeGPT zero-shot ranks among top-3 performers across all frequencies and both metrics.
- Monthly: TimeGPT achieves lowest rMAE (0.727) and rRMSE (0.685), beating NHITS (0.738/0.694).
- Weekly: TimeGPT achieves lowest rMAE (0.878), tied/close with NHITS (0.883).
- Daily: NHITS slightly outperforms (0.788 vs 0.804 rMAE), but TimeGPT remains competitive.
- Hourly: LGBM leads (0.733 rMAE), NHITS second (0.829), TimeGPT third (0.852).
- Statistical models (Theta, DOTheta, ETS, CES) are consistently outperformed by top DL models.
- LGBM (ML) performance is highly variable across frequencies (excellent hourly, poor daily).
- Deep learning methods as a group outperform statistical methods as a group (shown in violin/bean plot, Figure 4).

### Fine-Tuning Results

- Fine-tuning on a subset of the test set reduces rMAE from ~0.820 (zero-shot) to ~0.780 after approximately 200-300 steps.
- Diminishing returns observed beyond ~300 steps; curve flattens around 0.780-0.785.

### Computational Efficiency

| Method Category              | Average Speed per Series         |
|------------------------------|----------------------------------|
| TimeGPT (zero-shot GPU)      | **0.6 milliseconds** (inference only) |
| Seasonal Naive               | ~0.6 milliseconds (comparable)   |
| Statistical methods (Numba)  | ~600 milliseconds (train + inference) |
| Global models (LGBM, LSTM, NHITS) | ~57 milliseconds (train + inference) |

TimeGPT achieves orders-of-magnitude speedup over traditional methods because it requires only inference, no training.

---

## Capabilities

### Forecasting Horizons Supported

The model accommodates different forecasting horizons. Tested configurations: 12 (monthly), 1 (weekly), 7 (daily), 24 (hourly). The architecture is designed to handle variable horizon lengths.

### Frequency Handling

Handles time series of varied frequencies: monthly, weekly, daily, hourly (and potentially others). The model processes varied frequencies without architectural changes.

### Exogenous Variables

The model accepts exogenous covariates `x[0:t+h]` and calendar/event variables as additional inputs alongside the target time series. These are fed into the encoder alongside the target variable.

### Multivariate Handling

The paper distinguishes between "single series forecasting" and "multiple series forecasting" (Figure 1). The model can forecast multiple series. Exogenous variables provide multivariate input capability.

### Anomaly Detection

Anomaly detection is listed as a supported feature in the SDK/API documentation (Appendix A), but no technical details or results are provided in this paper.

### Cross-Domain Transfer

The core claim is cross-domain transfer: a model trained on finance, economics, healthcare, weather, IoT, energy, web traffic, sales, transport, and banking can forecast unseen series from any of these (and potentially other) domains without re-training. The test set explicitly spans domains different from training.

### Fine-Tuning for Domain Adaptation

When zero-shot is insufficient, the model supports fine-tuning on task-specific data starting from pre-trained parameters, achieving further accuracy gains.

---

## Hyperparameters / Model Specs

### Disclosed Specifications

| Specification               | Value / Detail                          |
|-----------------------------|----------------------------------------|
| Architecture type           | Transformer encoder-decoder            |
| Attention mechanism         | Multi-head self-attention (Vaswani et al., 2017) |
| Internal blocks             | CNN blocks (replace standard FFN)      |
| Normalization               | Layer normalization                    |
| Residual connections        | Yes, in every layer                    |
| Output projection           | Linear layer to forecasting window dim |
| Positional encoding         | Local positional encoding              |
| Training hardware           | NVIDIA A10G GPU cluster                |
| Training duration           | Multi-day                              |
| Framework                   | PyTorch                                |
| Optimizer                   | Adam                                   |
| Learning rate schedule      | Decay to 12% of initial value          |
| Batch size strategy         | Larger batch sizes preferred           |
| Learning rate strategy      | Smaller learning rates preferred       |
| Training data scale         | 100+ billion data points               |
| Test data scale             | 300,000+ time series                   |

### Not Disclosed

The paper does NOT disclose:
- Exact model size (number of parameters)
- Number of layers
- Hidden dimensions / embedding dimensions
- Number of attention heads
- Context window length (maximum historical input length)
- Maximum prediction length
- Vocabulary / tokenization scheme for time series values
- Specific loss function name (e.g., MSE, MAE, likelihood-based)
- Exact conformal prediction implementation details (quantile levels, calibration set size)
- CNN kernel sizes or architecture details within the blocks

---

## Referenced Model Taxonomy

### Models Benchmarked (by category)

**Baselines:** ZeroModel, HistoricAverage, SeasonalNaive

**Statistical (local, per-series):** Theta, DOTheta, ETS, CES, ADIDA, IMAPA, CrostonClassic

**Machine Learning (global):** LGBM (LightGBM)

**Deep Learning (global):** LSTM, DeepAR, TFT (Temporal Fusion Transformer), NHITS

**Excluded from benchmarks (computational cost):** ARIMA, Prophet

### Training Paradigm Distinction

- Statistical models: individually trained on each test series using historical values before the last forecasting window (local approach).
- ML and DL models: trained globally using all time series in the test set for each frequency (global approach).
- TimeGPT: pre-trained on separate 100B+ point dataset; applied zero-shot to test set (no training on test data at all).
