# From Factor Models to Deep Learning: Technical Extraction

**Source:** Ye, Goswami, Gu, Uddin, Wang -- "From Factor Models to Deep Learning: Machine Learning in Reshaping Empirical Asset Pricing" (arXiv:2403.06779, Mar 2024)

---

## 1. Fundamental Problem Statement

The core asset pricing problem: estimate the conditional expected excess return over the risk-free rate.

**Excess return decomposition:**

```
y_{i,t} = E[y_{i,t}] + epsilon_{i,t}
```

where `y_{i,t}` is individual asset excess return over the risk-free rate (T-bill rate), with `E[epsilon_{i,t}] = 0`.

**Conditional expected return as parameterized function:**

```
y_{i,t} = g(x_{i,t-1}; theta) + epsilon_{i,t}
```

where `g(.)` is an unknown function of P-dimensional predictors `x_{i,t}`, parameterized by `theta`.

Predictors X: asset-specific characteristics + macroeconomic factors.

---

## 2. Traditional Factor Models

### 2.1 General Factor Model

Replace high-dimensional predictors X with low-dimensional factors F:

```
y_{i,t} = alpha_{i,t-1} + beta'_{i,t-1} * f_t + epsilon_{i,t}
```

- `beta_{i,t-1}`: asset i's loading (exposure) to common risk factor `f_t`
- `alpha_{i,t-1}`: mispricing component
- Null hypothesis: `alpha_{i,t-1} = 0` for all i, t

### 2.2 Two Approaches to Latent Factor Estimation

**Approach 1 -- Characteristic-Sorted Portfolios:**
- Construct long-short portfolios based on prior cross-sectional knowledge
- Long-short portfolio returns serve as observable factors
- Example: Fama-French Five-Factor Model (Market, Size, Value, Profitability, Investment)
- Limitation: requires complete knowledge of cross-section of returns; empirically fails to produce `alpha = 0`

**Approach 2 -- Statistical Factor Analysis (PCA-based):**
- Estimate latent factors from the panel of realized returns
- No ex-ante knowledge of cross-section required
- Static PCA: two-step estimation
  1. Combine large set of predictors into linear combinations of latent factors f
  2. Use latent factors to model excess returns via predictive regression
- Relationship: `x_t = beta * f_t + u_t`

### 2.3 Instrumented PCA (IPCA)

Time-varying factors and factor loadings (Kelly et al., 2019):

```
y_{i,t} = x_{i,t-1} * beta_i * f_t + epsilon_{i,t}
```

- `beta in R^{N x K}`: factor loadings
- `f_t in R^{K x T}`: latent factors
- `epsilon_{i,t}`: composite error including `alpha_{i,t-1}`

### 2.4 Limitations of Traditional Models

1. Ever-increasing factor universe leads to large number of free parameters -- inefficient estimation via regression
2. Linear approximation of risk exposure; theoretical models suggest nonlinearity in return dynamics

---

## 3. ML Objective Function for Asset Pricing

The general supervised learning objective:

```
g_hat(.) = argmin_{g in G} SUM_{i=1}^{N} SUM_{t=1}^{T} (y_{i,t} - g(x_{i,t-1}))^2
```

Model classes for `g(.)` (ordered by complexity):
- Linear regression
- Non-linear decision trees
- Ensemble random forests
- Gradient-boosted regression trees (GBRT)
- Deep neural networks

Application domains: equities, cryptocurrencies, futures, options.

---

## 4. Temporal Models

### 4.1 General Formulation

```
y_{i,t}, y_{i,t+1}, ... = g_t(x_{i,t-1}, x_{i,t-2}, ..., x_{i,1})
```

- `y_{i,t}, y_{i,t+1}, ...`: predicted values (future prices/returns) at future times
- `x_{i,t-1}, x_{i,t-2}, ..., x_{i,1}`: historical input feature vectors up to time t-1
- Each `x_{i,t}` encapsulates lagged prices, returns, and/or financial indicators
- `g_t`: captures intrinsic temporal patterns and dependencies

### 4.2 Architecture Progression

| Generation | Architecture | Key Property |
|---|---|---|
| Classical | ARIMA, VAR | Linear temporal modeling |
| Early DL | Feedforward NN | Non-linear static mapping |
| Sequence DL | RNN, LSTM | Sequential hidden state; long-term memory (LSTM) |
| All-MLP | N-BEATS | Neural basis expansion; residual connections |
| All-MLP | TS-Mixer | Mixer blocks; joint feature-temporal correlation learning; residual connections |

**N-BEATS:** Dominates time-series benchmark leaderboards.
**TS-Mixer:** Outperforms both RNNs and N-BEATS on S&P 500 return prediction.

### 4.3 Multi-Scale Approaches

Transition from single-scale to multi-scale (time, frequency, resolution):

| Technique | Scale Method | Capability |
|---|---|---|
| MTDNN | Wavelet-based + downsampling | Fine-grained to broad trend analysis |
| Multi-Scale Gaussian Transformer | Multi-Scale Gaussian Prior | Long-term, short-term, and hierarchical dependency capture |

Scale decomposition techniques: Fourier transform, wavelets, downsampling.

---

## 5. Spatio-Temporal Models

### 5.1 General Formulation

```
y_t, y_{t+1}, ... = g_t[g_s(X_{t-1}), g_s(X_{t-2}), ..., g_s(X_1)]
```

- `X_t, X_{t-1}, ..., X_1`: feature matrices of asset pool at past timestamps
- `g_s`: spatial correlation function (typically GNN)
- `g_t`: temporal evolution function (typically LSTM/GRU)
- Pipeline: spatial correlations first (via `g_s`), then temporal evolution (via `g_t`)

### 5.2 Graph Construction Methods

| Model | Graph Construction | Graph Method |
|---|---|---|
| Chen et al. (2018) | Shareholding ratios (corporate interconnections) | DeepWalk, LINE, node2vec, GCN |
| HATS | Ownership and organizational structures | Graph Attention Network (GAT) |
| AD-GAT | Attribute-sensitive momentum spillovers | Attribute-mattered aggregator (dynamic, not pre-defined) |
| RSR | Sector similarity + supply-chain network | Temporal Graph Convolution |
| STHAN-SR | Industry and corporate connections (hyperedges) | Spatio-Temporal Hypergraph Attention Network |
| Uddin et al. (2021) | Pearson Correlation (positive/negative graphs) | Attention-based evolving networks |
| DTML | Asymmetric, dynamic stock correlations | Transformer-based encoding |

### 5.3 Temporal Integration

Spatial models integrate recurrent networks (LSTM, GRU) to form full spatio-temporal architectures.

---

## 6. Portfolio Optimization

### 6.1 Modern Portfolio Theory (MPT)

Utility-maximizing portfolio solution:

```
w* = (1 / gamma) * Sigma^{-1} * mu
```

- `mu in R^N`: mean return vector for N assets
- `Sigma in R^{N x N}`: covariance matrix of returns
- `gamma`: investor risk aversion parameter
- `w* in R^N`: optimal portfolio weights

Two optimization approaches:
1. **Estimate-then-optimize:** Estimate `mu` and `Sigma` via models, then solve for `w`
2. **Direct weight parameterization:** Use models to directly optimize portfolio weights

### 6.2 Supervised Learning for Portfolio Construction

Given ML predictions (returns, ranks, movements):
- **Long-short from return/rank prediction:** Long highest predicted returns/ranks; short lowest
- **Long-short from movement prediction:** Long predicted upward; short predicted downward
- Position at t-1, liquidate/rebalance at t

**Weight allocation schemes:**
- Equal weight: `w_{i,t} = 1/N`
- Value-weighted: `w_{i,t} = v_{i,t} / SUM_{i=1}^{N} v_{i,t}`

### 6.3 Reinforcement Learning Framework

**State-Action-Reward formulation:**
- State `s_t`: (historical prices `x_t`, previous portfolio weight `w_{t-1}`)
- Action `a_t`: portfolio weight vector `w_t`
- Reward `r_t`: periodic logarithmic return

**Objective -- maximize action-value function:**

```
Q^pi(s, a) = E[ SUM_{i=t}^{inf} gamma^i * r_{t+i} | s_t = x_t, a_t = w_{t-1} ]
```

where `gamma^i` is the discount factor.

### 6.4 Key RL Architectures

| Model | Approach | Key Feature |
|---|---|---|
| EIIE | Financial-model-free RL (CNN/RNN/LSTM) | Online Stochastic Batch Learning (OSML); zero market impact assumption |
| AlphaStock / Xu et al. | Attention-based RL | Inter-asset relationship modeling; short sales under simple assumptions |
| iRDPG | Imitative Recurrent Deterministic Policy Gradient | Behavior cloning; intraday greedy actions |
| Hierarchical RL (Wang et al.) | Hierarchical framework | Small-sized limit orders at desired prices/quantities; minimizes trading costs |
| Smart Trader | Normalizing flows + GBM process | Portfolio weight + optimal trading point; excess intraday profit |
| Meta Trader | Meta-learning RL | Acquires diverse policies; selects best policy per market condition |
| Margin Trader | RL with margin accounts | Integrates margin constraints for realistic long/short positions |

---

## 7. Dimensionality Reduction Techniques

### 7.1 Classical Methods

| Method | Type | Mechanism |
|---|---|---|
| PCA | Unsupervised | Linear combination of latent factors from realized returns |
| PLS (Partial Least Squares) | Supervised | Maximizes covariance between predictors and response |
| LASSO (two-step) | Regularized regression | Identifies most influential factors among factor zoo (e.g., 150 factors) |

### 7.2 Deep Learning Methods

| Model | Architecture | Mechanism |
|---|---|---|
| Autoencoder (Gu et al., 2021) | Autoencoder | Joint modeling of asset pricing characteristics and excess return to learn latent factors |
| FactorVAE | VAE + Dynamic Factor Model | Factors as latent variables; addresses low signal-to-noise ratios |
| DiffusionVAE (D-Va) | Hierarchical VAE + Diffusion | Processes stochastic stock data; denoised predictions |

---

## 8. Missing Data Imputation Methods

| Method | Technique | Application |
|---|---|---|
| Coupled matrix factorization | Matrix factorization augmented with firm characteristics | Impute missing analyst earnings forecasts |
| Transformer imputation | Attention-based transformer | Impute missing firm characteristics |
| Tensor imputation | Coupled tensor completion | Spatio-temporal financial data |

Naive baselines (with known drawbacks): remove observations, impute with zeros, impute with class mean.

---

## 9. Alternate Data Integration

### 9.1 Image-Based

- Time-series recast as image classification task via CNN
- Multi-asset price data transformed to 2D images; video prediction algorithms for market dynamics

### 9.2 Text-Based

| Model | Data Source | Architecture |
|---|---|---|
| TRAN (Time-Aware Graph Relational Attention Network) | Stock descriptions + financial data | Graph relational attention |
| PEN (Prediction-Explanation Network) | Social media + inter-stock relations + financial indicators | Sentiment-aware prediction |

### 9.3 Audio-Based

| Model | Data Source | Architecture |
|---|---|---|
| VolTAGE | Executive earnings call vocal cues + financial data | GCN + inter-modal multi-utterance attention |

---

## 10. Denoising and Non-IID Adaptation

### 10.1 Contrastive Learning

| Model | Mechanism |
|---|---|
| Co-CPC (Copula-based Contrastive Predictive Coding) | Filters noise; contrasts consecutive time-point data; enhances stock representations |
| CMLF (Contrastive Multi-Granularity Learning Framework) | Dual contrastive approach; handles data granularity and temporal relationships |

### 10.2 Mixture of Experts (MoE)

Architecture: set of specialized sub-models (experts) + gating network (router).

| Model | Mechanism |
|---|---|
| TRA (Temporal Routing Adaptor) | Optimizes data-to-predictor assignment; temporal pattern recognition |
| PASN (Pattern Adaptive Specialist Network) | Pattern Adaptive Training; autonomous adaptation to evolving market dynamics |

Core advantage: breaks conventional i.i.d. assumption; dynamically adapts to market heterogeneity.

---

## 11. Evaluation Metrics -- Complete Reference

### 11.1 Return Prediction

| Metric | Formula | Direction |
|---|---|---|
| RMSE | `sqrt( (1/T)(1/N) * SUM (y_{i,t} - y_hat_{i,t})^2 )` | Lower is better |
| MAE | `(1/T)(1/N) * SUM |y_{i,t} - y_hat_{i,t}|` | Lower is better |
| MAPE | `(1/T)(1/N) * SUM |(y_{i,t} - y_hat_{i,t}) / y_{i,t}| * 100%` | Lower is better |

### 11.2 Movement Prediction (Classification)

| Metric | Formula | Direction |
|---|---|---|
| Accuracy | `SUM I(D_hat_i = D_i) / N` | Higher is better |
| Precision | `TP / (TP + FP)` | Higher is better |
| Recall | `TP / (TP + FN)` | Higher is better |
| F1 Score | `2 * Precision * Recall / (Precision + Recall)` | Higher is better |
| MCC | `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Higher is better |

MCC is preferred for imbalanced financial datasets.

### 11.3 Ranking Prediction

| Metric | Formula | Direction |
|---|---|---|
| MAP | `(1/T)(1/K) * SUM P(k)_t * rel(k)_t` | Higher is better |
| MRR | `(1/T) * SUM 1/rank_t` | Higher is better |
| NDCG | `DCG / IDCG` where `DCG = SUM rel(k) / log2(k+1)` | Higher is better |

### 11.4 Portfolio Optimization

| Metric | Formula | Direction |
|---|---|---|
| Accumulated Return | `PROD_{t=1}^{T} SUM_i w_{i,t-1} * y_{i,t}` | Higher is better |
| Volatility | `sigma(PR_t)` | Lower is better |
| Maximum Drawdown (MDD) | `max_{0<=a<=b<=T} 1 - (PROD_{t=1}^{b} SUM_i w_{i,t-1}*y_{i,t}) / (PROD_{t=1}^{a} SUM_i w_{i,t-1}*y_{i,t})` | Higher is better |
| Sharpe Ratio | `(SUM PR_t / T) / sigma(PR_t)` | Higher is better |
| Calmar Ratio | `(SUM PR_t / T) / MDD` | Higher is better |
| Sortino Ratio | `(SUM PR_t / T) / DD` | Higher is better |

Notation: `PR` = portfolio return, `DD` = downside deviation, `MDD` = maximum drawdown, `sigma(.)` = standard deviation.

---

## 12. Taxonomy of ML Methods in Asset Pricing

```
ML in Asset Pricing
|
|-- Risk Assessment & Price Prediction (Section 2)
|   |-- Traditional Factor Models
|   |   |-- Characteristic-sorted portfolios (Fama-French)
|   |   |-- PCA / IPCA
|   |
|   |-- AI-Augmented Prediction
|       |-- Cross-sectional return prediction (Eq. 5)
|       |-- Time-series return prediction (Eq. 6)
|       |-- Movement prediction (classification)
|       |-- Ranking prediction
|       |
|       |-- Temporal Models
|       |   |-- ARIMA, VAR (classical)
|       |   |-- RNN, LSTM, GRU
|       |   |-- All-MLP: N-BEATS, TS-Mixer
|       |   |-- Multi-scale: MTDNN, Multi-Scale Gaussian Transformer
|       |
|       |-- Spatio-Temporal Models (Eq. 7)
|           |-- GCN, GAT, Hypergraph Attention
|           |-- Temporal Graph Convolution
|           |-- Transformer-based (DTML)
|
|-- Portfolio Optimization (Section 3)
|   |-- Supervised Learning (estimate mu, Sigma -> optimize w)
|   |-- Reinforcement Learning (directly optimize w)
|       |-- EIIE, AlphaStock, iRDPG, Hierarchical RL
|       |-- Smart Trader, Meta Trader, Margin Trader
|
|-- Advanced Techniques (Section 4)
    |-- Dimensionality Reduction
    |   |-- PCA, PLS, LASSO
    |   |-- Autoencoder, FactorVAE, DiffusionVAE
    |
    |-- Missing Data Imputation
    |   |-- Matrix factorization, Transformer, Tensor completion
    |
    |-- Alternate Data Integration
    |   |-- Image (CNN, video prediction)
    |   |-- Text (TRAN, PEN)
    |   |-- Audio (VolTAGE)
    |
    |-- Denoising & Non-IID Adaptation
        |-- Contrastive Learning (Co-CPC, CMLF)
        |-- Mixture of Experts (TRA, PASN)
```

---

## 13. Key Hyperparameters and Design Choices

| Decision Point | Options / Parameters |
|---|---|
| Factor dimensionality K (IPCA) | Number of latent factors |
| Risk aversion gamma (MPT) | Investor-specific; scales portfolio weights |
| RL discount factor gamma | Controls future reward weighting in Q-function |
| Lookback window | Length of historical sequence `(x_{i,t-1}, ..., x_{i,1})` |
| Number of experts (MoE) | Specialized sub-models for market regimes |
| Graph construction method | Shareholding, sector, supply-chain, correlation, ownership |
| Scale decomposition | Fourier, wavelet, downsampling (multi-scale models) |
| Portfolio rebalancing frequency | Period between position establishment and liquidation |
| Weight scheme | Equal (1/N) vs. value-weighted |
| EIIE training scheme | Online Stochastic Batch Learning (OSML) |

---

## 14. Open Technical Challenges

1. **Overfitting in complex models** (Transformers, GNNs): good training performance, poor out-of-sample generalization. Mitigations: meta-learning, one-shot learning, ensemble methods, early stopping, dropout.
2. **No-arbitrage constraint**: identified mispricings evaporate quickly, rendering models obsolete. Mitigations: online learning, meta-learning.
3. **Low signal-to-noise ratio**: financial data inherently noisy. Mitigations: contrastive learning, VAE-based denoising, diffusion models.
4. **Non-IID data**: financial data violates i.i.d. assumption due to regime changes, structural breaks. Mitigations: MoE with temporal routing, pattern-adaptive training.
5. **Missing data**: random and non-random missingness. Mitigations: coupled matrix/tensor factorization, transformer imputation.
6. **No unified benchmark dataset**: unlike CV/NLP, no standard test set for asset pricing model evaluation.
