# Deep Portfolio Management -- Curated Technical Extraction

**Source**: Jiang, Xu, Liang (2017). "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem." arXiv:1706.10059v2.

**Scope**: Financial-model-free RL framework for continuous portfolio management. EIIE topology, PVM, OSBL, CNN/RNN/LSTM policy networks. Cryptocurrency backtests.

---

## 1. PROBLEM FORMULATION

### 1.1 Portfolio Management as Sequential Decision

Portfolio of m+1 assets (including cash as asset 0). Time divided into equal periods of length T. At each period boundary, agent reallocates fund across assets.

**Closing price vector** for period t:
```
v_t = (v_{0,t}, v_{1,t}, ..., v_{m,t})^T
```
Cash is always v_{0,t} = 1 for all t (quoted currency = Bitcoin in experiments).

**Highest/Lowest price vectors**: v_t^{hi}, v_t^{lo} with v_{0,t}^{hi} = v_{0,t}^{lo} = 1.

### 1.2 Price Relative Vector

Element-wise division of consecutive closing prices:
```
y_t := v_t / v_{t-1} = (1, v_{1,t}/v_{1,t-1}, v_{2,t}/v_{2,t-1}, ..., v_{m,t}/v_{m,t-1})^T     [Eq.1]
```
Captures per-asset return ratios for period t.

### 1.3 Portfolio Weight Vector

```
w_t in R^{m+1},   sum_i w_{t,i} = 1,   w_{t,i} >= 0   for all t
```
w_{t,i} = proportion of total portfolio value allocated to asset i after rebalancing at time t.

**Initial condition**:
```
w_0 = (1, 0, ..., 0)^T     [Eq.5]
```
All capital starts in cash (Bitcoin).

### 1.4 Portfolio Value Dynamics (No Transaction Cost)

```
p_t = p_{t-1} * (y_t . w_{t-1})     [Eq.2]
```

**Rate of return**:
```
rho_t = p_t / p_{t-1} - 1 = y_t . w_{t-1} - 1     [Eq.3]
```

**Log return**:
```
r_t = ln(p_t / p_{t-1}) = ln(y_t . w_{t-1})     [Eq.4]
```

**Final portfolio value (no cost)**:
```
p_f = p_0 * exp(sum_{t=1}^{t_f+1} r_t) = p_0 * prod_{t=1}^{t_f+1} (y_t . w_{t-1})     [Eq.6]
```

---

## 2. TRANSACTION COST MODEL

### 2.1 Weight Drift from Market Movement

After price movement during period t, portfolio weights drift from w_{t-1} to:
```
w'_t = (y_t * w_{t-1}) / (y_t . w_{t-1})     [Eq.7]
```
where * is element-wise multiplication. Rebalancing from w'_t to w_t incurs transaction cost.

### 2.2 Transaction Remainder Factor

Rebalancing shrinks portfolio value by factor mu_t in (0, 1]:
```
p_t = mu_t * p'_t     [Eq.8]
```

**Modified returns with transaction cost**:
```
rho_t = mu_t * (y_t . w_{t-1}) - 1     [Eq.9]
r_t = ln(mu_t * y_t . w_{t-1})     [Eq.10]
```

**Final portfolio value with cost**:
```
p_f = p_0 * prod_{t=1}^{t_f+1} mu_t * (y_t . w_{t-1})     [Eq.11]
```

### 2.3 Solving for mu_t

Selling revenue:
```
(1 - c_s) * p'_t * sum_{i=1}^{m} (w'_{t,i} - mu_t * w_{t,i})^+     [Eq.12]
```
where (x)^+ = max(x, 0) = ReLU(x), c_s = sell commission rate.

Buying expenditure (implicit equation from cash balance):
```
(1 - c_p) * [w'_{t,0} + (1 - c_s) * sum_{i=1}^m (w'_{t,i} - mu_t * w_{t,i})^+ - mu_t * w_{t,0}]
  = sum_{i=1}^m (mu_t * w_{t,i} - w'_{t,i})^+     [Eq.13]
```

Simplified (using identity (a-b)^+ - (b-a)^+ = a - b):
```
mu_t = [1 / (1 - c_p * w_{t,0})] * [1 - c_p * w'_{t,0} - (c_s + c_p - c_s*c_p) * sum_{i=1}^m (w'_{t,i} - mu_t * w_{t,i})^+]     [Eq.14]
```

**mu_t is implicit** (appears inside ReLU on RHS). Not analytically solvable.

### 2.4 Iterative Convergence (Theorem 1)

Define:
```
f(mu) := [1 / (1 - c_p * w_{t,0})] * [1 - c_p * w'_{t,0} - (c_s + c_p - c_s*c_p) * sum_{i=1}^m (w'_{t,i} - mu * w_{t,i})^+]
```

The sequence {mu_t^(k)} with mu_t^(0) = mu_star and mu_t^(k) = f(mu_t^(k-1)) converges to mu_t for any mu_star in [0, 1].

**Properties of f(mu)**:
- Monotonically increasing (Lemma A.1)
- f(0) > 0 for commission rates < 38% (Lemma A.2)
- f(1) <= 1 (Lemma A.3)
- Convergence proved via Monotone Convergence Theorem

**Practical initial guess** (when c_p = c_s = c):
```
mu_star = c * sum_{i=1}^m |w'_{t,i} - w_{t,i}|     [Eq.16]
```

**Training**: Fixed k iterations of the sequence.
**Backtesting**: Dynamic k determined by tolerance delta: first k where |mu_t^(k) - mu_t^(k-1)| < delta.

General dependency:
```
mu_t = mu_t(w_{t-1}, w_t, y_t)     [Eq.17]
```

---

## 3. STATE AND INPUT REPRESENTATION

### 3.1 Asset Pre-Selection

- Top 11 non-cash assets by 30-day trading volume before backtest start
- Portfolio size m+1 = 12 (11 assets + Bitcoin cash)
- Volume measured over 30 days preceding experiment to avoid survival bias
- Volume from before backtest start, never from future

### 3.2 Price Tensor

Input tensor X_t has shape **(f, n, m)** where:
- f = 3 features (closing, highest, lowest prices)
- n = 50 periods (lookback window = 25 hours at 30-min periods)
- m = 11 non-cash assets

**Normalization**: All prices divided by latest closing price of respective asset:
```
X_t = stack[V_t^{lo}, V_t^{hi}, V_t]     [Eq.18]
```

**Normalized closing price matrix**:
```
V_t = [v_{t-n+1} / v_t,  v_{t-n+2} / v_t,  ...,  v_{t-1} / v_t,  1]
```

**Normalized high price matrix**:
```
V_t^{hi} = [v_{t-n+1}^{hi} / v_t,  ...,  v_t^{hi} / v_t]
```

**Normalized low price matrix**:
```
V_t^{lo} = [v_{t-n+1}^{lo} / v_t,  ...,  v_t^{lo} / v_t]
```

Division is element-wise across asset dimension. Last column of V_t is always vector of 1s.

### 3.3 Missing Data Treatment

Assets lacking early history (appeared recently): NANs replaced with flat price movements (decay rate 0). EIIE architecture prevents networks from learning asset identity, so flat fill does not cause avoidance behavior.

---

## 4. REINFORCEMENT LEARNING FORMULATION

### 4.1 MDP Components

**State**:
```
s_t = (X_t, w_{t-1})     [Eq.20]
```
Two parts:
- External state: price tensor X_t (market observation)
- Internal state: previous portfolio vector w_{t-1} (agent's position)

Portfolio value p_t excluded from state under zero-market-impact assumption.

**Action**:
```
a_t = w_t     [Eq.19]
```
Continuous action space: (m+1)-dimensional simplex (non-negative, sum to 1).

**Transition**: External state transitions independent of agent actions (zero market impact). Internal state determined by agent's action.

**Reward**: Logarithmic return for period t+1:
```
r_{t+1} = ln(mu_{t+1} * y_{t+1} . w_t)     [Eq.10]
```

### 4.2 Cumulated Reward (Objective)

Average log-cumulated return:
```
R = (1 / t_f) * ln(p_f / p_0) = (1 / t_f) * sum_{t=1}^{t_f+1} ln(mu_t * y_t . w_{t-1})     [Eq.21-22]
```

Denominator t_f ensures fairness across different-length episodes, enabling mini-batch training.

### 4.3 Full Exploitation (No Exploration)

Two key distinctions from standard RL:
1. **Exact reward function**: Domain knowledge fully known. No need to estimate value functions. Reward is explicit, not approximated.
2. **All episodic rewards equally important**: Discount factor = 0. r_t/t_f serves as action-value function directly.

Zero market impact means same market history segment can evaluate different action sequences. No exploration needed -- local optima avoided via random parameter initialization.

### 4.4 Deterministic Policy Gradient

Policy: pi_theta : S -> A (deterministic mapping from state to action)

**Performance metric**:
```
J_{[0,t_f]}(pi_theta) = R(s_1, pi_theta(s_1), ..., s_{t_f}, pi_theta(s_{t_f}), s_{t_f+1})     [Eq.23]
```

**Parameter update (gradient ascent)**:
```
theta <- theta + lambda * grad_theta J_{[0,t_f]}(pi_theta)     [Eq.24]
```

**Mini-batch update** for batch covering [t_{b1}, t_{b2}]:
```
theta <- theta + lambda * grad_theta J_{[t_{b1},t_{b2}]}(pi_theta)     [Eq.25]
```

Optimizer: Adam (Kingma and Ba, 2014) with learning rate 3e-5.

---

## 5. EIIE ARCHITECTURE (Ensemble of Identical Independent Evaluators)

### 5.1 Core Concept

Each asset gets its own **Identical Independent Evaluator (IIE)** -- a sub-network that:
- Inspects the price history of ONE asset
- Outputs a scalar "voting score" for that asset's growth potential
- Shares all parameters with IIEs for other assets

Voting scores from all m assets + cash bias -> softmax -> portfolio weight vector w_t.

**Key constraint**: All convolutional kernels have height 1, making asset rows completely independent until softmax. Parameters shared across rows (assets).

### 5.2 CNN-EIIE Architecture

```
Input: Price tensor (3, m, n) = (3, 11, 50)
  |
  v
Conv Layer 1: kernel (1 x 3), 2 feature maps, ReLU
  Output: (2, 11, 48)
  |
  v
Conv Layer 2: kernel (1 x 48), 20 feature maps, ReLU
  Output: (20, 11, 1)
  |
  v
[Concatenate w_{t-1} as extra feature map -> (21, 11, 1)]
  |
  v
Conv Layer 3: kernel (1 x 1), 1 feature map (scoring layer)
  Output: (1, 11, 1) = 11 voting scores
  |
  v
[Add cash bias scalar]
  |
  v
Softmax -> w_t (12 elements = 11 assets + cash)
```

All convolution kernels have height 1: each row (asset) is processed independently. Weight sharing across rows enforced by the convolutional structure.

### 5.3 RNN-EIIE / LSTM-EIIE Architecture

```
Input: Price tensor (3, m, n) = (3, 11, 50)
  |
  v
Per-asset: RNN/LSTM subnet (20 hidden units, 50 time steps)
  Each asset's (3, 50) slice processed by identical RNN/LSTM
  Output: (20, 11, 1) -- last hidden state per asset
  |
  v
[Concatenate w_{t-1} as extra feature map -> (21, 11, 1)]
  |
  v
Conv Layer: kernel (1 x 1), 1 feature map (scoring layer)
  Output: (1, 11, 1) = 11 voting scores
  |
  v
[Add cash bias]
  |
  v
Softmax -> w_t (12 elements)
```

RNN subnets are identical with shared parameters. Post-RNN structure identical to CNN version's last layers.

### 5.4 EIIE Advantages Over Integrated Networks

1. **Scalability**: Training time scales linearly with m (number of assets)
2. **Data efficiency**: Each IIE trains on m different assets per interval -- experience shared across time and asset dimensions
3. **Plasticity**: Can change asset selection or portfolio size at runtime without retraining from scratch
4. **No asset identity**: IIE cannot determine which specific asset it evaluates, preventing memorization of historical asset-specific biases

---

## 6. PORTFOLIO-VECTOR MEMORY (PVM)

### 6.1 Purpose

Store previous portfolio vectors w_{t-1} to feed into current network, enabling transaction cost awareness.

### 6.2 Mechanism

- PVM = chronologically ordered stack of portfolio vectors
- Initialized with uniform weights (1/(m+1) for each asset)
- Each training step: read w_{t-1} from PVM[t-1], write new w_t to PVM[t]
- Values converge as network parameters converge

### 6.3 Advantages Over Alternatives

| Approach | Problem |
|---|---|
| RNN memory for w_{t-1} | Requires abandoning price normalization scheme; gradient vanishing |
| Direct Reinforcement (Moody & Saffell 2001) | Gradient vanishing; requires serialized training |
| **PVM** | Enables parallel mini-batch training; avoids gradient vanishing by inserting w_{t-1} after recurrent blocks |

Previous portfolio weights inserted AFTER recurrent blocks -> gradients do not backpropagate through deep RNN structures for the transaction cost signal.

---

## 7. ONLINE STOCHASTIC BATCH LEARNING (OSBL)

### 7.1 Mini-Batch Structure

Unlike standard supervised learning where batches are random disjoint subsets:
- Data points within a batch must maintain temporal order
- Overlapping batches are valid: [t_b, t_b + n_b) and [t_b+1, t_b + n_b + 1) are distinct valid batches
- Batch size n_b = 50

### 7.2 Online Learning Protocol

At end of period t:
1. Add period t's price movement to training set
2. Complete orders for period t+1
3. Train policy network on N_b randomly sampled mini-batches

### 7.3 Batch Sampling Distribution

Batch starting at period t_b (where t_b <= t - n_b) selected with geometric probability:
```
P_beta(t_b) = beta * (1 - beta)^{t - t_b - n_b}     [Eq.26]
```

beta in (0, 1) = probability decay rate controlling recency bias.
- Higher beta: more weight on recent market events
- Exponential decay reflects belief that price correlation decays exponentially with temporal distance

---

## 8. PERFORMANCE METRICS

### 8.1 Accumulated Portfolio Value (APV)

```
p_t = p_t / p_0     [Eq.27]
```
Normalized by initial investment. Final APV (fAPV) = p_f / p_0.

### 8.2 Sharpe Ratio (SR)

```
S = E_t[rho_t - rho_F] / sqrt(Var_t(rho_t - rho_F))     [Eq.28]
```
Risk-free rate rho_F = 0 (cash = Bitcoin = quoted currency).

### 8.3 Maximum Drawdown (MDD)

```
D = max_{tau > t} (p_t - p_tau) / p_t     [Eq.29]
```
Largest peak-to-trough decline as fraction of peak value.

---

## 9. EXPERIMENTAL SETUP

### 9.1 Market and Data

- Exchange: Poloniex (cryptocurrency)
- ~65 available cryptocurrencies, ~80 tradable pairs
- Trading period: T = 30 minutes (1800 seconds)
- Cash currency: Bitcoin
- Commission rate: c_s = c_p = 0.25% (maximum at Poloniex)

### 9.2 Backtest Time Ranges

| Experiment | Backtest Period | Training Data |
|---|---|---|
| Cross-Validation | 2016-05-07 to 2016-06-27 | 2014-07-01 to 2016-05-07 |
| Back-Test 1 | 2016-09-07 to 2016-10-28 | 2014-11-01 to 2016-09-07 |
| Back-Test 2 | 2016-12-08 to 2017-01-28 | 2015-02-01 to 2016-12-08 |
| Back-Test 3 | 2017-03-07 to 2017-04-27 | 2015-05-01 to 2017-03-07 |

Each backtest spans ~50 days. Training sets span ~2 years.

### 9.3 Assumptions

1. **Zero slippage**: Sufficient liquidity; trades execute at last price
2. **Zero market impact**: Invested capital negligible relative to market volume

---

## 10. HYPERPARAMETERS

| Parameter | Value | Description |
|---|---|---|
| Batch size | 50 | Mini-batch size for training |
| Window size (n) | 50 | Input periods in price tensor (25 hours) |
| Number of assets (m+1) | 12 | 11 coins + Bitcoin cash |
| Trading period | 1800 sec | 30-minute rebalancing |
| Total pre-training steps | 2 x 10^6 | Steps on training set before backtest |
| L2 regularization | 10^{-8} | Weight decay coefficient |
| Learning rate (Adam alpha) | 3 x 10^{-5} | Adam optimizer step size |
| Volume observation window | 30 days | For asset pre-selection |
| Commission rate | 0.25% | Per transaction (buy and sell) |
| Rolling training steps | 30 | Online training steps per period during backtest |
| Sample bias (beta) | 5 x 10^{-5} | Geometric distribution parameter [Eq.26] |

Deeper architectures than presented did not improve cross-validation scores.

---

## 11. BENCHMARK RESULTS

### 11.1 Compared Algorithms

**Neural network methods**: CNN-EIIE, bRNN-EIIE, LSTM-EIIE, iCNN (integrated CNN, authors' prior work)

**Benchmarks**: Best Stock, UBAH (Uniform Buy and Hold), UCRP (Uniform Constant Rebalanced Portfolios)

**Follow-the-Loser**: OLMAR, PAMR, CWMR, WMAMR, RMR

**Follow-the-Winner**: UP, EG, ONS

**Pattern-Matching / Other**: Anticor, BK, CORN, M0

### 11.2 Results Table (fAPV = final portfolio value / initial)

| Algorithm | BT1 fAPV | BT1 SR | BT2 fAPV | BT2 SR | BT3 fAPV | BT3 SR |
|---|---|---|---|---|---|---|
| **CNN-EIIE** | **29.695** | **0.087** | **8.026** | **0.059** | 31.747 | 0.076 |
| **bRNN-EIIE** | 13.348 | 0.074 | 4.623 | 0.043 | **47.148** | **0.082** |
| **LSTM-EIIE** | 6.692 | 0.053 | 4.073 | 0.038 | 21.173 | 0.060 |
| iCNN | 4.542 | 0.053 | 1.573 | 0.022 | 3.958 | 0.044 |
| Best Stock | 1.223 | 0.012 | 1.401 | 0.018 | 4.594 | 0.033 |
| UCRP | 0.867 | -0.014 | 1.101 | 0.010 | 2.412 | 0.049 |
| UBAH | 0.821 | -0.015 | 1.029 | 0.004 | 2.230 | 0.036 |
| OLMAR | 0.142 | -0.039 | 0.123 | -0.038 | 4.582 | 0.034 |
| PAMR | 0.003 | -0.137 | 0.003 | -0.121 | 0.021 | -0.055 |
| WMAMR | 0.742 | -0.001 | 0.895 | 0.005 | 6.692 | 0.042 |
| RMR | 0.127 | -0.043 | 0.090 | -0.045 | 7.008 | 0.041 |
| ONS | 0.923 | -0.006 | 1.188 | 0.012 | 1.609 | 0.027 |
| CORN | 0.001 | -0.129 | 0.0001 | -0.179 | 0.001 | -0.125 |

### 11.3 MDD (Maximum Drawdown)

| Algorithm | BT1 | BT2 | BT3 |
|---|---|---|---|
| CNN-EIIE | 0.224 | 0.216 | 0.406 |
| bRNN-EIIE | 0.241 | 0.262 | 0.393 |
| LSTM-EIIE | 0.280 | 0.319 | 0.487 |
| Best Stock | 0.654 | 0.236 | 0.668 |
| PAMR | 0.997 | 0.998 | 0.981 |

### 11.4 Key Findings

- All three EIIEs monopolize top 3 in fAPV and SR across ALL three backtests
- CNN-EIIE best in BT1 and BT2; bRNN-EIIE best in BT3
- With 0.25% commission, all model-based strategies have negative returns in BT1
- EIIEs achieve minimum 4-fold returns in ~50 days across all experiments
- EIIE framework significantly outperforms iCNN (integrated network), proving EIIE topology advantage
- bRNN outperforms LSTM: suggests price patterns repeat (vanilla RNN better at exploiting repetitive patterns by not forgetting). May also reflect shared hyperparameters not tuned for LSTM.

---

## 12. COMPLETE EQUATION REFERENCE

| Eq. | Formula | Description |
|---|---|---|
| 1 | y_t = v_t / v_{t-1} | Price relative vector |
| 2 | p_t = p_{t-1} (y_t . w_{t-1}) | Portfolio value update (no cost) |
| 3 | rho_t = y_t . w_{t-1} - 1 | Rate of return (no cost) |
| 4 | r_t = ln(y_t . w_{t-1}) | Log return (no cost) |
| 5 | w_0 = (1, 0, ..., 0)^T | Initial portfolio (all cash) |
| 6 | p_f = p_0 prod(y_t . w_{t-1}) | Final value (no cost) |
| 7 | w'_t = (y_t * w_{t-1}) / (y_t . w_{t-1}) | Weights after price drift |
| 8 | p_t = mu_t * p'_t | Value after transaction cost |
| 9 | rho_t = mu_t(y_t . w_{t-1}) - 1 | Return with cost |
| 10 | r_t = ln(mu_t * y_t . w_{t-1}) | Log return with cost |
| 11 | p_f = p_0 prod(mu_t * y_t . w_{t-1}) | Final value with cost |
| 14 | mu_t = f(mu_t) implicit equation | Transaction remainder factor |
| 15 | mu^(k) = f(mu^(k-1)) | Iterative solution for mu_t |
| 16 | mu_star = c sum|w'_{t,i} - w_{t,i}| | Initial guess for iteration |
| 18 | X_t = stack[V^lo, V^hi, V] | Input price tensor |
| 19 | a_t = w_t | Action = portfolio vector |
| 20 | s_t = (X_t, w_{t-1}) | State = (prices, last weights) |
| 21-22 | R = (1/t_f) sum r_t | Objective: avg log return |
| 23 | J(pi_theta) = R(trajectory) | Policy performance metric |
| 24-25 | theta <- theta + lambda grad J | Gradient ascent update |
| 26 | P_beta(t_b) = beta(1-beta)^{...} | Batch sampling distribution |
| 28 | S = E[rho_t] / sqrt(Var[rho_t]) | Sharpe ratio |
| 29 | D = max (p_t - p_tau) / p_t | Maximum drawdown |

---

## 13. ARCHITECTURAL DESIGN PRINCIPLES

### 13.1 Why Not Discrete Actions
- Discrete actions introduce unknown risk (e.g., all-in on one asset)
- Discretization scales badly with number of assets
- Portfolio management is inherently continuous (weight simplex)

### 13.2 Why Not Actor-Critic
- Training two networks is difficult and sometimes unstable
- EIIE framework achieves continuous output with a single policy network
- Reward function is exactly known (no need for critic to estimate)

### 13.3 Why Not Price Prediction
- Prediction accuracy is difficult to achieve
- Converting predictions to actions requires hand-coded logic (not end-to-end)
- Cannot naturally incorporate transaction costs

### 13.4 Why EIIE Over Integrated Network
- Integrated CNN (iCNN) memorizes per-asset histories, becomes reluctant to invest in historically unfavorable assets even when current trend is promising
- EIIE processes each asset identically, judges purely on recent price patterns
- Empirically: iCNN fAPV = 4.5 vs CNN-EIIE fAPV = 29.7 in BT1

---

## 14. TRAINING PIPELINE

```
[Pre-Training Phase]
1. Initialize PVM with uniform weights
2. Initialize network parameters randomly
3. For 2,000,000 steps:
   a. Sample mini-batch start t_b from geometric distribution P_beta
   b. Load w_{t-1} from PVM for each t in [t_b, t_b + n_b)
   c. Forward pass: X_t, w_{t-1} -> w_t for each t in batch
   d. Compute R = (1/n_b) * sum ln(mu_t * y_t . w_{t-1})
   e. Backprop: theta <- theta + lambda * grad_theta R
   f. Write new w_t to PVM for each t in batch

[Online Trading / Backtest Phase]
For each new period t:
1. Observe X_t, load w_{t-1} from PVM
2. Compute w_t = pi_theta(X_t, w_{t-1})
3. Execute trades (rebalance from w'_t to w_t)
4. Store w_t in PVM
5. Add new price data to training set
6. Perform 30 online training steps (OSBL with geometric sampling)
```

---

## 15. IMPLEMENTATION SPECIFICS

### 15.1 CNN Layer Details

| Layer | Kernel | Maps/Filters | Input Size | Output Size | Activation |
|---|---|---|---|---|---|
| Conv1 | 1 x 3 | 2 | (3, 11, 50) | (2, 11, 48) | ReLU |
| Conv2 | 1 x 48 | 20 | (2, 11, 48) | (20, 11, 1) | ReLU |
| [Insert w_{t-1}] | -- | +1 | (20, 11, 1) | (21, 11, 1) | -- |
| Conv3 (score) | 1 x 1 | 1 | (21, 11, 1) | (1, 11, 1) | -- |
| Cash bias + Softmax | -- | -- | 11 scores + 1 bias | 12 weights | Softmax |

### 15.2 RNN/LSTM Layer Details

| Component | Units/Steps | Input | Output |
|---|---|---|---|
| Per-asset RNN/LSTM | 20 units, 50 steps | (3, 50) per asset | (20, 1) per asset |
| Ensemble output | -- | m x (20, 1) | (20, 11, 1) |
| [Insert w_{t-1}] | +1 map | (20, 11, 1) | (21, 11, 1) |
| Conv (score) | 1 x 1 kernel | (21, 11, 1) | (1, 11, 1) |
| Cash bias + Softmax | -- | 11 + 1 | 12 weights |

### 15.3 Cash Bias

A learnable scalar parameter added alongside the m asset voting scores before softmax. Allows the network to learn a baseline tendency to hold cash. Not derived from any IIE -- represents the "do nothing" option.
