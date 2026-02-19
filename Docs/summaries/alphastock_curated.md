# AlphaStock: Buying-Winners-and-Selling-Losers via Deep Reinforcement Attention Networks
**Source:** KDD 2019 (Wang, Zhang, Tang, Wu, Xiong -- Beihang University / Tsinghua University)

---

## Core Idea

A policy-gradient RL agent learns to construct a zero-investment long/short portfolio each month. Two attention networks produce a "winner score" per stock: LSTM-HA (temporal history attention) feeds into CAAN (cross-asset self-attention). The Sharpe ratio of the entire trajectory is the RL reward. No per-step supervision; the model is trained end-to-end to maximize risk-adjusted return.

---

## 1. Problem Formulation

### Definitions

- **Holding period:** Minimum time unit for one investment cycle (set to 1 month).
- **Sequential investment:** Chain of T holding periods. Capital + profit from period t becomes capital for period t+1.
- **Asset price:** Time series p^(i) = {p^(i)_1, p^(i)_2, ...} for asset i.
- **Long position on asset i:** Buy at t1, sell at t2. Profit = u_i * (p^(i)_{t2} - p^(i)_{t1}).
- **Short position on asset i:** Sell at t1, buy back at t2. Profit = u_i * (p^(i)_{t1} - p^(i)_{t2}).
- **Portfolio:** Vector b = (b^(1), ..., b^(I))^T where b^(i) is proportion of investment on asset i, sum(b^(i)) = 1.
- **Zero-investment portfolio:** Collection of portfolios {b^(1), ..., b^(J)} where total investment sum(M^(j)) = 0.

### BWSL (Buy-Winners-Sell-Losers) Strategy

At each period t, construct two portfolios:
- **b+_t** (long portfolio): buy "winner" stocks
- **b-_t** (short portfolio): sell "loser" stocks

Given budget constraint M_tilde:

**Short sell volume of stock i:**
```
u^{-(i)}_t = M_tilde * b^{-(i)}_t / p^(i)_t          ... (1)
```

**Long buy volume of stock i:**
```
u^{+(i)}_t = M_tilde * b^{+(i)}_t / p^(i)_t          ... (2)
```

Proceeds from selling short = M_tilde. This funds the long purchases. Net investment = 0.

**Proceeds from selling long portfolio at end of period:**
```
M+_t = sum_i [ M_tilde * b^{+(i)}_t * p^(i)_{t+1} / p^(i)_t ]    ... (3)
```

**Cost to buy back short portfolio at end of period:**
```
M-_t = sum_i [ M_tilde * b^{-(i)}_t * p^(i)_{t+1} / p^(i)_t ]    ... (4)
```

**Ensemble profit:** M_t = M+_t - M-_t

Let **z^(i)_t = p^(i)_{t+1} / p^(i)_t** (price rising rate of stock i in period t).

**Rate of return:**
```
R_t = M_t / M_tilde = sum_i [ b^{+(i)}_t * z^(i)_t ] - sum_i [ b^{-(i)}_t * z^(i)_t ]    ... (5)
```

**Key insight:** Profit is positive when the weighted average price rising rate of longs exceeds that of shorts. Even in falling markets, if longs fall slower than shorts, the strategy profits. Therefore *relative* price relations among stocks matter, not absolute prices.

---

## 2. Optimization Objective: Sharpe Ratio

**Sharpe Ratio:**
```
H_T = (A_T - Theta) / V_T                             ... (7)
```

where:
- Theta = risk-free return rate
- A_T = average rate of return per period:

```
A_T = (1/T) * sum_{t=1}^{T} (R_t - TC_t)             ... (8)
```

- TC_t = transaction cost in period t
- V_T = volatility (risk):

```
V_T = sqrt( sum_{t=1}^{T} (R_t - R_bar)^2 / T )      ... (9)
```

- R_bar = (1/T) * sum(R_t)

**Global objective:**
```
arg max_{B+, B-} H_T(B+, B-)                          ... (10)
```

where B+ = {b+_1, ..., b+_T} and B- = {b-_1, ..., b-_T}.

---

## 3. Architecture Overview

Three components in sequence:

```
[Raw Features X^(i)] --> [LSTM-HA] --> r^(i) --> [CAAN] --> s^(i) --> [Portfolio Generator] --> {b+, b-}
     per stock             shared         repr      cross-stock    winner score           long/short
```

All stocks share the same LSTM-HA parameters. CAAN operates across all stocks simultaneously.

---

## 4. Component 1: LSTM-HA (Long Short-Term Memory with History State Attention)

### Input

For stock i at time t, the look-back window covers K previous holding periods.
History states: X = {x_1, ..., x_K} where x_k = x_tilde_{t-K+k}.
Each x_k is a feature vector (Section 8 below).

### Sequential Encoding

```
h_k = LSTM(h_{k-1}, x_k),    k in [1, K]             ... (11)
```

h_K = final hidden state, encodes sequential dependence.

### History State Attention

Enhances h_K with global/long-range dependence from all intermediate hidden states:

```
r = sum_{k=1}^{K} ATT(h_K, h_k) * h_k                ... (12)
```

**Attention function:**
```
ATT(h_K, h_k) = exp(alpha_k) / sum_{k'=1}^{K} exp(alpha_{k'})    ... (13)

alpha_k = w^T * tanh( W^(1) * h_k + W^(2) * h_K )
```

Learnable parameters: w, W^(1), W^(2).

**Output:** r^(i)_t for stock i at time t. Contains both sequential and global dependence over the look-back window.

**Weight sharing:** All stocks use the same LSTM-HA parameters (w, W^(1), W^(2), LSTM weights). Representations are general across stocks, not stock-specific.

---

## 5. Component 2: CAAN (Cross-Asset Attention Network)

### Basic CAAN (Transformer-style self-attention across stocks)

Given stock representation r^(i), compute query, key, value:

```
q^(i) = W^(Q) * r^(i)
k^(i) = W^(K) * r^(i)                                 ... (14)
v^(i) = W^(V) * r^(i)
```

**Inter-stock attention score:**
```
beta_{ij} = (q^(i)^T * k^(j)) / sqrt(D_k)            ... (15)
```

D_k = rescaling parameter (as in Vaswani et al. 2017).

**Attention-weighted aggregation:**
```
a^(i) = sum_{j=1}^{I} SATT(q^(i), k^(j)) * v^(j)    ... (16)

SATT(q^(i), k^(j)) = exp(beta_{ij}) / sum_{j'=1}^{I} exp(beta_{ij'})    ... (17)
```

**Winner score:**
```
s^(i) = sigmoid( w^(s)^T * a^(i) + e^(s) )            ... (18)
```

s^(i) in (0, 1). Higher score = more likely a winner.

### Price Rising Rank Prior

Uses the rank of each stock's price rising rate from the previous period as positional prior knowledge.

Let c^(i)_{t-1} = rank of price rising rate of stock i in the last holding period.

**Discrete relative distance between stocks i and j:**
```
d_{ij} = floor( |c^(i)_{t-1} - c^(j)_{t-1}| / Q )   ... (19)
```

Q = preset quantization coefficient.

**Lookup embedding:** Matrix L = (l_1, ..., l_L). Use d_{ij} as index to get embedding vector l_{d_{ij}}.

**Prior relation coefficient:**
```
psi_{ij} = sigmoid( w^(L)^T * l_{d_{ij}} )            ... (20)
```

**Modified attention score (replacing Eq. 15):**
```
beta_{ij} = psi_{ij} * (q^(i)^T * k^(j)) / sqrt(D)   ... (21)
```

Stocks with similar recent price rising rates get stronger interrelationship in attention, producing similar winner scores.

---

## 6. Component 3: Portfolio Generator

Given winner scores {s^(1), ..., s^(I)} for I stocks:

1. Sort stocks in descending order by winner score. Let o^(i) = rank of stock i.
2. G = preset portfolio size (top-G and bottom-G).

**Long portfolio (top-G winners):** If o^(i) in [1, G]:
```
b^{+(i)} = exp(s^(i)) / sum_{o(i') in [1,G]} exp(s^(i'))      ... (22)
```

**Short portfolio (bottom-G losers):** If o^(i) in (I-G, I]:
```
b^{-(i)} = exp(1 - s^(i)) / sum_{o(i') in (I-G,I]} exp(1 - s^(i'))    ... (23)
```

Stocks with rank in (G, I-G] are unselected (no clear signal).

Combined vector b_c of length I: b_c^(i) = b+^(i) if winner, b-^(i) if loser, 0 otherwise.

---

## 7. Reinforcement Learning Optimization

### MDP Formulation

A T-period investment = one trajectory pi:
```
pi = {state_1, action_1, reward_1, ..., state_T, action_T, reward_T}
```

- **State_t:** History market state X_t = (X^(i)_t) for all stocks at time t.
- **Action_t:** I-dimensional binary vector. action^(i)_t = 1 if agent invests in stock i, 0 otherwise.
- **Action probability:**

```
Pr(action^(i)_t = 1 | X^n_t, theta) = (1/2) * G^(i)(X^n_t, theta) = (1/2) * b_c^(i)_t    ... (24)
```

G^(i) is the AlphaStock network output for stock i. Factor 1/2 ensures sum of probabilities = 1.

- **Reward_t:** Contribution of action_t to Sharpe ratio H_pi of trajectory pi, with sum_{t=1}^{T} reward_t = H_pi.

### Policy Gradient

Average reward across all trajectories:
```
J(theta) = integral_pi H_pi * Pr(pi | theta) d_pi     ... (25)
```

Objective: theta* = arg max_theta J(theta).

**Gradient ascent:** theta_tau = theta_{tau-1} + eta * grad_J(theta)|_{theta=theta_{tau-1}}

**REINFORCE gradient estimate** from N training trajectories {pi_1, ..., pi_N}:
```
grad_J(theta) approx (1/N) * sum_{n=1}^{N} [ H_{pi_n} * sum_{t=1}^{T_n} sum_{i=1}^{I} grad_theta log Pr(action^(i)_t = 1 | X^(n)_t, theta) ]    ... (26)
```

The inner gradient: grad_theta log Pr(action^(i)_t = 1 | X^(n)_t, theta) = grad_theta log G^(i)(X^n_t, theta), computed via backpropagation.

### Threshold Baseline (to beat the market)

```
grad_J(theta) = (1/N) * sum_{n=1}^{N} [ (H_{pi_n} - H_0) * sum_{t=1}^{T_n} sum_{i=1}^{I} grad_theta log G^(i)(X^n_t, theta) ]    ... (27)
```

H_0 = Sharpe ratio of the overall market (baseline). Gradient ascent only encourages parameters that outperform the market.

**Key property:** The reward (H_{pi_n} - H_0) weights ALL steps in trajectory pi_n equally. Reward is not decomposed per-step. This enforces far-sighted, steady strategy rather than short-term greedy optimization.

---

## 8. Raw Features (Input to LSTM-HA)

All features are Z-score standardized.

### Trading Features (at time t for stock i)

| Feature | Abbrev | Definition |
|---|---|---|
| Price Rising Rate | PR | p^(i)_t / p^(i)_{t-1} |
| Fine-grained Volatility | VOL | Std dev of daily prices within [t-1, t] |
| Trade Volume | TV | Total shares traded from t-1 to t |

### Company Features (at time t for stock i)

| Feature | Abbrev | Definition |
|---|---|---|
| Market Capitalization | MC | p^(i)_t * outstanding shares |
| Price-Earnings Ratio | PE | Market cap / annual earnings |
| Book-to-Market Ratio | BM | Book value / market value |
| Dividend | Div | Reward from earnings to holders in period t-1 |

Total: 7 features per stock per time step. Look-back window K=12 months. Feature matrix per stock: K x 7.

---

## 9. Model Interpretation via Sensitivity Analysis

Winner score as function of history features: s = F(X).

**Sensitivity of winner score s to feature element x_q:**
```
delta_{x_q}(X) = partial F(X) / partial x_q           ... (28)
```

**Average sensitivity across all stocks and time periods:**
```
delta_bar_{x_q} = (1 / (I * N)) * sum_{n=1}^{N} sum_{i=1}^{I} delta_{x_q}(X^(i)_n, X^{(not-i)}_n)    ... (30)
```

- delta_bar > 0: model favors stocks with high x_q as winners.
- delta_bar < 0: model favors stocks with low x_q as winners.

### Interpretation Results (U.S. Market)

**Price Rising Rate (PR):**
- Months -12 to -9 (long-term): POSITIVE influence on winner score. Model buys stocks with sustained long-term price growth.
- Months -8 to -1 (short-term): NEGATIVE influence. Model buys stocks with recent price retracement (undervalued).
- Implication: Long-term momentum + short-term mean reversion hybrid.

**Trade Volume (TV):** Similar pattern to PR (correlated with price movement).

**Fine-grained Volatility (VOL):** NEGATIVE influence across ALL history months. Model selects low-volatility stocks as winners.

**Company Features (averaged across history months):**
- MC (Market Cap): POSITIVE -- favors larger companies.
- PE (Price-Earnings): POSITIVE -- favors stocks with higher PE ratios.
- BM (Book-to-Market): POSITIVE -- favors stocks with high book-to-market (value stocks).
- DIV (Dividend): NEGATIVE -- dividends reduce intrinsic value, model avoids high-dividend stocks.

**Summary principle:** "Select stocks as winners with high long-term growth, low volatility, high intrinsic value, and being undervalued recently."

---

## 10. Experimental Setup

### U.S. Market Data
- **Source:** Wharton Research Data Services (WRDS)
- **Time range:** Jan 1970 -- Dec 2016
- **Exchanges:** NYSE, NYSE American, NASDAQ, NYSE Arca
- **Number of valid stocks:** >1000 per year
- **Training/validation:** Jan 1970 -- Jan 1990
- **Test:** Jan 1990 -- Dec 2016 (26 years)
- **Covers:** Dot-com bubble (1995-2000), subprime mortgage crisis (2007-2009)

### Chinese Market Data
- **Source:** WIND database
- **Time range:** Jun 2005 -- Dec 2018
- **Exchanges:** Shanghai Stock Exchange (SSE), Shenzhen Stock Exchange (SZSE)
- **Stock type:** RMB-priced ordinary shares (A-share), 1,131 stocks
- **Training/validation:** Jun 2005 -- Dec 2011
- **Test:** Jan 2012 -- Dec 2018
- **Note:** Chinese market does not allow short selling; only b+ (long) portfolio used.

### Hyperparameters
- Holding period: 1 month
- T (periods per trajectory for Sharpe ratio): 12 (= 1 year)
- K (look-back window): 12 months
- G (portfolio size): I/4 (one quarter of all stocks)
- Transaction cost TC: 0.1%

---

## 11. Evaluation Metrics

| Metric | Formula | Direction |
|---|---|---|
| Cumulative Wealth | CW_T = prod_{t=1}^{T} (R_t + 1 - TC) | Higher = better |
| Annualized Percentage Rate | APR_T = A_T * N_Y | Higher = better |
| Annualized Volatility | AVOL_T = V_T * sqrt(N_Y) | Lower = better |
| Annualized Sharpe Ratio | ASR_T = APR_T / AVOL_T | Higher = better |
| Maximum DrawDown | MDD_T = max_{tau in [1,T]} max_{t in [1,tau]} (APR_t - APR_tau) / APR_t | Lower = better |
| Calmar Ratio | CR_T = APR_T / MDD_T | Higher = better |
| Downside Deviation Ratio | DDR_T = APR_T / sqrt(E[min(R_t, MAR)]^2), MAR=0 | Higher = better |

---

## 12. Benchmark Results -- U.S. Market (Test: 1990-2016)

| Method | APR | AVOL | ASR | MDD | CR | DDR |
|---|---|---|---|---|---|---|
| Market | 0.042 | 0.174 | 0.239 | 0.569 | 0.073 | 0.337 |
| TSM | 0.047 | 0.223 | 0.210 | 0.523 | 0.090 | 0.318 |
| CSM | 0.044 | 0.096 | 0.456 | 0.126 | 0.350 | 0.453 |
| RMR | 0.074 | 0.134 | 0.551 | 0.098 | 1.249 | 0.757 |
| FDDR | 0.063 | 0.056 | 1.141 | 0.070 | 0.900 | 2.028 |
| AS-NC (no CAAN) | 0.101 | 0.052 | 1.929 | 0.068 | 1.492 | 1.685 |
| AS-NP (no prior) | 0.133 | 0.065 | 2.054 | 0.033 | 3.990 | 4.618 |
| **AlphaStock** | **0.143** | **0.067** | **2.132** | **0.027** | **5.296** | **6.397** |

**Key observations:**
- AlphaStock achieves 14.3% APR with ASR of 2.132 (nearly 2x FDDR's 1.141).
- MDD of 2.7% vs. Market's 56.9%. Extreme loss control is exceptional.
- CAAN contribution: AS-NP vs AS-NC shows CAAN doubles CR (1.492 -> 3.990) and nearly triples DDR.
- Price rising rank prior: AS vs AS-NP further improves MDD (3.3% -> 2.7%), CR (3.990 -> 5.296), DDR (4.618 -> 6.397).
- RL strategies (FDDR, all AS variants) are stable across bull and bear markets; traditional strategies (TSM, RMR) each only adapt to one market regime.

## 13. Benchmark Results -- Chinese Market (Test: 2012-2018)

| Method | APR | AVOL | ASR | MDD | CR | DDR |
|---|---|---|---|---|---|---|
| Market | 0.037 | 0.260 | 0.141 | 0.595 | 0.062 | 0.135 |
| TSM | 0.078 | 0.420 | 0.186 | 0.533 | 0.147 | 0.225 |
| CSM | 0.023 | 0.392 | 0.058 | 0.633 | 0.036 | 0.064 |
| RMR | 0.079 | 0.279 | 0.282 | 0.423 | 0.186 | 0.289 |
| FDDR | 0.084 | 0.152 | 0.553 | 0.231 | 0.365 | 0.801 |
| AS-NC | 0.104 | 0.113 | 0.916 | 0.163 | 0.648 | 1.103 |
| AS-NP | 0.122 | 0.105 | 1.163 | 0.136 | 0.895 | 1.547 |
| **AlphaStock** | **0.125** | **0.103** | **1.220** | **0.135** | **0.296** | **1.704** |

Higher AVOL and MDD across all methods compared to U.S. market (emerging market characteristics). AlphaStock still dominates all metrics except CR. Long-only constraint (no short selling in China) limits the BWSL mechanism.

---

## 14. Ablation Summary

| Ablation | What is removed | Impact |
|---|---|---|
| AS-NC | CAAN removed; LSTM-HA outputs feed directly to portfolio generator | ASR drops from 2.132 to 1.929; MDD worsens from 0.027 to 0.068; CR drops from 5.296 to 1.492 |
| AS-NP | Price rising rank prior removed; basic CAAN without positional bias | ASR drops from 2.132 to 2.054; MDD worsens from 0.027 to 0.033; CR drops from 5.296 to 3.990 |
| AS-NC vs FDDR | Both are RL-based BWSL; AS-NC uses LSTM-HA, FDDR uses fuzzy recurrent net | AS-NC dominates FDDR on all metrics, showing LSTM-HA superiority for representation |

---

## 15. Baselines Compared

| Method | Type | Description |
|---|---|---|
| Market | Passive | Uniform buy-and-hold |
| CSM | Traditional | Cross-sectional momentum (Jegadeesh & Titman 2002) |
| TSM | Traditional | Time-series momentum (Moskowitz et al. 2012) |
| RMR | Traditional | Robust Median Reversion (Huang et al. 2016) |
| FDDR | Deep RL | Fuzzy Deep Direct Reinforcement (Deng et al. 2017) |

---

## 16. Training Procedure Summary

```
1. Divide time axis into monthly holding periods.
2. For each trajectory pi_n (length T=12 months):
   a. For each time step t in trajectory:
      i.   For each stock i, extract look-back window X^(i) (K=12 months of 7 features).
      ii.  Z-score standardize all features.
      iii. Pass X^(i) through shared LSTM-HA -> get representation r^(i)_t.
      iv.  Compute price rising rank c^(i)_{t-1} for all stocks.
   b. Pass all {r^(i)_t} through CAAN with price rising rank prior -> winner scores {s^(i)_t}.
   c. Portfolio Generator: sort by s^(i), top-G -> long portfolio b+, bottom-G -> short portfolio b-.
   d. Execute investment: compute R_t from actual price changes z^(i)_t.
3. After T steps, compute trajectory Sharpe ratio H_{pi_n}.
4. Compute policy gradient using Eq. (27) with market Sharpe ratio baseline H_0.
5. Update theta via gradient ascent: theta <- theta + eta * grad_J(theta).
6. Repeat over N training trajectories until convergence.
```

**RL type:** Policy-based (REINFORCE with baseline).
**Reward structure:** Trajectory-level (Sharpe ratio of full 12-month trajectory), NOT step-level.
**Baseline:** Market Sharpe ratio H_0 -- only encourages parameters that beat the market.
**Gradient computation:** Backpropagation through LSTM-HA + CAAN + Portfolio Generator.
