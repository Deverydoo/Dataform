# FinRL: Deep Reinforcement Learning for Automated Stock Trading -- Curated Technical Extraction

Source: Liu et al., "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance," NeurIPS 2020 Deep RL Workshop.

---

## 1. Problem Formulation: Markov Decision Process (MDP)

Stock trading is modeled as an MDP. The agent observes the market state, selects a trading action, receives a reward, and transitions to the next state. The objective is to learn a policy that maximizes cumulative discounted reward over time.

---

## 2. State Space S

The state vector at time step t is composed of:

| Component | Symbol | Domain | Description |
|---|---|---|---|
| Account balance | b_t | R+ | Cash remaining in the account |
| Shares held | h_t | Z+^n | Integer share count for each of n stocks |
| Closing price | p_t | R+^n | Closing price per stock |
| Open/High/Low prices | o_t, h_t, l_t | R+^n | Intraday price tracking |
| Trading volume | v_t | R+^n | Total shares traded during the period |
| MACD | M_t | R^n | Moving Average Convergence Divergence indicator |
| RSI | R_t | R+^n | Relative Strength Index indicator |

**Time granularity**: daily, hourly, or minute-level.

Full state vector:
```
s_t = [b_t, h_t, p_t, o_t, h_t, l_t, v_t, M_t, R_t, ...]
```

---

## 3. Action Space A

Discrete-continuous hybrid:

```
a_i in {-k, ..., -1, 0, 1, ..., k}   for each stock i in {1, ..., n}
```

- **Negative values**: sell (e.g., -10 = sell 10 shares)
- **Zero**: hold
- **Positive values**: buy (e.g., +10 = buy 10 shares)
- k = maximum number of shares per trade per stock

Full action vector for n stocks:
```
a_t = [a_t^1, a_t^2, ..., a_t^n]    where a_t^i in {-k, ..., k}
```

---

## 4. Reward Functions

### 4.1 Portfolio Value Change (Primary)
```
r(s, a, s') = v' - v
```
Where v' = portfolio value at state s', v = portfolio value at state s.

### 4.2 Portfolio Log Return
```
r(s, a, s') = log(v' / v)
```

### 4.3 Sharpe Ratio (Period-based)
```
S_T = mean(R_t) / std(R_t)
```
Where R_t = v_t - v_{t-1} for periods t = {1, ..., T}.

### 4.4 User-Defined
Supports custom reward functions incorporating risk factors, transaction cost penalties, or volatility scaling.

---

## 5. DRL Algorithms

### 5.1 Algorithm Categories

| Category | Algorithms | Action Space | Key Property |
|---|---|---|---|
| Value-based | DQN | Discrete | Learns Q(s,a) value function |
| Policy-based | PPO | Continuous/Discrete | Directly optimizes policy |
| Actor-Critic | A2C, DDPG, TD3, SAC | Continuous | Combines value + policy learning |

### 5.2 Algorithm Specifications

**DQN (Deep Q-Network)**
- Type: Value-based, off-policy
- Action space: Discrete
- Learns Q-function: Q(s, a) -> expected return
- Uses experience replay and target network

**DDPG (Deep Deterministic Policy Gradient)**
- Type: Actor-Critic, off-policy
- Action space: Continuous
- Actor network: deterministic policy mu(s | theta_mu)
- Critic network: Q(s, a | theta_Q)
- Uses experience replay buffer and target networks (soft updates)

**Adaptive DDPG**
- Extension of DDPG with adaptive behavior for bull/bear markets
- Adjusts portfolio allocation strategy based on market regime

**Multi-Agent DDPG (MADDPG)**
- Extension for multi-agent cooperative/competitive settings
- Each agent has its own actor; centralized critic during training

**PPO (Proximal Policy Optimization)**
- Type: Actor-Critic, on-policy
- Action space: Continuous or Discrete
- Clips policy ratio to prevent large updates
- Stable training dynamics

**A2C (Advantage Actor-Critic)**
- Type: Actor-Critic, on-policy
- Synchronous variant of A3C
- Uses advantage function A(s, a) = Q(s, a) - V(s)
- Multiple parallel workers for stable gradient estimates

**SAC (Soft Actor-Critic)**
- Type: Actor-Critic, off-policy
- Maximum entropy framework
- Objective: maximize expected return + entropy bonus
- Stochastic policy; encourages exploration

**TD3 (Twin Delayed DDPG)**
- Type: Actor-Critic, off-policy
- Three key modifications over DDPG:
  1. Twin critics: two Q-networks, take minimum to reduce overestimation
  2. Delayed policy updates: update actor less frequently than critic
  3. Target policy smoothing: add noise to target actions

### 5.3 Implementation Foundation
- Built on OpenAI Baselines and Stable Baselines
- Uses OpenAI Gym-compatible environment interface

---

## 6. Trading Environment Design

### 6.1 Architecture: Three-Layer Design

```
Layer 3 (Top):     Applications        [Single stock | Multi-stock | Portfolio allocation]
                        |
Layer 2 (Middle):  DRL Agents           [DQN | DDPG | PPO | A2C | SAC | TD3 | ...]
                        |
Layer 1 (Bottom):  Market Environment   [Simulator | Data | Constraints | Indicators]
```

- Lower layers expose APIs to upper layers
- Modular: any module at any layer can be swapped independently

### 6.2 Time-Driven Simulation
- Simulates live stock markets with historical data
- Steps through time at configurable granularity (minute / hourly / daily)
- OpenAI Gym-compatible interface: reset(), step(action) -> (state, reward, done, info)

### 6.3 Supported Market Datasets

| Index | Exchange | Region |
|---|---|---|
| NASDAQ-100 | NASDAQ | US |
| DJIA (Dow 30) | NYSE/NASDAQ | US |
| S&P 500 | NYSE/NASDAQ | US |
| HSI | HKEX | Hong Kong |
| SSE 50 | Shanghai SE | China |
| CSI 300 | Shanghai/Shenzhen | China |

Plus user-imported custom datasets.

---

## 7. Trading Constraints and Market Frictions

### 7.1 Transaction Costs
Two models supported:

**Flat fee:**
```
cost = fixed_dollar_amount  (per trade, regardless of share count)
```

**Per-share percentage:**
```
cost = rate * num_shares * price_per_share
```
Common rates: 0.001 (1/1000) or 0.002 (2/1000) per trade.

### 7.2 Market Liquidity: Bid-Ask Spread
```
effective_buy_price  = closing_price + spread/2
effective_sell_price = closing_price - spread/2
```
Bid-ask spread is added as a simulation parameter.

### 7.3 Risk-Aversion via Financial Turbulence Index

```
turbulence_t = (y_t - mu) * Sigma^{-1} * (y_t - mu)^T    in R
```

Where:
- y_t in R^n : stock returns for current period t
- mu in R^n : average of historical returns
- Sigma in R^{n x n} : covariance matrix of historical returns

**Control logic:**
- If turbulence_t >= threshold: agent halts buying, begins gradual selling
- If turbulence_t < threshold: normal trading resumes

This is a Mahalanobis-distance-based measure of how extreme current returns are relative to historical distribution.

---

## 8. Portfolio Optimization Formulation

### 8.1 Multiple Stock Trading
- Action vector a_t = [a_t^1, ..., a_t^n] where n = number of stocks (e.g., 30 for DJIA)
- Each a_t^i in {-k, ..., k} determines buy/sell/hold per stock
- Portfolio value: V_t = b_t + sum_{i=1}^{n} h_t^i * p_t^i

### 8.2 Portfolio Allocation
- Weights w_t in R^n, sum(w_t) = 1
- Agent outputs target allocation; rebalancing executed against current holdings
- Compared against min-variance and mean-variance baseline strategies

### 8.3 Baseline Strategies for Comparison

| Strategy | Description |
|---|---|
| Passive Buy-and-Hold | Buy at start, hold through entire period |
| Mean-Variance (Markowitz) | Optimize E[R] - lambda * Var(R) |
| Min-Variance | Minimize portfolio variance regardless of return |
| Momentum | Buy recent winners, sell recent losers |
| Equal-Weighted | Allocate 1/n to each of n assets |

---

## 9. Training Methodology

### 9.1 Training-Validation-Testing Flow

```
|-------- Training --------|--- Validation ---|--- Testing (Trading) ---|
```

- **Training**: fit DRL model parameters
- **Validation**: hyperparameter tuning, overfitting prevention
- **Testing**: unbiased out-of-sample performance evaluation

### 9.2 Rolling Window
- Periodically retrain and rebalance: daily, monthly, quarterly, yearly, or user-defined
- Sliding window moves forward; model is retrained on updated historical data

### 9.3 Backtesting
- Automated via Quantopian pyfolio package
- Replays learned policy on historical test data
- Generates comprehensive performance plots and statistics

---

## 10. Performance Metrics

### 10.1 Five Standard Metrics

**1. Final Portfolio Value:**
```
V_T = b_T + sum_{i=1}^{n} h_T^i * p_T^i
```

**2. Annualized Return:**
```
R_annual = (V_T / V_0)^{252/T} - 1
```
(assuming 252 trading days/year)

**3. Annualized Standard Deviation:**
```
sigma_annual = std(daily_returns) * sqrt(252)
```

**4. Maximum Drawdown:**
```
MDD = max_{t in [0,T]} ( (peak_t - V_t) / peak_t )
```
Where peak_t = max_{tau in [0,t]} V_tau

**5. Sharpe Ratio:**
```
S = (R_annual - R_f) / sigma_annual
```
Where R_f = risk-free rate (often assumed 0 in the paper's context).

Simplified form used:
```
S_T = mean(R_t) / std(R_t)
```

---

## 11. Benchmark Results

### 11.1 Single Stock Trading (PPO) -- Test Period: 2019/01/01 to 2020/09/23

| Asset | Initial ($) | Final ($) | Annual Return | Annual Std | Sharpe | Max Drawdown |
|---|---|---|---|---|---|---|
| SPY | 100,000 | 127,044 | 14.89% | 9.63% | 1.49 | 20.93% |
| QQQ | 100,000 | 163,647 | 32.33% | 27.51% | 1.16 | 28.26% |
| GOOGL | 100,000 | 174,825 | 37.40% | 33.41% | 1.12 | 27.76% |
| AMZN | 100,000 | 192,031 | 44.94% | 29.62% | 1.40 | 21.13% |
| AAPL | 100,000 | 173,063 | 36.88% | 25.84% | 1.35 | 22.47% |
| MSFT | 100,000 | 172,797 | 36.49% | 33.41% | 1.10 | 28.11% |
| **S&P 500 (benchmark)** | **100,000** | **133,402** | **17.81%** | **27.00%** | **0.74** | **33.92%** |

**Key finding**: All PPO-trained single-stock agents outperformed the S&P 500 benchmark on Sharpe ratio (range: 1.10-1.49 vs 0.74).

### 11.2 Multiple Stock Trading & Portfolio Allocation (DJIA 30) -- Test Period: 2019/01/01 to 2020/09/23

| Strategy | Initial ($) | Final ($) | Annual Return | Annual Std | Sharpe | Max Drawdown |
|---|---|---|---|---|---|---|
| TD3 (multi-stock; portfolio) | 1,000,000 | 1,403,337; 1,381,120 | 21.40%; 17.61% | 14.60%; 17.01% | 1.38; 1.03 | 11.52%; 12.78% |
| DDPG (multi-stock; portfolio) | 1,000,000 | 1,396,607; 1,281,120 | 20.34%; 15.81% | 15.89%; 16.60% | 1.28; 0.98 | 13.72%; 13.68% |
| Min-Variance (baseline) | 1,000,000 | 1,171,120 | 8.38% | 26.21% | 0.44 | 34.34% |
| **DJIA Index (benchmark)** | **1,000,000** | **1,185,260** | **10.61%** | **28.63%** | **0.48** | **37.01%** |

**Key findings**:
- TD3 multi-stock trading achieved the highest Sharpe ratio (1.38) and lowest max drawdown (11.52%)
- Both TD3 and DDPG substantially outperformed DJIA index (Sharpe 0.48) and min-variance strategy (Sharpe 0.44)
- DRL agents achieved roughly 2-3x the Sharpe ratio of traditional strategies
- DRL agents had roughly 1/3 the maximum drawdown of traditional strategies

---

## 12. Technical Indicators

| Indicator | Formula Concept | Usage |
|---|---|---|
| MACD (M_t) | Difference between 12-period and 26-period EMA, plus 9-period signal line | Trend following / momentum |
| RSI (R_t) | 100 - 100/(1 + avg_gain/avg_loss) over 14 periods | Overbought/oversold detection |
| Open/High/Low/Close | Raw price data per period | Price action features |
| Volume | Total shares traded per period | Liquidity / activity signal |

---

## 13. Three Application Modes

### 13.1 Single Stock Trading
- Agent trades one stock at a time
- Action: scalar a_t in {-k, ..., k}
- State: [balance, shares, price, OHLCV, indicators]
- Demonstrated with PPO on SPY, QQQ, GOOGL, AMZN, AAPL, MSFT

### 13.2 Multiple Stock Trading
- Agent simultaneously trades n stocks (e.g., DJIA 30 constituents)
- Action: vector a_t in {-k, ..., k}^n
- Portfolio value tracks combined holdings
- Demonstrated with TD3 and DDPG

### 13.3 Portfolio Allocation
- Agent outputs portfolio weight vector w_t in R^n
- Rebalances holdings to match target allocation
- Compared against mean-variance, min-variance, equal-weight, momentum, buy-and-hold
- Demonstrated with TD3 and DDPG on DJIA 30

---

## 14. Key Equations Summary

**MDP transition:**
```
s_{t+1} ~ P(s_{t+1} | s_t, a_t)
```

**Objective:**
```
max_pi  E[ sum_{t=0}^{T} gamma^t * r(s_t, a_t, s_{t+1}) ]
```

**Portfolio value:**
```
V_t = b_t + sum_{i=1}^{n} h_t^i * p_t^i
```

**Reward (value change):**
```
r_t = V_{t+1} - V_t
```

**Reward (log return):**
```
r_t = log(V_{t+1} / V_t)
```

**Sharpe ratio:**
```
S_T = mean(R_t) / std(R_t),    R_t = V_t - V_{t-1}
```

**Financial turbulence index:**
```
turbulence_t = (y_t - mu) * Sigma^{-1} * (y_t - mu)^T
```

**Transaction cost (per-share):**
```
cost_t = rate * |a_t| * p_t
```

**Maximum drawdown:**
```
MDD = max_{t} [ (max_{tau <= t} V_tau  -  V_t) / (max_{tau <= t} V_tau) ]
```
