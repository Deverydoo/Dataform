# Empirical Asset Pricing via Machine Learning -- Curated Technical Extraction

**Source:** Gu, Kelly, Xiu (NBER WP 25398, 2018; Review of Financial Studies, 2020)

---

## 1. OVERARCHING MODEL

The additive prediction error model for excess stock returns:

```
r_{i,t+1} = E_t(r_{i,t+1}) + epsilon_{i,t+1}
```

where:

```
E_t(r_{i,t+1}) = g*(z_{i,t})
```

- Stocks indexed i = 1,...,N_t; months t = 1,...,T
- z_{i,t} is a P-dimensional vector of predictors
- g*() is a flexible function, invariant across stocks and time (depends only on z_{i,t})
- The function g*() depends on z only through z_{i,t} -- no history prior to t, no cross-stock information

### Feature Construction

The baseline covariate set is defined as:

```
z_{i,t} = x_t (Kronecker) c_{i,t}
```

where:
- c_{i,t} = P_c x 1 matrix of stock-level characteristics (94 total)
- x_t = P_x x 1 vector of macroeconomic predictors (8 variables + constant)
- z_{i,t} = P x 1 vector with P = P_c * P_x

**Total covariates:** 94 x (8 + 1) + 74 industry dummies = **920 baseline features**

This nests the conditional beta-pricing representation:

```
E_t(r_{i,t+1}) = beta'_{i,t} * lambda_t
```

If beta_{i,t} = theta_1 * c_{i,t} and lambda_t = theta_2 * x_t, then:

```
g*(z_{i,t}) = c'_{i,t} * theta'_1 * theta_2 * x_t = (x_t (Kronecker) c_{i,t})' * vec(theta'_1 * theta_2) = z'_{i,t} * theta
```

---

## 2. SAMPLE SPLITTING AND TUNING

Three disjoint time periods maintaining temporal ordering:

| Subsample | Period | Purpose |
|---|---|---|
| Training | 1957-1974 (18 years) | Estimate model given fixed hyperparameters |
| Validation | 1975-1986 (12 years) | Tune hyperparameters by simulating OOS |
| Testing | 1987-2016 (30 years) | True out-of-sample evaluation |

**Recursive scheme:** Refit once per year. Each refit increases training sample by one year; validation sample rolls forward (fixed 12-month window). No cross-validation -- temporal ordering preserved.

---

## 3. SIMPLE LINEAR MODEL (OLS)

### Model

```
g(z_{i,t}; theta) = z'_{i,t} * theta
```

### Objective Function (L2 Loss)

```
L(theta) = (1 / NT) * SUM_{i=1}^{N} SUM_{t=1}^{T} (r_{i,t+1} - g(z_{i,t}; theta))^2
```

### Weighted Least Squares Variant

```
L_W(theta) = (1 / NT) * SUM_{i=1}^{N} SUM_{t=1}^{T} w_{i,t} * (r_{i,t+1} - g(z_{i,t}; theta))^2
```

Weighting options:
- w_{i,t} inversely proportional to number of stocks at time t (equal month weighting)
- w_{i,t} proportional to market equity of stock i at time t (value weighting)

### Huber Robust Objective

```
L_H(theta) = (1 / NT) * SUM_{i=1}^{N} SUM_{t=1}^{T} H(r_{i,t+1} - g(z_{i,t}; theta), xi)
```

where:

```
H(x; xi) = { x^2,              if |x| <= xi
            { 2*xi*|x| - xi^2,  if |x| > xi
```

- Hybrid of squared loss (small errors) and absolute loss (large errors)
- xi is a tuning parameter (set to 99.9% quantile of errors)
- OLS is special case with xi = infinity

---

## 4. PENALIZED LINEAR MODELS

### Model

Same as simple linear: g(z_{i,t}; theta) = z'_{i,t} * theta

### Penalized Objective

```
L(theta; .) = L(theta) + phi(theta; .)
```

### Elastic Net Penalty

```
phi(theta; lambda, rho) = lambda*(1-rho) * SUM_{j=1}^{P} |theta_j| + (1/2)*lambda*rho * SUM_{j=1}^{P} theta_j^2
```

**Special cases:**
- rho = 0: **Lasso** (L1 penalty) -- variable selection, sets coefficients exactly to zero
- rho = 1: **Ridge** (L2 penalty) -- shrinkage, draws all coefficients toward zero
- 0 < rho < 1: **Elastic Net** -- both shrinkage and selection

**Hyperparameters:** lambda and rho, optimized via validation sample.

**In practice:** rho = 0.5 fixed; lambda searched over (10^{-4}, 10^{-1}).

### Computational Algorithm: Accelerated Proximal Gradient (APG)

Rewrite objective as L(theta; .) = L(theta) + phi(theta; .)

The proximal operator:

```
prox_{gamma*f}(theta) = argmin_z { f(z) + (1/(2*gamma)) * ||z - theta||^2 }
```

Fixed point condition: theta* minimizes L(theta; .) iff:

```
theta* = prox_{gamma*phi}(theta* - gamma * grad(L(theta*)))
```

**Closed-form proximal operators:**

| Method | prox_{gamma*phi}(theta) |
|---|---|
| Ridge | theta / (1 + lambda*gamma) |
| Lasso | S(theta, lambda*gamma) |
| Elastic Net | (1/(1 + lambda*gamma*rho)) * S(theta, (1-rho)*lambda*gamma) |
| Group Lasso | (S_tilde(theta_1, lambda*gamma)', ..., S_tilde(theta_P, lambda*gamma)')' |

where the soft-thresholding operator S(x, mu):

```
S(x, mu)_i = { x_i - mu,  if x_i > 0 and mu < |x_i|
              { x_i + mu,  if x_i < 0 and mu < |x_i|
              { 0,          if mu >= |x_i|
```

**APG Algorithm:**

```
Initialize: theta_0 = 0, m = 0, gamma (step size)
While theta_m not converged:
    theta_bar  <-- theta_m - gamma * grad(L(theta))|_{theta=theta_m}
    theta_tilde <-- prox_{gamma*phi}(theta_bar)
    theta_{m+1} <-- theta_tilde + (m/(m+3)) * (theta_tilde - theta_m)   [Nesterov momentum]
    m <-- m + 1
Return theta_m
```

---

## 5. DIMENSION REDUCTION: PCR AND PLS

### Model

Starting from R = Z*theta + E (vectorized), reduce dimension:

```
R = (Z * Omega_K) * theta_K + E_tilde
```

where:
- Omega_K is P x K matrix of combination weights [w_1, w_2, ..., w_K]
- Z * Omega_K is the dimension-reduced predictor set
- theta_K is K x 1 (instead of P x 1)

### PCR Objective

The j-th linear combination solves:

```
w_j = argmax_w Var(Z*w)
  s.t. w'w = 1
       Cov(Z*w, Z*w_l) = 0,  l = 1,...,j-1
```

PCR seeks components that retain maximal common variation in predictors. Solved via SVD of Z.

**Hyperparameter:** K (number of components), tuned via validation. Typical range: 20-40 components.

### PLS Objective

```
w_j = argmax_w Cov^2(R, Z*w)
  s.t. w'w = 1
       Cov(Z*w, Z*w_l) = 0,  l = 1,...,j-1
```

PLS maximizes predictive association with the forecast target. Solved via SIMPLS algorithm (de Jong, 1993).

**Hyperparameter:** K (number of components), tuned via validation. Typical range: 3-6 components.

Given Omega_K, theta_K estimated via OLS regression of R on Z*Omega_K.

---

## 6. GENERALIZED LINEAR MODEL

### Model

Adds K-term spline series expansion:

```
g(z; theta, p(.)) = SUM_{j=1}^{P} p(z_j)' * theta_j
```

where p(.) = (p_1(.), ..., p_K(.))' is a vector of basis functions. Order-2 spline:

```
{1, z, (z - c_1)^2, (z - c_2)^2, ..., (z - c_{K-2})^2}
```

with c_1, ..., c_{K-2} as knots.

### Group Lasso Penalty

```
phi(theta; lambda, K) = lambda * SUM_{j=1}^{P} (SUM_{k=1}^{K} theta_{j,k}^2)^{1/2}
```

Selects either all K spline terms for a characteristic or none of them.

**Hyperparameters:** lambda in (10^{-4}, 10^{-1}), K = 3 knots.

---

## 7. REGRESSION TREES

### Model

A tree T with K leaves and depth L:

```
g(z_{i,t}; theta, K, L) = SUM_{k=1}^{K} theta_k * 1{z_{i,t} in C_k(L)}
```

where:
- C_k(L) is one of K partitions of the data
- theta_k = (1/|C_k|) * SUM_{z_{i,t} in C_k} r_{i,t+1} (sample average within partition)
- Each partition is a product of up to L indicator functions

### Impurity Function (L2)

```
H(theta, C) = (1/|C|) * SUM_{z_{i,t} in C} (r_{i,t+1} - theta)^2
```

### Tree Growing Algorithm (Breiman et al., 1984)

```
Initialize stump. C_1(0) = range of all covariates.
For d from 1 to L:
  For each node C_l(d-1):
    i) For each feature j and threshold alpha, define split s = (j, alpha):
       C_left(s) = {z_j <= alpha} intersect C_l(d-1)
       C_right(s) = {z_j > alpha} intersect C_l(d-1)

    ii) Impurity function:
       L(C, C_left, C_right) = (|C_left|/|C|)*H(C_left) + (|C_right|/|C|)*H(C_right)

    iii) Select optimal split:
       s* = argmin_s L(C(s), C_left(s), C_right(s))

    iv) Update nodes:
       C_{2l-1}(d) = C_left(s*)
       C_{2l}(d) = C_right(s*)

Output: g(z_{i,t}; theta, L) = SUM_{k=1}^{2^L} theta_k * 1{z_{i,t} in C_k(L)}
```

### Variable Importance for Trees

```
VIP(z_j, T) = SUM_{d=1}^{L-1} SUM_{i=1}^{2^{d-1}} Delta_im(C_i(d-1), C_{2i-1}(d), C_{2i}(d)) * 1{z_j in T(i,d)}
```

where Delta_im(C, C_left, C_right) = H(C) - L(C, C_left, C_right).

---

## 8. RANDOM FORESTS

### Algorithm

```
For b from 1 to B:
  Generate bootstrap sample from training data.
  Grow tree using Algorithm 2 with modification:
    At each split, use only a random subsample of sqrt(P) features ("dropout").
  Write b-th tree: g_b(z_{i,t}; theta_b, L) = SUM_{k=1}^{2^L} theta_{k,b} * 1{z_{i,t} in C_k(L)}

Final output: g(z_{i,t}; L, B) = (1/B) * SUM_{b=1}^{B} g_b(z_{i,t}; theta_b, L)
```

**Hyperparameters:** Depth L (1-6), number of trees B = 300. Features per split = sqrt(P).

---

## 9. GRADIENT BOOSTED REGRESSION TREES (GBRT)

### Algorithm

```
Initialize: g_0(.) = 0
For b from 1 to B:
  Compute negative gradient of loss:
    epsilon_{i,t+1} = -d(l(r_{i,t+1}, g)) / dg |_{g = g_{b-1}(z_{i,t})}

  Grow a SHALLOW regression tree of depth L on {(z_{i,t}, epsilon_{i,t+1})}:
    f_b(.) = g(z_{i,t}; theta, L)

  Update model:
    g_b(.) = g_{b-1}(.) + nu * f_b(.)

Final output: g_B(z_{i,t}; B, nu, L) = SUM_{b=1}^{B} nu * f_b(.)
```

- nu in (0,1] is the learning rate (shrinkage factor)
- l(.,.) is L2 or Huber loss

**Hyperparameters:**
- Depth L: 1-2
- Number of trees B: 1-1000
- Learning rate nu: {0.01, 0.1}

---

## 10. NEURAL NETWORKS

### Architecture

Feed-forward networks with:
- Input layer: raw predictors x^{(0)} = (1, z_1, ..., z_P)'
- L hidden layers with K^{(l)} neurons in layer l
- Output layer: scalar prediction

**Tested architectures (geometric pyramid rule):**

| Model | Hidden Layers | Neurons per Layer |
|---|---|---|
| NN1 | 1 | 32 |
| NN2 | 2 | 32, 16 |
| NN3 | 3 | 32, 16, 8 |
| NN4 | 4 | 32, 16, 8, 4 |
| NN5 | 5 | 32, 16, 8, 4, 2 |

All architectures are fully connected.

### Recursive Output Formula

For each neuron k in layer l > 0:

```
x^{(l)}_k = ReLU(x^{(l-1)'} * theta^{(l-1)}_k)
```

where ReLU(x) = max(0, x)

Final output:

```
g(z; theta) = x^{(L-1)'} * theta^{(L-1)}
```

**Parameter count per layer l:** K^{(l)} * (1 + K^{(l-1)}) plus (1 + K^{(L-1)}) for output layer.

### Optimization: Adam (Adaptive Moment Estimation)

```
Initialize: theta_0, m_0 = 0, v_0 = 0, t = 0
While theta_t not converged:
  t <-- t + 1
  g_t <-- grad_theta L_t(theta)|_{theta=theta_{t-1}}
  m_t <-- beta_1 * m_{t-1} + (1 - beta_1) * g_t
  v_t <-- beta_2 * v_{t-1} + (1 - beta_2) * g_t (elementwise) g_t
  m_hat_t <-- m_t / (1 - beta_1^t)
  v_hat_t <-- v_t / (1 - beta_2^t)
  theta_t <-- theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
Return theta_t
```

Where (elementwise) denotes Hadamard product. Default Adam parameters used.

### Early Stopping Algorithm

```
Initialize: j = 0, epsilon = infinity, patience parameter p = 5
While j < p:
  Update theta using training algorithm for h steps
  Calculate validation prediction error epsilon'
  If epsilon' < epsilon:
    j <-- 0
    epsilon <-- epsilon'
    theta' <-- theta
  Else:
    j <-- j + 1
Return theta'
```

### Batch Normalization (per activation, per batch)

```
Input: B = {x_1, ..., x_N}
mu_B = (1/N) * SUM x_i
sigma^2_B = (1/N) * SUM (x_i - mu_B)^2
x_hat_i = (x_i - mu_B) / sqrt(sigma^2_B + epsilon)
y_i = gamma * x_hat_i + beta := BN_{gamma,beta}(x_i)
```

Applied to each activation after ReLU transformation.

### Regularization Stack for Neural Networks

1. **L1 penalty** on weight parameters: lambda_1 in (10^{-5}, 10^{-3})
2. **Learning rate shrinkage** via Adam: LR in {0.001, 0.01}
3. **Early stopping** with patience p = 5
4. **Batch normalization** at each hidden layer
5. **Ensemble of 10 networks** with different random seeds, predictions averaged

**Additional settings:** Batch size = 10,000; Epochs = 100.

---

## 11. EVALUATION METRICS

### Out-of-Sample R-Squared

```
R^2_oos = 1 - SUM_{(i,t) in T_3} (r_{i,t+1} - r_hat_{i,t+1})^2
               / SUM_{(i,t) in T_3} r^2_{i,t+1}
```

- T_3 = testing subsample only
- Denominator uses raw squared returns (NOT demeaned) -- benchmark is a forecast of zero
- Pools across firms and time into a panel-level assessment

### Diebold-Mariano Test (Modified for Panel)

To compare method (1) vs (2):

```
d_{12,t+1} = (1/n_{3,t+1}) * SUM_{i=1}^{n_3} [ (e^{(1)}_{i,t+1})^2 - (e^{(2)}_{i,t+1})^2 ]
```

```
DM_{12} = d_bar_{12} / sigma_hat_{d_bar_{12}}
```

where d_bar_{12} and sigma_hat are the mean and Newey-West standard error of d_{12,t} over the testing sample. Cross-sectional averaging handles strong cross-sectional dependence in stock returns.

Under H_0 (no difference): DM_{12} ~ N(0,1). Bonferroni-adjusted critical value for 12 comparisons at 5%: **2.64**.

### Campbell-Thompson Sharpe Ratio Improvement

An active investor exploiting predictive R^2 information:

```
SR* = sqrt( SR^2 + R^2 / (1 - R^2) )
```

### Maximum Drawdown

```
MaxDD = max_{0 <= t1 <= t2 <= T} (Y_{t1} - Y_{t2})
```

where Y_t = cumulative log return from date 0 through t.

### Portfolio Turnover

```
Turnover = (1/T) * SUM_{t=1}^{T} SUM_i |w_{i,t+1} - w_{i,t}*(1+r_{i,t+1}) / SUM_j w_{j,t}*(1+r_{j,t+1})|
```

---

## 12. VARIABLE IMPORTANCE MEASURES

### Method 1: R-Squared Reduction

For predictor j: set all values of predictor j to zero, hold remaining model estimates fixed, measure reduction in panel predictive R^2.

### Method 2: Sum of Squared Derivatives (SSD)

```
SSD_j = SUM_{(i,t) in T_1} ( d g(z; theta) / d z_j |_{z=z_{i,t}} )^2
```

Measured in the training set T_1. Not applicable to tree-based models (non-differentiable); for trees, use mean decrease in impurity instead.

Correlation between SSD and R^2 measures: 84.0% (NN1) to 97.7% (random forest).

---

## 13. FEATURE ENGINEERING

### 94 Stock-Level Characteristics

Organized by update frequency:
- **Monthly (20):** baspread, beta, betasq, chmom, dolvol, idiovol, ill, indmom, maxret, mom1m, mom12m, mom36m, mom6m, mvel1, pricedelay, retvol, std_dolvol, std_turn, turn, zerotrade
- **Quarterly (13):** aeavol, cash, chtx, cinvest, ear, ms, nincr, roaq, roavol, roeq, rsup, stdacc, stdcf
- **Annual (61):** absacc, acc, age, agr, bm, bm_ia, cashdebt, cashpr, cfp, cfp_ia, chatoia, chcsho, chempia, chinv, chpmia, convind, currat, depr, divi, divo, dy, egr, ep, gma, grcapx, grltnoa, herf, hire, invest, lev, lgr, mve_ia, operprof, orgcap, pchcapx_ia, pchcurrat, pchdepr, pchgm_pchsale, pchquick, pchsale_pchinvt, pchsale_pchrect, pchsale_pchxsga, pchsaleinv, pctacc, ps, quick, rd, rd_mve, rd_sale, realestate, roic, salecash, saleinv, salerec, secured, securedind, sgr, sin, sp, tang, tb

Plus **74 industry dummies** (2-digit SIC codes).

### Data Preprocessing

- All characteristics cross-sectionally ranked period-by-period and mapped to [-1, 1] interval
- Missing characteristics replaced with cross-sectional median each month
- Monthly characteristics lagged 1 month; quarterly lagged 4 months; annual lagged 6 months

### 8 Macroeconomic Predictors

| Variable | Acronym | Description |
|---|---|---|
| dp | Dividend-price ratio | log(dividends) - log(price) |
| ep | Earnings-price ratio | log(earnings) - log(price) |
| bm | Book-to-market ratio | Aggregate book-to-market |
| ntis | Net equity expansion | Net equity issuance / GDP |
| tbl | Treasury-bill rate | 3-month T-bill yield |
| tms | Term spread | Long-term yield - T-bill rate |
| dfy | Default spread | BAA yield - AAA yield |
| svar | Stock variance | Sum of squared daily returns |

---

## 14. KEY RESULTS: MODEL COMPARISON

### Table 1: Monthly Out-of-Sample R^2_oos (%) -- Individual Stock Returns

| Model | All Stocks | Top 1000 | Bottom 1000 |
|---|---|---|---|
| OLS (all 920) | -3.46 | -11.28 | -1.30 |
| OLS-3+H | 0.16 | 0.31 | 0.17 |
| PLS | 0.27 | -0.14 | 0.42 |
| PCR | 0.26 | 0.06 | 0.34 |
| ENet+H | 0.11 | 0.25 | 0.20 |
| GLM+H | 0.19 | 0.14 | 0.30 |
| RF | 0.33 | 0.63 | 0.35 |
| GBRT+H | 0.34 | 0.52 | 0.32 |
| NN1 | 0.33 | 0.49 | 0.38 |
| NN2 | 0.39 | 0.62 | 0.46 |
| **NN3** | **0.40** | **0.70** | **0.45** |
| NN4 | 0.39 | 0.67 | 0.47 |
| NN5 | 0.36 | 0.64 | 0.42 |

### Table 2: Annual Out-of-Sample R^2_oos (%) -- Individual Stock Returns

| Model | All | Top 1000 | Bottom 1000 |
|---|---|---|---|
| OLS (all) | -34.86 | -54.86 | -19.22 |
| OLS-3+H | 2.50 | 2.48 | 4.88 |
| PLS | 2.93 | 1.84 | 5.36 |
| PCR | 3.08 | 1.64 | 5.44 |
| ENet+H | 1.78 | 1.90 | 3.94 |
| GLM+H | 2.60 | 1.82 | 5.00 |
| RF | 3.28 | 4.80 | 5.08 |
| GBRT+H | 3.09 | 4.07 | 4.61 |
| NN3 | 3.40 | 4.73 | 5.17 |
| **NN4** | **3.60** | **4.91** | 5.01 |

### Table 5: Monthly Portfolio-Level R^2_oos (%) -- Select Results

| Portfolio | OLS-3+H | ENet+H | GLM+H | RF | GBRT+H | NN3 |
|---|---|---|---|---|---|---|
| S&P 500 | -0.22 | 0.75 | 0.71 | 1.37 | 1.40 | **1.80** |
| SMB | 0.81 | 1.72 | 2.36 | 0.57 | 0.35 | 1.31 |
| HML | 0.66 | 0.46 | 0.84 | 0.98 | 0.21 | 1.06 |
| RMW | -2.35 | -1.07 | -0.06 | -0.54 | -0.92 | 0.84 |
| CMA | 0.80 | -1.07 | 1.24 | -0.11 | -1.04 | 1.06 |
| UMD | -0.90 | 0.47 | -0.37 | 1.37 | -0.25 | 0.19 |

NN3 produces positive R^2_oos for every portfolio analyzed.

### Table 7: Long-Short Decile Spread Performance (Value-Weighted, Monthly)

| Model | H-L Pred (%) | H-L Avg Ret (%) | H-L Std (%) | H-L Ann. SR |
|---|---|---|---|---|
| OLS-3+H | 1.67 | 0.94 | 5.33 | 0.61 |
| PLS | 3.09 | 1.02 | 4.88 | 0.72 |
| PCR | 2.70 | 1.22 | 4.82 | 0.88 |
| ENet+H | 1.70 | 0.60 | 5.37 | 0.39 |
| GLM+H | 2.27 | 1.06 | 4.79 | 0.76 |
| RF | 0.83 | 1.62 | 5.75 | 0.98 |
| GBRT+H | 1.56 | 0.99 | 4.22 | 0.81 |
| NN1 | 2.57 | 1.81 | 5.34 | 1.17 |
| NN2 | 2.22 | 1.92 | 5.75 | 1.16 |
| NN3 | 1.86 | 2.12 | 6.13 | 1.20 |
| **NN4** | **2.01** | **2.26** | **5.80** | **1.35** |
| NN5 | 2.22 | 1.97 | 5.93 | 1.15 |

### Table 8: Risk-Adjusted Performance (Value-Weighted H-L)

| Model | Mean Ret (%) | FF5+Mom alpha (%) | t(alpha) | R^2 (%) | IR |
|---|---|---|---|---|---|
| OLS-3+H | 0.94 | 0.39 | 2.76 | 78.60 | 0.54 |
| RF | 1.62 | 1.20 | 3.95 | 13.43 | 0.77 |
| GBRT+H | 0.99 | 0.66 | 3.11 | 20.68 | 0.61 |
| NN1 | 1.81 | 1.20 | 4.68 | 27.67 | 0.92 |
| NN3 | 1.97 | 1.52 | 4.92 | 20.84 | 0.96 |
| **NN4** | **2.26** | **1.76** | **6.00** | **20.47** | **1.18** |

### Drawdowns (Value-Weighted H-L)

| Model | Max Drawdown (%) | Max 1M Loss (%) | Turnover (%) |
|---|---|---|---|
| OLS-3+H | 69.60 | 24.72 | 58.20 |
| NN3 | 30.84 | 30.84 | 123.50 |
| NN4 | 51.78 | 33.03 | 126.81 |

---

## 15. HYPERPARAMETER SUMMARY TABLE

| Method | Hyperparameters |
|---|---|
| OLS-3+H | Huber xi = 99.9% quantile |
| PLS | K components (tuned via validation) |
| PCR | K components (tuned via validation) |
| ENet+H | rho = 0.5 fixed; lambda in (10^{-4}, 10^{-1}); Huber xi = 99.9% quantile |
| GLM+H | #Knots = 3; lambda in (10^{-4}, 10^{-1}); Huber xi = 99.9% quantile |
| RF | Depth = 1-6; #Trees = 300; #Features per split = {3, 5, 10, 20, 30, 50, ...} |
| GBRT+H | Depth = 1-2; #Trees = 1-1000; LR = {0.01, 0.1}; Huber xi = 99.9% quantile |
| NN1-NN5 | L1 penalty lambda_1 in (10^{-5}, 10^{-3}); LR = {0.001, 0.01}; Batch = 10,000; Epochs = 100; Patience = 5; Adam defaults; Ensemble = 10 seeds |

---

## 16. DOMINANT PREDICTIVE SIGNALS (RANKED BY IMPORTANCE)

All models converge on the same top predictors across both R^2-reduction and SSD importance measures:

### Tier 1: Price Trends (Most Influential)
1. **mom1m** -- short-term reversal (1-month momentum)
2. **mom12m** -- 12-month momentum
3. **chmom** -- change in 6-month momentum
4. **indmom** -- industry momentum
5. **maxret** -- maximum daily return

### Tier 2: Liquidity
6. **mvel1** -- log market equity (size)
7. **dolvol** -- dollar trading volume
8. **turn** -- share turnover
9. **std_turn** -- turnover volatility
10. **baspread** -- bid-ask spread
11. **ill** -- Amihud illiquidity
12. **zerotrade** -- zero trading days

### Tier 3: Risk Measures
13. **retvol** -- total return volatility
14. **idiovol** -- idiosyncratic return volatility
15. **beta** -- market beta
16. **betasq** -- beta squared

### Tier 4: Valuation and Fundamentals
17. **ep** -- earnings-to-price
18. **sp** -- sales-to-price
19. **agr** -- asset growth
20. **nincr** -- number of recent earnings increases

### Most Important Macroeconomic Predictors
- **bm** (aggregate book-to-market) -- dominant across all models
- **tbl** (Treasury-bill rate) -- important for neural networks
- **ntis** (net equity expansion) -- important for nonlinear models
- **dfy** (default spread) -- important for linear models
- **svar** (stock variance) -- least important across all models

---

## 17. MONTE CARLO SIMULATION DESIGN

### Data Generating Process

Latent 3-factor model:

```
r_{i,t+1} = g*(z_{i,t}) + e_{i,t+1}
e_{i,t+1} = beta_{i,t} * v_{t+1} + epsilon_{i,t+1}
z_{i,t} = (1, x_t)' (Kronecker) c_{i,t}
beta_{i,t} = (c_{i1,t}, c_{i2,t}, c_{i3,t})
```

where v_{t+1} ~ N(0, 0.05^2 * I_3) and epsilon_{i,t+1} ~ t_5(0, 0.05^2).

Calibrated: average time-series R^2 = 40%, average annualized volatility = 30%.

### Characteristic Simulation

```
c_{ij,t} = (2/(N+1)) * CSrank(c_bar_{ij,t}) - 1
c_bar_{ij,t} = rho_j * c_bar_{ij,t-1} + epsilon_{ij,t}
```

where rho_j ~ U[0.9, 1] and epsilon_{ij,t} ~ N(0, 1 - rho_j^2).

### Two Test Cases for g*()

**(a) Linear and sparse:**
```
g*(z_{i,t}) = (c_{i1,t}, c_{i2,t}, c_{i3,t} * x_t) * theta_0
theta_0 = (0.02, 0.02, 0.02)'
```

**(b) Nonlinear with interactions:**
```
g*(z_{i,t}) = (c^2_{i1,t}, c_{i1,t} * c_{i2,t}, sgn(c_{i3,t} * x_t)) * theta_0
theta_0 = (0.04, 0.03, 0.012)'
```

Calibrated: cross-sectional R^2 = 50%, predictive R^2 = 5%.

**Settings:** N = 200, T = 180, P_x = 2, P_c = {50, 100}.

### Simulation Results (OOS R^2, P_c = 100)

| Method | Model (a) | Model (b) |
|---|---|---|
| OLS | -1.35 | -7.75 |
| Lasso | 4.25 | 0.61 |
| ENet | 4.25 | 0.61 |
| Ridge | 3.39 | 0.23 |
| PLS | 2.82 | -0.17 |
| PCR | 0.43 | -0.01 |
| GLM | 4.08 | 1.17 |
| RF | 3.30 | 3.02 |
| GBRT+H | 3.37 | 3.22 |
| NN1 | 4.28 | 2.59 |
| NN2 | 4.26 | **2.91** |
| NN3 | 4.27 | 2.68 |
| Oracle | 5.52 | 5.40 |

**Key finding:** In linear DGP (a), penalized linear methods match neural networks. In nonlinear DGP (b), trees and neural networks dominate by a wide margin.

---

## 18. FORECAST ERROR DECOMPOSITION

```
r_{i,t+1} - r_hat_{i,t+1} = [g*(z_{i,t}) - g(z_{i,t}; theta)]    (approximation error)
                            + [g(z_{i,t}; theta) - g(z_{i,t}; theta_hat)]  (estimation error)
                            + epsilon_{i,t+1}                              (intrinsic error)
```

- **Intrinsic error:** irreducible noise
- **Estimation error:** due to sampling variation, reducible with more data
- **Approximation error:** controlled by model flexibility, reducible with richer specifications (but risks overfitting)

---

## 19. META-STRATEGIES

### Strategy 1: Equal-Weight Ensemble
- Average decile long-short portfolios from all 11 ML methods
- Stock-level R^2: 0.43% per month
- EW decile spread SR: 2.49; VW decile spread SR: 1.33

### Strategy 2: Best-Model Rotation
- Select best ML model each year based on validation R^2
- Over 30 years: NN3 selected 11 times, NN1 7 times, GBRT 6 times, NN2 5 times, NN4 once
- Panel R^2: 0.45%; VW decile spread SR: 1.23
