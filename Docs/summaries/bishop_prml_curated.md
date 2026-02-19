# Bishop - Pattern Recognition and Machine Learning (2006) - Curated Knowledge

Christopher M. Bishop | 758 pages | Springer | Chapters 1-14 + Appendix

---

## Curated Technical Summary (Equations, Algorithms, Derivations)

---

# Chapter 1: Introduction

## 1.1 Polynomial Curve Fitting

### Model
Polynomial function:
```
y(x, w) = w0 + w1*x + w2*x^2 + ... + wM*x^M = SUM_{j=0}^{M} wj * x^j    (1.1)
```
- **x**: input variable (scalar)
- **w** = (w0, w1, ..., wM)^T: weight vector (M+1 parameters)
- **M**: polynomial order

### Error Function (Sum-of-Squares)
```
E(w) = (1/2) * SUM_{n=1}^{N} { y(xn, w) - tn }^2    (1.2)
```
- **N**: number of training observations
- **xn**: nth input
- **tn**: nth target value
- Factor of 1/2 included for convenience in differentiation

### Root-Mean-Square Error
```
E_RMS = sqrt(2*E(w*)/N)    (1.3)
```
- **w***: weight vector that minimizes E(w)
- Dividing by N allows comparison across different data set sizes
- Square root ensures same units and scale as target variable t

### Regularized Error Function
```
E_tilde(w) = (1/2) * SUM_{n=1}^{N} {y(xn,w) - tn}^2 + (lambda/2) * ||w||^2    (1.4)
```
- **lambda**: regularization coefficient (controls overfitting)
- **||w||^2** = w^T * w = SUM_j wj^2: squared L2 norm of weight vector
- Penalizes large weight magnitudes
- Equivalent to ridge regression in statistics

### Normal Equations for Polynomial Fitting
Minimizing (1.2) yields linear system:
```
SUM_{j=0}^{M} Aij * wj = Ti    (1.122)
```
where:
```
Aij = SUM_{n=1}^{N} (xn)^{i+j}    (1.123a)
Ti  = SUM_{n=1}^{N} (xn)^i * tn    (1.123b)
```

---

## 1.2 Probability Theory

### Fundamental Rules
**Sum rule** (marginalization):
```
p(X) = SUM_Y p(X, Y)    (1.10)
```

**Product rule**:
```
p(X, Y) = p(Y|X) * p(X)    (1.11)
```

**Bayes' theorem**:
```
p(Y|X) = p(X|Y) * p(Y) / p(X)    (1.12)
```
where:
```
p(X) = SUM_Y p(X|Y) * p(Y)    (1.13)
```
- **p(Y)**: prior probability
- **p(X|Y)**: likelihood
- **p(Y|X)**: posterior probability
- **p(X)**: evidence (normalization constant)

### Independence
Two variables x, y are **independent** iff:
```
p(x, y) = p(x) * p(y)
```

### Probability Density (Continuous)
For continuous variable x:
```
p(x in (a,b)) = INTEGRAL_a^b p(x) dx
```
Constraints:
```
p(x) >= 0
INTEGRAL_{-inf}^{inf} p(x) dx = 1
```

Transformation under change of variable x = g(y):
```
p_y(y) = p_x(g(y)) * |g'(y)|    (1.27)
```

### Expectations and Covariances
**Expectation** (discrete):
```
E[f] = SUM_x p(x) * f(x)    (1.33)
```

**Expectation** (continuous):
```
E[f] = INTEGRAL p(x) * f(x) dx    (1.34)
```

**Approximate expectation** from N samples:
```
E[f] ~ (1/N) * SUM_{n=1}^{N} f(xn)    (1.35)
```

**Conditional expectation**:
```
E_x[f|y] = SUM_x p(x|y) * f(x)    (1.36)
```

**Variance**:
```
var[f] = E[(f(x) - E[f(x)])^2] = E[f(x)^2] - E[f(x)]^2    (1.38, 1.39)
```

**Covariance** (vectors x, y):
```
cov[x, y] = E_{x,y}[ {x - E[x]} {y - E[y]}^T ] = E_{x,y}[x*y^T] - E[x]*E[y]^T    (1.41, 1.42)
```

---

## 1.2.4 Gaussian Distribution

### Univariate Gaussian
```
N(x|mu, sigma^2) = (1 / sqrt(2*pi*sigma^2)) * exp( -(x - mu)^2 / (2*sigma^2) )    (1.46)
```
- **mu**: mean, E[x] = mu    (1.49)
- **sigma^2**: variance, var[x] = sigma^2    (1.51)
- **sigma**: standard deviation
- Mode = mean = mu
- E[x^2] = mu^2 + sigma^2    (1.50)

### Multivariate Gaussian
```
N(x|mu, Sigma) = (1 / ((2*pi)^{D/2} * |Sigma|^{1/2})) * exp( -(1/2) * (x-mu)^T * Sigma^{-1} * (x-mu) )    (1.52)
```
- **x**: D-dimensional vector
- **mu**: D-dimensional mean vector
- **Sigma**: D x D covariance matrix
- **|Sigma|**: determinant of Sigma
- **(x-mu)^T * Sigma^{-1} * (x-mu)**: Mahalanobis distance squared

### Maximum Likelihood for Gaussian
Given i.i.d. observations x = {x1, ..., xN} from N(x|mu, sigma^2):

**Likelihood**:
```
p(x|mu, sigma^2) = PROD_{n=1}^{N} N(xn|mu, sigma^2)    (1.53)
```

**Log likelihood**:
```
ln p(x|mu, sigma^2) = -(1/(2*sigma^2)) * SUM_{n=1}^{N} (xn - mu)^2 - (N/2)*ln(sigma^2) - (N/2)*ln(2*pi)    (1.54)
```

**MLE for mean**:
```
mu_ML = (1/N) * SUM_{n=1}^{N} xn    (1.55)
```

**MLE for variance**:
```
sigma^2_ML = (1/N) * SUM_{n=1}^{N} (xn - mu_ML)^2    (1.56)
```

**Bias of MLE variance**:
```
E[mu_ML] = mu    (1.57)
E[sigma^2_ML] = ((N-1)/N) * sigma^2    (1.58)
```
MLE variance is biased: systematically underestimates true variance.

**Unbiased variance estimator**:
```
sigma^2_hat = (1/(N-1)) * SUM_{n=1}^{N} (xn - mu_ML)^2 = (N/(N-1)) * sigma^2_ML    (1.59)
```

---

## 1.2.5 Curve Fitting Re-visited (Probabilistic View)

### Probabilistic Model
Target variable with Gaussian noise:
```
p(t|x, w, beta) = N(t | y(x,w), beta^{-1})    (1.60)
```
- **beta** = 1/sigma^2: precision (inverse variance)
- **y(x,w)**: polynomial function (1.1)

### Likelihood for Curve Fitting
```
p(t|x, w, beta) = PROD_{n=1}^{N} N(tn | y(xn,w), beta^{-1})    (1.61)
```

**Log likelihood**:
```
ln p(t|x, w, beta) = -(beta/2) * SUM_{n=1}^{N} {y(xn,w) - tn}^2 + (N/2)*ln(beta) - (N/2)*ln(2*pi)    (1.62)
```
Maximizing log likelihood w.r.t. w is equivalent to minimizing sum-of-squares error E(w) in (1.2).

### Predictive Distribution (MLE)
```
p(t|x, w_ML, beta_ML) = N(t | y(x, w_ML), beta_ML^{-1})    (1.64)
```

### MAP Estimation with Gaussian Prior
Prior over w:
```
p(w|alpha) = N(w | 0, alpha^{-1} * I) = ((alpha/(2*pi))^{(M+1)/2}) * exp(-(alpha/2) * w^T * w)    (1.65)
```
- **alpha**: precision of prior

MAP (maximum a posteriori) maximizes posterior p(w|x, t, alpha, beta):
```
(beta/2) * SUM_{n=1}^{N} {y(xn,w) - tn}^2 + (alpha/2) * w^T * w    (1.67)
```
This is equivalent to regularized least squares with lambda = alpha/beta.

### Bayesian Predictive Distribution
```
p(t|x, x, t) = INTEGRAL p(t|x, w) * p(w|x, t) dw    (1.68)
```

For the curve fitting problem (details in Ch 3):
```
p(t|x, x, t) = N(t | m(x), s^2(x))    (1.69)
```
where the mean and variance depend on x, expressing both noise and parameter uncertainty.

### Model Selection Criteria
**Akaike Information Criterion (AIC)**:
```
ln p(D|w_ML) - M    (1.73)
```
- **M**: number of adjustable parameters
- Penalty for model complexity

---

## 1.5 Decision Theory

### Classification: Minimizing Misclassification Rate
For K classes C1, ..., CK, assign x to class Ck that maximizes posterior p(Ck|x).

**Misclassification probability** (two classes):
```
p(mistake) = p(x in R1, C2) + p(x in R2, C1)
            = INTEGRAL_{R1} p(x, C2) dx + INTEGRAL_{R2} p(x, C1) dx    (1.78, 1.79)
```
Minimized when each x is assigned to class with highest posterior probability.

### Classification: Minimizing Expected Loss
With loss matrix Lkj (loss for assigning class Cj when true class is Ck):
```
E[L] = SUM_k SUM_j INTEGRAL_{Rj} Lkj * p(x, Ck) dx    (1.80)
```
Optimal decision: assign each x to class j that minimizes:
```
SUM_k Lkj * p(Ck|x)    (1.81)
```

### Reject Option
Introduce threshold theta. Reject input x if:
```
max_k p(Ck|x) <= theta
```
- theta = 1: reject all
- theta < 1/K: reject none

### Three Approaches to Classification
**(a) Generative model**: Model p(x|Ck) and p(Ck), use Bayes to get p(Ck|x):
```
p(Ck|x) = p(x|Ck) * p(Ck) / p(x)    (1.82)
p(x) = SUM_k p(x|Ck) * p(Ck)    (1.83)
```

**(b) Discriminative model**: Model p(Ck|x) directly.

**(c) Discriminant function**: Find f(x) mapping x directly to class labels.

### Combining Models (Naive Bayes)
Conditional independence assumption:
```
p(xI, xB | Ck) = p(xI|Ck) * p(xB|Ck)    (1.84)
```
Then:
```
p(Ck|xI, xB) proportional_to p(Ck|xI) * p(Ck|xB) / p(Ck)    (1.85)
```

### Regression: Minimizing Expected Loss
**General expected loss**:
```
E[L] = INTEGRAL INTEGRAL L(t, y(x)) * p(x, t) dx dt    (1.86)
```

**Squared loss** L(t, y(x)) = {y(x) - t}^2:
```
E[L] = INTEGRAL {y(x) - t}^2 * p(x, t) dx dt    (1.87)
```

**Optimal prediction** (minimizing expected squared loss):
```
y(x) = E_t[t|x] = INTEGRAL t * p(t|x) dt    (1.89)
```
This is the conditional mean (regression function).

**Decomposition of expected loss**:
```
E[L] = INTEGRAL {y(x) - E[t|x]}^2 * p(x) dx + INTEGRAL var[t|x] * p(x) dx    (1.90)
```
- First term: depends on choice of y(x), minimized when y(x) = E[t|x]
- Second term: irreducible noise (intrinsic variance of data)

**Minkowski loss** (generalization):
```
E[Lq] = INTEGRAL |y(x) - t|^q * p(x,t) dx dt    (1.91)
```
- q=2: conditional mean
- q=1: conditional median
- q->0: conditional mode

---

## 1.6 Information Theory

### Information Content
```
h(x) = -log2(p(x))    (1.92)
```
- Low probability events carry high information
- Units: bits (base 2) or nats (natural log)

### Entropy (Discrete)
```
H[x] = -SUM_x p(x) * ln(p(x))    (1.98)
```
- Non-negative: H[x] >= 0
- Maximum entropy = ln(M) for M equally likely states (uniform distribution)
- From multiplicity: W = N! / PROD_i ni!, H = (1/N) ln W    (1.94, 1.95)
- Using Stirling's approximation ln(N!) ~ N*ln(N) - N    (1.96)

### Differential Entropy (Continuous)
```
H[x] = -INTEGRAL p(x) * ln(p(x)) dx    (1.104)
```
- Can be negative (unlike discrete entropy)
- Maximum entropy distribution (given mean mu, variance sigma^2) is the Gaussian:
```
H[x] = (1/2) * {1 + ln(2*pi*sigma^2)}    (1.110)
```
Entropy increases with sigma^2.

### Conditional Entropy
```
H[y|x] = -INTEGRAL INTEGRAL p(y,x) * ln(p(y|x)) dy dx    (1.111)
```
Relation:
```
H[x, y] = H[y|x] + H[x]    (1.112)
```

### Kullback-Leibler Divergence
```
KL(p||q) = -INTEGRAL p(x) * ln(q(x)/p(x)) dx    (1.113)
```
- KL(p||q) >= 0 (Gibbs' inequality), with equality iff p(x) = q(x)
- NOT symmetric: KL(p||q) != KL(q||p)
- Measures "distance" between distributions p and q
- Minimizing KL(p||q) w.r.t. parameters of q is equivalent to maximizing likelihood    (1.119)

**Jensen's inequality** (convex function f):
```
f(E[x]) <= E[f(x)]    (1.116)
```

### Mutual Information
```
I[x,y] = KL(p(x,y) || p(x)*p(y)) = -INTEGRAL INTEGRAL p(x,y) * ln(p(x)*p(y)/p(x,y)) dx dy    (1.120)
```
- I[x,y] >= 0, with equality iff x and y are independent
- Relation to entropy:
```
I[x,y] = H[x] - H[x|y] = H[y] - H[y|x]    (1.121)
```

---

# Chapter 2: Probability Distributions

## 2.1 Binary Variables

### Bernoulli Distribution
```
Bern(x|mu) = mu^x * (1 - mu)^{1-x}    (2.2)
```
- **x** in {0, 1}
- **mu** = p(x=1), where 0 <= mu <= 1
- E[x] = mu    (2.3)
- var[x] = mu*(1 - mu)    (2.4)

**MLE for mu**:
```
mu_ML = (1/N) * SUM_{n=1}^{N} xn = m/N    (2.7, 2.8)
```
where m = number of x=1 observations.

**Sufficient statistic**: SUM_n xn

### Binomial Distribution
Distribution of m successes in N trials:
```
Bin(m|N, mu) = C(N,m) * mu^m * (1 - mu)^{N-m}    (2.9)
```
where:
```
C(N,m) = N! / ((N-m)! * m!)    (2.10)
```
- E[m] = N*mu    (2.11)
- var[m] = N*mu*(1 - mu)    (2.12)

### Beta Distribution (Conjugate Prior for Bernoulli/Binomial)
```
Beta(mu|a, b) = Gamma(a+b) / (Gamma(a)*Gamma(b)) * mu^{a-1} * (1-mu)^{b-1}    (2.13)
```
- **a, b**: hyperparameters (effective prior observations)
- E[mu] = a/(a+b)    (2.15)
- var[mu] = ab / ((a+b)^2 * (a+b+1))    (2.16)
- Normalized: INTEGRAL_0^1 Beta(mu|a,b) dmu = 1    (2.14)

### Posterior with Beta Prior
Given m heads (x=1) and l = N-m tails from binomial likelihood:
```
p(mu|m, l, a, b) = Beta(mu | m+a, l+b)    (2.18)
```
- Posterior Beta(mu | m+a, l+b): hyperparameters updated by observed counts
- a, b act as effective prior counts for x=1, x=0 respectively
- Sequential updating: each observation updates the Beta parameters

### Predictive Distribution (next observation)
```
p(x=1|D) = INTEGRAL_0^1 p(x=1|mu) * p(mu|D) dmu = (m+a)/(m+a+l+b)    (2.20)
```

---

## 2.2 Multinomial Variables

### 1-of-K Representation
x = (0, ..., 0, 1, 0, ..., 0)^T, where xk = 1 for one k and all others 0.
```
p(x|mu) = PROD_{k=1}^{K} mu_k^{xk}    (2.26)
```
- **mu** = (mu1, ..., muK)^T, where SUM_k mu_k = 1 and mu_k >= 0
- E[x|mu] = mu    (2.28)

### Multinomial Distribution
```
Mult(m1, ..., mK | mu, N) = (N! / (m1! * m2! * ... * mK!)) * PROD_{k=1}^{K} mu_k^{mk}    (2.34)
```
- **mk**: count of observations where xk=1
- Constraint: SUM_k mk = N    (2.36)

**MLE**:
```
mu_k^ML = mk / N    (2.33)
```

**Sufficient statistics**: {m1, ..., mK}    (2.30)

### Dirichlet Distribution (Conjugate Prior for Multinomial)
```
Dir(mu|alpha) = (Gamma(alpha_0) / (Gamma(alpha_1) * ... * Gamma(alpha_K))) * PROD_{k=1}^{K} mu_k^{alpha_k - 1}    (2.38)
```
where:
```
alpha_0 = SUM_{k=1}^{K} alpha_k    (2.39)
```
- **alpha** = (alpha_1, ..., alpha_K)^T: hyperparameters (effective prior counts)
- Defined on the simplex: mu_k >= 0, SUM_k mu_k = 1

### Posterior with Dirichlet Prior
```
p(mu|D, alpha) = Dir(mu | alpha + m) = Dir(mu | alpha_1+m1, ..., alpha_K+mK)    (2.41)
```
Explicit normalization:
```
p(mu|D, alpha) = Gamma(alpha_0 + N) / (Gamma(alpha_1+m1) * ... * Gamma(alpha_K+mK)) * PROD_k mu_k^{alpha_k + mk - 1}
```

---

## 2.3 The Gaussian Distribution

### Multivariate Gaussian
```
N(x|mu, Sigma) = (1 / ((2*pi)^{D/2} * |Sigma|^{1/2})) * exp(-(1/2) * (x-mu)^T * Sigma^{-1} * (x-mu))    (2.43)
```
- **(x-mu)^T * Sigma^{-1} * (x-mu)**: Mahalanobis distance squared (= Delta^2)    (2.44)
- Constant-density contours are ellipsoids defined by Delta^2 = const

### Key Geometric Properties
- Sigma = U * Lambda * U^T (eigendecomposition), where Lambda = diag(lambda_i)
- Sigma^{-1} = U * Lambda^{-1} * U^T
- In rotated coordinates y_i = u_i^T * (x - mu):
```
Delta^2 = SUM_{i=1}^{D} y_i^2 / lambda_i    (2.50)
```
- Gaussian is a product of D independent univariate Gaussians in y-space

### Partitioned Gaussians
Given joint Gaussian N(x|mu, Sigma) with x = (xa, xb)^T:
```
mu = (mu_a, mu_b)^T    (2.94)
Sigma = [[Sigma_aa, Sigma_ab], [Sigma_ba, Sigma_bb]]    (2.95)
Lambda = Sigma^{-1} = [[Lambda_aa, Lambda_ab], [Lambda_ba, Lambda_bb]]    (2.95)
```

**Conditional distribution** p(xa|xb):
```
p(xa|xb) = N(xa | mu_{a|b}, Lambda_aa^{-1})    (2.96)
mu_{a|b} = mu_a - Lambda_aa^{-1} * Lambda_ab * (xb - mu_b)    (2.97)
```

Equivalently in terms of covariance matrix:
```
mu_{a|b} = mu_a + Sigma_ab * Sigma_bb^{-1} * (xb - mu_b)    (2.81)
Sigma_{a|b} = Sigma_aa - Sigma_ab * Sigma_bb^{-1} * Sigma_ba    (2.82)
```
- Conditional mean is **linear** in xb
- Conditional covariance is **independent** of xb

**Marginal distribution** p(xa):
```
p(xa) = N(xa | mu_a, Sigma_aa)    (2.98)
```
```
E[xa] = mu_a    (2.92)
cov[xa] = Sigma_aa    (2.93)
```

### Partitioned Matrix Inverse (Schur Complement)
```
[[A, B], [C, D]]^{-1} = [[M, -M*B*D^{-1}], [-D^{-1}*C*M, D^{-1}+D^{-1}*C*M*B*D^{-1}]]    (2.76)
```
where M = (A - B*D^{-1}*C)^{-1} is the Schur complement of D.    (2.77)

### Bayes' Theorem for Gaussian Variables (Linear Gaussian Model)
Given:
```
p(x) = N(x | mu, Lambda^{-1})    (2.113)
p(y|x) = N(y | A*x + b, L^{-1})    (2.114)
```
where A is D x M matrix, Lambda and L are precision matrices.

**Marginal** p(y):
```
p(y) = N(y | A*mu + b, L^{-1} + A*Lambda^{-1}*A^T)    (2.115)
```

**Posterior** p(x|y):
```
p(x|y) = N(x | Sigma_post * {A^T * L * (y-b) + Lambda * mu}, Sigma_post)    (2.116)
```
where:
```
Sigma_post = (Lambda + A^T * L * A)^{-1}    (2.117)
```

### MLE for Multivariate Gaussian
Given X = {x1, ..., xN}:

**Log likelihood**:
```
ln p(X|mu, Sigma) = -(ND/2)*ln(2*pi) - (N/2)*ln|Sigma| - (1/2)*SUM_{n=1}^{N} (xn-mu)^T * Sigma^{-1} * (xn-mu)    (2.118)
```

**Sufficient statistics**: SUM_n xn and SUM_n xn*xn^T    (2.119)

**MLE solutions**:
```
mu_ML = (1/N) * SUM_{n=1}^{N} xn    (2.121)
Sigma_ML = (1/N) * SUM_{n=1}^{N} (xn - mu_ML) * (xn - mu_ML)^T    (2.122)
```

**Bias**:
```
E[mu_ML] = mu    (2.123)
E[Sigma_ML] = ((N-1)/N) * Sigma    (2.124)
```
Unbiased estimator: Sigma_hat = (1/(N-1)) * SUM_n (xn - mu_ML)(xn - mu_ML)^T    (2.125)

### Sequential Estimation for Mean
```
mu_ML^{(N)} = mu_ML^{(N-1)} + (1/N) * (xN - mu_ML^{(N-1)})    (2.126)
```

### Robbins-Monro Algorithm
General sequential root-finding for regression function f(theta) = E[z|theta]:
```
theta^{(N)} = theta^{(N-1)} + a_{N-1} * z(theta^{(N-1)})    (2.129)
```
Conditions on {aN}:
```
lim_{N->inf} aN = 0    (2.130)
SUM_{N=1}^{inf} aN = inf    (2.131)
SUM_{N=1}^{inf} aN^2 < inf    (2.132)
```
Applied to MLE:
```
theta^{(N)} = theta^{(N-1)} + a_{N-1} * d/d(theta^{(N-1)}) ln p(xN | theta^{(N-1)})    (2.135)
```

---

## 2.3.6 Bayesian Inference for the Gaussian

### Known Variance, Unknown Mean
Likelihood:
```
p(X|mu) = (1/(2*pi*sigma^2)^{N/2}) * exp(-(1/(2*sigma^2)) * SUM_n (xn - mu)^2)    (2.137)
```

Gaussian prior on mu:
```
p(mu) = N(mu | mu_0, sigma_0^2)    (2.138)
```

**Posterior**:
```
p(mu|X) = N(mu | mu_N, sigma_N^2)    (2.140)
```
where:
```
mu_N = (sigma^2 / (N*sigma_0^2 + sigma^2)) * mu_0 + (N*sigma_0^2 / (N*sigma_0^2 + sigma^2)) * mu_ML    (2.141)

1/sigma_N^2 = 1/sigma_0^2 + N/sigma^2    (2.142)
```
- Posterior mean: weighted average of prior mean and MLE mean
- Posterior precision: sum of prior precision and N * data precision
- As N -> inf: posterior mean -> mu_ML, posterior variance -> 0
- As sigma_0^2 -> inf: posterior -> MLE solution

### Known Mean, Unknown Precision (Gamma Conjugate Prior)
Precision: lambda = 1/sigma^2

**Gamma distribution** (conjugate prior for Gaussian precision):
```
Gam(lambda|a, b) = (1/Gamma(a)) * b^a * lambda^{a-1} * exp(-b*lambda)    (2.146)
```
- E[lambda] = a/b    (2.147)
- var[lambda] = a/b^2    (2.148)

**Posterior**:
```
p(lambda|X) = Gam(lambda | a_N, b_N)
```
where:
```
a_N = a_0 + N/2    (2.150)
b_N = b_0 + (1/2) * SUM_{n=1}^{N} (xn - mu)^2 = b_0 + (N/2)*sigma^2_ML    (2.151)
```

### Unknown Mean and Precision (Normal-Gamma Prior)
```
p(mu, lambda) = N(mu | mu_0, (beta*lambda)^{-1}) * Gam(lambda | a, b)    (2.154)
```
Note: precision of mu is proportional to lambda (not independent).

### Multivariate: Wishart Distribution
For D-dimensional Gaussian with known mean, the conjugate prior for the precision matrix Lambda = Sigma^{-1} is the **Wishart distribution**:
```
W(Lambda | W, nu) = B(W, nu) * |Lambda|^{(nu-D-1)/2} * exp(-(1/2)*Tr(W^{-1}*Lambda))    (2.155)
```
- **nu**: degrees of freedom (nu > D-1)
- **W**: D x D scale matrix
- E[Lambda] = nu * W    (2.156)

### Normal-Wishart (unknown mu and Lambda)
```
p(mu, Lambda | mu_0, beta, W, nu) = N(mu | mu_0, (beta*Lambda)^{-1}) * W(Lambda | W, nu)    (2.157)
```

---

## 2.3.8 Student's t-Distribution

Obtained by marginalizing over the precision of a Gaussian:
```
INTEGRAL_0^{inf} N(x|mu, (eta*lambda)^{-1}) * Gam(lambda|nu/2, nu/2) dlambda
```

### Univariate Student-t
```
St(x|mu, lambda, nu) = Gamma(nu/2 + 1/2) / Gamma(nu/2) * (lambda/(pi*nu))^{1/2} * [1 + (lambda*(x-mu)^2)/nu]^{-(nu/2 + 1/2)}    (2.159)
```
- **mu**: location parameter
- **lambda**: precision parameter
- **nu**: degrees of freedom
- E[x] = mu (for nu > 1)    (2.160)
- var[x] = 1/(lambda * (1 - 2/nu)) (for nu > 2)    (2.161)
- As nu -> inf: Student-t -> Gaussian N(x|mu, lambda^{-1})
- More robust to outliers than Gaussian (heavier tails)

### Multivariate Student-t
```
St(x|mu, Lambda, nu) = Gamma(nu/2 + D/2) / Gamma(nu/2) * |Lambda|^{1/2} / ((nu*pi)^{D/2}) * [1 + (Delta^2)/nu]^{-(nu+D)/2}    (2.162)
```
where Delta^2 = (x-mu)^T * Lambda * (x-mu).

---

## 2.3.9 Mixture of Gaussians
```
p(x) = SUM_{k=1}^{K} pi_k * N(x | mu_k, Sigma_k)    (2.188)
```
- **pi_k**: mixing coefficients, where SUM_k pi_k = 1, pi_k >= 0    (2.189, 2.190)
- Can approximate any continuous density given sufficient components

**Responsibilities** (posterior probability of component k):
```
gamma_k(x) = p(k|x) = pi_k * N(x|mu_k, Sigma_k) / SUM_l pi_l * N(x|mu_l, Sigma_l)    (2.192)
```

**Log likelihood** (no closed form for MLE):
```
ln p(X|pi, mu, Sigma) = SUM_{n=1}^{N} ln { SUM_{k=1}^{K} pi_k * N(xn|mu_k, Sigma_k) }    (2.193)
```

---

## 2.4 The Exponential Family

### General Form
```
p(x|eta) = h(x) * g(eta) * exp(eta^T * u(x))    (2.194)
```
- **eta**: natural parameters
- **u(x)**: sufficient statistic function
- **h(x)**: scaling function
- **g(eta)**: normalization coefficient satisfying:
```
g(eta) * INTEGRAL h(x) * exp(eta^T * u(x)) dx = 1    (2.195)
```

### Key Property: Moments from Normalization
```
-grad ln g(eta) = E[u(x)]    (2.226)
```

### Examples
**Bernoulli** in exponential family form:
```
eta = ln(mu/(1-mu))    (2.198)    [logit function]
mu = sigma(eta) = 1/(1 + exp(-eta))    (2.199)    [logistic sigmoid]
u(x) = x, h(x) = 1, g(eta) = sigma(-eta)    (2.201-2.203)
```

**Multinomial** (M-1 independent parameters):
```
eta_k = ln(mu_k / (1 - SUM_j mu_j))    (2.212)
mu_k = exp(eta_k) / (1 + SUM_j exp(eta_j))    (2.213)    [softmax function]
```

**Univariate Gaussian**:
```
eta = (mu/sigma^2, -1/(2*sigma^2))^T    (2.220)
u(x) = (x, x^2)^T    (2.221)
h(x) = (2*pi)^{-1/2}    (2.222)
```

### Sufficient Statistics for MLE
From likelihood of N i.i.d. samples:
```
p(X|eta) = {PROD_n h(xn)} * g(eta)^N * exp(eta^T * SUM_n u(xn))    (2.227)
```
MLE condition:
```
-grad ln g(eta_ML) = (1/N) * SUM_{n=1}^{N} u(xn)    (2.228)
```
Data enters only through sufficient statistic SUM_n u(xn).

### Conjugate Priors for Exponential Family
```
p(eta | chi, nu) = f(chi, nu) * g(eta)^nu * exp(nu * eta^T * chi)    (2.229)
```
- **f(chi, nu)**: normalization constant
- **nu**: effective number of pseudo-observations
- **chi**: prior sufficient statistic per pseudo-observation

**Posterior** (after N observations):
```
p(eta|X, chi, nu) proportional_to g(eta)^{nu+N} * exp(eta^T * {SUM_n u(xn) + nu*chi})    (2.230)
```
Same form as prior -- confirming conjugacy.

### Noninformative Priors
- **Location parameter** (e.g., Gaussian mean): p(mu) = const    (2.235)
- **Scale parameter** (e.g., Gaussian sigma): p(sigma) proportional_to 1/sigma    (2.239)
- Equivalently for precision: p(lambda) proportional_to 1/lambda
- Gaussian mean: take conjugate Gaussian prior with sigma_0^2 -> inf
- Gaussian precision: take Gamma prior with a_0 = b_0 = 0

---

## 2.5 Nonparametric Methods

### Histogram Density Estimation
```
p_i = n_i / (N * Delta_i)    (2.241)
```
- **n_i**: count in bin i
- **N**: total observations
- **Delta_i**: bin width
- Limitation: M^D bins needed for D-dimensional space (curse of dimensionality)

### General Density Estimation Framework
For region R containing x with volume V, and K points from N total falling in R:
```
P = INTEGRAL_R p(x) dx ~ p(x) * V    (2.245)
K ~ N * P    (2.244)
```
Therefore:
```
p(x) = K / (N * V)    (2.246)
```
Two approaches: fix V (kernel methods) or fix K (nearest neighbours).

### Kernel Density Estimation (Parzen Window)
**Parzen window function** (hypercube kernel):
```
k(u) = 1  if |u_i| <= 1/2 for all i, else 0    (2.247)
```

**Kernel density estimate**:
```
p(x) = (1/N) * SUM_{n=1}^{N} (1/h^D) * k((x - xn)/h)    (2.249)
```
- **h**: bandwidth (smoothing parameter)
- **h^D**: volume of hypercube in D dimensions

**Gaussian kernel density estimator**:
```
p(x) = (1/N) * SUM_{n=1}^{N} (1/(2*pi*h^2)^{D/2}) * exp(-||x - xn||^2 / (2*h^2))    (2.250)
```

Kernel requirements:
```
k(u) >= 0    (2.251)
INTEGRAL k(u) du = 1    (2.252)
```

### K-Nearest-Neighbours (KNN)
Fix K, find volume V of smallest sphere around x containing K points:
```
p(x) = K / (N * V)    (using 2.246)
```

**KNN Classification**: For sphere of volume V around x containing K total points:
```
p(x|Ck) = Kk / (Nk * V)    (2.253)
p(x) = K / (N * V)    (2.254)
p(Ck) = Nk / N    (2.255)
```
By Bayes' theorem:
```
p(Ck|x) = Kk / K    (2.256)
```
Assign x to class with largest Kk.

Property: For K=1 nearest-neighbour classifier, as N -> inf, error rate <= 2 * (Bayes optimal error rate).

---

# Chapter 3: Linear Models for Regression

## 3.1 Linear Basis Function Models

### Model Definition
```
y(x, w) = w0 + SUM_{j=1}^{M-1} wj * phi_j(x) = w^T * phi(x)    (3.2, 3.3)
```
- **phi_j(x)**: basis functions (M total, with phi_0(x) = 1)
- **w** = (w0, w1, ..., w_{M-1})^T: weight vector
- **phi(x)** = (phi_0(x), ..., phi_{M-1}(x))^T: basis function vector
- Linear in parameters w, but can be nonlinear in x

### Common Basis Functions
**Polynomial**: phi_j(x) = x^j

**Gaussian**:
```
phi_j(x) = exp(-(x - mu_j)^2 / (2*s^2))    (3.4)
```
- mu_j: centre location, s: spatial scale

**Sigmoidal**:
```
phi_j(x) = sigma((x - mu_j)/s)    (3.5)
```
where sigma(a) = 1/(1 + exp(-a))    (3.6)

**Fourier basis**: sinusoidal functions (infinite spatial extent, localized frequency)

**Wavelets**: localized in both space and frequency

---

## 3.1.1 Maximum Likelihood and Least Squares

### Probabilistic Model
```
t = y(x, w) + epsilon    (3.7)
p(t|x, w, beta) = N(t | y(x,w), beta^{-1})    (3.8)
```
- **epsilon**: zero-mean Gaussian noise with precision beta
- Conditional mean: E[t|x] = y(x, w)    (3.9)

### Likelihood
```
p(t|X, w, beta) = PROD_{n=1}^{N} N(tn | w^T * phi(xn), beta^{-1})    (3.10)
```

### Log Likelihood
```
ln p(t|w, beta) = (N/2)*ln(beta) - (N/2)*ln(2*pi) - beta*ED(w)    (3.11)
```

### Sum-of-Squares Error Function
```
ED(w) = (1/2) * SUM_{n=1}^{N} {tn - w^T * phi(xn)}^2    (3.12)
```

### Gradient of Log Likelihood
```
grad ln p(t|w, beta) = SUM_{n=1}^{N} {tn - w^T * phi(xn)} * phi(xn)^T    (3.13)
```

### Normal Equations (MLE Solution)
Setting gradient to zero:
```
w_ML = (Phi^T * Phi)^{-1} * Phi^T * t    (3.15)
```
- **Normal equations**: Phi^T * Phi * w = Phi^T * t

### Design Matrix
```
Phi = [[phi_0(x1), phi_1(x1), ..., phi_{M-1}(x1)],
       [phi_0(x2), phi_1(x2), ..., phi_{M-1}(x2)],
       ...
       [phi_0(xN), phi_1(xN), ..., phi_{M-1}(xN)]]    (3.16)
```
Phi is N x M matrix with elements Phi_{nj} = phi_j(xn).

### Moore-Penrose Pseudo-Inverse
```
Phi^dagger = (Phi^T * Phi)^{-1} * Phi^T    (3.17)
```

### Bias Parameter
```
w0 = t_bar - SUM_{j=1}^{M-1} wj * phi_j_bar    (3.19)
```
where t_bar = (1/N)*SUM tn and phi_j_bar = (1/N)*SUM phi_j(xn)    (3.20)

### MLE for Noise Precision
```
1/beta_ML = (1/N) * SUM_{n=1}^{N} {tn - w_ML^T * phi(xn)}^2    (3.21)
```

### Stochastic Gradient Descent (Sequential Learning)
For error E = SUM_n En:
```
w^{(tau+1)} = w^{(tau)} - eta * grad(En)    (3.22)
```
For sum-of-squares error (LMS algorithm):
```
w^{(tau+1)} = w^{(tau)} + eta * (tn - w^{(tau)T} * phi_n) * phi_n    (3.23)
```
- **eta**: learning rate
- **phi_n** = phi(xn)

---

## 3.1.4 Regularized Least Squares

### General Regularized Error
```
E(w) = ED(w) + lambda * EW(w)    (3.24)
```

### L2 Regularization (Ridge / Weight Decay)
```
EW(w) = (1/2) * w^T * w    (3.25)
```

Total error:
```
(1/2) * SUM_{n=1}^{N} {tn - w^T*phi(xn)}^2 + (lambda/2) * w^T * w    (3.27)
```

**Closed-form solution**:
```
w = (lambda*I + Phi^T*Phi)^{-1} * Phi^T * t    (3.28)
```

### General Lq Regularization
```
(1/2) * SUM_n {tn - w^T*phi(xn)}^2 + (lambda/2) * SUM_j |wj|^q    (3.29)
```
- **q = 2**: Ridge regression (quadratic, closed-form solution)
- **q = 1**: Lasso (promotes sparsity, some wj driven exactly to zero)

Equivalent constrained form:
```
minimize ED(w) subject to SUM_j |wj|^q <= eta    (3.30)
```

### Multiple Outputs
For K targets with shared basis functions:
```
y(x, W) = W^T * phi(x)    (3.31)
```
- W: M x K parameter matrix
- Conditional: p(t|x, W, beta) = N(t | W^T*phi(x), beta^{-1}*I)    (3.32)

MLE decouples across targets:
```
W_ML = (Phi^T * Phi)^{-1} * Phi^T * T    (3.34)
wk = Phi^dagger * tk    (3.35)
```

---

## 3.2 Bias-Variance Decomposition

### Setup
Optimal prediction h(x) = E[t|x]:
```
h(x) = INTEGRAL t * p(t|x) dt    (3.36)
```

### Expected Loss Decomposition
```
E[L] = INTEGRAL {y(x) - h(x)}^2 * p(x) dx + INTEGRAL {h(x) - t}^2 * p(x,t) dx dt    (3.37)
```
- First term: model-dependent (reducible)
- Second term: intrinsic noise (irreducible)

### Bias-Variance Decomposition
For prediction y(x; D) learned from data set D:
```
E_D[{y(x;D) - h(x)}^2] = {E_D[y(x;D)] - h(x)}^2 + E_D[{y(x;D) - E_D[y(x;D)]}^2]    (3.40)
                          =       (bias)^2         +              variance
```

**Overall decomposition**:
```
expected loss = (bias)^2 + variance + noise    (3.41)
```
where:
```
(bias)^2 = INTEGRAL {E_D[y(x;D)] - h(x)}^2 * p(x) dx    (3.42)
variance = INTEGRAL E_D[{y(x;D) - E_D[y(x;D)]}^2] * p(x) dx    (3.43)
noise    = INTEGRAL {h(x) - t}^2 * p(x,t) dx dt    (3.44)
```

**Trade-off**:
- Flexible models: low bias, high variance
- Rigid models: high bias, low variance
- Optimal: best balance between bias and variance

---

## 3.3 Bayesian Linear Regression

### Prior
```
p(w) = N(w | m0, S0)    (3.48)
```

### Posterior
```
p(w|t) = N(w | mN, SN)    (3.49)
```
where:
```
mN = SN * (S0^{-1} * m0 + beta * Phi^T * t)    (3.50)
SN^{-1} = S0^{-1} + beta * Phi^T * Phi    (3.51)
```
- wMAP = mN (mode = mean for Gaussian posterior)

### Isotropic Gaussian Prior (simplified)
```
p(w|alpha) = N(w | 0, alpha^{-1} * I)    (3.52)
```

**Posterior parameters**:
```
mN = beta * SN * Phi^T * t    (3.53)
SN^{-1} = alpha*I + beta * Phi^T * Phi    (3.54)
```

**Log posterior**:
```
ln p(w|t) = -(beta/2) * SUM_n {tn - w^T*phi(xn)}^2 - (alpha/2) * w^T*w + const    (3.55)
```
Maximizing posterior = minimizing regularized error with lambda = alpha/beta.

### Predictive Distribution
```
p(t|x, t, alpha, beta) = INTEGRAL p(t|w, beta) * p(w|t, alpha, beta) dw    (3.57)
```
Result (convolution of two Gaussians):
```
p(t|x, t, alpha, beta) = N(t | mN^T * phi(x), sigma_N^2(x))    (3.58)
```
where:
```
sigma_N^2(x) = 1/beta + phi(x)^T * SN * phi(x)    (3.59)
```
- **1/beta**: noise on data
- **phi(x)^T * SN * phi(x)**: uncertainty from parameter w
- As N -> inf: second term -> 0, variance -> 1/beta (noise only)
- Property: sigma_{N+1}^2(x) <= sigma_N^2(x) (variance decreases with more data)

### Equivalent Kernel
The predictive mean can be written:
```
y(x, mN) = SUM_{n=1}^{N} k(x, xn) * tn    (3.61)
```
where:
```
k(x, x') = beta * phi(x)^T * SN * phi(x')    (3.62)
```
- **k(x, x')**: equivalent kernel (smoother matrix)
- Localized around x even for non-local basis functions
- Predictive covariance: cov[y(x), y(x')] = beta^{-1} * k(x, x')    (3.63)

---

## 3.4 Bayesian Model Comparison

### Model Evidence (Marginal Likelihood)
For model Mi with parameters w:
```
p(D|Mi) = INTEGRAL p(D|w, Mi) * p(w|Mi) dw    (3.68)
```

### Bayes Factor
```
p(D|M1) / p(D|M2)    (3.67)
```

### Approximation of Log Evidence
For single parameter:
```
ln p(D) ~ ln p(D|wMAP) + ln(Delta_w_posterior / Delta_w_prior)    (3.71)
```
For M parameters:
```
ln p(D) ~ ln p(D|wMAP) + M * ln(Delta_w_posterior / Delta_w_prior)    (3.72)
```
- First term: data fit
- Second term: complexity penalty (increases with M)
- Optimal model: best trade-off between fit and complexity

---

## 3.5 The Evidence Approximation (Empirical Bayes / Type-II MLE)

### Objective
Maximize marginal likelihood p(t|alpha, beta) w.r.t. hyperparameters alpha, beta:
```
p(t|alpha, beta) = INTEGRAL p(t|w, beta) * p(w|alpha) dw    (3.77)
```

### Evaluation of Evidence Function
```
p(t|alpha, beta) = (beta/(2*pi))^{N/2} * (alpha/(2*pi))^{M/2} * INTEGRAL exp{-E(w)} dw    (3.78)
```
where:
```
E(w) = beta*ED(w) + alpha*EW(w) = (beta/2)*||t - Phi*w||^2 + (alpha/2)*w^T*w    (3.79)
```

Completing the square:
```
E(w) = E(mN) + (1/2)*(w - mN)^T * A * (w - mN)    (3.80)
```
where:
```
A = alpha*I + beta * Phi^T * Phi    (3.81)    [= SN^{-1}, the Hessian]
E(mN) = (beta/2)*||t - Phi*mN||^2 + (alpha/2)*mN^T*mN    (3.82)
mN = beta * A^{-1} * Phi^T * t    (3.84)
```

### Log Marginal Likelihood
```
ln p(t|alpha, beta) = (M/2)*ln(alpha) + (N/2)*ln(beta) - E(mN) - (1/2)*ln|A| - (N/2)*ln(2*pi)    (3.86)
```

### Re-estimation Equations for alpha

Define eigenvalues of beta*Phi^T*Phi:
```
(beta * Phi^T * Phi) * ui = lambda_i * ui    (3.87)
```
Then A has eigenvalues (alpha + lambda_i).

**Effective number of parameters**:
```
gamma = SUM_i lambda_i / (alpha + lambda_i)    (3.91)
```
- 0 <= gamma <= M
- lambda_i >> alpha: parameter well-determined by data (contributes ~1 to gamma)
- lambda_i << alpha: parameter set by prior (contributes ~0 to gamma)

**Re-estimation for alpha**:
```
alpha = gamma / (mN^T * mN)    (3.92)
```

### Re-estimation Equations for beta
```
1/beta = (1/(N - gamma)) * SUM_{n=1}^{N} {tn - mN^T * phi(xn)}^2    (3.95)
```
- Compare to MLE: 1/beta_ML = (1/N) * SUM {tn - wML^T*phi(xn)}^2
- Factor (N - gamma) corrects for bias (gamma effective parameters absorbed by fitting)
- Analogous to Bessel's correction (N-1) for variance estimation

### Iterative Algorithm
1. Initialize alpha, beta
2. Compute mN using (3.53): mN = beta * SN * Phi^T * t
3. Compute SN using (3.54): SN^{-1} = alpha*I + beta*Phi^T*Phi
4. Compute eigenvalues lambda_i of beta*Phi^T*Phi
5. Compute gamma using (3.91)
6. Update alpha using (3.92)
7. Update beta using (3.95)
8. Repeat 2-7 until convergence

### Large Data Limit (N >> M)
When N >> M, all parameters are well-determined (gamma -> M):
```
alpha = M / (2 * EW(mN))    (3.98)
beta = N / (2 * ED(mN))    (3.99)
```

---

## Key Cross-Chapter Relationships

| Concept | Ch 1 | Ch 2 | Ch 3 |
|---------|------|------|------|
| MLE mean | (1.55) | (2.121) | (3.15) |
| MLE variance bias | (1.58) | (2.124) | (3.95) N-gamma correction |
| Gaussian noise model | (1.60) | (2.42) | (3.8) |
| Regularization | (1.4) lambda\|\|w\|\|^2 | -- | (3.27) equivalent to alpha/beta |
| Bayesian prediction | (1.68-1.69) | (2.113-2.117) | (3.57-3.59) |
| Conjugate priors | -- | Beta-Binomial, Dirichlet-Multinomial, Gaussian-Gaussian, Gamma-Gaussian | (3.52) Gaussian prior on w |
| Model complexity | AIC (1.73) | -- | Evidence (3.86), gamma (3.91) |
| Information theory | H, KL, MI (1.93-1.121) | Exponential family (2.194) | Expected Bayes factor (3.73) |

---

## Summary of All Distributions

| Distribution | Form | Parameters | Conjugate Prior |
|-------------|------|------------|-----------------|
| Bernoulli | mu^x * (1-mu)^{1-x} | mu in [0,1] | Beta(a,b) |
| Binomial | C(N,m) * mu^m * (1-mu)^{N-m} | mu, N | Beta(a,b) |
| Multinomial | (N!/PROD mk!) * PROD mu_k^{mk} | mu (K-simplex), N | Dirichlet(alpha) |
| Univariate Gaussian | (2*pi*sigma^2)^{-1/2} exp(-(x-mu)^2/(2*sigma^2)) | mu, sigma^2 | Gaussian (for mu), Gamma (for precision) |
| Multivariate Gaussian | (2*pi)^{-D/2}\|Sigma\|^{-1/2} exp(-1/2 (x-mu)^T Sigma^{-1} (x-mu)) | mu, Sigma | Gaussian (for mu), Wishart (for precision) |
| Beta | Gamma(a+b)/(Gamma(a)Gamma(b)) * mu^{a-1}(1-mu)^{b-1} | a, b > 0 | -- |
| Dirichlet | Gamma(alpha_0)/PROD Gamma(alpha_k) * PROD mu_k^{alpha_k-1} | alpha_k > 0 | -- |
| Gamma | b^a/Gamma(a) * lambda^{a-1} exp(-b*lambda) | a > 0, b > 0 | -- |
| Student-t | Complex (see 2.159) | mu, lambda, nu | -- |
| Wishart | Complex (see 2.155) | W (scale matrix), nu | -- |
| Normal-Gamma | N(mu\|mu_0,(beta*lambda)^{-1}) * Gam(lambda\|a,b) | mu_0, beta, a, b | -- |
| Exponential family | h(x)*g(eta)*exp(eta^T*u(x)) | eta (natural params) | g(eta)^nu * exp(nu*eta^T*chi) |

---

## Chapter 4: Linear Models for Classification

### 4.1 Discriminant Functions

#### 4.1.1 Two-Class Discriminant

**Linear discriminant function:**
```
y(x) = w^T x + w0                                                    (4.4)
```
- `w`: weight vector (D-dimensional)
- `w0`: bias (threshold = -w0)
- Assign x to C1 if y(x) >= 0, else C2
- Decision boundary: y(x) = 0 is a (D-1)-dimensional hyperplane

**Geometric properties:**
- `w` is orthogonal to the decision surface
- Normal distance from origin to decision surface: `w^T x / ||w|| = -w0 / ||w||`    (4.5)
- Signed perpendicular distance of point x from surface: `r = y(x) / ||w||`          (4.7)
- Compact notation with augmented vectors: `y(x) = w_tilde^T x_tilde`                (4.8)
  where `x_tilde = (1, x^T)^T`, `w_tilde = (w0, w^T)^T`

#### 4.1.2 Multiple Classes (K > 2)

**K-class discriminant (single K-class approach):**
```
y_k(x) = w_k^T x + w_{k0}                                           (4.9)
```
- Assign x to C_k if y_k(x) > y_j(x) for all j != k

**Decision boundary between classes C_k and C_j:**
```
(w_k - w_j)^T x + (w_{k0} - w_{j0}) = 0                            (4.10)
```

**Property:** Decision regions are always singly connected and convex (proof via convex combination argument using linearity).

#### 4.1.3 Least Squares for Classification

**Matrix formulation:** Each class k has `y_k(x) = w_k^T x_tilde + w_{k0}`, grouped as:
```
y(x) = W_tilde^T x_tilde                                             (4.14)
```
- `W_tilde`: matrix whose k-th column is `w_tilde_k = (w_{k0}, w_k^T)^T`

**Sum-of-squares error:**
```
E_D(W_tilde) = (1/2) Tr{(X W_tilde - T)^T (X W_tilde - T)}         (4.15)
```

**Closed-form solution:**
```
W_tilde = (X^T X)^{-1} X^T T = X^dagger T                           (4.16)
```
where `X^dagger` is the pseudo-inverse.

**Limitation:** Lacks robustness to outliers; penalizes predictions that are "too correct."

#### 4.1.4 Fisher's Linear Discriminant

**Projection:** y = w^T x (D-dimensional to 1-dimensional)                           (4.20)

**Class means:**
```
m_1 = (1/N_1) sum_{n in C_1} x_n,    m_2 = (1/N_2) sum_{n in C_2} x_n   (4.21)
```

**Projected means:** `m_k = w^T m_k`                                                  (4.23)

**Within-class variance (projected):**
```
s_k^2 = sum_{n in C_k} (y_n - m_k)^2                                (4.24)
```

**Fisher criterion:**
```
J(w) = (m_2 - m_1)^2 / (s_1^2 + s_2^2)                             (4.25)
```

**Equivalent matrix form:**
```
J(w) = (w^T S_B w) / (w^T S_W w)                                    (4.26)
```
where:
- Between-class covariance: `S_B = (m_2 - m_1)(m_2 - m_1)^T`                        (4.27)
- Within-class covariance: `S_W = sum_{n in C_1} (x_n - m_1)(x_n - m_1)^T + sum_{n in C_2} (x_n - m_2)(x_n - m_2)^T`   (4.28)

**Optimal projection (Fisher's solution):**
```
w propto S_W^{-1} (m_2 - m_1)                                       (4.30)
```

#### 4.1.5 Relation to Least Squares

With target coding: t = N/N_1 for C_1, t = -N/N_2 for C_2:
- Bias: `w_0 = -w^T m`    (m = total data mean)                                      (4.34)
- Weight equation: `(S_W + (N_1 N_2 / N) S_B) w = N(m_1 - m_2)`                     (4.37)
- Solution: `w propto S_W^{-1}(m_2 - m_1)` -- same as Fisher                         (4.38)

#### 4.1.6 Fisher's Discriminant for Multiple Classes

**Projection to D'-dimensional space:**
```
y = W^T x                                                           (4.39)
```

**Generalized within-class covariance:**
```
S_W = sum_{k=1}^{K} S_k,    S_k = sum_{n in C_k} (x_n - m_k)(x_n - m_k)^T   (4.40-4.41)
```

**Between-class covariance:**
```
S_B = sum_{k=1}^{K} N_k (m_k - m)(m_k - m)^T                       (4.46)
```

**Multi-class criterion:**
```
J(W) = Tr{s_W^{-1} s_B} = Tr{(W^T S_W W)^{-1} (W^T S_B W)}       (4.50-4.51)
```

**Solution:** Eigenvectors of `S_W^{-1} S_B` corresponding to D' largest eigenvalues. Maximum D' = K-1 (rank of S_B).

#### 4.1.7 Perceptron Algorithm

**Model:**
```
y(x) = f(w^T phi(x))                                                (4.52)
```
where activation function:
```
f(a) = +1 if a >= 0, -1 if a < 0                                   (4.53)
```
Target coding: t = +1 for C_1, t = -1 for C_2.

**Perceptron criterion (error function):**
```
E_P(w) = -sum_{n in M} w^T phi_n t_n                                (4.54)
```
where M = set of misclassified patterns.

**Perceptron update rule (SGD):**
```
w^{(tau+1)} = w^{(tau)} + eta phi_n t_n                             (4.55)
```
- eta: learning rate (can set eta=1 WLOG)

**Perceptron convergence theorem:** If training data is linearly separable, the algorithm converges in a finite number of steps. Does NOT converge for non-separable data.

---

### 4.2 Probabilistic Generative Models

#### Posterior via Bayes' Theorem

**Two-class posterior:**
```
p(C_1|x) = sigma(a)                                                 (4.57)
```
where:
```
a = ln [p(x|C_1)p(C_1) / (p(x|C_2)p(C_2))]                       (4.58)
```

**Logistic sigmoid function:**
```
sigma(a) = 1 / (1 + exp(-a))                                        (4.59)
```
Properties:
- `sigma(-a) = 1 - sigma(a)`                                                         (4.60)
- Inverse (logit): `a = ln(sigma / (1-sigma))`                                       (4.61)
- Derivative: `d sigma/da = sigma(1 - sigma)`                                        (4.88)

**Multi-class posterior (softmax / normalized exponential):**
```
p(C_k|x) = exp(a_k) / sum_j exp(a_j)                               (4.62)
```
where `a_k = ln p(x|C_k)p(C_k)`                                                     (4.63)

#### 4.2.1 Gaussian Class-Conditional Densities (Shared Covariance)

Class-conditional:
```
p(x|C_k) = N(x|mu_k, Sigma)                                        (4.64)
```

**Two-class result:**
```
p(C_1|x) = sigma(w^T x + w_0)                                      (4.65)
w = Sigma^{-1}(mu_1 - mu_2)                                        (4.66)
w_0 = -(1/2)mu_1^T Sigma^{-1} mu_1 + (1/2)mu_2^T Sigma^{-1} mu_2 + ln(p(C_1)/p(C_2))   (4.67)
```
Quadratic terms cancel (shared Sigma) => linear decision boundary.

**K-class result:**
```
a_k(x) = w_k^T x + w_{k0}                                          (4.68)
w_k = Sigma^{-1} mu_k                                               (4.69)
w_{k0} = -(1/2)mu_k^T Sigma^{-1} mu_k + ln p(C_k)                 (4.70)
```

**Non-shared covariance:** Quadratic terms remain => quadratic discriminant.

#### 4.2.2 Maximum Likelihood Solution

**Likelihood:**
```
p(t|pi, mu_1, mu_2, Sigma) = prod_{n=1}^{N} [pi N(x_n|mu_1, Sigma)]^{t_n} [(1-pi) N(x_n|mu_2, Sigma)]^{1-t_n}   (4.71)
```

**ML estimates:**
```
pi = N_1/N                                                           (4.73)
mu_1 = (1/N_1) sum_{n=1}^{N} t_n x_n                               (4.75)
mu_2 = (1/N_2) sum_{n=1}^{N} (1-t_n) x_n                           (4.76)
Sigma = S = (N_1/N)S_1 + (N_2/N)S_2                                (4.78)
S_k = (1/N_k) sum_{n in C_k} (x_n - mu_k)(x_n - mu_k)^T           (4.79-4.80)
```

#### 4.2.3 Discrete Features (Naive Bayes)

For binary features x_i in {0,1}:
```
p(x|C_k) = prod_{i=1}^{D} mu_{ki}^{x_i} (1-mu_{ki})^{1-x_i}     (4.81)
```
Gives `a_k(x)` linear in x_i.                                                       (4.82)

#### 4.2.4 Exponential Family

For class-conditionals from exponential family with shared scale s:
```
p(x|lambda_k, s) = (1/s) h(x/s) g(lambda_k) exp((1/s) lambda_k^T x)   (4.84)
```
- Two-class: posterior is logistic sigmoid of linear function of x                    (4.85)
- K-class: posterior is softmax of linear function of x                               (4.86)

---

### 4.3 Probabilistic Discriminative Models

#### 4.3.2 Logistic Regression

**Model:**
```
p(C_1|phi) = y(phi) = sigma(w^T phi)                                (4.87)
```

**Likelihood:**
```
p(t|w) = prod_{n=1}^{N} y_n^{t_n} (1-y_n)^{1-t_n}                (4.89)
```

**Cross-entropy error function:**
```
E(w) = -sum_{n=1}^{N} {t_n ln y_n + (1-t_n) ln(1-y_n)}            (4.90)
```

**Gradient:**
```
nabla E(w) = sum_{n=1}^{N} (y_n - t_n) phi_n = Phi^T (y - t)      (4.91/4.96)
```

**Over-fitting for linearly separable data:** ||w|| -> infinity, sigma becomes step function.

#### 4.3.3 Iterative Reweighted Least Squares (IRLS)

**Newton-Raphson update:**
```
w^{new} = w^{old} - H^{-1} nabla E(w)                              (4.92)
```

**Hessian for logistic regression:**
```
H = nabla nabla E(w) = Phi^T R Phi                                  (4.97)
```
where R is N x N diagonal: `R_{nn} = y_n(1-y_n)`                                    (4.98)

**IRLS update formula:**
```
w^{new} = (Phi^T R Phi)^{-1} Phi^T R z                             (4.99)
```
where:
```
z = Phi w^{old} - R^{-1}(y - t)                                    (4.100)
```

**Properties:**
- H is positive definite => error function is concave => unique minimum
- Mean: `E[t] = y`, Variance: `var[t] = y(1-y)`                                     (4.101-4.102)

#### 4.3.4 Multiclass Logistic Regression

**Model:**
```
p(C_k|phi) = y_k(phi) = exp(a_k) / sum_j exp(a_j)                 (4.104)
a_k = w_k^T phi                                                     (4.105)
```

**Softmax derivative:**
```
partial y_k / partial a_j = y_k(I_{kj} - y_j)                     (4.106)
```

**Multiclass cross-entropy error:**
```
E(w_1,...,w_K) = -sum_{n=1}^{N} sum_{k=1}^{K} t_{nk} ln y_{nk}   (4.108)
```

**Gradient:**
```
nabla_{w_j} E = sum_{n=1}^{N} (y_{nj} - t_{nj}) phi_n             (4.109)
```

**Hessian block (j,k):**
```
nabla_{w_k} nabla_{w_j} E = -sum_{n=1}^{N} y_{nk}(I_{kj} - y_{nj}) phi_n phi_n^T   (4.110)
```

#### 4.3.5 Probit Regression

**Probit activation function (cumulative Gaussian):**
```
Phi(a) = integral_{-inf}^{a} N(theta|0,1) d theta                  (4.114)
```

**Relation to erf function:**
```
Phi(a) = (1/2)(1 + (1/sqrt(2)) erf(a))                             (4.116)
```

**Mislabelling model:**
```
p(t|x) = epsilon + (1-2*epsilon) sigma(x)                           (4.117)
```

#### 4.3.6 Canonical Link Functions

For target from exponential family `p(t|eta, s)`:
```
y = E[t|eta] = -s (d/d eta) ln g(eta)                              (4.119)
```

Canonical link: `f^{-1}(y) = psi(y)` where `eta = psi(y)`.

With canonical link, gradient simplifies to:
```
nabla E(w) = (1/s) sum_{n=1}^{N} (y_n - t_n) phi_n                (4.124)
```
- Gaussian: s = beta^{-1}
- Logistic: s = 1

---

### 4.4 The Laplace Approximation

**Goal:** Approximate `p(z) = f(z)/Z` with a Gaussian centered on a mode z_0.

**1D case:**
```
ln f(z) ~ ln f(z_0) - (A/2)(z - z_0)^2                             (4.127)
A = -d^2/dz^2 ln f(z)|_{z=z_0}                                     (4.128)
q(z) = (A/2pi)^{1/2} exp(-(A/2)(z-z_0)^2)                         (4.130)
```

**M-dimensional case:**
```
ln f(z) ~ ln f(z_0) - (1/2)(z-z_0)^T A (z-z_0)                    (4.131)
A = -nabla nabla ln f(z)|_{z=z_0}                                   (4.132)
q(z) = N(z|z_0, A^{-1})                                            (4.134)
```

**Normalization constant approximation:**
```
Z ~ f(z_0) (2pi)^{M/2} / |A|^{1/2}                                (4.135)
```

#### 4.4.1 Model Comparison and BIC

**Laplace approximation to model evidence:**
```
ln p(D) ~ ln p(D|theta_MAP) + ln p(theta_MAP) + (M/2)ln(2pi) - (1/2)ln|A|   (4.137)
```
where:
```
A = -nabla nabla ln p(theta_MAP|D)                                  (4.138)
```

**Bayesian Information Criterion (BIC) / Schwarz criterion:**
```
ln p(D) ~ ln p(D|theta_MAP) - (M/2) ln N                           (4.139)
```
- M: number of parameters, N: number of data points
- Penalizes complexity more heavily than AIC

---

### 4.5 Bayesian Logistic Regression

#### 4.5.1 Laplace Approximation

**Gaussian prior:**
```
p(w) = N(w|m_0, S_0)                                                (4.140)
```

**Log posterior:**
```
ln p(w|t) = -(1/2)(w-m_0)^T S_0^{-1}(w-m_0) + sum_n {t_n ln y_n + (1-t_n)ln(1-y_n)} + const   (4.142)
```

**Hessian of negative log posterior:**
```
S_N = -nabla nabla ln p(w|t) = S_0^{-1} + sum_n y_n(1-y_n) phi_n phi_n^T   (4.143)
```

**Gaussian approximation to posterior:**
```
q(w) = N(w|w_MAP, S_N)                                              (4.144)
```

#### 4.5.2 Predictive Distribution

```
p(C_1|phi,t) = integral sigma(w^T phi) q(w) dw ~ integral sigma(a) N(a|mu_a, sigma_a^2) da   (4.151)
```
where:
```
mu_a = w_MAP^T phi                                                   (4.149)
sigma_a^2 = phi^T S_N phi                                           (4.150)
```

**Probit approximation for sigmoid-Gaussian convolution:**
```
integral sigma(a) N(a|mu, sigma^2) da ~ sigma(kappa(sigma^2) mu)   (4.153)
kappa(sigma^2) = (1 + pi sigma^2 / 8)^{-1/2}                      (4.154)
```

**Approximate predictive distribution:**
```
p(C_1|phi,t) = sigma(kappa(sigma_a^2) mu_a)                        (4.155)
```

---

## Chapter 5: Neural Networks

### 5.1 Feed-Forward Network Functions

#### Two-Layer Network Architecture

**First layer (hidden units):**
```
a_j = sum_{i=1}^{D} w_{ji}^{(1)} x_i + w_{j0}^{(1)}              (5.2)
z_j = h(a_j)                                                        (5.3)
```
- `w_{ji}^{(1)}`: first-layer weights, `w_{j0}^{(1)}`: first-layer biases
- `h(.)`: activation function (sigmoid, tanh)
- `z_j`: hidden unit outputs

**Second layer (output units):**
```
a_k = sum_{j=1}^{M} w_{kj}^{(2)} z_j + w_{k0}^{(2)}              (5.4)
```

**Full two-layer network function:**
```
y_k(x,w) = sigma(sum_{j=1}^{M} w_{kj}^{(2)} h(sum_{i=1}^{D} w_{ji}^{(1)} x_i + w_{j0}^{(1)}) + w_{k0}^{(2)})   (5.7)
```

**Compact form (bias absorbed):**
```
y_k(x,w) = sigma(sum_{j=0}^{M} w_{kj}^{(2)} h(sum_{i=0}^{D} w_{ji}^{(1)} x_i))   (5.9)
```
with x_0 = 1 (bias input).

**General feed-forward unit:**
```
z_k = h(sum_j w_{kj} z_j)                                          (5.10)
```

**Universal approximation:** Two-layer networks with linear outputs can approximate any continuous function on a compact domain to arbitrary accuracy, given sufficient hidden units.

#### 5.1.1 Weight-Space Symmetries

For a two-layer network with M tanh hidden units:
- **Sign-flip symmetry:** 2^M equivalent weight vectors (from tanh(-a) = -tanh(a))
- **Interchange symmetry:** M! equivalent weight vectors (from reordering hidden units)
- **Total symmetry factor:** M! 2^M

---

### 5.2 Network Training

#### Error Functions

**Regression (Gaussian noise model):**
```
p(t|x,w) = N(t|y(x,w), beta^{-1})                                 (5.12)
```

**Sum-of-squares error:**
```
E(w) = (1/2) sum_{n=1}^{N} {y(x_n,w) - t_n}^2                    (5.14)
```

**ML noise precision:**
```
1/beta_ML = (1/N) sum_n {y(x_n, w_ML) - t_n}^2                    (5.15)
```

**Multiple targets with shared precision:**
```
1/beta_ML = (1/(NK)) sum_n ||y(x_n, w_ML) - t_n||^2               (5.17)
```

**Binary classification (cross-entropy):**
```
E(w) = -sum_n {t_n ln y_n + (1-t_n) ln(1-y_n)}                    (5.21)
```

**Multiclass classification (softmax cross-entropy):**
```
E(w) = -sum_n sum_k t_{kn} ln y_k(x_n, w)                         (5.24)
y_k = exp(a_k) / sum_j exp(a_j)                                    (5.25)
```

**Key property (canonical link):** For all three cases:
```
partial E / partial a_k = y_k - t_k                                (5.18)
```

#### Optimization

**Gradient descent:**
```
w^{(tau+1)} = w^{(tau)} - eta nabla E(w^{(tau)})                   (5.41)
```

**Stochastic gradient descent (SGD):**
```
w^{(tau+1)} = w^{(tau)} - eta nabla E_n(w^{(tau)})                 (5.43)
```

**Local quadratic approximation:**
```
E(w) ~ E(w*) + (1/2)(w-w*)^T H (w-w*)                              (5.32)
```
- Eigenvalue decomposition: `H u_i = lambda_i u_i`                                   (5.33)
- Contours: ellipses with axes along eigenvectors, lengths proportional to `lambda_i^{-1/2}`

---

### 5.3 Error Backpropagation

#### 5.3.1 Evaluation of Error-Function Derivatives

**Forward propagation:**
```
a_j = sum_i w_{ji} z_i                                              (5.48)
z_j = h(a_j)                                                        (5.49)
```

**Error signal (delta):**
```
delta_j = partial E_n / partial a_j                                 (5.51)
```

**Key derivative formula:**
```
partial E_n / partial w_{ji} = delta_j z_i                          (5.53)
```

**Output unit deltas (canonical link):**
```
delta_k = y_k - t_k                                                 (5.54)
```

**Backpropagation formula (hidden units):**
```
delta_j = h'(a_j) sum_k w_{kj} delta_k                             (5.56)
```

#### Backpropagation Algorithm Summary

1. Forward propagate input x_n using (5.48) and (5.49)
2. Evaluate delta_k for output units using (5.54)
3. Backpropagate deltas using (5.56) for hidden units
4. Compute derivatives using (5.53)

**Computational complexity:** O(W) per pattern (W = total weights), same as forward pass.

#### 5.3.2 Simple Example (tanh hidden, linear output, SSE)

```
a_j = sum_{i=0}^{D} w_{ji}^{(1)} x_i                              (5.62)
z_j = tanh(a_j)                                                     (5.63)
y_k = sum_{j=0}^{M} w_{kj}^{(2)} z_j                              (5.64)
delta_k = y_k - t_k                                                 (5.65)
delta_j = (1 - z_j^2) sum_k w_{kj} delta_k                         (5.66)
```
where `tanh'(a) = 1 - tanh^2(a)`                                                    (5.60)

**First and second layer derivatives:**
```
partial E_n / partial w_{ji}^{(1)} = delta_j x_i
partial E_n / partial w_{kj}^{(2)} = delta_k z_j                   (5.67)
```

#### 5.3.4 Jacobian Matrix

```
J_{ki} = partial y_k / partial x_i                                 (5.70)
```

**Backpropagation formula for Jacobian:**
```
J_{ki} = sum_j w_{ji} (partial y_k / partial a_j)                  (5.73)
partial y_k / partial a_j = h'(a_j) sum_l w_{lj} (partial y_k / partial a_l)   (5.74)
```

**Error propagation:**
```
Delta y_k ~ sum_i (partial y_k / partial x_i) Delta x_i            (5.72)
```

---

### 5.4 The Hessian Matrix

Elements: `H_{ij} = partial^2 E / partial w_i partial w_j`

**Applications:** (1) Second-order optimization, (2) Fast re-training, (3) Network pruning, (4) Bayesian neural networks (Laplace approx).

#### 5.4.1 Diagonal Approximation

```
partial^2 E_n / partial w_{ji}^2 = (partial^2 E_n / partial a_j^2) z_i^2   (5.79)
```

Recursive:
```
partial^2 E_n / partial a_j^2 = h'(a_j)^2 sum_k w_{kj}^2 (partial^2 E_n / partial a_k^2)
                                + h''(a_j) sum_k w_{kj} (partial E_n / partial a_k)   (5.81)
```
Complexity: O(W) but strongly nondiagonal Hessians are poorly approximated.

#### 5.4.2 Outer Product Approximation (Levenberg-Marquardt)

For SSE: `E = (1/2) sum_n (y_n - t_n)^2`:
```
H = sum_n nabla y_n nabla y_n^T + sum_n (y_n - t_n) nabla nabla y_n   (5.83)
```

**Outer product approximation (neglect second term):**
```
H ~ sum_n b_n b_n^T                                                 (5.84)
```
where `b_n = nabla y_n = nabla a_n`.

**Cross-entropy version:**
```
H ~ sum_n y_n(1-y_n) b_n b_n^T                                     (5.85)
```

#### 5.4.3 Sequential Inverse Hessian

Using Woodbury identity:
```
H_{L+1}^{-1} = H_L^{-1} - (H_L^{-1} b_{L+1} b_{L+1}^T H_L^{-1}) / (1 + b_{L+1}^T H_L^{-1} b_{L+1})   (5.89)
```
Initialize H_0 = alpha I.

#### 5.4.5 Exact Hessian Evaluation

For two-layer network, define:
```
delta_k = partial E_n / partial a_k,    M_{kk'} = partial^2 E_n / partial a_k partial a_{k'}   (5.92)
```

**Three blocks:**

1. Both weights in second layer:
```
partial^2 E_n / (partial w_{kj}^{(2)} partial w_{k'j'}^{(2)}) = z_j z_{j'} M_{kk'}   (5.93)
```

2. Both weights in first layer:
```
partial^2 E_n / (partial w_{ji}^{(1)} partial w_{j'i'}^{(1)}) = x_i x_{i'} h''(a_{j'}) I_{jj'} sum_k w_{kj'}^{(2)} delta_k
    + x_i x_{i'} h'(a_{j'}) h'(a_j) sum_k sum_{k'} w_{k'j'}^{(2)} w_{kj}^{(2)} M_{kk'}   (5.94)
```

3. One weight in each layer:
```
partial^2 E_n / (partial w_{ji}^{(1)} partial w_{kj'}^{(2)}) = x_i h'(a_{j'}) {delta_k I_{jj'} + z_j sum_{k'} w_{k'j'}^{(2)} M_{kk'}}   (5.95)
```

Complexity: O(W^2).

#### 5.4.6 Fast Multiplication by the Hessian (R{.} operator)

To compute v^T H in O(W) operations, define `R{.} = v^T nabla`:

**Forward R-propagation:**
```
R{a_j} = sum_i v_{ji} x_i                                          (5.101)
R{z_j} = h'(a_j) R{a_j}                                            (5.102)
R{y_k} = sum_j w_{kj} R{z_j} + sum_j v_{kj} z_j                   (5.103)
```

**Backward R-propagation:**
```
R{delta_k} = R{y_k}                                                 (5.106)
R{delta_j} = h''(a_j) R{a_j} sum_k w_{kj} delta_k + h'(a_j) sum_k v_{kj} delta_k + h'(a_j) sum_k w_{kj} R{delta_k}   (5.107)
```

**Result:**
```
R{partial E / partial w_{kj}} = R{delta_k} z_j + delta_k R{z_j}   (5.110)
R{partial E / partial w_{ji}} = x_i R{delta_j}                     (5.111)
```

---

### 5.5 Regularization in Neural Networks

#### Weight Decay

```
E_tilde(w) = E(w) + (lambda/2) w^T w                               (5.112)
```

#### 5.5.1 Consistent Gaussian Priors

Simple weight decay is inconsistent with input/output scaling. Consistent regularizer:
```
(lambda_1 / 2) sum_{w in W_1} w^2 + (lambda_2 / 2) sum_{w in W_2} w^2   (5.121)
```
- W_1: first-layer weights, W_2: second-layer weights, biases excluded.

**Corresponding prior:**
```
p(w|alpha_1, alpha_2) propto exp(-(alpha_1/2) sum_{W_1} w^2 - (alpha_2/2) sum_{W_2} w^2)   (5.122)
```

**General grouped prior:**
```
p(w) propto exp(-(1/2) sum_k alpha_k ||w||_k^2)                    (5.123)
```
Leads to automatic relevance determination (ARD) when groups correspond to input units.

#### 5.5.2 Early Stopping

- Training error decreases monotonically; validation error decreases then increases
- Stop at minimum validation error
- Effective number of degrees of freedom grows with training iterations
- For quadratic error surface with gradient descent: `(rho tau)^{-1}` is analogous to regularization parameter lambda

#### 5.5.4 Tangent Propagation

**Regularizer:**
```
Omega = (1/2) sum_n sum_k (sum_i tau_i (partial y_{nk} / partial x_i))^2   (5.128)
```
where `tau = partial s(x,xi) / partial xi |_{xi=0}` is the tangent vector of transformation.

**Tikhonov regularization (random noise case):**
```
Omega = (1/2) integral ||nabla y(x)||^2 p(x) dx                    (5.135)
```

#### 5.5.6 Convolutional Networks

Three key mechanisms:
1. **Local receptive fields:** Each unit connects to small subregion of image
2. **Weight sharing:** All units in a feature map share weights (evaluation = convolution)
3. **Subsampling:** Pooling layer reduces spatial resolution (e.g., 2x2 averaging)

Architecture: alternating convolutional and subsampling layers, final fully-connected layer with softmax.

#### 5.5.7 Soft Weight Sharing

**Mixture of Gaussians prior:**
```
p(w_i) = sum_{j=1}^{M} pi_j N(w_i|mu_j, sigma_j^2)               (5.137)
```

**Regularizer:**
```
Omega(w) = -sum_i ln(sum_j pi_j N(w_i|mu_j, sigma_j^2))           (5.138)
```

**Posterior responsibility:**
```
gamma_j(w) = pi_j N(w|mu_j, sigma_j^2) / sum_k pi_k N(w|mu_k, sigma_k^2)   (5.140)
```

**Derivatives:**
```
partial E_tilde / partial w_i = partial E / partial w_i + lambda sum_j gamma_j(w_i)(w_i - mu_j)/sigma_j^2   (5.141)
partial E_tilde / partial mu_j = lambda sum_i gamma_j(w_i)(mu_j - w_i)/sigma_j^2                             (5.142)
partial E_tilde / partial sigma_j = lambda sum_i gamma_j(w_i)(1/sigma_j - (w_i-mu_j)^2/sigma_j^3)           (5.143)
```

---

### 5.6 Mixture Density Networks

**Conditional density model:**
```
p(t|x) = sum_{k=1}^{K} pi_k(x) N(t|mu_k(x), sigma_k^2(x))       (5.148)
```

**Output parameterization:**
- Mixing coefficients (softmax): `pi_k(x) = exp(a_k^pi) / sum_l exp(a_l^pi)`        (5.150)
- Variances (exponential): `sigma_k(x) = exp(a_k^sigma)`                              (5.151)
- Means (identity): `mu_{kj}(x) = a_{kj}^mu`                                         (5.152)

**Total network outputs:** (K+2)L for L components, K target dimensions.

**Error function:**
```
E(w) = -sum_n ln(sum_k pi_k(x_n,w) N(t_n|mu_k(x_n,w), sigma_k^2(x_n,w)))   (5.153)
```

**Posterior responsibilities:**
```
gamma_k(t|x) = pi_k N_{nk} / sum_l pi_l N_{nl}                    (5.154)
```

**Output-unit derivatives:**
```
partial E_n / partial a_k^pi = pi_k - gamma_k                      (5.155)
partial E_n / partial a_{kl}^mu = gamma_k (mu_{kl} - t_l) / sigma_k^2   (5.156)
partial E_n / partial a_k^sigma = -gamma_k (||t-mu_k||^2/sigma_k^3 - 1/sigma_k)   (5.157)
```

**Conditional mean:**
```
E[t|x] = sum_k pi_k(x) mu_k(x)                                    (5.158)
```

**Conditional variance:**
```
s^2(x) = sum_k pi_k(x) {sigma_k^2(x) + ||mu_k(x) - sum_l pi_l(x) mu_l(x)||^2}   (5.160)
```

---

### 5.7 Bayesian Neural Networks

#### 5.7.1 Posterior Parameter Distribution

**Model:**
```
p(t|x,w,beta) = N(t|y(x,w), beta^{-1})                            (5.161)
p(w|alpha) = N(w|0, alpha^{-1} I)                                  (5.162)
```

**Log posterior:**
```
ln p(w|D) = -(alpha/2) w^T w - (beta/2) sum_n {y(x_n,w) - t_n}^2 + const   (5.165)
```

**Laplace approximation:**
```
A = alpha I + beta H                                                 (5.166)
q(w|D) = N(w|w_MAP, A^{-1})                                        (5.167)
```
where H = Hessian of sum-of-squares error at w_MAP.

**Predictive distribution (linearized):**
```
y(x,w) ~ y(x,w_MAP) + g^T(w - w_MAP)                              (5.169)
g = nabla_w y(x,w)|_{w=w_MAP}                                      (5.170)

p(t|x,D,alpha,beta) = N(t|y(x,w_MAP), sigma^2(x))                 (5.172)
sigma^2(x) = beta^{-1} + g^T A^{-1} g                             (5.173)
```
- First term: intrinsic noise variance
- Second term: parameter uncertainty (input-dependent)

#### 5.7.2 Hyperparameter Optimization

**Log evidence (Laplace approximation):**
```
ln p(D|alpha,beta) ~ -E(w_MAP) - (1/2)ln|A| + (W/2)ln alpha + (N/2)ln beta - (N/2)ln(2pi)   (5.175)
```

**Regularized error:**
```
E(w_MAP) = (beta/2) sum_n {y(x_n,w_MAP) - t_n}^2 + (alpha/2) w_MAP^T w_MAP   (5.176)
```

**Alpha re-estimation:**
```
alpha = gamma / (w_MAP^T w_MAP)                                     (5.178)
```

**Effective number of parameters:**
```
gamma = sum_{i=1}^{W} lambda_i / (alpha + lambda_i)                (5.179)
```
where lambda_i are eigenvalues of beta H.

**Beta re-estimation:**
```
1/beta = (1/(N-gamma)) sum_n {y(x_n,w_MAP) - t_n}^2               (5.180)
```

**Model comparison:** Multiply evidence by M! 2^M to account for weight-space symmetries.

#### 5.7.3 Bayesian NNs for Classification

**Log likelihood:**
```
ln p(D|w) = sum_n {t_n ln y_n + (1-t_n) ln(1-y_n)}                (5.181)
```

**Regularized error:**
```
E(w) = -ln p(D|w) + (alpha/2) w^T w                                (5.182)
```

**Log evidence:**
```
ln p(D|alpha) ~ -E(w_MAP) - (1/2)ln|A| + (W/2)ln alpha + const   (5.183)
```

**Predictive distribution:**
```
p(t=1|x,D) ~ sigma(kappa(sigma_a^2) b^T w_MAP)                    (5.190)
```
where:
```
sigma_a^2(x) = b^T(x) A^{-1} b(x)                                 (5.188)
b = nabla a(x,w_MAP)
kappa(sigma^2) = (1 + pi sigma^2/8)^{-1/2}                        (4.154)
```

---

## Chapter 6: Kernel Methods

### 6.1 Dual Representations

**Kernel function:**
```
k(x, x') = phi(x)^T phi(x')                                        (6.1)
```
Symmetric: k(x, x') = k(x', x).

**Regularized least squares (primal):**
```
J(w) = (1/2) sum_n (w^T phi(x_n) - t_n)^2 + (lambda/2) w^T w     (6.2)
```

**Dual representation:**
```
w = sum_n a_n phi(x_n) = Phi^T a                                    (6.3)
```

**Gram matrix:**
```
K = Phi Phi^T,    K_{nm} = k(x_n, x_m)                             (6.6)
```

**Dual solution:**
```
a = (K + lambda I_N)^{-1} t                                        (6.8)
```

**Prediction for new input:**
```
y(x) = k(x)^T (K + lambda I_N)^{-1} t                             (6.9)
```
where `k(x)` has elements `k_n(x) = k(x_n, x)`.

**Primal:** invert M x M matrix. **Dual:** invert N x N matrix. Advantage of dual: work implicitly in high/infinite-dimensional feature spaces.

---

### 6.2 Constructing Kernels

**Validity criterion (Mercer's condition):** k(x,x') is a valid kernel iff the Gram matrix K is positive semidefinite for all possible sets {x_n}.

#### Kernel Construction Rules

Given valid kernels k_1, k_2:

| Rule | Kernel |
|------|--------|
| (6.13) | k(x,x') = c k_1(x,x'),  c > 0 |
| (6.14) | k(x,x') = f(x) k_1(x,x') f(x') |
| (6.15) | k(x,x') = q(k_1(x,x')),  q polynomial with nonneg coefficients |
| (6.16) | k(x,x') = exp(k_1(x,x')) |
| (6.17) | k(x,x') = k_1(x,x') + k_2(x,x') |
| (6.18) | k(x,x') = k_1(x,x') k_2(x,x') |
| (6.19) | k(x,x') = k_3(phi(x), phi(x')),  k_3 valid in R^M |
| (6.20) | k(x,x') = x^T A x',  A symmetric positive semidefinite |
| (6.21) | k(x,x') = k_a(x_a, x'_a) + k_b(x_b, x'_b) |
| (6.22) | k(x,x') = k_a(x_a, x'_a) k_b(x_b, x'_b) |

#### Common Kernels

**Polynomial kernel:**
```
k(x,x') = (x^T x')^M                   -- degree M monomials only
k(x,x') = (x^T x' + c)^M, c > 0        -- all terms up to degree M
```

Example for D=2: `k(x,z) = (x^T z)^2` gives `phi(x) = (x_1^2, sqrt(2) x_1 x_2, x_2^2)^T`   (6.12)

**Gaussian (RBF) kernel:**
```
k(x,x') = exp(-||x-x'||^2 / (2 sigma^2))                          (6.23)
```
Feature space is infinite-dimensional. Decomposition:
```
k(x,x') = exp(-x^T x/(2 sigma^2)) exp(x^T x'/sigma^2) exp(-(x')^T x'/(2 sigma^2))   (6.25)
```

**Generalized Gaussian (with nonlinear kernel):**
```
k(x,x') = exp(-(1/(2 sigma^2))(kappa(x,x) + kappa(x',x') - 2 kappa(x,x')))   (6.26)
```

**Kernel from generative model:**
```
k(x,x') = p(x) p(x')                                               (6.28)
k(x,x') = sum_i p(x|i) p(x'|i) p(i)                               (6.29)
k(x,x') = integral p(x|z) p(x'|z) p(z) dz                         (6.30)
```

**Fisher kernel:**
```
g(theta, x) = nabla_theta ln p(x|theta)                            (6.32)
k(x,x') = g(theta,x)^T F^{-1} g(theta,x')                         (6.33)
F = E_x[g(theta,x) g(theta,x)^T]                                   (6.34)
```
Invariant under reparameterization theta -> psi(theta).

**Stationary kernels:** k(x,x') = k(x-x') (translation invariant).
**Homogeneous (radial basis function):** k(x,x') = k(||x-x'||).

---

### 6.3 Radial Basis Function Networks

**Exact interpolation:**
```
f(x) = sum_{n=1}^{N} w_n h(||x - x_n||)                           (6.38)
```

**Nadaraya-Watson model (normalized RBF):**
```
y(x) = sum_n k(x, x_n) t_n                                        (6.45)
```
where:
```
k(x, x_n) = g(x - x_n) / sum_m g(x - x_m)                        (6.46)
```
satisfying `sum_n k(x, x_n) = 1`.

Derives from Parzen density estimation of joint p(x,t) with zero-mean kernel components.

---

### 6.4 Gaussian Processes

**Definition:** A Gaussian process is a probability distribution over functions y(x) such that the set of values y(x_1),...,y(x_N) at any finite set of points jointly have a Gaussian distribution.

#### 6.4.1 Linear Regression Revisited

Given `y(x) = w^T phi(x)`, prior `p(w) = N(w|0, alpha^{-1}I)`:
```
E[y] = 0                                                            (6.52)
cov[y] = (1/alpha) Phi Phi^T = K                                   (6.53)
K_{nm} = k(x_n, x_m) = (1/alpha) phi(x_n)^T phi(x_m)             (6.54)
```

**Covariance specification:**
```
E[y(x_n) y(x_m)] = k(x_n, x_m)                                    (6.55)
```

#### 6.4.2 Gaussian Processes for Regression

**Observation model:**
```
t_n = y_n + epsilon_n,    p(t_n|y_n) = N(t_n|y_n, beta^{-1})     (6.57-6.58)
```

**Marginal distribution of targets:**
```
p(t) = N(t|0, C)                                                    (6.61)
C(x_n, x_m) = k(x_n, x_m) + beta^{-1} delta_{nm}                 (6.62)
```

**Common kernel:**
```
k(x_n, x_m) = theta_0 exp(-(theta_1/2)||x_n - x_m||^2) + theta_2 + theta_3 x_n^T x_m   (6.63)
```

**Predictive distribution** for new input x_{N+1}:

Partition covariance:
```
C_{N+1} = [[C_N, k], [k^T, c]]                                    (6.65)
```
where k has elements k(x_n, x_{N+1}), and c = k(x_{N+1}, x_{N+1}) + beta^{-1}.

```
m(x_{N+1}) = k^T C_N^{-1} t                                       (6.66)
sigma^2(x_{N+1}) = c - k^T C_N^{-1} k                             (6.67)
```

**Mean as RBF expansion:**
```
m(x_{N+1}) = sum_n a_n k(x_n, x_{N+1})                            (6.68)
```
where `a = C_N^{-1} t`.

**Computational complexity:** O(N^3) for matrix inversion (once), O(N^2) per test point.

#### 6.4.3 Learning the Hyperparameters

**Log likelihood:**
```
ln p(t|theta) = -(1/2) ln|C_N| - (1/2) t^T C_N^{-1} t - (N/2) ln(2pi)   (6.69)
```

**Gradient:**
```
partial/partial theta_i ln p(t|theta) = -(1/2) Tr(C_N^{-1} partial C_N/partial theta_i) + (1/2) t^T C_N^{-1} (partial C_N/partial theta_i) C_N^{-1} t   (6.70)
```

Generally nonconvex => multiple maxima possible.

#### 6.4.4 Automatic Relevance Determination (ARD)

**ARD kernel:**
```
k(x,x') = theta_0 exp(-(1/2) sum_i eta_i (x_i - x'_i)^2)         (6.71)
```
- Separate precision eta_i per input dimension
- Optimizing eta_i by ML: small eta_i => input x_i is irrelevant

**Full ARD kernel:**
```
k(x_n, x_m) = theta_0 exp(-(1/2) sum_{i=1}^{D} eta_i (x_{ni} - x_{mi})^2) + theta_2 + theta_3 sum_i x_{ni} x_{mi}   (6.72)
```

#### 6.4.5 Gaussian Processes for Classification

**Two-class model:**
- GP prior on function a(x)
- Transform: y = sigma(a) (logistic sigmoid)
- Target distribution: `p(t|a) = sigma(a)^t (1-sigma(a))^{1-t}`                     (6.73)

**GP prior:**
```
p(a_{N+1}) = N(a_{N+1}|0, C_{N+1})                                (6.74)
C(x_n, x_m) = k(x_n, x_m) + nu delta_{nm}                         (6.75)
```

**Predictive distribution:**
```
p(t_{N+1}=1|t_N) = integral sigma(a_{N+1}) p(a_{N+1}|t_N) da_{N+1}   (6.76)
```

This integral is intractable; requires approximation (variational, EP, or Laplace).

#### 6.4.6 Laplace Approximation for GP Classification

**Log posterior (unnormalized):**
```
Psi(a_N) = -(1/2) a_N^T C_N^{-1} a_N + t_N^T a_N - sum_n ln(1 + exp(a_n)) + const   (6.80)
```

**Gradient:**
```
nabla Psi(a_N) = t_N - sigma_N - C_N^{-1} a_N                     (6.81)
```

**Hessian:**
```
nabla nabla Psi(a_N) = -W_N - C_N^{-1}                             (6.82)
```
where `W_N = diag(sigma(a_n)(1-sigma(a_n)))`.

**Properties:** A = W_N + C_N^{-1} is positive definite => posterior is log-concave => unique mode.

**IRLS update:**
```
a_N^{new} = C_N (I + W_N C_N)^{-1} {t_N - sigma_N + W_N a_N}     (6.83)
```

**At mode a_N^*:**
```
a_N^* = C_N (t_N - sigma_N)                                        (6.84)
```

**Gaussian approximation to posterior:**
```
q(a_N) = N(a_N|a_N^*, H^{-1})                                     (6.86)
H = W_N + C_N^{-1}                                                  (6.85)
```

**Predictive distribution:**
```
E[a_{N+1}|t_N] = k^T (t_N - sigma_N)                              (6.87)
var[a_{N+1}|t_N] = c - k^T (W_N^{-1} + C_N)^{-1} k               (6.88)
```

Then apply sigmoid-Gaussian convolution approximation (4.153) to get class probability.

**Approximate log likelihood for hyperparameter optimization:**
```
ln p(t_N|theta) = Psi(a_N^*) - (1/2) ln|W_N + C_N^{-1}| + (N/2) ln(2pi)   (6.90)
```

**Gradient of log likelihood:**
```
partial ln p(t_N|theta) / partial theta_j = (1/2) a_N^{*T} C_N^{-1} (partial C_N/partial theta_j) C_N^{-1} a_N^*
    - (1/2) Tr((I + C_N W_N)^{-1} W_N (partial C_N/partial theta_j))   (6.91)
```

#### 6.4.7 Connection to Neural Networks

**Neal's result (1996):** A single hidden layer neural network with:
- i.i.d. zero-mean Gaussian priors on weights
- M hidden units
- In the limit M -> infinity

converges to a Gaussian process. The kernel function is determined by the hidden unit activation function and the prior hyperparameters. This provides a deep connection between neural networks and kernel methods.

---

## Summary of Key Algorithms

| Algorithm | Section | Complexity | Key Equation |
|-----------|---------|------------|--------------|
| Fisher's discriminant | 4.1.4 | O(D^2 N) | w propto S_W^{-1}(m_2-m_1) |
| Perceptron | 4.1.7 | O(W) per update | w <- w + eta phi_n t_n |
| Logistic regression (IRLS) | 4.3.3 | O(M^3) per iteration | w = (Phi^T R Phi)^{-1} Phi^T R z |
| Backpropagation | 5.3.1 | O(W) per pattern | delta_j = h'(a_j) sum_k w_{kj} delta_k |
| Exact Hessian | 5.4.5 | O(W^2) per pattern | Three-block formulas (5.93-5.95) |
| Fast Hv product | 5.4.6 | O(W) per pattern | R{.} operator method |
| GP regression | 6.4.2 | O(N^3) training | m = k^T C_N^{-1} t |
| GP classification (Laplace) | 6.4.6 | O(N^3) training | IRLS on a_N, then sigma approx |

---

## Chapter 7: Sparse Kernel Machines

> **Note**: The source file (`bishop_part3.txt`) contains only Ch7 exercises (pages 377-378), not the main chapter body. The equations below are reconstructed from exercise references and standard PRML content.

### 7.1 Maximum Margin Classifiers (SVM)

**Decision boundary**:
```
y(x) = w^T phi(x) + b
```
- `w`: weight vector in feature space
- `phi(x)`: feature-space mapping of input `x`
- `b`: bias parameter

**Classification rule**: `t_n in {-1, +1}`, correct classification requires `t_n y(x_n) > 0`.

**Margin**: The perpendicular distance from a point `x_n` to the decision boundary is:
```
t_n y(x_n) / ||w|| = t_n (w^T phi(x_n) + b) / ||w||
```

**Maximum margin optimization** (primal):
```
argmin_{w,b}  (1/2) ||w||^2
subject to:   t_n (w^T phi(x_n) + b) >= 1,   for all n = 1,...,N
```
- The constraint `t_n y(x_n) >= 1` fixes the scale of `w` so the margin equals `1/||w||`.

**Lagrangian**:
```
L(w, b, a) = (1/2)||w||^2 - sum_{n=1}^{N} a_n { t_n (w^T phi(x_n) + b) - 1 }    (7.10)
```
- `a_n >= 0`: Lagrange multipliers (one per data point)

**Dual formulation** (maximize w.r.t. `a`):
```
L_tilde(a) = sum_{n=1}^{N} a_n - (1/2) sum_{n=1}^{N} sum_{m=1}^{N} a_n a_m t_n t_m k(x_n, x_m)
```
subject to:
```
a_n >= 0,   for all n
sum_{n=1}^{N} a_n t_n = 0
```
- `k(x_n, x_m) = phi(x_n)^T phi(x_m)`: kernel function

**Solution for w**:
```
w = sum_{n=1}^{N} a_n t_n phi(x_n)
```

**KKT conditions**:
```
a_n >= 0
t_n y(x_n) - 1 >= 0
a_n { t_n y(x_n) - 1 } = 0
```
- Either `a_n = 0` (point is not a support vector) or `t_n y(x_n) = 1` (point lies on the margin).
- **Support vectors**: points with `a_n > 0`.

**Prediction**:
```
y(x) = sum_{n in SV} a_n t_n k(x, x_n) + b
```
- `SV`: set of support vector indices
- `b` determined from any support vector: `t_n y(x_n) = 1`; typically averaged over all SVs for numerical stability.

### 7.1.1 Solving the SVM: Sequential Minimal Optimization (SMO)

- At each step, pick two multipliers `a_i`, `a_j` and optimize them jointly while holding all others fixed.
- The constraint `sum_n a_n t_n = 0` means changing one multiplier requires adjusting another.
- Repeat until convergence (all KKT conditions satisfied within tolerance).

### 7.1.2 Multiclass SVM Approaches

**One-versus-the-rest**: Train `K` binary SVMs, each separating class `k` from all others. Assign test point to class with largest `y_k(x)`.

**One-versus-one**: Train `K(K-1)/2` binary SVMs for all pairs. Assign test point by majority voting.

### 7.2 Soft Margin SVM (Overlapping Class Distributions)

**Slack variables**: Introduce `xi_n >= 0` for each data point:
```
t_n y(x_n) >= 1 - xi_n
```
- `xi_n = 0`: point on or beyond correct margin
- `0 < xi_n < 1`: point inside margin but correctly classified
- `xi_n >= 1`: point is misclassified

**Soft margin objective**:
```
argmin_{w,b,xi}  (1/2)||w||^2 + C sum_{n=1}^{N} xi_n
subject to:  t_n y(x_n) >= 1 - xi_n,   xi_n >= 0
```
- `C > 0`: regularization parameter controlling trade-off between margin maximization and misclassification penalty.
- `C -> infinity`: recovers hard margin SVM.

**Dual formulation** (soft margin):
```
L_tilde(a) = sum_{n=1}^{N} a_n - (1/2) sum_{n=1}^{N} sum_{m=1}^{N} a_n a_m t_n t_m k(x_n, x_m)
```
subject to:
```
0 <= a_n <= C,   for all n
sum_{n=1}^{N} a_n t_n = 0
```
- The only difference from hard margin: upper bound `C` on Lagrange multipliers.

**KKT conditions** (soft margin):
```
a_n = 0      =>  t_n y(x_n) >= 1     (correctly outside margin)
0 < a_n < C  =>  t_n y(x_n) = 1      (on the margin, xi_n = 0)
a_n = C      =>  t_n y(x_n) <= 1     (inside margin or misclassified)
```

**Nu-SVM** alternative parameterization: Replace `C` with `nu in (0,1]`:
```
argmin_{w,b,xi,rho}  (1/2)||w||^2 - nu*rho + (1/N) sum_n xi_n
subject to:  t_n y(x_n) >= rho - xi_n,   xi_n >= 0
```
- `nu`: upper bound on fraction of margin errors, lower bound on fraction of support vectors.

### 7.3 Kernel SVM

**Kernel function**: `k(x, x') = phi(x)^T phi(x')` computes inner product in feature space without explicit mapping.

**Common kernels**:

| Kernel | Formula | Parameters |
|--------|---------|------------|
| Linear | `k(x,x') = x^T x'` | None |
| Polynomial | `k(x,x') = (x^T x' + c)^d` | `c >= 0`, degree `d` |
| Gaussian RBF | `k(x,x') = exp(-||x-x'||^2 / (2 sigma^2))` | Width `sigma` |
| Sigmoid/Tanh | `k(x,x') = tanh(a x^T x' + b)` | `a > 0`, `b < 0` |

**Mercer's theorem**: A kernel `k(x,x')` corresponds to a valid inner product in some feature space if and only if the Gram matrix `K` (with `K_nm = k(x_n, x_m)`) is positive semidefinite for all possible choices of `{x_n}`.

**Kernel construction rules**: If `k_1` and `k_2` are valid kernels, then:
- `c * k_1(x,x')` is valid for `c > 0`
- `k_1(x,x') + k_2(x,x')` is valid
- `k_1(x,x') * k_2(x,x')` is valid
- `exp(k_1(x,x'))` is valid
- `f(x) k_1(x,x') f(x')` is valid for any function `f`

### 7.4 Relevance Vector Machines (RVM)

**RVM model** (regression):
```
y(x, w) = sum_{n=1}^{N} w_n k(x, x_n) + b
```

**Prior over weights**: Each weight has its own precision hyperparameter:
```
p(w|alpha) = prod_{i=0}^{N} N(w_i | 0, alpha_i^{-1})
```
- `alpha = {alpha_0, alpha_1, ..., alpha_N}`: individual precision hyperparameters (one per weight)

**Likelihood** (regression with noise precision `beta`):
```
p(t|X, w, beta) = prod_{n=1}^{N} N(t_n | y(x_n, w), beta^{-1})
```

**Posterior over weights**:
```
p(w|t, X, alpha, beta) = N(w|m, S)
```
where:
```
m = beta S Phi^T t
S = (A + beta Phi^T Phi)^{-1}
A = diag(alpha_0, alpha_1, ..., alpha_N)
```
- `Phi`: design matrix with `Phi_ni = k(x_n, x_i)` (plus bias column)

**Hyperparameter re-estimation** (type-II maximum likelihood / EM):
```
alpha_i^new = gamma_i / m_i^2                                    (7.87)
```
where:
```
gamma_i = 1 - alpha_i S_ii                                       (7.89)
```
- `m_i`: i-th component of posterior mean
- `S_ii`: i-th diagonal element of posterior covariance
- `gamma_i in [0,1]`: measures how well-determined parameter `i` is by data

```
beta^new = (N - sum_i gamma_i) / ||t - Phi m||^2                 (7.88)
```

**Sparsity mechanism**: During optimization, many `alpha_i -> infinity`, driving corresponding `w_i -> 0`. Only a few basis functions (relevance vectors) survive with finite `alpha_i`.

**RVM for classification** (via Laplace approximation):
```
p(t|w) = prod_{n=1}^{N} sigma(y_n)^{t_n} [1 - sigma(y_n)]^{1-t_n}
```
- `sigma(y) = 1/(1 + exp(-y))`: logistic sigmoid
- Use iteratively reweighted least squares (IRLS) to find MAP weight vector, then Laplace approximation for the posterior.

**RVM vs SVM comparison**:

| Property | SVM | RVM |
|----------|-----|-----|
| Sparsity | Support vectors | Relevance vectors |
| Typically sparser | No | Yes |
| Probabilistic outputs | No (requires Platt scaling) | Yes (natural) |
| Kernel must be Mercer | Yes | No |
| Training complexity | O(N^2) to O(N^3) | O(N^3) (but fewer vectors) |
| Regularization | C parameter (cross-validation) | Automatic (ARD) |

---

## Chapter 8: Graphical Models

### 8.1 Bayesian Networks (Directed Graphical Models)

**Joint distribution factorization** over a directed acyclic graph (DAG):
```
p(x_1, ..., x_K) = prod_{k=1}^{K} p(x_k | pa_k)                (8.5)
```
- `x_k`: variable associated with node `k`
- `pa_k`: set of parent variables of node `k` in the DAG
- Each factor `p(x_k | pa_k)` is a conditional distribution

**Key property**: The graph encodes conditional independence relationships. If there is no link from node `i` to node `j`, then `x_j` is conditionally independent of `x_i` given the parents of `x_j`.

**Number of parameters example**:
- Full joint over `K` binary variables: `2^K - 1` parameters
- Fully factorized (no edges): `K` parameters
- Chain graph `x_1 -> x_2 -> ... -> x_K`: `2K - 1` parameters

### 8.1.1 Example: Polynomial Regression as Graphical Model

Nodes: `t_n` (observed targets), `x_n` (observed inputs), `w` (weights), `alpha` (precision prior on `w`), `sigma^2` (noise variance).

Factorization:
```
p(t, w | x, alpha, sigma^2) = p(w|alpha) prod_{n=1}^{N} p(t_n | w, x_n, sigma^2)
```

Plate notation: The `N` repeated nodes `{t_n, x_n}` are drawn inside a box (plate) labeled `N`.

### 8.2 Conditional Independence

**Notation**: `a` is conditionally independent of `b` given `c`:
```
a _||_ b | c                                                      (8.22)
```
Formally: `p(a,b|c) = p(a|c) p(b|c)`
Equivalently: `p(a|b,c) = p(a|c)`

**Three canonical structures** (for three nodes `a`, `b`, `c`):

**Tail-to-tail** at node `c`: `a <- c -> b`
```
p(a,b,c) = p(a|c) p(b|c) p(c)
```
- `c` unobserved: `a` and `b` are DEPENDENT (marginally)
- `c` observed: `a _||_ b | c` (INDEPENDENT)
- Observing `c` BLOCKS the path.

**Head-to-tail** at node `c`: `a -> c -> b`
```
p(a,b,c) = p(a) p(c|a) p(b|c)
```
- `c` unobserved: `a` and `b` are DEPENDENT
- `c` observed: `a _||_ b | c` (INDEPENDENT)
- Observing `c` BLOCKS the path.

**Head-to-head** at node `c`: `a -> c <- b`
```
p(a,b,c) = p(a) p(b) p(c|a,b)
```
- `c` unobserved: `a _||_ b` (INDEPENDENT)
- `c` observed (or any descendant of `c` observed): `a` and `b` become DEPENDENT
- Observing `c` OPENS the path (explaining away).

### 8.2.1 D-Separation

**Definition**: In a directed graph, nodes `A` and `B` are **d-separated** by node set `C` if every undirected path from any node in `A` to any node in `B` is blocked. A path is blocked if it contains a node such that either:

1. The arrows on the path meet **tail-to-tail** or **head-to-tail** at that node, AND the node is in `C`, OR
2. The arrows meet **head-to-head** at that node, AND neither the node nor any of its descendants is in `C`.

If `A` and `B` are d-separated by `C`, then: `A _||_ B | C` in every distribution that factorizes according to the graph.

### 8.2.2 Markov Blanket

The **Markov blanket** of a node `x_i` in a directed graph consists of:
1. Parents of `x_i`
2. Children of `x_i`
3. Co-parents of `x_i` (other parents of `x_i`'s children)

**Property**: `x_i` is conditionally independent of all other nodes given its Markov blanket:
```
p(x_i | MB(x_i)) = p(x_i | x_1, ..., x_{i-1}, x_{i+1}, ..., x_K)
```

### 8.3 Markov Random Fields (Undirected Graphical Models)

**Joint distribution** over an undirected graph:
```
p(x) = (1/Z) prod_{C} psi_C(x_C)                                (8.39)
```
- `C`: index over maximal cliques of the graph
- `x_C`: subset of variables in clique `C`
- `psi_C(x_C) >= 0`: potential function (clique potential) for clique `C`
- `Z`: partition function (normalizing constant)

**Partition function**:
```
Z = sum_x prod_{C} psi_C(x_C)                                   (8.40)
```
(integral for continuous variables)

**Energy representation** (Boltzmann distribution):
```
psi_C(x_C) = exp{ -E(x_C) }                                     (8.41)
```

**Total energy**:
```
E(x) = sum_C E_C(x_C)
```
so:
```
p(x) = (1/Z) exp{ -E(x) }                                       (8.43)
```

### 8.3.1 Conditional Independence in MRFs

**Pairwise Markov property**: If no direct edge between `x_i` and `x_j`, then:
```
x_i _||_ x_j | x_{\{i,j}}
```
(conditionally independent given all other variables)

**Markov blanket in an MRF**: The set of neighbors of node `x_i` (all nodes directly connected by an edge).

### 8.3.2 Converting Directed to Undirected Graphs (Moralization)

1. For every pair of nodes that share a common child (head-to-head), add an undirected edge between them (**marry the parents**).
2. Drop all arrow directions.
3. The resulting undirected graph is the **moral graph**.

Note: Moralization may lose some conditional independence information (the undirected graph may assert fewer independencies).

### 8.3.3 Image De-noising Energy Model

Binary pixel model with observed noisy pixels `y_i in {-1, +1}` and hidden true pixels `x_i in {-1, +1}`:
```
E(x, y) = h sum_i x_i - beta sum_{i,j} x_i x_j - eta sum_i x_i y_i    (8.42)
```
- `h`: bias term (prior preference for +1 or -1)
- `beta > 0`: coupling between neighboring pixels (encourages smoothness)
- `eta > 0`: coupling between observed and hidden (data fidelity)
- `{i,j}`: sum over neighboring pixel pairs

**ICM (Iterated Conditional Modes)**: Greedy optimization -- for each pixel, set `x_i` to the value that maximizes `p(x_i | {x_j : j != i}, y)`. Repeat until convergence.

### 8.4 Factor Graphs

**Motivation**: Factor graphs make the factorization structure explicit, resolving ambiguity that graphical models may have.

**Structure**: A bipartite graph with two types of nodes:
- **Variable nodes** (circles): represent variables `x_i`
- **Factor nodes** (squares): represent factors `f_s`

An edge connects variable `x_i` to factor `f_s` if and only if `x_i` is an argument of `f_s`.

**Joint distribution**:
```
p(x) = prod_s f_s(x_s)
```
- `x_s`: subset of variables connected to factor `f_s`

### 8.4.1 Converting Directed Graphs to Factor Graphs

Each conditional `p(x_i | pa_i)` becomes a factor node `f_i` connected to `x_i` and all variables in `pa_i`.

### 8.4.2 Converting Undirected Graphs to Factor Graphs

Each clique potential `psi_C(x_C)` becomes a factor node connected to all variables in the clique.

Chains: `p(x) = f_a(x_1, x_2) f_b(x_2, x_3) f_c(x_3, x_4) ...`

**Tree structure**: A factor graph is a tree (no cycles) if and only if exact inference via message passing is possible in linear time.

### 8.5 The Sum-Product Algorithm (Belief Propagation)

**Goal**: Compute marginals `p(x)` for each variable `x` efficiently on tree-structured factor graphs.

**Marginal at variable node `x`**:
```
p(x) = prod_{s in ne(x)} mu_{f_s -> x}(x)                       (8.63)
```
- `ne(x)`: set of factor nodes neighboring variable `x`
- `mu_{f_s -> x}(x)`: message from factor `f_s` to variable `x`

**Factor-to-variable message**:
```
mu_{f_s -> x}(x) = sum_{x_1,...,x_M} f_s(x, x_1, ..., x_M) prod_{m in ne(f_s)\x} mu_{x_m -> f_s}(x_m)    (8.66)
```
- `ne(f_s)\x`: all variable nodes neighboring `f_s` except `x`
- `x_1, ..., x_M`: the other variables in factor `f_s` besides `x`
- Summation (or integration for continuous) over all configurations of `x_1, ..., x_M`

**Variable-to-factor message**:
```
mu_{x_m -> f_s}(x_m) = prod_{l in ne(x_m)\f_s} mu_{f_l -> x_m}(x_m)    (8.69)
```
- Product of all incoming factor-to-variable messages at `x_m`, excluding the target factor `f_s`

**Leaf node initialization**:
```
mu_{x -> f}(x) = 1          (variable leaf)                      (8.70)
mu_{f -> x}(x) = f(x)       (factor leaf)                        (8.71)
```

**Algorithm** (for trees):
1. Designate any node as root.
2. Start from leaf nodes, send messages toward root.
3. Root sends messages back toward leaves.
4. After both passes, all marginals are available.

**Computational complexity**: For a tree with `N` nodes, each with at most `M` states: `O(N * M^2)`.

**Joint marginals** (over variables in a single factor):
```
p(x_s) = f_s(x_s) prod_{i in ne(f_s)} [ prod_{l in ne(x_i)\f_s} mu_{f_l -> x_i}(x_i) ]
```

### 8.5.1 Normalizing the Messages

In practice, messages can be normalized after each step to prevent numerical overflow/underflow:
```
mu(x) <- mu(x) / sum_x mu(x)
```

### 8.6 The Max-Sum Algorithm

**Goal**: Find the configuration `x^*` that maximizes the joint probability (MAP inference).

**Key insight**: Replace sum with max, and work in log space to convert products to sums.

**Objective**:
```
x^max = argmax_x p(x) = argmax_x prod_s f_s(x_s)
```
Equivalently:
```
x^max = argmax_x sum_s ln f_s(x_s)
```

**Factor-to-variable message** (max-sum):
```
mu_{f -> x}(x) = max_{x_1,...,x_M} [ ln f(x, x_1, ..., x_M) + sum_{m in ne(f)\x} mu_{x_m -> f}(x_m) ]    (8.93)
```

**Variable-to-factor message** (max-sum):
```
mu_{x -> f}(x) = sum_{l in ne(x)\f} mu_{f_l -> x}(x)
```

**Leaf initialization**:
```
mu_{x -> f}(x) = 0           (variable leaf)                     (8.95)
mu_{f -> x}(x) = ln f(x)     (factor leaf)                       (8.96)
```

**Finding the MAP configuration**:
```
x^max = argmax_x [ sum_{s in ne(x)} mu_{f_s -> x}(x) ]
```
- This gives the globally optimal `x^max` for each variable in the root subtree.

**Back-tracking** (for non-root variables):
- During the forward pass, store:
```
phi_n(x_n) = argmax_{x_{n-1}} [ ln f(x_{n-1}, x_n) + mu_{x_{n-1} -> f}(x_{n-1}) ]    (derived from 8.100)
```
- Back-track from root:
```
x^max_{n-1} = phi_n(x^max_n)                                     (8.102)
```

### 8.7 Junction Tree Algorithm

**Purpose**: Perform exact inference on graphs with loops by converting to a tree of clusters.

**Steps**:
1. **Moralize** the directed graph (marry parents, drop directions).
2. **Triangulate**: Add edges to eliminate all cycles of length >= 4 without a chord. The resulting graph is **chordal** (also called triangulated or decomposable).
3. **Identify maximal cliques** of the triangulated graph.
4. **Build junction tree**: Create a tree where each node is a maximal clique. Edges between cliques are labeled with their **separator** (intersection of the two cliques).
5. **Running intersection property**: For any variable, the set of cliques containing that variable must form a connected subtree.
6. **Run sum-product** (or max-sum) on the junction tree using clique-to-clique messages.

**Complexity**: Exponential in the size of the largest clique (treewidth + 1). The **treewidth** of a graph determines the computational cost of exact inference.

### 8.8 Loopy Belief Propagation

**Idea**: Apply sum-product algorithm to factor graphs with cycles (no theoretical guarantees of correctness or convergence).

**Procedure**:
1. Initialize all messages to uniform (or random).
2. Apply sum-product message update rules iteratively.
3. Iterate until messages converge (or max iterations).
4. Compute approximate marginals using converged messages.

**Properties**:
- Not guaranteed to converge on general graphs.
- When it converges, provides approximate marginals.
- Fixed points correspond to stationary points of the **Bethe free energy**.
- Often works well in practice (e.g., turbo codes, LDPC codes).

---

## Chapter 9: Mixture Models and EM

### 9.1 K-Means Clustering

**Distortion measure** (objective function):
```
J = sum_{n=1}^{N} sum_{k=1}^{K} r_nk ||x_n - mu_k||^2          (9.1)
```
- `x_n in R^D`: n-th data point, `n = 1,...,N`
- `mu_k in R^D`: k-th cluster prototype/mean, `k = 1,...,K`
- `r_nk in {0, 1}`: binary indicator, `r_nk = 1` if `x_n` assigned to cluster `k`
- `||.||^2`: squared Euclidean distance

**E-step (assignment)**:
```
r_nk = { 1  if k = argmin_j ||x_n - mu_j||^2                    (9.2)
        { 0  otherwise
```

**M-step (update means)**:
```
mu_k = (sum_{n=1}^{N} r_nk x_n) / (sum_{n=1}^{N} r_nk)         (9.4)
```
- Numerator: sum of all points assigned to cluster `k`
- Denominator: number of points assigned to cluster `k`

**K-Means Algorithm**:
1. Initialize `mu_1, ..., mu_K` (e.g., random data points, or K-means++).
2. **E-step**: Assign each `x_n` to nearest `mu_k` using (9.2).
3. **M-step**: Recompute each `mu_k` as mean of assigned points using (9.4).
4. Repeat steps 2-3 until assignments do not change (or `J` decreases below threshold).

**Convergence**: `J` is guaranteed to decrease (or stay the same) at each step. Converges to a local minimum (not necessarily global).

**Online/sequential K-Means** (Robbins-Monro):
```
mu_k^new = mu_k^old + eta_n (x_n - mu_k^old)                    (9.5)
```
- `eta_n > 0`: learning rate (should satisfy `sum eta_n = infinity`, `sum eta_n^2 < infinity`)

**K-Means++** initialization:
1. Choose first center `mu_1` uniformly at random from data.
2. For each subsequent center `mu_k`: choose data point `x_n` with probability proportional to `D(x_n)^2`, where `D(x_n)` is the distance from `x_n` to the nearest existing center.
3. Guarantees `O(log K)` competitive ratio in expectation.

### 9.2 Gaussian Mixture Models (GMM)

**Mixture density**:
```
p(x) = sum_{k=1}^{K} pi_k N(x | mu_k, Sigma_k)                 (9.7)
```
- `pi_k`: mixing coefficient for component `k`, with `0 <= pi_k <= 1` and `sum_k pi_k = 1`
- `N(x | mu_k, Sigma_k)`: Gaussian (normal) distribution with mean `mu_k` and covariance `Sigma_k`

**Latent variable formulation**: Introduce `z_n` (1-of-K binary vector):
```
p(z) = prod_{k=1}^{K} pi_k^{z_k}                                (9.10)
```
```
p(x | z) = prod_{k=1}^{K} N(x | mu_k, Sigma_k)^{z_k}           (9.11)
```

**Marginal**:
```
p(x) = sum_z p(z) p(x|z) = sum_{k=1}^{K} pi_k N(x | mu_k, Sigma_k)    (9.12)
```

**Responsibilities** (posterior over latent variables):
```
gamma(z_nk) = p(z_k = 1 | x_n) = (pi_k N(x_n | mu_k, Sigma_k)) / (sum_{j=1}^{K} pi_j N(x_n | mu_j, Sigma_j))    (9.13)
```

**Log likelihood**:
```
ln p(X | pi, mu, Sigma) = sum_{n=1}^{N} ln { sum_{k=1}^{K} pi_k N(x_n | mu_k, Sigma_k) }    (9.14)
```
- Cannot be solved in closed form due to the sum inside the logarithm.

### 9.2.1 Maximum Likelihood for GMM (Direct)

Setting derivatives of log likelihood to zero gives **coupled** equations:

**Means**:
```
mu_k = (1/N_k) sum_{n=1}^{N} gamma(z_nk) x_n                   (9.17)
```

**Effective number of points**:
```
N_k = sum_{n=1}^{N} gamma(z_nk)                                  (9.18)
```

**Covariances**:
```
Sigma_k = (1/N_k) sum_{n=1}^{N} gamma(z_nk) (x_n - mu_k)(x_n - mu_k)^T    (9.19)
```

**Mixing coefficients**:
```
pi_k = N_k / N                                                    (9.22)
```

These are not closed-form solutions because `gamma(z_nk)` depends on all parameters -- motivating the EM algorithm.

### 9.2.2 Singularities in GMM Likelihood

If `Sigma_k` collapses onto a single data point `x_n`, the likelihood `-> infinity`. This is a pathological singularity, not a meaningful maximum.

**Remedies**:
- Use Bayesian approach (prior prevents singularities)
- Reset collapsed components
- Use diagonal or spherical covariance constraints
- Minimum eigenvalue constraint on `Sigma_k`

### 9.3 EM Algorithm for GMM

**E-step**: Compute responsibilities using current parameters:
```
gamma(z_nk) = (pi_k N(x_n | mu_k, Sigma_k)) / (sum_j pi_j N(x_n | mu_j, Sigma_j))    (9.23)
```

**M-step**: Re-estimate parameters using current responsibilities:
```
mu_k^new = (1/N_k) sum_{n=1}^{N} gamma(z_nk) x_n                (9.24)

Sigma_k^new = (1/N_k) sum_{n=1}^{N} gamma(z_nk) (x_n - mu_k^new)(x_n - mu_k^new)^T    (9.25)

pi_k^new = N_k / N                                                (9.26)
```
where:
```
N_k = sum_{n=1}^{N} gamma(z_nk)                                  (9.27)
```

**Evaluate log likelihood**:
```
ln p(X | pi, mu, Sigma) = sum_{n=1}^{N} ln { sum_{k=1}^{K} pi_k N(x_n | mu_k, Sigma_k) }    (9.28)
```

**EM Algorithm for GMM**:
1. Initialize means `mu_k`, covariances `Sigma_k`, mixing coefficients `pi_k`.
2. **E-step**: Evaluate responsibilities `gamma(z_nk)` using (9.23).
3. **M-step**: Re-estimate parameters using (9.24)-(9.27).
4. Evaluate log likelihood (9.28). Check for convergence.
5. If not converged, return to step 2.

**Properties**:
- Log likelihood is guaranteed to increase (or stay the same) at each EM iteration.
- Converges to a local maximum of the likelihood.
- Initialization sensitive -- use multiple restarts or K-means initialization.

### 9.4 General EM Algorithm

**Complete-data log likelihood**: `ln p(X, Z | theta)`

**Expected complete-data log likelihood** (Q-function):
```
Q(theta, theta^old) = sum_Z p(Z | X, theta^old) ln p(X, Z | theta)    (9.30)
```
- Expectation of complete-data log likelihood w.r.t. posterior of latent variables under old parameters.

**General E-step**:
```
Evaluate p(Z | X, theta^old)                                      (9.33)
```
or equivalently compute `Q(theta, theta^old)`.

**General M-step**:
```
theta^new = argmax_theta Q(theta, theta^old)                      (9.31, 9.32)
```

**Theorem** (EM monotonically increases likelihood):
```
ln p(X | theta^new) >= ln p(X | theta^old)
```
with equality iff `theta^new = theta^old` at a stationary point.

### 9.4.1 ELBO Decomposition (Variational View of EM)

For any distribution `q(Z)`:
```
ln p(X | theta) = L(q, theta) + KL(q || p)                       (9.70)
```

**Lower bound** (ELBO):
```
L(q, theta) = sum_Z q(Z) ln { p(X, Z | theta) / q(Z) }          (9.71)
```

**KL divergence**:
```
KL(q || p) = - sum_Z q(Z) ln { p(Z | X, theta) / q(Z) }         (9.72)
```

Since `KL(q||p) >= 0`, we have `L(q, theta) <= ln p(X | theta)`.

**EM as coordinate ascent on L**:
- **E-step**: Maximize `L(q, theta)` w.r.t. `q(Z)` holding `theta` fixed.
  - Optimal: `q(Z) = p(Z | X, theta^old)`, which makes `KL = 0`, so `L = ln p(X | theta^old)`.
- **M-step**: Maximize `L(q, theta)` w.r.t. `theta` holding `q` fixed.
  - This is equivalent to maximizing `Q(theta, theta^old)`.

**Generalized EM (GEM)**: Instead of full maximization in M-step, just increase `L` w.r.t. `theta`. Guaranteed to increase likelihood.

### 9.5 Bernoulli Mixtures

**Bernoulli distribution** for binary `x in {0,1}^D`:
```
p(x | mu) = prod_{i=1}^{D} mu_i^{x_i} (1 - mu_i)^{1-x_i}      (9.44)
```
- `mu_i in [0,1]`: probability that `x_i = 1`

**Mean**: `E[x] = mu`
**Covariance**: `cov[x] = diag{ mu_i(1-mu_i) }`

**Mixture of Bernoullis**:
```
p(x | mu, pi) = sum_{k=1}^{K} pi_k prod_{i=1}^{D} mu_{ki}^{x_i} (1 - mu_{ki})^{1-x_i}    (9.47)
```
- `mu_{ki}`: probability of dimension `i` being 1 in component `k`

**Mean**: `E[x] = sum_k pi_k mu_k` (9.49)
**Covariance**: `cov[x] = sum_k pi_k { Sigma_k + mu_k mu_k^T } - E[x] E[x]^T` (9.50)
where `Sigma_k = diag{ mu_{ki}(1-mu_{ki}) }`

Note: The mixture has a full covariance matrix even though individual components are diagonal (captures correlations).

**Log likelihood**:
```
ln p(X | mu, pi) = sum_{n=1}^{N} ln { sum_{k=1}^{K} pi_k p(x_n | mu_k) }    (9.51)
```

**EM for Bernoulli mixtures**:

**E-step** -- responsibilities:
```
gamma(z_nk) = (pi_k p(x_n | mu_k)) / (sum_j pi_j p(x_n | mu_j))    (9.56)
```

**M-step**:
```
mu_k = x_bar_k = (1/N_k) sum_{n=1}^{N} gamma(z_nk) x_n          (9.59)

pi_k = N_k / N                                                    (9.60)

N_k = sum_{n=1}^{N} gamma(z_nk)                                  (9.57)
```

### 9.6 EM for Bayesian Linear Regression

**Model**:
```
p(t | X, w, beta) = prod_n N(t_n | w^T phi(x_n), beta^{-1})
p(w | alpha) = N(w | 0, alpha^{-1} I)
```

**E-step**: Compute posterior statistics:
```
p(w | t, X, alpha, beta) = N(w | m, S)
m = beta S Phi^T t                                                (9.63)
S = (alpha I + beta Phi^T Phi)^{-1}                              (9.64)
```

**M-step**: Re-estimate hyperparameters:
```
alpha^new = M / (m^T m + Tr(S))                                   (analogy to 9.62)
```
where `M` = dimensionality of `w`.

For RVM (individual `alpha_i`):
```
alpha_i^new = 1 / (m_i^2 + S_ii)                                 (9.67)
```
- `m_i`: i-th element of posterior mean
- `S_ii`: i-th diagonal element of posterior covariance

```
beta^new = N / ( ||t - Phi m||^2 + beta^{-1} Tr(Phi^T Phi S) )
```

Equivalent to evidence-function re-estimation from Chapter 7, but derived as EM.

---

## Chapter 10: Approximate Inference

### 10.1 Variational Inference Framework

**Goal**: Approximate intractable posterior `p(Z|X)` with a simpler distribution `q(Z)`.

**Decomposition of log marginal likelihood** (for any `q(Z)`):
```
ln p(X) = L(q) + KL(q || p)                                      (10.2)
```

**Evidence Lower Bound (ELBO)**:
```
L(q) = integral q(Z) ln { p(X, Z) / q(Z) } dZ                   (10.3)
```

**KL divergence**:
```
KL(q || p) = - integral q(Z) ln { p(Z|X) / q(Z) } dZ            (10.4)
```

Since `KL(q||p) >= 0` (with equality iff `q(Z) = p(Z|X)`):
```
L(q) <= ln p(X)
```

**Variational inference**: Maximize `L(q)` over a restricted family of distributions `q`, which is equivalent to minimizing `KL(q||p)`.

### 10.1.1 Reverse vs Forward KL

**Reverse KL** (used in variational inference): `KL(q||p)`
- `q` seeks regions where `p` is large.
- **Mode-seeking**: `q` tends to concentrate on one mode of `p`.
- Where `p(Z) = 0`, forces `q(Z) = 0` (zero-avoiding in `q`).

**Forward KL**: `KL(p||q)`
- `q` must cover all regions where `p` is large.
- **Mean-seeking / inclusive**: `q` tends to spread over all modes.
- Where `p(Z) > 0`, requires `q(Z) > 0` (zero-forcing in `q`).

### 10.2 Factorized Distributions (Mean Field)

**Factorized approximation**: Partition `Z` into disjoint groups `Z_1, ..., Z_M`:
```
q(Z) = prod_{i=1}^{M} q_i(Z_i)                                  (10.5)
```
No restriction on functional form of individual `q_i`.

**Optimal factor** (holding all other factors fixed):
```
ln q*_j(Z_j) = E_{i != j} [ ln p(X, Z) ] + const                (10.9)
```
- `E_{i != j}[.]`: expectation w.r.t. all factors `q_i` for `i != j`
- The constant ensures normalization.

Equivalently:
```
q*_j(Z_j) = exp{ E_{i != j} [ ln p(X, Z) ] } / (normalization)
```

**Iterative optimization**:
1. Initialize all factors `q_i(Z_i)`.
2. For each `j = 1, ..., M`: update `q_j` using equation (10.9), computing expectations w.r.t. current values of all other factors.
3. Repeat until ELBO converges.

**Guaranteed convergence**: ELBO is convex w.r.t. each factor individually, so coordinate ascent converges to a local maximum.

### 10.2.1 Example: Factorized Gaussian

**True distribution**: `p(z) = N(z | mu, Lambda^{-1})` where `z = (z_1, z_2)^T`, `Lambda` = precision matrix.

**Factorized approximation**: `q(z) = q_1(z_1) q_2(z_2)`

**Optimal factors** (applying eq. 10.9):
```
q*(z_1) = N(z_1 | m_1, Lambda_11^{-1})                           (10.12)
```
where:
```
m_1 = mu_1 - Lambda_11^{-1} Lambda_12 (E[z_2] - mu_2)           (10.13)
```

```
q*(z_2) = N(z_2 | m_2, Lambda_22^{-1})                           (10.14)
```
where:
```
m_2 = mu_2 - Lambda_22^{-1} Lambda_21 (E[z_1] - mu_1)           (10.15)
```

**Key observations**:
- Means are correct: `m_1 = mu_1`, `m_2 = mu_2` (at convergence)
- Marginal variances are **underestimated**: `q` uses diagonal of precision, ignoring correlations
- This is a general property of reverse KL: the factorized approximation underestimates uncertainty

### 10.3 Variational Mixture of Gaussians

**Model** (conjugate Bayesian GMM):

**Priors**:
```
p(pi) = Dir(pi | alpha_0)                                         (10.39)
```
- `Dir`: Dirichlet distribution with concentration `alpha_0 = (alpha_0, ..., alpha_0)` (symmetric)

```
p(mu_k, Lambda_k) = N(mu_k | m_0, (beta_0 Lambda_k)^{-1}) W(Lambda_k | W_0, nu_0)    (10.40)
```
- `N-W`: Gaussian-Wishart prior (conjugate prior for Gaussian with unknown mean and precision)
- `m_0`: prior mean
- `beta_0`: prior precision scaling
- `W_0`: prior scale matrix for Wishart
- `nu_0`: prior degrees of freedom

**Likelihood**:
```
p(X | Z, mu, Lambda) = prod_{n=1}^{N} prod_{k=1}^{K} N(x_n | mu_k, Lambda_k^{-1})^{z_nk}    (10.38)
```

**Variational factorization**:
```
q(Z, pi, mu, Lambda) = q(Z) q(pi, mu, Lambda)                    (10.42)
```
```
q(pi, mu, Lambda) = q(pi) prod_{k=1}^{K} q(mu_k, Lambda_k)      (10.43)
```

**Optimal factor q*(Z)** -- responsibilities:
```
r_nk = rho_nk / sum_j rho_nj                                     (10.49)
```
```
ln rho_nk = E[ln pi_k] + (1/2) E[ln |Lambda_k|] - (D/2) ln(2*pi)
           - (1/2) E_{mu_k, Lambda_k}[ (x_n - mu_k)^T Lambda_k (x_n - mu_k) ]    (10.46)
```

**Component expectations** used in responsibilities:
```
E[ln pi_k] = psi(alpha_k) - psi(sum_j alpha_j)                   (10.68)
```
- `psi(.)`: digamma function

```
E[ln |Lambda_k|] = sum_{i=1}^{D} psi((nu_k + 1 - i)/2) + D ln 2 + ln |W_k|    (10.65)
```

```
E[(x_n - mu_k)^T Lambda_k (x_n - mu_k)] = D/beta_k + nu_k (x_n - m_k)^T W_k (x_n - m_k)    (10.64)
```

**Optimal factor q*(pi)** -- Dirichlet:
```
q*(pi) = Dir(pi | alpha)                                          (10.57)
```
```
alpha_k = alpha_0 + N_k                                           (10.58)
```

**Optimal factor q*(mu_k, Lambda_k)** -- Gaussian-Wishart:
```
q*(mu_k, Lambda_k) = N(mu_k | m_k, (beta_k Lambda_k)^{-1}) W(Lambda_k | W_k, nu_k)    (10.57)
```
**Parameter updates**:
```
beta_k = beta_0 + N_k                                             (10.60)

m_k = (1/beta_k)(beta_0 m_0 + N_k x_bar_k)                      (10.61)

W_k^{-1} = W_0^{-1} + N_k S_k + (beta_0 N_k)/(beta_0 + N_k) (x_bar_k - m_0)(x_bar_k - m_0)^T    (10.62)

nu_k = nu_0 + N_k                                                 (10.63)
```

**Sufficient statistics**:
```
N_k = sum_{n=1}^{N} r_nk                                          (10.51)

x_bar_k = (1/N_k) sum_{n=1}^{N} r_nk x_n                         (10.52)

S_k = (1/N_k) sum_{n=1}^{N} r_nk (x_n - x_bar_k)(x_n - x_bar_k)^T    (10.53)
```

**Variational lower bound** (ELBO for GMM):
```
L = E[ln p(X|Z, mu, Lambda)] + E[ln p(Z|pi)] + E[ln p(pi)]
  + E[ln p(mu, Lambda)] - E[ln q(Z)] - E[ln q(pi)] - E[ln q(mu, Lambda)]    (10.70)
```

Each term can be computed analytically using standard results for Dirichlet and Gaussian-Wishart distributions.

**Automatic component pruning**: Components with `N_k -> 0` effectively have their mixing coefficient driven to zero. The effective number of components is determined automatically.

**Predictive density**:
```
p(x_new | X) = sum_k pi_k^* St(x_new | m_k, L_k, nu_k + 1 - D)    (10.81-10.83)
```
where `St` is the Student-t distribution, and:
```
L_k = ((nu_k + 1 - D) beta_k) / (1 + beta_k) * W_k               (10.83)
pi_k^* = alpha_k / sum_j alpha_j
```

### 10.4 Variational Linear Regression

**Model**:
```
p(t | w, beta) = prod_n N(t_n | w^T phi_n, beta^{-1})
p(w | alpha) = N(w | 0, alpha^{-1} I)
p(alpha) = Gam(alpha | a_0, b_0)
p(beta) = Gam(beta | c_0, d_0) [optional]
```
- `phi_n = phi(x_n)`: basis function vector for input `x_n`
- `alpha`: weight precision hyperparameter
- `beta`: noise precision

**Variational factorization**:
```
q(w, alpha) = q(w) q(alpha)
```

**Optimal q*(w)** -- Gaussian:
```
q*(w) = N(w | m_N, S_N)                                           (10.99)
```
```
m_N = beta S_N Phi^T t                                            (10.100)
S_N = (E[alpha] I + beta Phi^T Phi)^{-1}                         (10.101)
```
- Note: `E[alpha]` replaces the fixed `alpha` from non-variational case.

**Optimal q*(alpha)** -- Gamma:
```
q*(alpha) = Gam(alpha | a_N, b_N)                                 (10.93)
```
```
a_N = a_0 + M/2                                                   (10.94)
b_N = b_0 + (1/2) E_w[w^T w] = b_0 + (1/2)(m_N^T m_N + Tr(S_N))    (10.95)
```
- `M`: number of basis functions (dimensionality of `w`)

**Expected hyperparameter**:
```
E[alpha] = a_N / b_N                                              (10.96)
```

**Iterative algorithm**:
1. Initialize hyperparameters.
2. Update `q*(w)`: compute `m_N`, `S_N` using (10.100-10.101).
3. Update `q*(alpha)`: compute `a_N`, `b_N` using (10.94-10.95).
4. Evaluate ELBO. Repeat until convergence.

**Comparison with evidence framework** (Ch3):
- Variational approach and evidence approximation give very similar results.
- Variational approach is more principled (full posterior over hyperparameters).
- As `N -> infinity`, both converge to the same answer.

### 10.5 Local Variational Methods

**Idea**: Obtain bounds on individual functions (e.g., sigmoid, log-sum-exp) rather than the entire distribution.

### 10.5.1 Convex Duality

For a convex function `f(x)`, the **conjugate function** is:
```
f*(lambda) = sup_x { lambda x - f(x) }
```

**Double conjugate**: `f**(x) = f(x)` for convex `f`.

This gives:
```
f(x) = sup_lambda { lambda x - f*(lambda) }
```

For a **concave** function `-f(x)`, we get an **upper bound**:
```
-f(x) <= -lambda x + f*(lambda)    for all lambda
```

### 10.5.2 Bounds on the Logistic Sigmoid

**Logistic sigmoid**: `sigma(a) = 1 / (1 + exp(-a))`

**Log sigmoid**: `ln sigma(a)` is a concave function of `a^2`.

**Variational lower bound on sigma(a)**:
```
sigma(a) >= sigma(xi) exp{ (a - xi)/2 - lambda(xi)(a^2 - xi^2) }    (10.144)
```
where:
```
lambda(xi) = [sigma(xi) - 1/2] / (2 xi)                           (10.141)
```
- `xi`: variational parameter (to be optimized)
- The bound is tight when `xi^2 = a^2` (i.e., `xi = |a|`)

**Equivalently** (for log sigmoid):
```
ln sigma(a) >= a/2 - a^2 lambda(xi) + ln sigma(xi) - xi/2 + xi^2 lambda(xi)
```

### 10.6 Variational Logistic Regression

**Model**:
```
p(t | w) = prod_{n=1}^{N} sigma(a_n)^{t_n} [1 - sigma(a_n)]^{1-t_n}
```
where `a_n = w^T phi_n`, `t_n in {0,1}`.

**Prior**: `p(w) = N(w | m_0, S_0)`

**Variational bound on likelihood** (using local bound on each sigmoid):
```
p(t | w) >= prod_n h(a_n, xi_n)
```
where `h(a_n, xi_n)` uses the sigmoid bound with variational parameters `{xi_n}`.

**Variational posterior** (Gaussian):
```
q(w) = N(w | m_N, S_N)                                            (10.156)
```
```
m_N = S_N { S_0^{-1} m_0 + sum_{n=1}^{N} (t_n - 1/2) phi_n }    (10.157)

S_N^{-1} = S_0^{-1} + 2 sum_{n=1}^{N} lambda(xi_n) phi_n phi_n^T    (10.158)
```

**Optimizing variational parameters**:
```
xi_n^2 = phi_n^T E[w w^T] phi_n = phi_n^T (S_N + m_N m_N^T) phi_n    (10.163)
```

**Variational logistic regression algorithm**:
1. Initialize `{xi_n}`.
2. Compute `S_N` using (10.158).
3. Compute `m_N` using (10.157).
4. Update `{xi_n}` using (10.163).
5. Evaluate lower bound. Repeat until convergence.

**Variational lower bound**:
```
L = -KL(q(w) || p(w)) + sum_n { ln sigma(xi_n) - xi_n/2 + lambda(xi_n) xi_n^2 + (t_n - 1/2) phi_n^T m_N }
```

**Predictive distribution**:
```
p(t_new = 1 | phi_new, X) >= sigma(kappa(xi_new) m_N^T phi_new)
```
where:
```
kappa(xi) = (1 - 2 lambda(xi) sigma_a^2)^{1/2}    or    kappa(sigma^2) = (1 + pi sigma^2 / 8)^{-1/2}
```
(probit approximation to account for uncertainty in `w`)

### 10.7 Expectation Propagation (EP)

**Setting**: Posterior has the form:
```
p(theta | D) = (1/Z) prod_{i=0}^{N} f_i(theta)
```
- `f_0(theta)`: prior
- `f_i(theta)` for `i >= 1`: likelihood factors (one per data point)

**EP approximation**: Approximate each factor with a simpler (e.g., exponential family) function:
```
q(theta) = (1/Z_EP) prod_{i=0}^{N} f_tilde_i(theta)              (10.191/10.203)
```
- `f_tilde_i(theta)`: approximate (unnormalized) factor

### 10.7.1 Moment Matching

**Core principle**: Match sufficient statistic moments between the true and approximate distributions.

For exponential family with sufficient statistics `u(z)`:
```
E_{q(z)}[u(z)] = E_{p(z)}[u(z)]                                   (10.187)
```

For Gaussian `q`: match mean and covariance of `p`.

This is equivalent to minimizing `KL(p || q)` (forward/inclusive KL):
```
KL(p || q) = - integral p(z) ln { q(z) / p(z) } dz               (10.188)
```
- Contrast with variational inference which minimizes reverse KL `KL(q || p)`.
- Forward KL is **mean-seeking** (tries to cover all of `p`), reverse KL is **mode-seeking**.

### 10.7.2 EP Algorithm

**Cavity distribution** (leave-one-out):
```
q^{\j}(theta) = q(theta) / f_tilde_j(theta)                       (10.195/10.205)
```
- Remove factor `j` from the current approximation.

**Augmented distribution** (include true factor):
```
q^{\j}(theta) f_j(theta) / Z_j                                    (10.196)
```
where:
```
Z_j = integral q^{\j}(theta) f_j(theta) d theta                   (10.197/10.206)
```

**EP update** -- project back to exponential family (moment matching):
```
q^new(theta) = proj[ q^{\j}(theta) f_j(theta) / Z_j ]
```
where `proj[.]` denotes moment matching (matching mean and covariance for Gaussian).

**Updated approximate factor**:
```
f_tilde_j^new(theta) = Z_j * q^new(theta) / q^{\j}(theta)         (10.199/10.207)
```

**EP Algorithm (pseudocode)**:
```
Initialize all f_tilde_i(theta) (e.g., to uniform or prior-based)
Compute initial q(theta) = (1/Z_EP) prod_i f_tilde_i(theta)

Repeat until convergence:
    For each factor j = 1, ..., N:
        1. Compute cavity: q^{\j}(theta) = q(theta) / f_tilde_j(theta)     (10.205)
        2. Compute normalization: Z_j = integral q^{\j}(theta) f_j(theta)   (10.206)
        3. Moment match: compute mean, covariance of q^{\j}(theta) f_j(theta) / Z_j
        4. Set q^new(theta) to exponential family with matched moments
        5. Update factor: f_tilde_j(theta) = Z_j q^new(theta) / q^{\j}(theta)    (10.207)
        6. Update Z_EP: multiply by Z_j / (previous normalization)           (10.208)
```

### 10.7.3 EP for the Clutter Problem

**Model**: Observations are either from a signal Gaussian or uniform clutter:
```
p(x | theta, nu) = (1-w) N(x | theta, 1) + w * nu^{-1}
```
- `theta`: signal mean (to be inferred)
- `w`: probability of clutter
- `nu`: range of uniform clutter distribution

**Prior**: `p(theta) = N(theta | 0, 100)` (broad prior)

Each likelihood factor: `f_n(theta) = (1-w) N(x_n | theta, 1) + w / nu`

EP approximation: `f_tilde_n(theta) = s_n N(theta | mu_n, sigma_n^2)` (unnormalized Gaussian)

The cavity distribution is Gaussian, and the moment matching step involves computing expectations w.r.t. a mixture of a Gaussian and a constant -- done analytically.

### 10.7.4 EP vs Variational Inference

| Property | Variational Inference | Expectation Propagation |
|----------|----------------------|------------------------|
| KL direction | Minimizes `KL(q\|\|p)` (reverse) | Minimizes `KL(p\|\|q)` (forward, locally) |
| Behavior | Mode-seeking | Mean-seeking / inclusive |
| Uncertainty | Tends to underestimate | Tends to be better calibrated |
| Bound on evidence | Yes (ELBO is a lower bound) | No strict bound (can overestimate) |
| Convergence | Guaranteed (coordinate ascent) | Not guaranteed |
| Negative variances | Never | Possible (numerical issue) |

### 10.7.5 Alpha-Family of Divergences

**Alpha-divergence**:
```
D_alpha(p || q) = (4 / (1 - alpha^2)) { 1 - integral p(x)^{(1+alpha)/2} q(x)^{(1-alpha)/2} dx }
```

**Special cases**:
- `alpha -> 1`: `KL(p || q)` (forward KL, EP-like)
- `alpha -> -1`: `KL(q || p)` (reverse KL, variational-like)
- `alpha = 0`: Hellinger distance (symmetric)

**Power EP**: Uses alpha-divergence instead of KL for the local projection step, giving a family of algorithms interpolating between EP and variational methods.

### 10.7.6 Model Evidence Approximation in EP

EP provides an approximation to the log model evidence:
```
ln p(X) approx ln Z_EP = ln Z_0 + sum_{j=1}^{N} ln Z_j - sum_{j=1}^{N} ln z_j
```
where:
- `Z_0`: normalization of prior times all approximate factors
- `Z_j`: local normalizations from EP updates
- `z_j`: normalization constants of individual approximate factors

This can be used for model comparison and hyperparameter optimization (analogous to evidence framework).

---

## Summary of Key Algorithms

| Algorithm | Chapter | Complexity | Type |
|-----------|---------|------------|------|
| SVM (dual, SMO) | 7 | O(N^2)-O(N^3) per pass | Discriminative, max-margin |
| RVM (EM) | 7 | O(N^3) | Probabilistic, sparse |
| Sum-Product | 8 | O(N M^2) on trees | Exact marginal inference |
| Max-Sum | 8 | O(N M^2) on trees | Exact MAP inference |
| Junction Tree | 8 | O(N exp(treewidth)) | Exact inference, general graphs |
| Loopy BP | 8 | O(iterations * edges * M^2) | Approximate inference |
| K-Means | 9 | O(N K D) per iteration | Hard clustering |
| EM for GMM | 9 | O(N K D^2) per iteration | Soft clustering, MLE |
| General EM | 9 | Problem-dependent | MLE with latent variables |
| Mean Field VI | 10 | Problem-dependent | Approximate posterior |
| Variational GMM | 10 | O(N K D^2) per iteration | Bayesian clustering |
| Variational LinReg | 10 | O(M^3) per iteration | Bayesian regression |
| Variational LogReg | 10 | O(N M^2) per iteration | Bayesian classification |
| EP | 10 | O(N M^3) per iteration | Approximate posterior |

---

## Key Mathematical Identities Used Throughout

**Gaussian identity** (completing the square):
```
N(x|mu, Sigma) = (2pi)^{-D/2} |Sigma|^{-1/2} exp{ -(1/2)(x-mu)^T Sigma^{-1} (x-mu) }
```

**KL divergence between Gaussians**:
```
KL(N_1 || N_2) = (1/2) { ln(|Sigma_2|/|Sigma_1|) - D + Tr(Sigma_2^{-1} Sigma_1) + (mu_2-mu_1)^T Sigma_2^{-1} (mu_2-mu_1) }
```

**Dirichlet distribution**:
```
Dir(pi | alpha) = C(alpha) prod_{k=1}^{K} pi_k^{alpha_k - 1}
```
```
E[pi_k] = alpha_k / sum_j alpha_j
E[ln pi_k] = psi(alpha_k) - psi(sum_j alpha_j)
```

**Wishart distribution** `W(Lambda | W, nu)`:
```
E[Lambda] = nu W
E[ln |Lambda|] = sum_{i=1}^{D} psi((nu+1-i)/2) + D ln 2 + ln |W|
```

**Gamma distribution** `Gam(x | a, b)`:
```
E[x] = a/b
E[ln x] = psi(a) - ln(b)
```

**Digamma function**: `psi(x) = d/dx ln Gamma(x)`

---

## Chapter 11: Sampling Methods

### 11.1 Basic Sampling Algorithms

#### 11.1.1 Inverse Transform Sampling
- Given: target distribution p(z) with CDF h(z)
- Draw u ~ Uniform(0,1)
- Set z = h^{-1}(u)
- Result: z ~ p(z)

**Exponential distribution example:**
- p(z) = lambda * exp(-lambda * z), z >= 0
- CDF: h(z) = 1 - exp(-lambda * z)
- Inverse: z = -ln(1 - u) / lambda
- Since (1-u) ~ Uniform(0,1), simplify: z = -ln(u) / lambda

**Cauchy distribution example:**
- p(z) = (1/pi) * 1/(1 + z^2)
- CDF: h(z) = (1/pi) * arctan(z) + 1/2
- Inverse: z = tan(pi*(u - 1/2))

#### 11.1.2 Box-Muller Method (Gaussian Sampling)
- Draw z1, z2 ~ Uniform(0,1)
- Compute:
  - y1 = sqrt(-2 ln(z1)) * cos(2*pi*z2)
  - y2 = sqrt(-2 ln(z1)) * sin(2*pi*z2)
- Result: y1, y2 ~ N(0,1) (independent standard normals)

**Multivariate Gaussian:**
- To sample x ~ N(mu, Sigma):
  1. Compute Cholesky decomposition: Sigma = L L^T
  2. Draw z ~ N(0, I)
  3. Set x = mu + L z

#### 11.1.3 Rejection Sampling
- **Goal:** Sample from p(z) = p_tilde(z) / Z_p where Z_p is unknown
- **Requires:** Proposal distribution q(z) from which we can sample, and constant k such that:
  - k * q(z) >= p_tilde(z) for all z
- **Algorithm:**
  1. Sample z_0 ~ q(z)
  2. Sample u_0 ~ Uniform(0, k*q(z_0))
  3. If u_0 <= p_tilde(z_0): accept z_0
  4. Else: reject z_0 and repeat
- **Acceptance rate:** p(accept) = integral of p_tilde(z) dz / (k * integral of q(z) dz) = Z_p / k
- **Limitation:** In D dimensions, acceptance rate is exponentially small: k scales as exp(D)

#### 11.1.4 Adaptive Rejection Sampling
- **Requirement:** log p(z) must be concave (log-concave distribution)
- **Method:**
  1. Construct piecewise-linear upper bound of ln p(z) using tangent lines at evaluated points
  2. Exponentiate to get envelope function
  3. Use rejection sampling with this envelope
  4. When a point is rejected, add evaluation point to refine the envelope
- **Advantage:** Envelope becomes tighter over time, improving acceptance rate

#### 11.1.5 Importance Sampling
- **Goal:** Evaluate E_p[f(z)] = integral f(z) p(z) dz without sampling from p(z) directly
- **Method:** Use proposal q(z):
  - E_p[f(z)] = integral f(z) * (p(z)/q(z)) * q(z) dz
  - Approximate: E_p[f(z)] ~= (1/L) sum_{l=1}^{L} f(z_l) * r_l
  - where z_l ~ q(z) and importance weights r_l = p(z_l) / q(z_l)
- **When Z_p unknown (p(z) = p_tilde(z)/Z_p):**
  - E_p[f(z)] ~= sum_{l=1}^{L} w_l f(z_l)
  - where w_l = r_tilde_l / sum_{m=1}^{L} r_tilde_m
  - and r_tilde_l = p_tilde(z_l) / q(z_l)

**Variables:**
- z_l: samples drawn from proposal q(z)
- r_l: importance weights (ratio p/q)
- w_l: normalized importance weights
- L: number of samples

#### 11.1.6 Sampling-Importance-Resampling (SIR)
- **Two-stage procedure:**
  1. **Stage 1:** Draw z_1, ..., z_L from q(z); compute weights w_l = p_tilde(z_l) / q(z_l); normalize: w_l = w_l / sum_m w_m
  2. **Stage 2:** Resample L points from {z_1, ..., z_L} with probabilities {w_1, ..., w_L}
- **Result:** In the limit L -> infinity, resampled points are drawn from p(z)

---

### 11.2 Markov Chain Monte Carlo (MCMC)

#### Core Concepts
- **Markov chain:** Sequence z^(1), z^(2), ... where z^(tau+1) ~ T(z^(tau+1) | z^(tau))
  - T is the transition kernel
- **Invariance condition:** p*(z) = integral T(z|z') p*(z') dz' (ensures p* is stationary)
- **Detailed balance (sufficient for invariance):**
  - p*(z) T(z'|z) = p*(z') T(z|z')
- **Ergodicity:** The chain must be ergodic (irreducible and aperiodic) for convergence to p*

#### 11.2.1 Metropolis Algorithm
- **For symmetric proposal:** q(z|z') = q(z'|z)
- **Algorithm:**
  1. Given current state z^(tau), draw candidate z* ~ q(z*|z^(tau))
  2. Compute acceptance probability:
     - A(z*, z^(tau)) = min(1, p_tilde(z*) / p_tilde(z^(tau)))
  3. Accept z* with probability A: set z^(tau+1) = z*
  4. Otherwise: set z^(tau+1) = z^(tau)

#### 11.2.2 Metropolis-Hastings Algorithm
- **General (asymmetric) proposal q(z*|z^(tau))**
- **Acceptance probability:**
  - A(z*, z^(tau)) = min(1, [p_tilde(z*) * q(z^(tau)|z*)] / [p_tilde(z^(tau)) * q(z*|z^(tau))])

**Variables:**
- z^(tau): current state at step tau
- z*: proposed candidate state
- q(z*|z^(tau)): proposal distribution
- p_tilde(z): unnormalized target distribution
- A: acceptance probability

**Proof of detailed balance:**
- p(z) q(z'|z) min(1, [p(z')q(z|z')] / [p(z)q(z'|z)]) = min(p(z)q(z'|z), p(z')q(z|z'))
- This is symmetric in z, z', so detailed balance holds

---

### 11.3 Gibbs Sampling

#### Algorithm
- **For target distribution p(z) = p(z_1, z_2, ..., z_M):**
  1. Initialize z^(0) = (z_1^(0), ..., z_M^(0))
  2. For tau = 0, 1, 2, ...:
     - Sample z_1^(tau+1) ~ p(z_1 | z_2^(tau), z_3^(tau), ..., z_M^(tau))
     - Sample z_2^(tau+1) ~ p(z_2 | z_1^(tau+1), z_3^(tau), ..., z_M^(tau))
     - ...
     - Sample z_M^(tau+1) ~ p(z_M | z_1^(tau+1), z_2^(tau+1), ..., z_{M-1}^(tau+1))

**Variables:**
- z_i: the i-th component of the state vector
- p(z_i | z_{\i}): full conditional distribution of z_i given all other variables

**Properties:**
- Special case of Metropolis-Hastings where every proposal is accepted (A = 1)
- Each step updates one variable from its full conditional
- Requires ability to sample from each conditional p(z_k | z_{\k})

#### 11.3.1 Over-Relaxation
- **Goal:** Reduce random-walk behavior in Gibbs sampling
- **Method for Gaussian conditionals p(z_i | z_{\i}) = N(mu_i, sigma_i^2):**
  - Standard Gibbs gives z_i' ~ N(mu_i, sigma_i^2)
  - Over-relaxation: z_i^(new) = alpha * z_i^(old) + (1 - alpha) * mu_i + sqrt((1 - alpha^2)) * sigma_i * noise
  - where -1 < alpha < 0 produces over-relaxation (anti-correlated successive samples)
  - alpha = 0 reduces to standard Gibbs
- **Ordered over-relaxation (general distributions):**
  1. Draw multiple samples from conditional
  2. Rank them
  3. If current value has rank r, choose the sample with rank (L+1-r) (reversed rank)

---

### 11.4 Slice Sampling

#### Algorithm
- **Goal:** Sample from p(z) = p_tilde(z)/Z_p (avoids choosing step size)
- **Augmented distribution:** p(z, u) = {1/Z_p if 0 <= u <= p_tilde(z); 0 otherwise}
  - Marginalizing over u gives p(z) = p_tilde(z) / Z_p
- **Procedure:**
  1. Given current z^(tau), sample u ~ Uniform(0, p_tilde(z^(tau)))
  2. Define the "slice": S = {z : p_tilde(z) > u}
  3. Sample z^(tau+1) uniformly from S
- **Practical implementation:** Finding the slice S exactly is hard; use stepping-out and shrinkage procedures

---

### 11.5 Hamiltonian Monte Carlo (HMC)

#### Setup
- **Target:** p(z) = (1/Z_p) exp(-E(z))  where E(z) is the "potential energy"
- **Augmented with momentum r:** H(z, r) = E(z) + K(r) where K(r) = (1/2) r^T M^{-1} r
- **Joint distribution:** p(z, r) proportional to exp(-H(z, r))
- **Variables:**
  - z: position (parameters of interest)
  - r: momentum (auxiliary variables), r ~ N(0, M)
  - M: mass matrix (positive definite)
  - H: Hamiltonian (total energy)
  - E(z): potential energy = -ln p_tilde(z)
  - K(r): kinetic energy

#### Hamilton's Equations of Motion
- dz_i/dtau = dH/dr_i = [M^{-1} r]_i
- dr_i/dtau = -dH/dz_i = -dE/dz_i

#### Leapfrog Integration
- Half-step momentum: r_i(tau + epsilon/2) = r_i(tau) - (epsilon/2) * dE/dz_i(tau)
- Full-step position: z_i(tau + epsilon) = z_i(tau) + epsilon * [M^{-1} r(tau + epsilon/2)]_i
- Half-step momentum: r_i(tau + epsilon) = r_i(tau + epsilon/2) - (epsilon/2) * dE/dz_i(tau + epsilon)

**Variables:**
- epsilon: step size for leapfrog integrator
- tau: fictitious time variable

#### HMC Algorithm
1. Sample r ~ N(0, M)
2. Compute H_current = E(z^(tau)) + K(r)
3. Perform L leapfrog steps starting from (z^(tau), r) to get (z*, r*)
4. Compute H_proposed = E(z*) + K(r*)
5. Accept (z*, r*) with probability min(1, exp(H_current - H_proposed))
6. If accepted: z^(tau+1) = z*; else z^(tau+1) = z^(tau)

**Key properties:**
- Leapfrog preserves volume (det of Jacobian = 1)
- Leapfrog is time-reversible
- Hamiltonian is approximately conserved (exact for continuous dynamics)
- L and epsilon are tuning parameters

---

### 11.6 Estimating the Partition Function

#### Method of Chaining
- **Goal:** Estimate ratio Z_2/Z_1 for distributions p_1(z) and p_2(z)
- **Introduce intermediate distributions:**
  - p_beta(z) proportional to p_1(z)^{1-beta} * p_2(z)^{beta}  for 0 = beta_0 < beta_1 < ... < beta_K = 1
- **Chain formula:**
  - Z_2/Z_1 = (Z_{beta_1}/Z_{beta_0}) * (Z_{beta_2}/Z_{beta_1}) * ... * (Z_{beta_K}/Z_{beta_{K-1}})
  - Each ratio estimated by importance sampling:
  - Z_{beta_{k+1}}/Z_{beta_k} = E_{p_{beta_k}}[p_tilde_{beta_{k+1}}(z) / p_tilde_{beta_k}(z)]

---

## Chapter 12: Continuous Latent Variables

### 12.1 Principal Component Analysis (PCA)

#### 12.1.1 Maximum Variance Formulation
- **Data:** {x_n}, n = 1, ..., N with x_n in R^D
- **Sample mean:** x_bar = (1/N) sum_{n=1}^{N} x_n
- **Sample covariance:** S = (1/N) sum_{n=1}^{N} (x_n - x_bar)(x_n - x_bar)^T
- **Goal:** Find direction u_1 (unit vector, ||u_1|| = 1) that maximizes variance of projected data
- **Variance of projection:** u_1^T S u_1
- **Optimization with Lagrange multiplier:**
  - L = u_1^T S u_1 + lambda_1(1 - u_1^T u_1)
  - Setting dL/du_1 = 0: S u_1 = lambda_1 u_1
  - u_1 is the eigenvector of S with largest eigenvalue lambda_1
- **Projected variance = lambda_1**
- **For M principal components:** u_1, u_2, ..., u_M are the M eigenvectors of S with largest eigenvalues lambda_1 >= lambda_2 >= ... >= lambda_M

**Variables:**
- x_n: D-dimensional data point
- S: D x D sample covariance matrix
- u_i: i-th principal component (eigenvector of S)
- lambda_i: i-th eigenvalue of S (variance along u_i)
- M: number of retained components (M < D)

#### 12.1.2 Minimum Error Formulation
- **Represent each x_n in complete orthonormal basis {u_i}:**
  - x_n = sum_{i=1}^{D} (x_n^T u_i) u_i = sum_{i=1}^{D} z_{ni} u_i
  - where z_{ni} = x_n^T u_i
- **M-dimensional approximation (centered data):**
  - x_n_hat = x_bar + sum_{i=1}^{M} z_{ni} u_i + sum_{i=M+1}^{D} b_i u_i
  - Optimal: b_i = x_bar^T u_i and z_{ni} = x_n^T u_i
- **Reconstruction error (distortion):**
  - J = (1/N) sum_{n=1}^{N} ||x_n - x_n_hat||^2 = sum_{i=M+1}^{D} lambda_i
  - Minimized by choosing u_1, ..., u_M as eigenvectors with largest eigenvalues
  - Error = sum of discarded eigenvalues

#### 12.1.3 PCA via SVD
- **Centered data matrix:** X (N x D), rows are (x_n - x_bar)^T
- **SVD:** X = U_X Sigma V^T
  - V columns are eigenvectors of X^T X = N*S
  - Principal components = first M columns of V

#### 12.1.4 Computational Shortcuts
- **D >> N case:** Instead of D x D eigenvalue problem, solve N x N problem
  - Compute (1/N) X X^T v_i = lambda_i v_i  (N x N matrix)
  - Then u_i = (1/sqrt(N*lambda_i)) X^T v_i

---

### 12.2 Probabilistic PCA

#### 12.2.1 Generative Model
- **Latent variable:** z ~ N(z | 0, I)  where z is M-dimensional
- **Observation model:** x | z ~ N(x | W z + mu, sigma^2 I)

**Variables:**
- z: M-dimensional latent variable
- x: D-dimensional observed variable
- W: D x M loading matrix (maps latent to observed space)
- mu: D-dimensional mean
- sigma^2: isotropic noise variance
- I: identity matrix

#### 12.2.2 Marginal Distribution
- p(x) = integral p(x|z) p(z) dz = N(x | mu, C)
- where C = W W^T + sigma^2 I  (D x D covariance)

#### 12.2.3 Posterior Distribution
- p(z | x) = N(z | M^{-1} W^T (x - mu), sigma^2 M^{-1})
- where M = W^T W + sigma^2 I  (M x M matrix)

#### 12.2.4 Maximum Likelihood Solution
- **Log likelihood:**
  - ln p(X | W, mu, sigma^2) = -(N/2) {D ln(2*pi) + ln|C| + Tr(C^{-1} S)}
  - where S = (1/N) sum_{n=1}^{N} (x_n - mu)(x_n - mu)^T
- **MLE for mean:** mu_ML = x_bar
- **MLE for W:**
  - W_ML = U_M (L_M - sigma^2 I)^{1/2} R
  - where:
    - U_M: D x M matrix of eigenvectors of S (corresponding to M largest eigenvalues)
    - L_M: M x M diagonal matrix of M largest eigenvalues (lambda_1, ..., lambda_M)
    - R: arbitrary M x M orthogonal rotation matrix
- **MLE for sigma^2:**
  - sigma^2_ML = (1/(D - M)) sum_{i=M+1}^{D} lambda_i
  - = average of discarded eigenvalues

#### 12.2.5 EM Algorithm for PCA
- **E-step:**
  - E[z_n] = M^{-1} W^T (x_n - x_bar)
  - E[z_n z_n^T] = sigma^2 M^{-1} + E[z_n] E[z_n]^T
  - where M = W^T W + sigma^2 I

- **M-step:**
  - W_new = [sum_{n=1}^{N} (x_n - x_bar) E[z_n]^T] [sum_{n=1}^{N} E[z_n z_n^T]]^{-1}
  - sigma^2_new = (1/(ND)) sum_{n=1}^{N} {||x_n - x_bar||^2 - 2 E[z_n]^T W_new^T (x_n - x_bar) + Tr(E[z_n z_n^T] W_new^T W_new)}

**Advantages of EM for PCA:**
- O(M) per iteration instead of O(D^3) eigendecomposition
- Can handle missing data
- Extends naturally to mixtures of PPCA

---

### 12.3 Bayesian PCA

#### Automatic Relevance Determination (ARD)
- **Prior on columns w_i of W:** p(w_i | alpha_i) = N(w_i | 0, alpha_i^{-1} I)
- **Hyperprior:** p(alpha_i) = Gamma(alpha_i | a, b) (with a, b -> 0 for broad prior)
- **Effect:** During optimization, some alpha_i -> infinity, driving corresponding w_i -> 0
- **Result:** Automatic selection of effective dimensionality M_eff < M
- **The effective dimensionality is determined by data, not set a priori**

---

### 12.4 Factor Analysis

#### Model
- Same as PPCA but with diagonal (not isotropic) noise:
- z ~ N(0, I)
- x | z ~ N(W z + mu, Psi)
- where Psi = diag(psi_1, ..., psi_D) is diagonal

**Variables:**
- Psi: D x D diagonal noise covariance (each dimension has its own noise variance)
- W: D x M factor loading matrix

**Marginal:**
- p(x) = N(x | mu, W W^T + Psi)

**Key difference from PPCA:** Not rotationally invariant (unique W up to column sign)

#### Posterior
- p(z | x) = N(z | G W^T Psi^{-1} (x - mu), G)
- where G = (I + W^T Psi^{-1} W)^{-1}

---

### 12.5 Kernel PCA

#### Algorithm
- **Standard PCA:** S u = lambda u where S = (1/N) X^T X
- **Kernel PCA:** Project data into feature space phi(x), perform PCA there
- **Key equation:** K a_i = lambda_i N a_i
  - where K is the N x N kernel (Gram) matrix: K_{nm} = k(x_n, x_m) = phi(x_n)^T phi(x_m)
  - a_i is the N-dimensional coefficient vector for the i-th component
  - Normalize: a_i^T a_i = 1/(N * lambda_i)

**Projection of new point x:**
- Component i: y_i(x) = phi(x)^T v_i = sum_{n=1}^{N} a_{in} k(x_n, x)
  - where v_i = sum_n a_{in} phi(x_n) is the i-th eigenvector in feature space

**Centering in feature space:**
- K_tilde = K - 1_N K - K 1_N + 1_N K 1_N
  - where (1_N)_{nm} = 1/N for all n, m

---

### 12.6 Independent Component Analysis (ICA)

#### Model
- **Generative model:** x = A s  (noiseless linear mixing)
- **Assumptions:**
  - Sources s = (s_1, ..., s_M) are statistically independent: p(s) = prod_{j=1}^{M} p(s_j)
  - At most one source is Gaussian
  - A is M x M mixing matrix

**Goal:** Recover unmixing matrix W = A^{-1} so that s = W x

**Key difference from PCA:** ICA uses non-Gaussianity; PCA uses only second-order statistics (covariance)

#### Relation to PPCA
- If p(s_j) = N(0, 1) for all j: model is identical to PPCA with sigma^2 -> 0
- ICA requires non-Gaussian source distributions (e.g., super-Gaussian, sub-Gaussian)

#### Whitening Preprocessing
1. Center data: x_centered = x - x_bar
2. Compute covariance: Sigma = E[x x^T]
3. Whiten: x_w = Sigma^{-1/2} x_centered (so that E[x_w x_w^T] = I)
4. After whitening, mixing matrix is orthogonal: W^T W = I

---

## Chapter 13: Sequential Data

### 13.1 Markov Models

#### 13.1.1 First-Order Markov Chain
- **Joint distribution:**
  - p(x_1, x_2, ..., x_N) = p(x_1) * prod_{n=2}^{N} p(x_n | x_{n-1})

#### 13.1.2 Second-Order Markov Chain
- p(x_1, ..., x_N) = p(x_1) p(x_2|x_1) prod_{n=3}^{N} p(x_n | x_{n-1}, x_{n-2})

#### 13.1.3 General State-Space Model
- **Latent variables z_n, observations x_n:**
  - p(x_1, ..., x_N, z_1, ..., z_N) = p(z_1) [prod_{n=2}^{N} p(z_n | z_{n-1})] [prod_{n=1}^{N} p(x_n | z_n)]

**Two conditional independence assumptions:**
1. z_n depends only on z_{n-1} (Markov property in latent space)
2. x_n depends only on z_n (observation depends only on current latent state)

---

### 13.2 Hidden Markov Models (HMM)

#### 13.2.1 Model Definition
- **Latent states:** z_n is a K-dimensional 1-of-K binary vector
- **Transition matrix A:** A_{jk} = p(z_{n,k} = 1 | z_{n-1,j} = 1) with sum_k A_{jk} = 1
  - p(z_n | z_{n-1}, A) = prod_{k=1}^{K} prod_{j=1}^{K} A_{jk}^{z_{n-1,j} * z_{n,k}}
- **Initial state:** pi_k = p(z_{1,k} = 1) with sum_k pi_k = 1
  - p(z_1 | pi) = prod_{k=1}^{K} pi_k^{z_{1,k}}
- **Emission probabilities:** p(x_n | z_n, phi) = prod_{k=1}^{K} p(x_n | phi_k)^{z_{n,k}}
  - phi_k: parameters of k-th emission distribution (e.g., mu_k, Sigma_k for Gaussian)

**Variables:**
- z_n: latent state at time n (1-of-K encoding)
- x_n: observation at time n
- A_{jk}: probability of transitioning from state j to state k
- pi_k: probability of starting in state k
- phi_k: emission distribution parameters for state k
- K: number of hidden states
- N: sequence length

#### 13.2.2 Likelihood
- p(X | theta) = sum_{all Z} p(X, Z | theta)
- where theta = {pi, A, phi}
- Direct computation: O(K^N) -- intractable for long sequences

---

### 13.3 Forward-Backward Algorithm (Alpha-Beta)

#### 13.3.1 Forward Recursion (Alpha)
- **Definition:** alpha(z_n) = p(x_1, ..., x_n, z_n)
- **Initialization:** alpha(z_1) = p(x_1 | z_1) p(z_1) = prod_k [pi_k p(x_1 | phi_k)]^{z_{1,k}}
- **Recursion:**
  - alpha(z_n) = p(x_n | z_n) sum_{z_{n-1}} alpha(z_{n-1}) p(z_n | z_{n-1})
  - In component form: alpha(z_{n,k}=1) = p(x_n | phi_k) sum_{j=1}^{K} alpha(z_{n-1,j}=1) A_{jk}
- **Likelihood:** p(X) = sum_{z_N} alpha(z_N)
- **Complexity:** O(K^2 * N)

#### 13.3.2 Backward Recursion (Beta)
- **Definition:** beta(z_n) = p(x_{n+1}, ..., x_N | z_n)
- **Initialization:** beta(z_N) = 1 (vector of ones)
- **Recursion:**
  - beta(z_n) = sum_{z_{n+1}} beta(z_{n+1}) p(x_{n+1} | z_{n+1}) p(z_{n+1} | z_n)
  - In component form: beta(z_{n,k}=1) = sum_{j=1}^{K} A_{kj} p(x_{n+1} | phi_j) beta(z_{n+1,j}=1)

#### 13.3.3 Posterior Marginals
- **Single-node marginal:**
  - gamma(z_n) = p(z_n | X) = alpha(z_n) beta(z_n) / p(X)
  - gamma(z_{n,k}) = E[z_{n,k}] = alpha(z_{n,k}=1) beta(z_{n,k}=1) / sum_{j} alpha(z_{n,j}=1) beta(z_{n,j}=1)

- **Two-slice marginal:**
  - xi(z_{n-1}, z_n) = p(z_{n-1}, z_n | X) = alpha(z_{n-1}) p(x_n | z_n) p(z_n | z_{n-1}) beta(z_n) / p(X)
  - xi(z_{n-1,j}, z_{n,k}) = alpha(z_{n-1,j}=1) A_{jk} p(x_n | phi_k) beta(z_{n,k}=1) / p(X)

#### 13.3.4 Scaling Factors (Numerical Stability)
- **Problem:** alpha values underflow for long sequences
- **Solution:** Use scaled variables:
  - alpha_hat(z_n) = p(z_n | x_1, ..., x_n)
  - c_n = p(x_n | x_1, ..., x_{n-1}) (scaling factor)
  - alpha(z_n) = c_n * alpha_hat(z_n) * c_{n-1} * ... * c_1
  - Recursion: alpha_hat(z_n) = (1/c_n) * p(x_n | z_n) * sum_{z_{n-1}} alpha_hat(z_{n-1}) p(z_n | z_{n-1})
  - where c_n = sum_{z_n} p(x_n | z_n) sum_{z_{n-1}} alpha_hat(z_{n-1}) p(z_n | z_{n-1})
- **Log likelihood:** ln p(X) = sum_{n=1}^{N} ln c_n

---

### 13.4 Baum-Welch Algorithm (EM for HMMs)

#### E-step
- Run forward-backward to compute gamma(z_n) and xi(z_{n-1}, z_n)

#### M-step (for Gaussian emissions p(x|phi_k) = N(x|mu_k, Sigma_k))
- **Initial state probabilities:**
  - pi_k = gamma(z_{1,k}) / sum_j gamma(z_{1,j})

- **Transition probabilities:**
  - A_{jk} = sum_{n=2}^{N} xi(z_{n-1,j}, z_{n,k}) / sum_{n=2}^{N} gamma(z_{n-1,j})

- **Emission means:**
  - mu_k = sum_{n=1}^{N} gamma(z_{n,k}) x_n / sum_{n=1}^{N} gamma(z_{n,k})

- **Emission covariances:**
  - Sigma_k = sum_{n=1}^{N} gamma(z_{n,k}) (x_n - mu_k)(x_n - mu_k)^T / sum_{n=1}^{N} gamma(z_{n,k})

**Variables:**
- gamma(z_{n,k}): posterior probability that z_n is in state k given all observations
- xi(z_{n-1,j}, z_{n,k}): posterior probability of transition j->k at time n

---

### 13.5 Viterbi Algorithm (Most Probable State Sequence)

#### Goal
- Find z* = argmax_{z_1,...,z_N} p(z_1, ..., z_N | X)
- Equivalently: argmax_{z_1,...,z_N} ln p(X, Z)

#### Forward Pass (Max-Sum)
- **Define:**
  - omega(z_1) = ln p(z_1) + ln p(x_1 | z_1)
  - omega(z_n) = ln p(x_n | z_n) + max_{z_{n-1}} [omega(z_{n-1}) + ln p(z_n | z_{n-1})]
  - In component form: omega_k(n) = ln p(x_n | phi_k) + max_j [omega_j(n-1) + ln A_{jk}]
- **Store backtrack pointers:**
  - psi_k(n) = argmax_j [omega_j(n-1) + ln A_{jk}]

#### Backward Pass (Backtracking)
- z_N* = argmax_k omega_k(N)
- z_n* = psi_{z_{n+1}*}(n+1)  for n = N-1, N-2, ..., 1

**Complexity:** O(K^2 * N) -- same as forward-backward

---

### 13.6 Linear Dynamical Systems (LDS)

#### 13.6.1 Model Definition
- **Latent transition:** p(z_n | z_{n-1}) = N(z_n | A z_{n-1}, Gamma)
- **Emission:** p(x_n | z_n) = N(x_n | C z_n, Sigma)
- **Initial state:** p(z_1) = N(z_1 | mu_0, V_0)

**Variables:**
- z_n: M-dimensional latent state vector
- x_n: D-dimensional observation vector
- A: M x M state transition matrix
- Gamma: M x M process noise covariance
- C: D x M observation matrix
- Sigma: D x D observation noise covariance
- mu_0, V_0: initial state mean and covariance

---

### 13.7 Kalman Filter (Forward Inference)

#### Prediction Step
- p(z_n | x_1, ..., x_{n-1}) = N(z_n | mu_{n|n-1}, P_{n|n-1})
- mu_{n|n-1} = A mu_{n-1}
- P_{n|n-1} = A V_{n-1} A^T + Gamma

#### Update Step (Correction)
- p(z_n | x_1, ..., x_n) = N(z_n | mu_n, V_n)
- **Kalman gain:**
  - K_n = P_{n|n-1} C^T (C P_{n|n-1} C^T + Sigma)^{-1}
- **Updated mean:**
  - mu_n = A mu_{n-1} + K_n (x_n - C A mu_{n-1})
  - = mu_{n|n-1} + K_n (x_n - C mu_{n|n-1})
- **Updated covariance:**
  - V_n = (I - K_n C) P_{n|n-1}

**Variables:**
- mu_n = E[z_n | x_1, ..., x_n]: filtered state mean
- V_n = cov[z_n | x_1, ..., x_n]: filtered state covariance
- P_{n|n-1}: predicted (prior) covariance
- K_n: Kalman gain matrix (D x M -> M x D, weights innovation)
- Innovation: x_n - C A mu_{n-1} (prediction error)

#### Predicted Observation
- p(x_n | x_1, ..., x_{n-1}) = N(x_n | C mu_{n|n-1}, C P_{n|n-1} C^T + Sigma)

#### Log Likelihood
- ln p(x_1, ..., x_N) = sum_{n=1}^{N} ln p(x_n | x_1, ..., x_{n-1})
- Each term is a Gaussian evaluated at x_n with mean C mu_{n|n-1} and covariance C P_{n|n-1} C^T + Sigma

---

### 13.8 Rauch-Tung-Striebel (RTS) Smoother (Backward Pass)

#### Smoothed Estimates
- p(z_n | x_1, ..., x_N) = N(z_n | mu_hat_n, V_hat_n)
- **Initialize:** mu_hat_N = mu_N, V_hat_N = V_N (from forward Kalman filter)
- **Backward recursion (n = N-1, N-2, ..., 1):**
  - J_n = V_n A^T P_{n+1|n}^{-1}
  - mu_hat_n = mu_n + J_n (mu_hat_{n+1} - A mu_n)
  - V_hat_n = V_n + J_n (V_hat_{n+1} - P_{n+1|n}) J_n^T

**Variables:**
- mu_hat_n: smoothed state mean (conditioned on ALL observations)
- V_hat_n: smoothed state covariance
- J_n: smoother gain matrix
- P_{n+1|n} = A V_n A^T + Gamma: predicted covariance at step n+1

#### Two-Slice Smoothed Marginal
- E[z_n z_{n-1}^T | X] = J_{n-1} V_hat_n + mu_hat_n mu_hat_{n-1}^T
  (needed for EM M-step)

---

### 13.9 EM for Linear Dynamical Systems

#### E-Step
- Run Kalman filter (forward) then RTS smoother (backward) to compute:
  - E[z_n], E[z_n z_n^T], E[z_n z_{n-1}^T] for all n

#### M-Step
- **Observation matrix:**
  - C_new = [sum_{n=1}^{N} x_n E[z_n]^T] [sum_{n=1}^{N} E[z_n z_n^T]]^{-1}

- **Observation noise:**
  - Sigma_new = (1/N) sum_{n=1}^{N} {x_n x_n^T - C_new E[z_n] x_n^T - x_n E[z_n]^T C_new^T + C_new E[z_n z_n^T] C_new^T}

- **Transition matrix:**
  - A_new = [sum_{n=2}^{N} E[z_n z_{n-1}^T]] [sum_{n=2}^{N} E[z_{n-1} z_{n-1}^T]]^{-1}

- **Process noise:**
  - Gamma_new = (1/(N-1)) sum_{n=2}^{N} {E[z_n z_n^T] - A_new E[z_{n-1} z_n^T] - E[z_n z_{n-1}^T] A_new^T + A_new E[z_{n-1} z_{n-1}^T] A_new^T}

- **Initial state:**
  - mu_0_new = E[z_1]
  - V_0_new = E[z_1 z_1^T] - E[z_1] E[z_1]^T

---

### 13.10 Particle Filters (Sequential Monte Carlo)

#### Bootstrap Filter
- **For nonlinear/non-Gaussian state-space models**
- **Represent posterior p(z_n | x_1, ..., x_n) by weighted particles {z_n^{(l)}, w_n^{(l)}}**
- **Algorithm:**
  1. For each particle l = 1, ..., L:
     - Propagate: z_n^{(l)} ~ p(z_n | z_{n-1}^{(l)})  (sample from transition prior)
     - Update weight: w_n^{(l)} = p(x_n | z_n^{(l)})
  2. Normalize: w_n^{(l)} = w_n^{(l)} / sum_m w_n^{(m)}
  3. Resample L particles from {z_n^{(l)}} with probabilities {w_n^{(l)}}

**Problem:** Particle degeneracy (few particles carry most weight)
**Solution:** Resample when effective sample size N_eff = 1/sum(w^2) falls below threshold

---

## Chapter 14: Combining Models

### 14.1 Bayesian Model Averaging

- **Predictive distribution given data D:**
  - p(t | x, D) = sum_{h=1}^{H} p(t | x, D, M_h) p(M_h | D)
  - where p(M_h | D) proportional to p(D | M_h) p(M_h)
  - p(D | M_h) = integral p(D | theta_h, M_h) p(theta_h | M_h) d theta_h  (model evidence)

**Note:** Bayesian model averaging selects the single best model in the limit of infinite data. It does NOT combine diverse models -- that is model combination (committees, boosting, etc.).

---

### 14.2 Committees (Averaging)

#### Basic Committee Prediction
- **Combine M models:** y_COM(x) = (1/M) sum_{m=1}^{M} y_m(x)
- **Expected error decomposition (assuming models have uncorrelated errors):**
  - E_COM = (1/M) E_AV
  - where E_AV = (1/M) sum_{m=1}^{M} E_m is the average individual error
  - and E_m = E[(y_m(x) - h(x))^2] for true function h(x)
- **Key insight:** Committee error is at most (1/M) times average individual error
- **In practice:** Errors are correlated, so improvement is less than factor M

#### Bagging (Bootstrap Aggregating)
- Train each model on a different bootstrap sample (sample with replacement from training set)
- Average predictions: reduces variance while bias stays approximately the same

---

### 14.3 Boosting

#### 14.3.1 AdaBoost Algorithm
- **Given:** Training set {(x_n, t_n)}, n = 1, ..., N where t_n in {-1, +1}
- **Initialize:** w_n^{(1)} = 1/N for all n
- **For m = 1, 2, ..., M:**
  1. Fit weak classifier y_m(x) to training data with weights {w_n^{(m)}}
  2. Compute weighted error rate:
     - epsilon_m = sum_{n=1}^{N} w_n^{(m)} I(y_m(x_n) != t_n) / sum_{n=1}^{N} w_n^{(m)}
  3. Compute classifier coefficient:
     - alpha_m = ln((1 - epsilon_m) / epsilon_m)
  4. Update weights:
     - w_n^{(m+1)} = w_n^{(m)} exp(alpha_m I(y_m(x_n) != t_n))

- **Final classifier:**
  - Y_M(x) = sign(sum_{m=1}^{M} alpha_m y_m(x))

**Variables:**
- w_n^{(m)}: weight of data point n at round m
- epsilon_m: weighted error rate of m-th weak learner (must be < 0.5)
- alpha_m: coefficient for m-th weak learner (larger for more accurate classifiers)
- y_m(x): m-th weak classifier output ({-1, +1})
- I(.): indicator function (1 if argument is true, 0 otherwise)
- M: number of boosting rounds

#### 14.3.2 Exponential Error Function Interpretation
- AdaBoost minimizes the exponential error function:
  - E = sum_{n=1}^{N} exp(-t_n f_m(x_n))
  - where f_m(x) = (1/2) sum_{l=1}^{m} alpha_l y_l(x)
- **Sequential minimization:** At step m, minimize E w.r.t. alpha_m and y_m(x):
  - Optimal y_m: minimizes sum w_n^{(m)} I(y_m(x_n) != t_n)
  - Optimal alpha_m = (1/2) ln((1 - epsilon_m) / epsilon_m)

**Relation to logistic loss:**
- Exponential loss: exp(-t*f(x))
- Logistic (cross-entropy) loss: ln(1 + exp(-t*f(x)))
- Exponential loss gives more weight to misclassified examples (less robust to outliers)

---

### 14.4 Decision Trees

#### 14.4.1 CART (Classification and Regression Trees)
- **Recursive binary partitioning of input space**
- **Each leaf tau contains data points; predict using majority class or mean**

#### 14.4.2 Splitting Criteria (for classification with K classes)
- **p_k = proportion of class k in region tau**

- **Cross-entropy (deviance):**
  - Q_tau = -sum_{k=1}^{K} p_k ln p_k

- **Gini index:**
  - Q_tau = sum_{k=1}^{K} p_k (1 - p_k) = 1 - sum_{k=1}^{K} p_k^2

- **Misclassification rate:**
  - Q_tau = 1 - max_k p_k

**All three equal zero when a node is pure (all same class)**

#### 14.4.3 Greedy Splitting
- For each candidate split:
  - Compute sum of impurity over child nodes, weighted by fraction of data
  - Choose split that gives maximum reduction in impurity

#### 14.4.4 Pruning (Cost-Complexity)
- **Regularized cost:**
  - C(T) = sum_{tau=1}^{|T|} Q_tau + lambda |T|
  - where |T| = number of terminal (leaf) nodes, lambda = regularization parameter
- **Cross-validation** to choose lambda

#### 14.4.5 Regression Trees
- **Prediction in leaf tau:** y_tau = (1/N_tau) sum_{x_n in tau} t_n (mean of targets in leaf)
- **Splitting criterion:** Minimize sum of squared residuals in children

---

### 14.5 Mixtures of Experts

#### 14.5.1 Conditional Mixture Model
- **Mixing coefficients depend on input:**
  - p(t | x, theta) = sum_{k=1}^{K} pi_k(x) p_k(t | x)
  - where pi_k(x) = exp(a_k(x)) / sum_j exp(a_j(x))  (softmax gating)
  - a_k(x) = v_k^T x + v_{k0} (linear gating function)

- **Expert models:** p_k(t | x) can be:
  - Linear regression: N(t | w_k^T x, sigma_k^2)
  - Logistic regression: Bernoulli with logistic output

**Variables:**
- pi_k(x): gating function for expert k (input-dependent mixing coefficient)
- p_k(t | x): expert k's predictive distribution
- v_k: gating network parameters
- w_k: expert k's parameters

#### 14.5.2 EM for Mixtures of Linear Regression
- **E-step:**
  - gamma_{nk} = pi_k(x_n) N(t_n | w_k^T x_n, sigma_k^2) / sum_j pi_j(x_n) N(t_n | w_j^T x_n, sigma_j^2)

- **M-step:**
  - w_k: weighted least squares: (X^T R_k X)^{-1} X^T R_k t
    where R_k = diag(gamma_{1k}, ..., gamma_{Nk})
  - sigma_k^2 = sum_n gamma_{nk} (t_n - w_k^T x_n)^2 / sum_n gamma_{nk}
  - Gating parameters: IRLS to maximize sum_n sum_k gamma_{nk} ln pi_k(x_n)

#### 14.5.3 Hierarchical Mixture of Experts (HME)
- **Tree-structured model:**
  - Internal nodes: gating networks (softmax over children)
  - Leaf nodes: expert models
  - Path from root to leaf l gives mixing coefficient: product of gating probabilities along path
  - p(t | x) = sum_{leaves l} [prod_{nodes j on path to l} g_j(x)] * p_l(t | x)

---

## Appendix B: Probability Distributions (Reference Table)

### Bernoulli Distribution
- p(x | mu) = mu^x (1 - mu)^{1-x}, x in {0, 1}
- E[x] = mu; var[x] = mu(1 - mu); H[x] = -mu ln mu - (1 - mu) ln(1 - mu)

### Beta Distribution
- Beta(mu | a, b) = Gamma(a+b) / (Gamma(a) Gamma(b)) * mu^{a-1} (1-mu)^{b-1}
- E[mu] = a/(a+b); var[mu] = ab/((a+b)^2 (a+b+1)); mode = (a-1)/(a+b-2)

### Binomial Distribution
- Bin(m | N, mu) = C(N,m) mu^m (1-mu)^{N-m}
- E[m] = N*mu; var[m] = N*mu*(1-mu)

### Dirichlet Distribution
- Dir(mu | alpha) = [Gamma(alpha_0) / prod_k Gamma(alpha_k)] prod_{k=1}^{K} mu_k^{alpha_k - 1}
- where alpha_0 = sum_k alpha_k and sum_k mu_k = 1
- E[mu_k] = alpha_k / alpha_0; var[mu_k] = alpha_k(alpha_0 - alpha_k) / (alpha_0^2 (alpha_0 + 1))

### Gamma Distribution
- Gam(tau | a, b) = b^a tau^{a-1} exp(-b*tau) / Gamma(a)
- E[tau] = a/b; var[tau] = a/b^2; mode = (a-1)/b for a >= 1

### Gaussian (Univariate)
- N(x | mu, sigma^2) = (1/sqrt(2*pi*sigma^2)) exp(-(x-mu)^2 / (2*sigma^2))
- E[x] = mu; var[x] = sigma^2; H = (1/2) ln(2*pi*e*sigma^2)

### Gaussian (Multivariate)
- N(x | mu, Sigma) = (2*pi)^{-D/2} |Sigma|^{-1/2} exp(-(1/2)(x-mu)^T Sigma^{-1} (x-mu))
- E[x] = mu; cov[x] = Sigma; H = (D/2) ln(2*pi*e) + (1/2) ln|Sigma|

#### Gaussian Conditional and Marginal (Partitioned)
- Given x = (x_a, x_b)^T with mu = (mu_a, mu_b)^T and Sigma partitioned:
  - Sigma = [[Sigma_aa, Sigma_ab], [Sigma_ba, Sigma_bb]]
  - Lambda = Sigma^{-1} = [[Lambda_aa, Lambda_ab], [Lambda_ba, Lambda_bb]]

- **Conditional:**
  - p(x_a | x_b) = N(x_a | mu_{a|b}, Sigma_{a|b})
  - mu_{a|b} = mu_a + Sigma_ab Sigma_bb^{-1} (x_b - mu_b) = mu_a - Lambda_aa^{-1} Lambda_ab (x_b - mu_b)
  - Sigma_{a|b} = Sigma_aa - Sigma_ab Sigma_bb^{-1} Sigma_ba = Lambda_aa^{-1}

- **Marginal:**
  - p(x_a) = N(x_a | mu_a, Sigma_aa)

### Gaussian-Gamma Distribution
- NG(mu, tau | mu_0, lambda, a, b) = N(mu | mu_0, (lambda * tau)^{-1}) Gam(tau | a, b)

### Gaussian-Wishart Distribution
- NW(mu, Lambda | mu_0, beta, W, nu) = N(mu | mu_0, (beta * Lambda)^{-1}) W(Lambda | W, nu)

### Multinomial Distribution
- Mult(m_1, ..., m_K | mu, N) = (N! / prod_k m_k!) prod_{k=1}^{K} mu_k^{m_k}
- E[m_k] = N*mu_k; var[m_k] = N*mu_k*(1-mu_k); cov[m_j, m_k] = -N*mu_j*mu_k

### Student's t-Distribution
- St(x | mu, lambda, nu) = [Gamma(nu/2 + 1/2) / Gamma(nu/2)] * (lambda/(pi*nu))^{1/2} * [1 + lambda*(x-mu)^2/nu]^{-(nu/2 + 1/2)}
- E[x] = mu (for nu > 1); var[x] = nu / (lambda*(nu-2)) (for nu > 2)
- nu = degrees of freedom; nu -> infinity gives Gaussian; nu = 1 gives Cauchy

### Uniform Distribution
- U(x | a, b) = 1/(b - a) for a <= x <= b
- E[x] = (a+b)/2; var[x] = (b-a)^2/12

### Von Mises Distribution
- p(theta | theta_0, m) = (1/(2*pi*I_0(m))) exp(m cos(theta - theta_0))
- theta in [0, 2*pi); theta_0: mean direction; m >= 0: concentration parameter
- I_0(m): modified Bessel function of first kind, order 0
- E[cos(theta)] = I_1(m)/I_0(m); ML: A(m_ML) = (1/N) sum cos(theta_n - theta_0_ML)
  where A(m) = I_1(m)/I_0(m)

### Wishart Distribution
- W(Lambda | W, nu) = B(W, nu) |Lambda|^{(nu-D-1)/2} exp(-(1/2) Tr(W^{-1} Lambda))
- Lambda: D x D positive definite precision matrix
- nu >= D: degrees of freedom; W: D x D scale matrix (positive definite)
- E[Lambda] = nu * W; E[Lambda^{-1}] = (nu - D - 1)^{-1} W^{-1} (for nu > D + 1)

---

## Appendix C: Properties of Matrices

### Basic Identities
- (AB)^T = B^T A^T
- (AB)^{-1} = B^{-1} A^{-1}
- (A^T)^{-1} = (A^{-1})^T

### Trace Properties
- Tr(A) = sum_i A_{ii}
- Tr(AB) = Tr(BA)
- Tr(ABC) = Tr(CAB) = Tr(BCA)  (cyclic permutation)

### Determinant Properties
- |A^T| = |A|
- |AB| = |A| |B|
- |A^{-1}| = 1/|A|
- |I_M + A B| = |I_N + B A|  (Sylvester's determinant identity, A: MxN, B: NxM)

### Woodbury Matrix Identity
- (A + B D^{-1} C)^{-1} = A^{-1} - A^{-1} B (D + C A^{-1} B)^{-1} C A^{-1}

**Special case (Matrix Inversion Lemma):**
- (A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)

### Related Determinant Identity
- |A + B D^{-1} C| = |A| |D + C A^{-1} B| |D|^{-1}

### Matrix Derivatives
- d/dA Tr(AB) = B^T
- d/dA ln|A| = (A^{-1})^T = (A^T)^{-1}
- d/dA Tr(A^{-1} B) = -(A^{-1} B A^{-1})^T = -(A^{-T} B^T A^{-T})
- d/dA Tr(A B A^T) = A(B + B^T)  (when A is not symmetric)
- d/dA Tr(A^k) = k (A^{k-1})^T

### Eigenvalue Decomposition (Symmetric Matrix)
- **For real symmetric A (D x D):**
  - A u_i = lambda_i u_i  (eigenvalue equation)
  - U = [u_1, ..., u_D] (orthogonal: U^T U = I)
  - A = U Lambda U^T  (diag: Lambda = diag(lambda_1, ..., lambda_D))
  - A^{-1} = U Lambda^{-1} U^T

- **Spectral decomposition:** A = sum_{i=1}^{D} lambda_i u_i u_i^T
- **Inverse:** A^{-1} = sum_{i=1}^{D} (1/lambda_i) u_i u_i^T
- **Determinant:** |A| = prod_{i=1}^{D} lambda_i
- **Trace:** Tr(A) = sum_{i=1}^{D} lambda_i

### Positive Definite and Semidefinite
- **Positive definite (A > 0):** w^T A w > 0 for all w != 0; equivalently, all lambda_i > 0
- **Positive semidefinite (A >= 0):** w^T A w >= 0 for all w; equivalently, all lambda_i >= 0

### Orthogonal Matrices
- U^T U = U U^T = I
- U^{-1} = U^T
- Preserves lengths: ||U x|| = ||x||
- Preserves angles: (Ux)^T(Uy) = x^T y
- |U| = +/- 1

---

## Appendix D: Calculus of Variations

### Functional and Functional Derivative
- **Functional:** F[y] maps a function y(x) to a scalar
- **Functional derivative:** Defined by:
  - F[y(x) + epsilon*eta(x)] = F[y(x)] + epsilon * integral (delta F / delta y(x)) eta(x) dx + O(epsilon^2)
  - where eta(x) is an arbitrary perturbation function

### Stationarity Condition
- F[y] is stationary when: delta F / delta y(x) = 0 for all x
- (Since integral (delta F / delta y(x)) eta(x) dx = 0 must hold for all eta)

### Euler-Lagrange Equations
- **For functionals of the form:**
  - F[y] = integral G(y(x), y'(x), x) dx
  - where y'(x) = dy/dx
- **Stationarity requires:**
  - dG/dy - (d/dx)(dG/dy') = 0

**Example:**
- G = y(x)^2 + (y'(x))^2
- Euler-Lagrange: y - d^2y/dx^2 = 0

### Simple Case (No Derivatives)
- F[y] = integral G(y, x) dx
- Stationarity: dG/dy(x) = 0 for all x

### Application to Probability
- When optimizing F[p] subject to integral p(x) dx = 1:
  - Use Lagrange multiplier lambda for the normalization constraint
  - Optimize L[p] = F[p] + lambda (integral p(x) dx - 1)

---

## Appendix E: Lagrange Multipliers

### Equality Constraints
- **Problem:** maximize f(x) subject to g(x) = 0
- **Lagrangian:** L(x, lambda) = f(x) + lambda g(x)
- **Stationarity conditions:**
  - grad_x L = 0  =>  grad f + lambda grad g = 0
  - dL/dlambda = 0  =>  g(x) = 0
- **Geometric interpretation:** At the optimum, grad f and grad g are parallel (or anti-parallel)

### Inequality Constraints
- **Problem:** maximize f(x) subject to g(x) >= 0
- **Lagrangian:** L(x, lambda) = f(x) + lambda g(x)
- **Karush-Kuhn-Tucker (KKT) conditions:**
  1. g(x) >= 0  (primal feasibility)
  2. lambda >= 0  (dual feasibility)
  3. lambda g(x) = 0  (complementary slackness)

**Two cases:**
- **Inactive constraint:** Solution in interior (g(x) > 0), lambda = 0, grad f = 0
- **Active constraint:** Solution on boundary (g(x) = 0), lambda > 0, grad f = -lambda grad g

### Multiple Constraints
- **Problem:** maximize f(x) subject to g_j(x) = 0 (j=1,...,J) and h_k(x) >= 0 (k=1,...,K)
- **Lagrangian:**
  - L(x, {lambda_j}, {mu_k}) = f(x) + sum_{j=1}^{J} lambda_j g_j(x) + sum_{k=1}^{K} mu_k h_k(x)
- **Conditions:** mu_k >= 0 and mu_k h_k(x) = 0 for all k

### Minimization Variant
- To minimize f(x) subject to g(x) >= 0:
  - L(x, lambda) = f(x) - lambda g(x) with lambda >= 0

---

## Cross-Chapter Algorithm Comparison Table

| Algorithm | Type | Latent States | Observations | Complexity |
|-----------|------|---------------|--------------|------------|
| HMM (Forward-Backward) | Discrete latent, any emission | K discrete states | Any | O(K^2 N) |
| HMM (Viterbi) | MAP sequence | K discrete states | Any | O(K^2 N) |
| Kalman Filter | Continuous latent, linear-Gaussian | R^M continuous | Linear Gaussian | O(M^3 + M^2 D) per step |
| RTS Smoother | Backward smoothing for LDS | R^M continuous | Linear Gaussian | O(M^3) per step |
| Particle Filter | Nonlinear/non-Gaussian | R^M continuous | Any | O(L) per step (L particles) |
| Gibbs Sampling | General MCMC | Any | Any | Depends on conditionals |
| HMC | Continuous MCMC | R^M continuous | Any | O(L * cost_of_gradient) |
| Metropolis-Hastings | General MCMC | Any | Any | O(cost_of_p_tilde) |

## Model Combination Methods Comparison

| Method | Diversity Mechanism | Weight Scheme | Error Reduction |
|--------|-------------------|---------------|-----------------|
| Committee | Different initializations | Equal (1/M) | Reduces variance |
| Bagging | Bootstrap samples | Equal (1/M) | Reduces variance |
| AdaBoost | Reweighted training data | alpha_m = ln((1-eps)/eps) | Reduces bias+variance |
| Decision Trees | Greedy feature splitting | N/A (single model) | Interpretable |
| Mixture of Experts | Input-dependent gating | pi_k(x) via softmax | Captures heterogeneity |
| Random Forest | Bootstrap + random features | Equal (1/M) | Reduces variance |

---

## Key Relationships Between Methods

### Unified View of Latent Variable Models
| Model | p(z) | p(x|z) | Noise |
|-------|-------|--------|-------|
| PCA | N(0, I) | N(Wz + mu, sigma^2 I) | Isotropic, sigma^2 -> 0 |
| Probabilistic PCA | N(0, I) | N(Wz + mu, sigma^2 I) | Isotropic |
| Factor Analysis | N(0, I) | N(Wz + mu, Psi) | Diagonal Psi |
| ICA | prod p_j(z_j) non-Gaussian | x = Az (noiseless) | None |

### Sampling Method Selection Guide
- **Known CDF inverse:** Inverse transform sampling
- **Log-concave target:** Adaptive rejection sampling
- **Evaluating expectations only:** Importance sampling
- **High-dimensional continuous:** HMC (if gradients available)
- **Discrete or mixed:** Gibbs sampling
- **General purpose:** Metropolis-Hastings
- **Sequential/online filtering:** Particle filters

---

