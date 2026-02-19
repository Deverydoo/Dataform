# The Transformer Cookbook -- Raw Technical Extraction

**Source:** Yang, Watson, Xue, Bhattamishra, Llarena, Merrill, Ferreira, Svete, Chiang (2025). arXiv:2510.00368v1

---

## 1. Self-Attention Mechanism

### 1.1 Scaled Dot-Product Self-Attention (Definition 2.3)

Given input sequence `(z_1, ..., z_n)` where `z_i in R^d`:

```
q_i = W^(Q) z_i                          # Query projection: R^d -> R^d_key
k_j = W^(K) z_j                          # Key projection:   R^d -> R^d_key
v_j = W^(V) z_j                          # Value projection: R^d -> R^d

s_{i,j} = (q_i^T k_j) / sqrt(d_key)     # Attention scores

alpha_{i,*} = S(s_{i,*})                 # Weighting function (e.g., softmax)

c_i = sum_{j=1}^{n} alpha_{i,j} v_j     # Weighted sum of values
```

**Variables:**
- `W^(Q), W^(K) in R^{d_key x d}` -- query and key projection matrices
- `W^(V) in R^{d x d}` -- value projection matrix
- `s_{i,j}` -- attention score from position i to position j
- `alpha_{i,j}` -- attention weight (post-weighting-function)
- `d_key` -- key/value dimension
- `d` -- model/embedding dimension

### 1.2 Causal (Future) Masking

```
s_{i,j} = { (q_i^T k_j) / sqrt(d_key)    if j <= i
           { -inf                           otherwise
```

**Strict future masking** (excludes self):
```
s_{i,j} = { (q_i^T k_j) / sqrt(d_key)    if j < i
           { -inf                           otherwise
```

### 1.3 Past Masking

```
s_{i,j} = { (q_i^T k_j) / sqrt(d_key)    if j >= i
           { -inf                           otherwise
```

### 1.4 Weighting Functions

**Softmax:**
```
[softmax(s_1, ..., s_n)]_j = exp(s_j) / sum_{k=1}^{n} exp(s_k)
```

**Left-Hardmax (UHAT):**
```
I(s) = { i in [|s|] | s_i = max(s) }
[lhardmax(s)]_i = I[i = min I(s)]
```

**Right-Hardmax:**
```
[rhardmax(s)]_i = I[i = max I(s)]
```

**Average-Hardmax (AHAT):**
```
[ahardmax(s)]_i = (1 / |I(s)|) * I[i in I(s)]
```

### 1.5 Multi-Head Attention

Multi-head attention with H heads and d_hid key/value dimensions per head: the result is the concatenation of H separate attention head outputs. Can be simulated by H single-headed attention layers with results summed together.

### 1.6 Attention Identity (Zero Attention)

```
W^(Q) = 0,  W^(K) = 0,  W^(V) = 0
```
With residual connection: `id(x) = att_zero(x) + x = x`.

### 1.7 Uniform Average via Attention

**Unmasked average:**
```
Avg(x_1, ..., x_n)_i = (1/n) * sum_{k=1}^{n} x_k

W^(Q) = 0,  W^(K) = 0,  W^(V) = I
```

**Future-masked prefix average:**
```
Avg_left(x_1, ..., x_n)_i = (1/i) * sum_{k=1}^{i} x_k
```
Same weights as above, but with future masking applied.

### 1.8 Simulating UHAT with AHAT (Tie-Breaking)

Given attention scores with minimum gap gamma between max and non-max scores:

```
q_hat_i = [q_i; gamma]
k_hat_j = [k_j; t(j)]
```

Where tie-breaking function `t(j)`:
```
t(j) = -1/j       (rightmost selection)
t(j) = +1/j       (leftmost selection)
t(j) = j/n        (rightmost selection, alternative)
t(j) = -j/n       (leftmost selection, alternative)
```

### 1.9 Simulating AHAT with Softmax (Lemma 5.3)

Given tieless scores with gap gamma:
```
||hardmax(s) - softmax(s)||_1 <= 2n * exp(-gamma)
```

To make softmax approximate hardmax for binary values with max length N:
```
W^(Q)' = (1/tau) * W^(Q)
tau = gamma / log(8N)
```

Result: `|c_i - v_{q_i}| <= 1/4`, allowing rounding via `GTZero_{1/2}(c_i - 1/4)`.

---

## 2. Positional Encoding

### 2.1 Sinusoidal Positional Encoding (Vaswani et al., 2017)

For even embedding dimension d, for `0 <= c <= d/2 - 1`:

```
pe(i, 2c+1) = sin(i / M^{2c/d})
pe(i, 2c)   = cos(i / M^{2c/d})
```

Where `M = 10000` (typical choice).

### 2.2 Simple Positional Encodings

| Encoding | Value at position i | Use case |
|---|---|---|
| `1/i` | Reciprocal of position | Via future-masked uniform attention + bos token |
| `i/n` | Length-normalized position | Via unmasked uniform attention + bos token; bounded |
| `(-1)^i` | Alternating sign | First-position detection, predecessor construction |
| `i` | Raw integer position | Index lookup |
| `i^2` | Quadratic position | Quadratic maximization lookup |
| `i^3` | Cubic position | Exact table-lookup via soft attention |

### 2.3 One-Hot Positional Encoding

Encode positions as `e_1, ..., e_N in R^N` (one-hot vectors up to max length N).

- Width requirement: Omega(N)
- Dot product: `e_{q_i} . e_j = I[j = q_i]`
- Minimum gap: gamma = 1

### 2.4 Almost-Orthogonal Positional Encoding

Family `x_1, ..., x_N in R^m` satisfying:
```
|x_i . x_j| <= epsilon    (i != j)
x_i . x_i  >= 1 - epsilon
```

Dimension requirement:
```
m = ceil(12k / epsilon^2 * log(2N)) = O(log N)
```
where `k > 0` controls failure probability `1/N^k <= delta`.

Constructed by sampling `x_i in {+/- 1/sqrt(m)}^m` uniformly. Equivalent to Johnson-Lindenstrauss transforms of one-hot vectors.

### 2.5 Layernorm Hash Positional Encoding

Store integer x as:
```
lh(x) = LN([x, 1, -x, -1]) = sqrt(2/(x^2+1)) * [x, 1, -x, -1]
```

**Key property:** Scale-invariant: `lh(kx) = lh(x)`.

Works even when only `x/i` and `1/i` are available:
```
lh(x) = LN([x/i, 1/i, -x/i, -1/i])
```

Width requirement: Theta(1) (constant width, 9 dimensions total with value).

### 2.6 Quadratic Maximization Positional Encoding

Using features `j` and `j^2`:
```
q_i = [q_i, 1]
k_j = [2j, -j^2]

s_{i,j} = 2*q_i*j - j^2
```

Uniquely maximized when `j = q_i`. Width: Theta(1) (5 dimensions).

| Approach | Width | Requirements |
|---|---|---|
| One-hot | Theta(N) | One-hot positional encodings |
| Almost-orthogonal | Theta(log N) | Near-orthogonal positional vectors |
| Layernorm-hash | Theta(1) | Selective layernorm |
| Quadratic maximization | Theta(1) | Positional features j and j^2 |

---

## 3. Feed-Forward Layers

### 3.1 Standard FFN (Definition 2.5)

```
FFN(x) = W_2 * ReLU(W_1 * x + b_1) + b_2
```

Parameters:
```
W_1 in R^{d_hid x d}
b_1 in R^{d_hid}
W_2 in R^{d x d_hid}
b_2 in R^{d}
```

### 3.2 ReLU (Definition 2.4)

```
ReLU(x) = max(0, x)
```

### 3.3 GELU (Definition 4.2)

```
GELU(x) = x * Phi(x)
         = (x/2) * (1 + erf(x/sqrt(2)))                         [exact]
         ~ (x/2) * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))  [approx 1]
         ~ x * sigmoid(1.702x)                                    [approx 2]
```

Where `Phi` = CDF of standard normal, `erf` = Gauss error function, `sigmoid(x) = 1/(1+e^{-x})`.

Taylor expansion:
```
GELU(z) = z/2 + z^2/sqrt(2*pi) + R(z)
|R(z)| <= (1/6)|z|^3
```

### 3.4 Continuous Piecewise Linear Functions (CPWL)

Any CPWL function `f: R -> R` with n pieces and knots at `(x_2, y_2), ..., (x_n, y_n)`:

```
f(x) = y_1 + m_1*(x - x_1) + sum_{k=2}^{n} (m_k - m_{k-1}) * ReLU(x - x_k)
```

where slopes `m_k = (y_{k+1} - y_k) / (x_{k+1} - x_k)`.

FFN parameters (hidden dim = n+1):
```
W_1 = [-1, 1, 1, ..., 1]^T        (n+1 rows)
b_1 = [0, 0, -x_2, -x_3, ..., -x_n]^T
W_2 = [-m_1, m_1, (m_2-m_1), ..., (m_n-m_{n-1})]
b_2 = y_1 - m_1*x_1
```

Multivariate CPWL with k pieces: exact with O(log k) layers.

### 3.5 Canceling Residual Connections

Given FFN `f`, construct `f'` such that `f'(x) + x = f(x)`:

```
f'.W_1 = [f.W_1; I; -I]
f'.b_1 = [f.b_1; 0; 0]
f'.W_2 = [f.W_2, -I, I]
f'.b_2 = f.b_2
```

### 3.6 Identity Function via FFN

For scalar (d=1):
```
W_1 = [1; -1],  b_1 = 0,  W_2 = [1, -1],  b_2 = 0
id(x) = ReLU(x) - ReLU(-x) = x
```

For vector (d dimensions):
```
W_1 = [I_d; -I_d],  b_1 = 0_{2d},  W_2 = [I_d, -I_d],  b_2 = 0_d
```

### 3.7 Min and Max via FFN

```
min(x,y) = ReLU(x) - ReLU(-x) - ReLU(x-y) = x - ReLU(x-y)
max(x,y) = ReLU(x) - ReLU(-x) + ReLU(y-x) = x + ReLU(y-x)
```

Min parameters:
```
W_1 = [[1,0],[-1,0],[1,-1]]   b_1 = 0   W_2 = [1,-1,-1]   b_2 = 0
```

Max parameters:
```
W_1 = [[1,0],[-1,0],[-1,1]]   b_1 = 0   W_2 = [1,-1,1]    b_2 = 0
```

### 3.8 Addition via FFN

```
add(x,y) = ReLU(x) - ReLU(-x) + ReLU(y) - ReLU(-y) = x + y
```

```
W_1 = [[1,0],[-1,0],[0,1],[0,-1]]   b_1 = 0   W_2 = [1,-1,1,-1]   b_2 = 0
```

### 3.9 Multiplication (Approximate, via GELU)

```
sqrt(pi/2) * (GELU(x+y) - GELU(x) - GELU(y)) = xy + epsilon(x,y)
|epsilon(x,y)| <= (1/4)*(|x|+|y|)^3
```

FFN parameters:
```
W_1 = [[1,1],[1,0],[0,1]]
b_1 = 0
W_2 = [sqrt(pi/2), -sqrt(pi/2), -sqrt(pi/2)]
b_2 = 0
```

### 3.10 Comparison Functions (Approximate)

**Greater-than-zero** (tolerance epsilon):
```
GTZero_eps(x) = { 0      if x <= 0
                { x/eps  if 0 < x < eps
                { 1      if eps <= x

W_1 = [1; 1]   b_1 = [0; -eps]   W_2 = [1/eps, -1/eps]   b_2 = 0
```

**Greater-or-equal-zero:**
```
GEZero_eps(x) = { 0          if x <= -eps
                { 1 + x/eps  if -eps < x < 0
                { 1          if 0 <= x

W_1 = [1; 1]   b_1 = [eps; 0]   W_2 = [1/eps, -1/eps]   b_2 = 0
```

**Equal-to-zero:**
```
EqZero_eps(x) = { 0          if |x| >= eps
                { 1 - |x/eps| if 0 < |x| < eps
                { 1          if x = 0

W_1 = [1; 1; 1]   b_1 = [eps; 0; -eps]   W_2 = [1/eps, -2/eps, 1/eps]   b_2 = 0
```

### 3.11 Boolean Functions (Arbitrary, Single FFN)

For `phi: {0,1}^m -> {0,1}`, enumerate all 2^m truth assignments `xi_0, ..., xi_{2^m-1}`:

```
W_1[k,:] = (2*xi_k - 1)^T          # Rows are +/-1 vectors
b_1[k]   = -1 . xi_k + 1           # Bias offsets
W_2       = [phi(xi_0), ..., phi(xi_{2^m-1})]
b_2       = 0
```

Logic: `h_k = ReLU((2*xi_k-1) . x - 1.xi_k + 1) = I[xi_k = x]` when `x in {0,1}^m`.

Then: `y = sum_k phi(xi_k) * I[xi_k = x] = phi(x)`.

**Simple Boolean connectives:**
- AND (conjunction): equivalent to `min(x,y)`
- OR (disjunction): equivalent to `max(x,y)`
- NOT (negation): `1 - x`

### 3.12 Conditional via FFN

```
if(p, x, y) = { x  if p = 1
              { y  if p = 0
```

For `x, y in [0,1]`:
```
W_1 = [[1,1,0],[-1,0,1]]   b_1 = [-1, 0]   W_2 = [1, 1]   b_2 = 0

if(1, x, y) = ReLU(x) + ReLU(y-1) = x
if(0, x, y) = ReLU(x-1) + ReLU(y) = y
```

---

## 4. Normalization

### 4.1 Layer Normalization (Definition 6.1)

```
LN(x) = ((x_i - mu) / sqrt(sigma^2 + epsilon)) * gamma + beta

mu     = (1/d) * sum_{i=1}^{d} x_i
sigma^2 = (1/d) * sum_{i=1}^{d} (x_i - mu)^2
```

Parameters: `gamma, beta in R^d`, `epsilon > 0` (small constant for stability).

### 4.2 Post-Norm vs Pre-Norm

**Post-norm** (original transformer):
```
y = LN(f(x) + x)
```

**Pre-norm** (current standard):
```
y = f(LN(x)) + x
```
Pre-norm avoids exploding gradients that make post-norm training unstable.

### 4.3 Circumventing LayerNorm

For any LN with `beta = 0`, if components come in additive-inverse pairs:
```
LN([x; -x]) = [cx; -cx]    for some scalar c
```

The layernorm can only scale the vector (zero mean is preserved).

### 4.4 Amplification via LayerNorm

To amplify a small value `+/-delta` to `+/-1`:

1. Clip all activations to `+/-delta` using a CPWL FFN:
```
clip_delta(x):
  W_1 = [1; 1]   b_1 = [delta; -delta]   W_2 = [1, -1]   b_2 = -delta
```

2. Apply layernorm with `epsilon = 0, beta = 0, gamma = 1` to normalize to `+/-1`.

**Key fact:** LayerNorm with `epsilon = 0` is NOT Lipschitz-continuous, enabling amplification of O(1/n) signals to O(1).

### 4.5 Properties

| Property | Definition |
|---|---|
| k-Lipschitz continuity | `||f(x_1) - f(x_2)|| <= k * ||x_1 - x_2||` for all x_1, x_2 |
| Positive k-homogeneity | `f(cx) = c^k * f(x)` for all x, c >= 0 |

- FFN with ReLU and no bias: positively 1-homogeneous.
- If all layers are 1-homogeneous and output is scale-invariant, layernorm has no effect.
- LN with epsilon > 0: Lipschitz continuous.
- LN with epsilon = 0: NOT Lipschitz continuous (enables signal amplification).

---

## 5. Attention Variants

### 5.1 Index Lookup via Attention

**Problem:** At position i, retrieve `v_{q_i}` (the value at the queried position q_i).

**One-Hot Lookup** (width Omega(N)):
```
W^(Q) = [I_{NxN}, 0_{NxN}, 0_{Nx1}]
W^(K) = [0_{NxN}, I_{NxN}, 0_{Nx1}]
W^(V) = [0_{1xN}, 0_{1xN}, 1]

q_i . k_j = e_{q_i} . e_j = I[j = q_i]
```

**Almost-Orthogonal Lookup** (width O(log N)):
```
Same structure as one-hot, but with almost-orthogonal vectors.
q_i . k_j >= 1-eps  if j = q_i
q_i . k_j <= eps    if j != q_i
Gap: gamma = 1 - 2*epsilon
```

**Layernorm-Hash Lookup** (width Theta(1)):
```
q_i = lh(q_i) = LN([q_i/i, 1/i, -q_i/i, -1/i])
k_j = lh(j)   = LN([1, 1/j, -1, -1/j])

s_{i,j} = lh(q_i) . lh(j)   uniquely maximized when q_i = j
```

Requires selective layernorm: `y = sa(LN(W^(N) x)) + x`.

**Quadratic Maximization Lookup** (width Theta(1)):
```
q_i = [q_i, 1]
k_j = [2j, -j^2]

s_{i,j} = 2*q_i*j - j^2     uniquely maximized at j = q_i
Gap: gamma = 1
```

### 5.2 Predecessor Function

Each position i attends to position i-1. Uses two future-masked attention heads with positional encoding `(-1)^i`:

```
Head 1 (odd i -> even i-1):
  W^(Q) = [1,0,0]   W^(K) = [0,1,0]   W^(V) = [0,0,1]

Head 2 (even i -> odd i-1):
  W^(Q) = [1,0,0]   W^(K) = [0,-1,0]  W^(V) = [0,0,1]
```

Final selection via conditional: `pred(z) = if(GTZero((-1)^i), pred.even(z), pred.odd(z))`.

With strict future masking + `i/n` encoding + UHAT: predecessor is trivial (UHAT selects max of `1/n, ..., (i-1)/n`).

### 5.3 First-Position Detection

Using `(-1)^i` encoding and future-masked uniform attention with `W^(V) = -I`:
- At i=1: output = 1 (only self-attention)
- At i>1: output <= 1/3 (average of +/-1)
- Threshold with `GTZero_{1/3}(c_i - 1/3)` to get indicator `(1, 0, ..., 0)`.

---

## 6. Architecture: Transformer Definition

### 6.1 Full Transformer (Definition 2.1)

```
z_i^(0) = WE(x_i) + PE(i)                              # Embedding

(z_1^(l), ..., z_n^(l)) = tl^(l)(z_1^(l-1), ..., z_n^(l-1))  # L layers

y_i = out(z_i^(L))                                      # Unembedding
```

### 6.2 Transformer Layer (Definition 2.2)

```
(c_bar_1, ..., c_bar_n) = tl.sa(z_1, ..., z_n) + (z_1, ..., z_n)   # Attention + residual
z'_i = tl.ff(c_bar_i) + c_bar_i                                      # FFN + residual
```

### 6.3 Unembedding Options

**Binary classification:**
```
W_out in R^{1 x d}
y_i = I[W_out * z_i^(L) > 0]
```

**Multi-class classification:**
```
W_out in R^{|Sigma| x d}
y_i = argmax(W_out * z_i^(L))
```

**Language modeling:**
```
W_out in R^{|Sigma| x d}
y_i = softmax(W_out * z_i^(L))

P(y | x) = prod_{i=1}^{n} [y_i]_{y_i}
```

Autoregressive: `x_1 = bos, x_i = y_{i-1}` for `i = 2,...,n`, `y_n = eos`.

### 6.4 Residual Stream

Each layer adds to (not replaces) its input. In constructions, each layer typically writes to previously-zero components, resembling a straight-line program where each component is set exactly once and all previous values remain accessible.

---

## 7. Assembly and Composition

### 7.1 Routing Lemma (Lemma 8.1)

If `ff: R^d -> R^d` is an FFN and `L, R: R^d -> R^d` are linear transformations, then `L o ff` and `ff o R` are also FFNs:

```
ff' = L o ff o R:
  ff'.W_1 = (ff.W_1) * R
  ff'.b_1 = ff.b_1
  ff'.W_2 = L * (ff.W_2)
  ff'.b_2 = L * (ff.b_2)
```

For self-attention `sa' = L o sa o R`:
```
sa'.W^(Q) = (sa.W^(Q)) * R
sa'.W^(K) = (sa.W^(K)) * R
sa'.W^(V) = L * (sa.W^(V)) * R
```

Allows reordering, duplicating, or zeroing out hidden vector components at will.

### 7.2 Serial Composition (Lemma 8.2)

Stack `tl_2` on top of `tl_1` to compute `tl_2(tl_1(w))`. Extends to arbitrary layer sequences.

### 7.3 Parallel Composition (Lemma 8.3)

Given `tf_1: Sigma* -> (R^{d_1})*` and `tf_2: Sigma* -> (R^{d_2})*` (without layernorm):

```
(tf_1 + tf_2)(w) = [ [tf_1(w)_i; tf_2(w)_i] for i=1..n ]
```

Width: `d = d_1 + d_2`. If layer counts differ, pad with identity layers.

Embeddings:
```
(tf_1 + tf_2).we(sigma) = [tf_1.we(sigma); tf_2.we(sigma)]
(tf_1 + tf_2).pen(i)    = [tf_1.pen(i);    tf_2.pen(i)]
```

Each sub-transformer operates on its half of the activation vector (top/bottom), with cross-terms zeroed out.

---

## 8. Boolean Representations

| Representation | false | true | Continuous ops | Min gap | Fixed mean | Fixed variance |
|---|---|---|---|---|---|---|
| `(-inf, 0] / (0, inf)` | `(-inf, 0]` | `(0, inf)` | No | No | No | No |
| `0 / 1` | 0 | 1 | Yes | Yes | No | No |
| `-1 / +1` | -1 | +1 | Yes | Yes | No | No |
| `(-inf,0) / (0,inf)` | `(-inf,0)` | `(0,inf)` | Yes | No | No | No |
| `0 / [1,inf)` | 0 | `[1,inf)` | Yes | Yes | No | No |
| `[+1,-1] / [-1,+1]` (2D) | `[+1,-1]` | `[-1,+1]` | Yes | Yes | Yes | Yes |
| `[+d,-d] / [-d,+d]` (2D) | `[+d,-d]` | `[-d,+d]` | Yes | No | Yes | No |

The 2D pair `[+1,-1]/[-1,+1]` is the only representation with all four desirable properties (continuous operations, minimum gap, fixed mean, fixed variance).

---

## 9. Error Analysis

### 9.1 Bounded Activations (Proposition 7.1)

If `||pen(i)|| <= P` for all n and i, then there exists X such that `||z_i^(l)|| <= X` for all layers l and positions i. This holds even with LN (epsilon = 0).

### 9.2 FFN Error (Proposition 7.2)

```
||x_hat - x|| <= delta  =>  ||ff(x_hat) - ff(x)|| <= K*delta
```
for some constant K (ReLU FFN).

### 9.3 Attention Error (Proposition 7.3)

```
||x|| <= X, ||x_hat - x|| <= delta <= 1  =>  ||[sa(x_hat)]_i - [sa(x)]_i|| <= K*delta
```
for some constant K.

### 9.4 Residual Error (Proposition 7.4)

```
||x_hat - x|| <= delta, ||f(x_hat) - f(x)|| <= eps
  =>  ||f(x_hat) + x_hat - (f(x) + x)|| <= eps + delta
```

### 9.5 LayerNorm Error (Proposition 7.5)

```
||x|| <= X, ||x_hat - x|| <= delta <= 1, LN.epsilon > 0
  =>  ||LN(x_hat) - LN(x)|| <= K*delta
```

### 9.6 Operator Norm (L_{inf,1})

```
||A||_{inf,1} = sum_i max_j |A_{i,j}|
```

---

## 10. All Equations (Consolidated)

| Eq | Formula | Description |
|---|---|---|
| (2) | `z_i^(0) = WE(x_i) + PE(i)` | Token + position embedding |
| (3-4) | `z^(l) = tl^(l)(z^(l-1))` | Layer-wise processing |
| (5) | `y_i = out(z_i^(L))` | Unembedding |
| (6) | `q_i = W^(Q) z_i` | Query projection |
| (7) | `k_j = W^(K) z_j` | Key projection |
| (8) | `v_j = W^(V) z_j` | Value projection |
| (9-10) | `s_{i,j} = q_i^T k_j / sqrt(d_key)` | Attention score |
| (11) | `alpha_{i,*} = S(s_{i,*})` | Attention weights |
| (12) | `c_i = sum_j alpha_{i,j} v_j` | Attention output |
| (13) | `ReLU(x) = max(0, x)` | ReLU activation |
| (14-16) | `FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2` | Feed-forward network |
| (17) | `y_i = I[W_out z_i^(L) > 0]` | Binary classification |
| (18) | `y_i = argmax(W_out z_i^(L))` | Multi-class classification |
| (19) | `y_i = softmax(W_out z_i^(L))` | Language model output |
| (20) | `P(y|x) = prod_i [y_i]_{y_i}` | String probability |
| (23-26) | `GELU(x) = x*Phi(x) = ...` | GELU activation (4 forms) |
| (27) | `GTZero_eps(x)` | Greater-than-zero comparator |
| (28) | `GEZero_eps(x)` | Greater-or-equal-zero comparator |
| (29) | `EqZero_eps(x)` | Equal-to-zero comparator |
| (33) | `phi(x) = sum_k phi(xi_k) I[xi_k = x]` | Arbitrary Boolean function |
| (35) | `|x_i.x_j| <= eps, x_i.x_i >= 1-eps` | Almost-orthogonality condition |
| (36-37) | `m = ceil(12k/eps^2 * log(2N))` | Almost-orthogonal dimension |
| (38) | `(tf_1+tf_2)(w) = [tf_1(w); tf_2(w)]` | Parallel composition |
| (39) | `o_i = +1 if "(", -1 if ")"` | Bracket encoding |
| (40) | `Avg_left(o)_i = B_i/i` | Running balance via prefix avg |
| (41) | `t_i = Avg_left(E)_i` | Error aggregation |

---

## 11. FFN Construction Reference Table

| Function | Type | Description | Exact? |
|---|---|---|---|
| Identity | `R^d -> R^d` | Returns input unchanged | Yes |
| Min / Max | `R^2 -> R` | `min(x,y)` or `max(x,y)` | Yes |
| Add / Subtract | `R^2 -> R` | `x + y` (or `x - y`) | Yes |
| Multiply by c | `R -> R` | Scales input: `x -> cx` | Yes |
| Multiply (xy) | `R^2 -> R` | Product via GELU Taylor expansion | No |
| Comparators | `R -> {0,1}` | `I[x>0]`, `I[x>=0]`, `I[x=0]` | No |
| Boolean functions | `{0,1}^m -> {0,1}` | Any Boolean function of m bits | Yes |
| Conditional (if) | `{0,1} x R^2 -> R` | `if p=1 then x else y` | Yes |
| CPWL f | `R^d -> R` | Any cont. piecewise-linear function | Yes |
| Cancel Residual | `R^d -> R^d` | Builds f' so f'(x)+x = f(x) | Yes |

---

## 12. Attention Construction Reference Table

| Function | Mapping | Behavior | Weighting |
|---|---|---|---|
| Identity | `(R^d)+ -> (R^d)+` | Pass each token unchanged | SMAT, AHAT, UHAT |
| Uniform Average | `(R^d)+ -> (R^d)+` | Mean of all unmasked tokens (prefix-mean if causal) | SMAT, AHAT |
| First | `(R^d)+ -> (R^d)+` | Retrieve value at first position | SMAT, AHAT, UHAT |
| Predecessor | `(R^d)+ -> (R^d)+` | Each token retrieves content at position i-1 | AHAT, UHAT |
| Index Lookup | `(R^d)+ -> R+` | Lookup values at a particular index | SMAT, AHAT, UHAT |
| Tie-breaking | -- | Add tiny bias (+/-1/(j+1)) to emulate left/right hardmax | SMAT, AHAT |
| Multi-head | -- | Simulate multi-headed attention using single head | SMAT, AHAT, UHAT |

---

## 13. Example Constructions

### 13.1 Induction Heads

**Task:** If current symbol is A and previous occurrence of A was followed by B, predict B.

**Most-Recent Induction** (rightmost match):
1. One-hot embedding: `WE(w) = (e_{w_1}, ..., e_{w_n})`
2. Predecessor retrieval: `Pred(w)_i = e_{w_{i-1}}` (via Section 5.4)
3. Attention lookup: `k_i = WE(x_i)`, `q_j = Pred(x_j)`, `v_j = WE(x_j)` with rightmost UHAT

**Most-Frequent Induction** (majority vote):
1. One-hot embedding + predecessor to encode bigrams
2. Bigram counting via uniform attention: `W^(Q) = [e_{sigma_1}; e_{sigma_2}]`, `W^(K) = Bigram(x_j)`, `W^(V) = 1`
3. Comparison of counts to find most frequent successor

**Lower bound:** Any single-layer transformer of size o(N) or independent of N cannot compute induction heads.

### 13.2 Dyck-1 Recognition

**Task:** Accept iff string of parentheses is well-balanced.

**Two-layer construction** (AHAT, no layernorm, future masking):

**Layer 1 -- Running balance:**
```
Embed: o_i = +1 if "(", -1 if ")"
Avg_left(o)_i = B_i/i = (opens - closes) / i
```

**Layer 1 FFN -- Error detection:**
```
E_i = ReLU(-B_i/i)        # positive iff prefix violation
```

**Layer 2 -- Error aggregation:**
```
t_i = Avg_left(E)_i = (1/i) * sum_{j=1}^{i} E_j
```

**Accept at position n iff:**
- `t_n = 0` (no prefix violations -- Property 1)
- `B_n/n = 0` (balanced counts -- Property 2)

**Uniform version:** Parameters independent of n; acceptance check is external.
**Nonuniform version:** Uses `GTZero_{1/n^2}` and `EqZero_{1/n}` to produce 0/1 decision; requires fixed max length.

### 13.3 Dyck-1-2 Recognition (Depth-Bounded)

**Requirements:** Hard attention, no layernorm, strict future + past masking, `i/n` encodings.

**Step 1 -- Depth-1 match:** Use predecessor + successor to check adjacent bracket pairs:
```
a_i = 0  if o_i = +1 and o_{i+1} = -1   (matched)
a_i = 0  if o_i = -1 and o_{i-1} = +1   (matched)
a_i = 1  otherwise                         (unmatched)
```

**Step 2 -- Depth-2 match:** Among unmatched positions (a_i = 1), find nearest active left/right neighbors via UHAT on `i/n - (a_i - 1)`. Repeat matching.

**Step 3:** Accept iff `a'_1 = ... = a'_n = 0` (all matched).

### 13.4 Dyck-k-D Generalization

Extend bracket encoding: `o_i in {-k, ..., -1, +1, ..., +k}` (type + openness). Repeat select-nearest-active and check-match D times. Same acceptance criterion.

---

## 14. Uniformity and Precision

**Three principles for parameter dependence on sequence length:**

1. Non-parameter properties (position embeddings, numerical precision) may depend on n.
2. Parameter count/values may depend on max length N. Must state scaling (O(N), O(log N), etc.).
3. Any dependence on n or N should have noted computational complexity. Preferably computable in poly(log N) time.

**Integer representation considerations:**
- Direct: Store C (unbounded; problematic with bounded position embeddings)
- Length-normalized: Store C/n (from unmasked uniform attention)
- Position-normalized: Store C/i (from future-masked uniform attention)
- Scale-invariant: Via layernorm hash `lh(C)`

**Key result (Hahn 2020):** With Lipschitz position-wise operations and bounded position embeddings, a change in a single input symbol produces at most O(1/n) change in any output activation.
