# Dive into Deep Learning - Curated Knowledge

Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola | 1185 pages | d2l.ai

---

## 1. PRELIMINARIES

### 1.1 Tensor Operations

**Creation:**
```python
x = torch.arange(12)              # [0, 1, ..., 11]
torch.zeros((2, 3, 4))            # all zeros, shape (2,3,4)
torch.ones((2, 3, 4))             # all ones
torch.randn(3, 4)                 # standard normal samples
torch.tensor([[2, 1, 4, 3],       # from nested Python list
              [1, 2, 3, 4],
              [4, 3, 2, 1]])
```

**Shape manipulation:**
```python
x.reshape(3, 4)      # reshape to (3,4) -- total elements must match
x.reshape(-1, 4)     # infer first dimension automatically
x.numel()            # total number of elements
x.shape              # dimension tuple
```

**Indexing and slicing:**
```python
X[-1]          # last row
X[1:3]         # rows 1 and 2 (exclusive end)
X[1, 2]        # element at row 1, col 2
X[1, 2] = 17   # write element
X[:2, :] = 12  # write to slice
```

**Elementwise operations:**
All standard arithmetic (+, -, *, /, **) operate elementwise on tensors of the same shape.

Unary: `torch.exp(x)` applies elementwise.

**Concatenation:**
```python
torch.cat((X, Y), dim=0)  # along rows (vertical stack)
torch.cat((X, Y), dim=1)  # along columns (horizontal stack)
```

**Logical operations:** `X == Y` produces elementwise boolean tensor.

**Summation:** `X.sum()` reduces all elements to scalar.

**Broadcasting:**
When operands have different shapes, arrays are expanded by copying along axes of length 1 to match shapes before elementwise operation.
```python
a = torch.arange(3).reshape((3, 1))  # shape (3,1)
b = torch.arange(2).reshape((1, 2))  # shape (1,2)
a + b  # broadcasts to shape (3,2)
```

**In-place operations (memory optimization):**
```python
Z[:] = X + Y          # write into pre-allocated Z
X += Y                 # in-place update
X[:] = X + Y           # in-place via slice assignment
```

**Conversion:**
```python
A = X.numpy()     # torch tensor -> numpy
B = torch.from_numpy(A)  # numpy -> torch tensor
a.item()           # size-1 tensor -> Python scalar
float(a)           # same
int(a)             # same
```

### 1.2 Linear Algebra

**Scalars:** Single numerical values. Denoted by lowercase (x, y, z). In R.

**Vectors:** Fixed-length arrays of scalars. Denoted by bold lowercase (**x**). Element x_i accessed by subscript.
- Length/dimensionality: number of elements.
- `len(x)`, `x.shape`

**Matrices:** 2D arrays. Denoted by bold uppercase (**A**). Element A_{ij} at row i, column j.
- Transpose: **A**^T where [A^T]_{ij} = A_{ji}
- Symmetric matrix: **A** = **A**^T
- `A.T` in code

**Tensors:** n-dimensional arrays with arbitrary number of axes. Denoted by bold uppercase with special font (e.g., **X**).

**Reduction operations:**
```python
A.sum()           # sum all elements -> scalar
A.sum(axis=0)     # sum along axis 0 (collapse rows) -> vector
A.sum(axis=1)     # sum along axis 1 (collapse columns) -> vector
A.sum(axis=[0,1]) # sum along both axes -> scalar
A.mean()          # mean of all elements
A.mean(axis=0)    # mean along axis 0
A.sum(axis=1, keepdims=True)  # keep dimension for broadcasting
A.cumsum(axis=0)  # cumulative sum along axis 0
```

**Dot product:**
Given vectors **x**, **y** in R^d:
$$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$$
```python
torch.dot(x, y)
torch.sum(x * y)  # equivalent
```

**Matrix-vector product:**
Given **A** in R^{m x n}, **x** in R^n:
$$\mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^\top \mathbf{x} \\ \mathbf{a}_2^\top \mathbf{x} \\ \vdots \\ \mathbf{a}_m^\top \mathbf{x} \end{bmatrix}$$
```python
torch.mv(A, x)
```

**Matrix-matrix multiplication:**
Given **A** in R^{n x k}, **B** in R^{k x m}:
$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$
Equivalently: C_{ij} = **a**_i^T **b**_j (dot product of i-th row of A with j-th column of B).
```python
torch.mm(A, B)
```

**Hadamard (elementwise) product:**
$$(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} B_{ij}$$

**Norms:**

L2 norm (Euclidean):
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$
```python
torch.norm(x)
```

L1 norm (Manhattan):
$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$
```python
torch.abs(x).sum()
```

Lp norm (general):
$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

Frobenius norm (for matrices):
$$\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} A_{ij}^2}$$
```python
torch.norm(A)  # Frobenius by default for matrices
```

### 1.3 Calculus

**Derivative definition:**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Differentiation rules:**
| Rule | Formula |
|---|---|
| Constant | d/dx [C] = 0 |
| Power | d/dx [x^n] = n x^{n-1} |
| Exponential | d/dx [e^x] = e^x |
| Logarithm | d/dx [ln x] = 1/x |
| Sum | d/dx [f(x) + g(x)] = f'(x) + g'(x) |
| Product | d/dx [f(x) g(x)] = f(x) g'(x) + f'(x) g(x) |
| Quotient | d/dx [f(x)/g(x)] = [g(x)f'(x) - f(x)g'(x)] / [g(x)]^2 |
| Chain | d/dx [f(g(x))] = f'(g(x)) g'(x) |

**Partial derivatives:**
For f: R^n -> R, the partial derivative with respect to x_i:
$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

**Gradient:**
$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top$$

**Gradient rules for multivariate functions:**
- For **A** in R^{m x n}, **x** in R^n:
  - nabla_x (**Ax**) = **A**^T
  - nabla_x (**x**^T **A**) = **A**
  - nabla_x (**x**^T **A** **x**) = (**A** + **A**^T)**x**
  - nabla_x ||**x**||^2 = 2**x**

**Chain rule (multivariate):**
For y = f(u_1, u_2, ..., u_m) where each u_k = g_k(x_1, ..., x_n):
$$\frac{\partial y}{\partial x_i} = \sum_{k=1}^{m} \frac{\partial y}{\partial u_k} \frac{\partial u_k}{\partial x_i}$$

### 1.4 Automatic Differentiation

**Concept:** Computational graph tracks operations. Forward pass computes values. Backward pass (reverse-mode autodiff) computes gradients via chain rule.

**PyTorch API:**
```python
x = torch.arange(4.0, requires_grad=True)  # enable gradient tracking
x.grad  # stores gradient after backward(), initially None

y = 2 * torch.dot(x, x)  # y = 2 * x^T x
y.backward()               # compute dy/dx
x.grad                     # contains 4*x (since d/dx[2x^Tx] = 4x)
```

**Gradient accumulation:** PyTorch accumulates gradients. Must zero before new computation:
```python
x.grad.zero_()
y = x.sum()       # y = sum(x_i)
y.backward()
x.grad             # all ones (d/dx_i [sum x_j] = 1)
```

**Non-scalar backward:** For non-scalar y, pass gradient tensor:
```python
y = x * x          # elementwise
y.backward(torch.ones(len(x)))  # equivalent to y.sum().backward()
```

**Detaching computation:**
```python
x.grad.zero_()
y = x * x
u = y.detach()    # treat u as constant (stops gradient flow)
z = u * x         # z = u*x where u is treated as constant
z.sum().backward()
x.grad == u       # True: dz/dx = u (since u treated as constant)
```

**Control flow:** Autodiff works through Python control flow (if/else, while loops). The computational graph is built dynamically per execution.

### 1.5 Probability and Statistics

**Sample space (S):** Set of all possible outcomes.

**Event:** Subset of sample space.

**Probability axioms:**
1. P(A) >= 0 for any event A
2. P(S) = 1
3. For mutually exclusive events A_1, A_2, ...: P(union A_i) = sum P(A_i)

**Conditional probability:**
$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

**Multiplication rule:**
$$P(A \cap B) = P(B \mid A) P(A) = P(A \mid B) P(B)$$

**Bayes' theorem:**
$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

**Law of total probability:**
$$P(B) = \sum_i P(B \mid A_i) P(A_i)$$

**Independence:**
Events A, B are independent iff P(A, B) = P(A) P(B).
Equivalently: P(A | B) = P(A).

**Random variable:** Function mapping outcomes to numerical values.

**Discrete probability distribution:**
$$P(X = x_i) = p_i, \quad \sum_i p_i = 1, \quad p_i \geq 0$$

**Continuous density function:**
$$P(a \leq X \leq b) = \int_a^b p(x)\,dx, \quad p(x) \geq 0, \quad \int_{-\infty}^{\infty} p(x)\,dx = 1$$

**Joint distribution:**
$$P(A = a, B = b) \leq P(A = a), \quad P(A = a, B = b) \leq P(B = b)$$

**Marginalization:**
$$P(B = b) = \sum_a P(A = a, B = b), \quad P(A = a) = \sum_b P(A = a, B = b)$$

**Expectation (mean):**
$$E[X] = \sum_x x \, P(X = x) \quad \text{(discrete)}$$
$$E[X] = \int x \, p(x)\,dx \quad \text{(continuous)}$$

**Linearity of expectation:** E[aX + b] = aE[X] + b

**Variance:**
$$\text{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - (E[X])^2$$

**Standard deviation:** sigma = sqrt(Var[X])

**Variance rules:** Var[aX + b] = a^2 Var[X]

**Covariance:**
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

**Covariance matrix:** For random vector **x** = (x_1, ..., x_n)^T:
$$\Sigma = E[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top]$$
where mu = E[**x**]. Diagonal entries are variances; off-diagonal are covariances. Sigma is symmetric positive semidefinite.

**Common distributions:**

Bernoulli: X in {0, 1}, P(X=1) = p, P(X=0) = 1-p.
- E[X] = p, Var[X] = p(1-p)

Multinomial: n trials, k outcomes with probabilities p_1,...,p_k.
- P(X_1=n_1,...,X_k=n_k) = (n! / (n_1!...n_k!)) * p_1^{n_1} ... p_k^{n_k}

Normal (Gaussian):
$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- E[X] = mu, Var[X] = sigma^2

---

## 2. LINEAR NEURAL NETWORKS FOR REGRESSION

### 2.1 Linear Regression Model

**Model:**
$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$
where:
- **x** in R^d: input feature vector
- **w** in R^d: weight vector (learnable parameters)
- b in R: bias (learnable parameter)
- y_hat in R: predicted output

**Vectorized form (batch):**
$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$$
where:
- **X** in R^{n x d}: design matrix (n examples, d features)
- **w** in R^d: weight vector
- b in R: bias (broadcast across n examples)
- **y_hat** in R^n: predictions

### 2.2 Loss Function

**Squared error (single example):**
$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$
The 1/2 factor is for convenience (cancels with derivative).

**Mean squared error (full dataset):**
$$L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} l^{(i)}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2$$

### 2.3 Analytic Solution

$$\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$
where:
- **X** in R^{n x d}: design matrix
- **y** in R^n: target vector
- Only exists when X^T X is invertible (full column rank)

### 2.4 Minibatch Stochastic Gradient Descent (SGD)

**Algorithm:**
1. Initialize parameters (w, b) (e.g., random)
2. Repeat until convergence:
   a. Randomly sample minibatch B of size |B| from training set
   b. Compute gradient of loss averaged over minibatch
   c. Update parameters:

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w}, b)} l^{(i)}(\mathbf{w}, b)$$

where:
- eta > 0: learning rate (hyperparameter)
- |B|: minibatch size (hyperparameter)
- The sum is over examples i in the randomly sampled minibatch B

**Expanded update rules:**
$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)$$
$$b \leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)$$

**Hyperparameters:** Learning rate eta, minibatch size |B|, number of epochs. These are tuned, not learned.

### 2.5 Normal Distribution and Maximum Likelihood

**Connection:** If we assume target y is generated as:
$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
then the likelihood of observing y given x is:
$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \mathbf{w}^\top \mathbf{x} - b)^2}{2\sigma^2}\right)$$

**Maximum likelihood estimation (MLE):**
$$\hat{\mathbf{w}}, \hat{b} = \arg\max_{\mathbf{w}, b} \prod_{i=1}^{n} P\left(y^{(i)} \mid \mathbf{x}^{(i)}\right)$$

Taking negative log-likelihood:
$$-\log P(\mathbf{y} \mid \mathbf{X}) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{n} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2$$

**Result:** Minimizing MSE is equivalent to maximum likelihood estimation under Gaussian noise assumption.

### 2.6 Training Pipeline (PyTorch)

```python
# Data
@d2l.add_to_class(d2l.DataModule)
def get_dataloader(self, train):
    dataset = torch.utils.data.TensorDataset(*self.get_tensorloader(train))
    return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

# Model definition
class LinearRegression(d2l.Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
```

### 2.7 Weight Decay (L2 Regularization)

**Regularized loss:**
$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$
where:
- lambda >= 0: regularization strength (hyperparameter)
- ||**w**||^2 = sum w_j^2: squared L2 norm of weights
- Bias b is typically NOT regularized

**Gradient of regularized objective:**
$$\frac{\partial}{\partial \mathbf{w}} \left(L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|^2\right) = \frac{\partial L}{\partial \mathbf{w}} + \lambda \mathbf{w}$$

**Update rule with weight decay:**
$$\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{\partial l^{(i)}}{\partial \mathbf{w}}$$
where:
- (1 - eta*lambda): shrinkage factor -- weights are multiplicatively reduced each step
- This is why it is called "weight decay"

**PyTorch implementation:**
```python
# weight_decay parameter in optimizer applies L2 penalty
trainer = torch.optim.SGD([
    {'params': net.weight, 'weight_decay': wd},
    {'params': net.bias}], lr=lr)
```

---

## 3. LINEAR NEURAL NETWORKS FOR CLASSIFICATION

### 3.1 Softmax Regression

**Setup:** Classification into q classes. Output is probability distribution over classes.

**Model:**
$$\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$$
where:
- **x** in R^d: input features
- **W** in R^{q x d}: weight matrix
- **b** in R^q: bias vector
- **o** in R^q: unnormalized log-probabilities (logits)

**Softmax function:**
$$\hat{y}_j = \text{softmax}(\mathbf{o})_j = \frac{\exp(o_j)}{\sum_{k=1}^{q} \exp(o_k)}$$
where:
- o_j: logit for class j
- y_hat_j: predicted probability for class j
- sum_j y_hat_j = 1, y_hat_j >= 0

**Properties:**
- arg max_j y_hat_j = arg max_j o_j (softmax preserves ordering)
- Softmax is a differentiable approximation to argmax

**Vectorized (batch):**
$$\mathbf{O} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}, \quad \hat{\mathbf{Y}} = \text{softmax}(\mathbf{O})$$
where:
- **X** in R^{n x d}: batch of n examples
- **O** in R^{n x q}: logits matrix
- **Y_hat** in R^{n x q}: predicted probabilities

### 3.2 Loss Function: Cross-Entropy

**One-hot encoding:** True label y as vector **y** in {0,1}^q with exactly one 1.

**Cross-entropy loss (single example):**
$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^{q} y_j \log \hat{y}_j$$

Since **y** is one-hot (y_j = 1 only for true class y):
$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_y$$

**Gradient with respect to logits:**
$$\frac{\partial l}{\partial o_j} = \text{softmax}(\mathbf{o})_j - y_j = \hat{y}_j - y_j$$

This gradient has an elegant form: the difference between the predicted probability and the true label (0 or 1).

**Maximum likelihood interpretation:**
Softmax + cross-entropy loss is equivalent to maximum likelihood estimation for the categorical distribution.

### 3.3 Information Theory Connections

**Self-information:**
$$I(x) = -\log P(x)$$
where P(x) is the probability of event x.

**Entropy:**
$$H[P] = -\sum_j P(j) \log P(j) = E_{x \sim P}[-\log P(x)]$$
where:
- P is a probability distribution
- H[P] measures the expected surprise / minimum average bits needed to encode samples from P
- Maximum entropy: uniform distribution
- Lower bound: H[P] >= 0

**Cross-entropy:**
$$H(P, Q) = -\sum_j P(j) \log Q(j)$$
where:
- P: true distribution
- Q: estimated distribution
- H(P, Q) >= H(P) (always at least as large as entropy)
- H(P, Q) = H(P) iff P = Q

**KL divergence:**
$$D_{KL}(P \| Q) = H(P, Q) - H(P) = \sum_j P(j) \log \frac{P(j)}{Q(j)}$$
- D_KL >= 0, with equality iff P = Q
- Minimizing cross-entropy H(P, Q) w.r.t. Q is equivalent to minimizing D_KL(P || Q) since H(P) is constant

### 3.4 Numerical Stability: Log-Sum-Exp Trick

**Problem:** Direct computation of softmax overflows/underflows for large/small logits.

**Solution:** Subtract max logit before exponentiation:
$$\log \hat{y}_j = o_j - \max_k(o_k) - \log \sum_k \exp(o_k - \max_k(o_k))$$

This is equivalent mathematically but numerically stable because:
- The largest exponent argument is 0
- All other exponent arguments are negative (no overflow)
- At least one term in the sum is exp(0) = 1 (no underflow in log)

**PyTorch combined function:**
```python
loss = nn.CrossEntropyLoss()     # takes raw logits, not softmax output
loss(o, y)                        # applies log-sum-exp internally
```

### 3.5 Classification Accuracy

$$\text{accuracy} = \frac{\text{number correct}}{\text{total examples}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(\hat{y}^{(i)} = y^{(i)})$$

where 1(.) is the indicator function.

Not used as training objective because it is not differentiable.

---

## 4. MULTILAYER PERCEPTRONS (MLPs)

### 4.1 Hidden Layers

**Problem with linear models:** Composition of linear transformations is still linear. For inputs **X**, weights **W**^(1), **W**^(2):
$$\mathbf{O} = (\mathbf{X}\mathbf{W}^{(1)})\mathbf{W}^{(2)} = \mathbf{X}\mathbf{W}$$
where **W** = **W**^(1)**W**^(2). No increase in model capacity.

**Solution:** Apply nonlinear activation function sigma(.) after each linear layer.

**Single hidden layer MLP:**
$$\mathbf{H} = \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$$
$$\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$
where:
- **X** in R^{n x d}: input (n examples, d features)
- **W**^(1) in R^{d x h}: input-to-hidden weights
- **b**^(1) in R^h: hidden biases
- **H** in R^{n x h}: hidden activations
- **W**^(2) in R^{h x q}: hidden-to-output weights
- **b**^(2) in R^q: output biases
- **O** in R^{n x q}: output
- sigma(.): activation function applied elementwise
- h: number of hidden units (hyperparameter)

**Multiple hidden layers:**
$$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$$
$$\mathbf{H}^{(l)} = \sigma_l(\mathbf{H}^{(l-1)}\mathbf{W}^{(l)} + \mathbf{b}^{(l)})$$
$$\mathbf{O} = \mathbf{H}^{(L)}\mathbf{W}^{(L+1)} + \mathbf{b}^{(L+1)}$$

### 4.2 Activation Functions

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(x) = \max(0, x)$$
- Derivative: 0 if x < 0, 1 if x > 0, undefined at x = 0 (convention: 0)
- Advantages: Mitigates vanishing gradients, computationally efficient
- Disadvantage: Dead neurons (units stuck at 0 for all inputs)

**Parametric ReLU (pReLU):**
$$\text{pReLU}(x) = \max(0, x) + \alpha \min(0, x)$$
where alpha is a learnable parameter.

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$
- Range: (0, 1)
- Derivative: sigma(x)(1 - sigma(x))
- Maximum derivative at x = 0: sigma(0)(1-sigma(0)) = 0.25
- Problem: Gradient vanishes for large |x|

**Tanh (Hyperbolic Tangent):**
$$\tanh(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)} = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$$
- Range: (-1, 1)
- Derivative: 1 - tanh^2(x)
- Zero-centered (unlike sigmoid)
- Relation to sigmoid: tanh(x) = 2*sigma(2x) - 1

**GELU (Gaussian Error Linear Unit):**
$$\text{GELU}(x) = x \cdot \Phi(x)$$
where Phi(x) is the CDF of the standard normal distribution.

**SiLU / Swish:**
$$\text{SiLU}(x) = x \cdot \sigma(x)$$

### 4.3 Universal Approximation Theorem

A single hidden layer MLP with sufficient hidden units can approximate any continuous function on compact subsets of R^n to arbitrary accuracy (Cybenko, 1989; Hornik, 1991).

Caveats:
- Does not specify how many hidden units are needed
- Does not guarantee the network is learnable via SGD
- Deeper networks can be exponentially more parameter-efficient than wider shallow ones

### 4.4 Forward Propagation

**Computation graph (single hidden layer with L2 regularization):**

Given input **x**, target y, weights **W**^(1), **W**^(2):

1. Intermediate: **z** = **W**^(1)**x**
2. Hidden: **h** = phi(**z**) where phi is the activation function
3. Output: o = **W**^(2)**h**
4. Loss: L = l(o, y) (e.g., cross-entropy or MSE)
5. Regularization: s = lambda/2 (||**W**^(1)||_F^2 + ||**W**^(2)||_F^2)
6. Objective: J = L + s

### 4.5 Backpropagation

**Compute gradients in reverse order using chain rule:**

$$\frac{\partial J}{\partial L} = 1, \quad \frac{\partial J}{\partial s} = 1$$

$$\frac{\partial J}{\partial o} = \frac{\partial J}{\partial L} \cdot \frac{\partial L}{\partial o} = \frac{\partial L}{\partial o}$$

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}} = \frac{\partial J}{\partial o} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}$$

$$\frac{\partial J}{\partial \mathbf{h}} = {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial o}$$

$$\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{h}} \odot \phi'(\mathbf{z})$$

$$\frac{\partial J}{\partial \mathbf{W}^{(1)}} = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}$$

where:
- The Hadamard product with phi'(**z**) applies the activation derivative elementwise
- Lambda terms come from the regularization gradient

**Computational cost:**
- Forward: O(n) multiplications per layer (n = number of parameters)
- Backward: approximately 2x forward cost
- Memory: must store all intermediate activations for backward pass

### 4.6 Numerical Stability

**Vanishing gradients:**
For a deep network with L layers, the gradient involves a product of L-l matrices:
$$\frac{\partial \mathbf{h}^{(L)}}{\partial \mathbf{h}^{(l)}} = \prod_{t=l}^{L-1} \frac{\partial \mathbf{h}^{(t+1)}}{\partial \mathbf{h}^{(t)}}$$

If gradients of individual layers are < 1, the product vanishes exponentially.

**Sigmoid problem:** sigmoid'(x) has max value 0.25. After L layers: 0.25^L -> 0 rapidly. Gradients vanish, training stalls.

**Exploding gradients:** If gradients > 1, the product explodes exponentially. Can cause NaN values, divergence.

**Symmetry breaking:** If all weights initialized identically, all hidden units compute the same function. Gradients are identical, so weights stay identical forever. Random initialization breaks this symmetry.

### 4.7 Parameter Initialization

**Default (PyTorch):** Random initialization, framework-specific.

**Xavier (Glorot) Initialization:**
Maintain variance of activations and gradients across layers.

Condition: For layer with n_in inputs and n_out outputs, if weights are i.i.d. with mean 0 and variance sigma^2:
- Forward variance preservation: n_in * sigma^2 = 1
- Backward variance preservation: n_out * sigma^2 = 1
- Compromise: sigma^2 = 2 / (n_in + n_out)

**Gaussian Xavier:**
$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

**Uniform Xavier:**
$$W_{ij} \sim U\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

Derivation assumes linear activations; in practice works well for tanh and reasonably for ReLU.

**He (Kaiming) Initialization (for ReLU):**
$$\sigma^2 = \frac{2}{n_{\text{in}}}$$

### 4.8 Dropout Regularization

**Training time:**
$$h' = \begin{cases} 0 & \text{with probability } p \\ \frac{h}{1-p} & \text{with probability } 1-p \end{cases}$$
where:
- h: original activation
- h': dropout-modified activation
- p: dropout probability (hyperparameter, typically 0.1 to 0.5)
- Division by (1-p): ensures E[h'] = h (expectation is preserved)

**Test time:** Dropout is disabled. All activations used directly (no scaling needed due to the 1/(1-p) factor during training).

**Properties:**
- Acts as regularization by injecting noise
- Prevents co-adaptation of hidden units
- Applied independently to each activation, each training example, each iteration
- Typically applied after activation, before next linear layer
- Common pattern: higher dropout in later layers

**PyTorch implementation:**
```python
net = nn.Sequential(
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.2),
    nn.LazyLinear(10))
```

### 4.9 Environment and Distribution Shift

**Types of distribution shift:**

- **Covariate shift:** P_train(x) != P_test(x), but P(y|x) unchanged. Correction via importance weighting: weight each training loss by P_test(x)/P_train(x).

- **Label shift:** P_train(y) != P_test(y), but P(x|y) unchanged. Correction: weight by P_test(y)/P_train(y).

- **Concept drift:** P(y|x) changes over time. Requires model retraining.

**Importance weighting correction for covariate shift:**
$$\min_f \frac{1}{n} \sum_{i=1}^{n} \frac{P_{\text{test}}(\mathbf{x}^{(i)})}{P_{\text{train}}(\mathbf{x}^{(i)})} l(f(\mathbf{x}^{(i)}), y^{(i)})$$

---

## 5. BUILDERS' GUIDE (PyTorch)

### 5.1 Layers and Modules

**nn.Module:** Base class for all neural network modules in PyTorch.

**Custom module pattern:**
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

**nn.Sequential:** Container that chains modules in order.
```python
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X)  # automatically calls forward through each layer
```

**Custom Sequential:**
```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
```

**Arbitrary forward logic:** Modules can contain any Python control flow in forward().
```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20))  # non-trainable
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        X = self.linear(X)  # reuse same layer (shared parameters)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

### 5.2 Parameter Management

**Access parameters:**
```python
net[2].state_dict()                    # layer's state dict
net[2].bias                            # Parameter object
net[2].bias.data                       # raw tensor value
net[2].weight.grad                     # gradient (after backward)
```

**Named parameters (all at once):**
```python
[(name, param.shape) for name, param in net.named_parameters()]
# e.g., [('0.weight', (256, 20)), ('0.bias', (256,)), ('2.weight', (10, 256)), ('2.bias', (10,))]
```

**Nested module access:**
```python
# For nested Sequential blocks:
rgnet[0][1][0].bias.data
```

### 5.3 Parameter Initialization

**Built-in initializers:**
```python
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
net.apply(init_normal)

def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)
net.apply(init_constant)

# Xavier uniform
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

# Kaiming (He) for ReLU
def init_kaiming(module):
    if type(module) == nn.Linear:
        nn.init.kaiming_uniform_(module.weight)
```

**Per-layer initialization:**
```python
net[0].apply(init_xavier)     # Xavier for first layer
net[2].apply(init_kaiming)    # Kaiming for third layer
```

**Custom initialization:**
```python
def my_init(module):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5  # zero out |w| < 5
net.apply(my_init)
```

**Direct parameter manipulation:**
```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
```

### 5.4 Tied (Shared) Parameters

```python
shared = nn.LazyLinear(8)
net = nn.Sequential(
    nn.LazyLinear(8), nn.ReLU(),
    shared, nn.ReLU(),
    shared, nn.ReLU(),        # same layer object reused
    nn.LazyLinear(1))
```
- Shared layers have identical parameters (same memory)
- Gradients from both usages are summed during backward pass

### 5.5 Custom Layers

**Without parameters:**
```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

**With parameters:**
```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```
- `nn.Parameter` wraps tensor and registers it for gradient computation and optimizer updates

### 5.6 Lazy Initialization

`nn.LazyLinear(out_features)`: Defers weight initialization until first forward pass, when input shape is known.
- Eliminates need to manually specify input dimensions
- Shape inference happens automatically on first call

### 5.7 File I/O

**Save/load tensors:**
```python
torch.save(x, 'x-file')           # save single tensor
x2 = torch.load('x-file')         # load tensor

torch.save([x, y], 'x-files')     # save list of tensors
x2, y2 = torch.load('x-files')

torch.save({'x': x, 'y': y}, 'mydict')  # save dict
mydict2 = torch.load('mydict')
```

**Save/load model parameters:**
```python
# Save
torch.save(net.state_dict(), 'mlp.params')

# Load (must create model architecture first)
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```
- Saves parameters only, NOT model architecture
- Must reconstruct the model class in code, then load parameters into it

### 5.8 GPU Usage

**Device specification:**
```python
torch.device('cpu')       # CPU device
torch.device('cuda:0')    # first GPU
torch.device('cuda:1')    # second GPU
torch.cuda.device_count() # number of available GPUs
```

**Tensor placement:**
```python
X = torch.ones(2, 3, device=torch.device('cuda:0'))  # create on GPU
x.device                  # query device
Z = X.cuda(1)             # copy tensor to GPU 1
```

**Rule:** All operands must be on the same device. Cross-device operations raise errors.

**Model placement:**
```python
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=torch.device('cuda:0'))  # move model to GPU
```

**Performance considerations:**
- GPU-CPU transfers are slow; minimize them
- Many small transfers are worse than one large transfer
- Avoid logging scalar losses to CPU every iteration (triggers synchronization)
- Keep data and model on the same device throughout training

**Trainer GPU support pattern:**
```python
def prepare_batch(self, batch):
    if self.gpus:
        batch = [a.to(self.gpus[0]) for a in batch]
    return batch

def prepare_model(self, model):
    if self.gpus:
        model.to(self.gpus[0])
    self.model = model
```

---

## Raw Algorithms, Equations, Architectures, and Technical Knowledge

---

# 1. CONVOLUTIONAL NEURAL NETWORKS -- CORE OPERATIONS

## 1.1 Cross-Correlation (Convolution) Operation

The fundamental operation in CNNs is **cross-correlation** (commonly called "convolution" in deep learning, though mathematically it omits the kernel flip).

**2D Cross-Correlation:**

```
(X * K)(i, j) = sum_{a=0}^{k_h - 1} sum_{b=0}^{k_w - 1} X(i + a, j + b) * K(a, b)
```

Where:
- `X` = input tensor (height x width)
- `K` = kernel/filter tensor (k_h x k_w)
- `(i, j)` = spatial position in output
- `k_h, k_w` = kernel height and width

**Output dimensions (no padding, stride=1):**
```
Output height = n_h - k_h + 1
Output width  = n_w - k_w + 1
```

Where:
- `n_h, n_w` = input height and width
- `k_h, k_w` = kernel height and width

## 1.2 Padding and Stride

**General output size formula with padding and stride:**

```
Output_h = floor((n_h - k_h + p_h + s_h) / s_h)
Output_w = floor((n_w - k_w + p_w + s_w) / s_w)
```

Where:
- `n_h, n_w` = input height, width
- `k_h, k_w` = kernel height, width
- `p_h` = total vertical padding (top + bottom)
- `p_w` = total horizontal padding (left + right)
- `s_h, s_w` = vertical stride, horizontal stride

**Common padding choice to preserve spatial dimensions (stride=1):**
```
p_h = k_h - 1,  p_w = k_w - 1
```
With odd kernel size k, pad k//2 on each side.

## 1.3 Multiple Input and Output Channels

**Multiple input channels (c_i input channels):**
```
Y(i, j) = sum_{c=0}^{c_i - 1} (X_c * K_c)(i, j) + b
```
Each input channel has its own kernel slice. Results are summed across channels.

**Multiple output channels:** Stack `c_o` independent filters, each of shape `(c_i, k_h, k_w)`.
- Full kernel tensor shape: `(c_o, c_i, k_h, k_w)`
- Parameters per conv layer: `c_o * c_i * k_h * k_w + c_o` (weights + biases)

## 1.4 1x1 Convolutions (Channel Mixing / Pointwise Convolution)

Kernel size = 1x1. Acts as a **fully connected layer applied independently at each spatial location** across channels.

```
Y(i, j, :) = W * X(i, j, :) + b
```

Where:
- `W` has shape `(c_o, c_i)` -- mixes channels without spatial interaction
- Used to: change channel count, add nonlinearity across channels, reduce dimensionality

Parameters: `c_o * c_i + c_o`

## 1.5 Pooling Operations

**Max Pooling:**
```
Y(i, j) = max_{a=0..p_h-1, b=0..p_w-1} X(i*s_h + a, j*s_w + b)
```

**Average Pooling:**
```
Y(i, j) = (1 / (p_h * p_w)) * sum_{a=0}^{p_h-1} sum_{b=0}^{p_w-1} X(i*s_h + a, j*s_w + b)
```

Where:
- `p_h, p_w` = pooling window height, width
- `s_h, s_w` = stride height, width

**Global Average Pooling:** Pool over entire spatial dimension, producing one value per channel.
```
Y(c) = (1 / (H * W)) * sum_{i=0}^{H-1} sum_{j=0}^{W-1} X(c, i, j)
```
Replaces fully connected layers at network head (introduced by NiN).

Output size formula for pooling is identical to the convolution output size formula.

## 1.6 Batch Normalization

**Core formula (Eq. 8.5.1):**
```
BN(x) = gamma * ((x - mu_hat_B) / sigma_hat_B) + beta
```

Where:
- `x` = input (element of minibatch B)
- `mu_hat_B` = sample mean of minibatch B
- `sigma_hat_B` = sample standard deviation of minibatch B
- `gamma` = learnable scale parameter (initialized to 1)
- `beta` = learnable shift parameter (initialized to 0)
- `*` denotes elementwise (Hadamard) product

**Minibatch statistics (Eq. 8.5.2):**
```
mu_hat_B    = (1 / |B|) * sum_{x in B} x
sigma_hat_B^2 = (1 / |B|) * sum_{x in B} (x - mu_hat_B)^2 + epsilon
```

Where:
- `|B|` = minibatch size
- `epsilon` > 0 = small constant to prevent division by zero (typically 1e-5)

**For fully connected layers:** normalize over feature dimension (dim=0).
- Placement: `h = phi(BN(Wx + b))` -- BN after affine transform, before activation.

**For convolutional layers:** normalize per channel across all spatial locations and batch elements.
- Compute mean/variance over `m * p * q` elements per channel (m=batch, p=height, q=width).
- Each channel has its own scalar gamma and beta.

**Training vs. Inference:**

| Mode | Mean/Variance Source |
|------|---------------------|
| Training | Computed from current minibatch |
| Inference | Running/moving average accumulated during training |

**Moving average update during training:**
```
moving_mean = (1 - momentum) * moving_mean + momentum * batch_mean
moving_var  = (1 - momentum) * moving_var  + momentum * batch_var
```
Where `momentum` is typically 0.1.

**Learnable parameters per BN layer:** `2 * C` (gamma and beta per channel, where C = number of channels).

## 1.7 Layer Normalization

**Formula (Eq. 8.5.4):**
```
LN(x) = (x - mu_hat) / sigma_hat
```

**Statistics computed per single observation (Eq. 8.5.5):**
```
mu_hat    = (1/n) * sum_{i=1}^{n} x_i
sigma_hat^2 = (1/n) * sum_{i=1}^{n} (x_i - mu_hat)^2 + epsilon
```

Where:
- `n` = number of features in vector x
- `x_i` = i-th element of the feature vector

**Key properties:**
- Scale-invariant: `LN(x) ~ LN(alpha * x)` for any alpha != 0
- Does NOT depend on minibatch size
- Identical behavior in training and inference (deterministic)
- No running averages needed

---

# 2. MODERN CNN ARCHITECTURES

## 2.1 LeNet-5 (LeCun et al., 1995)

**Key innovation:** First successful CNN for handwritten digit recognition (MNIST). Demonstrated that learned features via backpropagation outperform hand-crafted features.

**Architecture (input: 1x28x28 grayscale):**

| Layer | Type | Kernel | Stride | Padding | Output Shape | Parameters |
|-------|------|--------|--------|---------|-------------|------------|
| Input | -- | -- | -- | -- | 1x28x28 | 0 |
| Conv1 | Conv2D | 5x5 | 1 | 0 | 6x24x24 | 6*(1*25+1) = 156 |
| Act1 | Sigmoid | -- | -- | -- | 6x24x24 | 0 |
| Pool1 | AvgPool | 2x2 | 2 | 0 | 6x12x12 | 0 |
| Conv2 | Conv2D | 5x5 | 1 | 0 | 16x8x8 | 16*(6*25+1) = 2,416 |
| Act2 | Sigmoid | -- | -- | -- | 16x8x8 | 0 |
| Pool2 | AvgPool | 2x2 | 2 | 0 | 16x4x4 | 0 |
| Flatten | -- | -- | -- | -- | 256 | 0 |
| FC1 | Linear | -- | -- | -- | 120 | 256*120+120 = 30,840 |
| Act3 | Sigmoid | -- | -- | -- | 120 | 0 |
| FC2 | Linear | -- | -- | -- | 84 | 120*84+84 = 10,164 |
| Act4 | Sigmoid | -- | -- | -- | 84 | 0 |
| FC3 | Linear | -- | -- | -- | 10 | 84*10+10 = 850 |

**Total parameters: ~44,426**

**Key characteristics:**
- Sigmoid activations (later replaced by ReLU in modern nets)
- Average pooling (later replaced by max pooling)
- No dropout, no batch normalization
- Pattern: [Conv -> Act -> Pool] x 2 -> Flatten -> FC x 3

---

## 2.2 AlexNet (Krizhevsky et al., 2012)

**Key innovations:**
1. **ReLU activation** instead of sigmoid (avoids vanishing gradients in positive region; gradient always = 1 for positive inputs)
2. **Dropout** (p=0.5) for regularization on FC layers
3. **Data augmentation** (flipping, clipping, color changes)
4. **GPU training** (originally split across two NVIDIA GTX 580s)
5. Trained on ImageNet (1M images, 1000 classes, 224x224)

**Architecture (input: 1x224x224, streamlined version):**

| Layer | Type | Kernel | Stride | Padding | Output Shape | Params (approx) |
|-------|------|--------|--------|---------|-------------|-----------------|
| Input | -- | -- | -- | -- | 1x224x224 | 0 |
| Conv1 | Conv2D | 11x11 | 4 | 1 | 96x54x54 | 96*(1*121+1) = 11,712 |
| Act1 | ReLU | -- | -- | -- | 96x54x54 | 0 |
| Pool1 | MaxPool | 3x3 | 2 | 0 | 96x26x26 | 0 |
| Conv2 | Conv2D | 5x5 | 1 | 2 | 256x26x26 | 256*(96*25+1) = 614,656 |
| Act2 | ReLU | -- | -- | -- | 256x26x26 | 0 |
| Pool2 | MaxPool | 3x3 | 2 | 0 | 256x12x12 | 0 |
| Conv3 | Conv2D | 3x3 | 1 | 1 | 384x12x12 | 384*(256*9+1) = 885,120 |
| Act3 | ReLU | -- | -- | -- | 384x12x12 | 0 |
| Conv4 | Conv2D | 3x3 | 1 | 1 | 384x12x12 | 384*(384*9+1) = 1,327,488 |
| Act4 | ReLU | -- | -- | -- | 384x12x12 | 0 |
| Conv5 | Conv2D | 3x3 | 1 | 1 | 256x12x12 | 256*(384*9+1) = 885,248 |
| Act5 | ReLU | -- | -- | -- | 256x12x12 | 0 |
| Pool3 | MaxPool | 3x3 | 2 | 0 | 256x5x5 | 0 |
| Flatten | -- | -- | -- | -- | 6400 | 0 |
| FC1 | Linear | -- | -- | -- | 4096 | 6400*4096+4096 = 26,218,496 |
| Act6 | ReLU | -- | -- | -- | 4096 | 0 |
| Drop1 | Dropout(0.5) | -- | -- | -- | 4096 | 0 |
| FC2 | Linear | -- | -- | -- | 4096 | 4096*4096+4096 = 16,781,312 |
| Act7 | ReLU | -- | -- | -- | 4096 | 0 |
| Drop2 | Dropout(0.5) | -- | -- | -- | 4096 | 0 |
| FC3 | Linear | -- | -- | -- | num_classes | 4096*C+C |

**Total parameters: ~46.7M (for 10 classes) / ~62M (for 1000 classes)**

**Efficiency bottleneck:** FC layers dominate parameters. FC1 weight matrix alone is 6400x4096 = ~100MB (FP32). FC1+FC2 require ~164MB memory and ~81 MFLOPs.

---

## 2.3 VGG (Simonyan & Zisserman, 2014)

**Key innovation:** Systematic use of **repeated blocks** of small (3x3) convolutions. Demonstrated that **deep and narrow** networks outperform shallow and wide ones. Two stacked 3x3 convs have same receptive field as one 5x5 conv but fewer parameters: `2 * (9 * c^2)` = `18c^2` vs `25c^2`.

**VGG Block structure:**
```
VGG_Block(num_convs, out_channels):
    for i in range(num_convs):
        Conv2D(out_channels, kernel=3x3, padding=1)  # preserves spatial dims
        ReLU()
    MaxPool2D(kernel=2x2, stride=2)  # halves spatial dims
```

**VGG Family configurations (all use 224x224 input, 3 FC layers at end: 4096->4096->num_classes):**

| Variant | Block Config (num_convs, channels) | Conv Layers | Total Layers | Params |
|---------|-----------------------------------|-------------|-------------|--------|
| VGG-11 | (1,64), (1,128), (2,256), (2,512), (2,512) | 8 | 11 | ~133M |
| VGG-13 | (2,64), (2,128), (2,256), (2,512), (2,512) | 10 | 13 | ~133M |
| VGG-16 | (2,64), (2,128), (3,256), (3,512), (3,512) | 13 | 16 | ~138M |
| VGG-19 | (2,64), (2,128), (4,256), (4,512), (4,512) | 16 | 19 | ~144M |

**VGG-11 layer-by-layer output shapes (input 1x224x224):**

| Stage | Output Shape |
|-------|-------------|
| Block 1 (1 conv, 64 ch) | 64x112x112 |
| Block 2 (1 conv, 128 ch) | 128x56x56 |
| Block 3 (2 conv, 256 ch) | 256x28x28 |
| Block 4 (2 conv, 512 ch) | 512x14x14 |
| Block 5 (2 conv, 512 ch) | 512x7x7 |
| Flatten | 25088 |
| FC1 | 4096 |
| FC2 | 4096 |
| FC3 | num_classes |

**Design principle:** Each block doubles channels (64->128->256->512) while halving spatial resolution (224->112->56->28->14->7). Maximum log2(224) ~ 7-8 downsampling stages possible.

---

## 2.4 NiN -- Network in Network (Lin et al., 2013)

**Key innovations:**
1. **1x1 convolutions** after spatial convolutions to add cross-channel nonlinearity (acts as per-pixel fully connected layer)
2. **Global Average Pooling (GAP)** replaces FC layers at network head -- dramatically reduces parameters

**NiN Block structure:**
```
NiN_Block(out_channels, kernel_size, strides, padding):
    Conv2D(out_channels, kernel_size, strides, padding)
    ReLU()
    Conv2D(out_channels, kernel=1x1)    # channel mixing
    ReLU()
    Conv2D(out_channels, kernel=1x1)    # channel mixing
    ReLU()
```

**Architecture (input: 1x224x224):**

| Layer | Output Shape | Notes |
|-------|-------------|-------|
| NiN Block 1 (96, k=11, s=4, p=0) | 96x54x54 | |
| MaxPool (3x3, s=2) | 96x26x26 | |
| NiN Block 2 (256, k=5, s=1, p=2) | 256x26x26 | |
| MaxPool (3x3, s=2) | 256x12x12 | |
| NiN Block 3 (384, k=3, s=1, p=1) | 384x12x12 | |
| MaxPool (3x3, s=2) | 384x5x5 | |
| Dropout(0.5) | 384x5x5 | |
| NiN Block 4 (num_classes, k=3, s=1, p=1) | 10x5x5 | Output channels = num_classes |
| Global AvgPool | 10x1x1 | Averages over all spatial locations |
| Flatten | 10 | |

**Total parameters: dramatically fewer than AlexNet/VGG** (no FC layers; final NiN block outputs num_classes channels directly).

**Why GAP works:** Averaging across a low-resolution representation with many channels adds translation invariance. No learned parameters in the pooling head.

---

## 2.5 GoogLeNet / Inception (Szegedy et al., 2015)

**Key innovation:** **Inception block** -- parallel multi-branch convolutions with different kernel sizes, concatenated along channel dimension. Solves the kernel-size selection problem by using all sizes simultaneously. Introduced **stem/body/head** design pattern.

**Inception Block (4 parallel branches):**
```
Branch 1: Conv 1x1 (c1 channels)
Branch 2: Conv 1x1 (c2[0] channels) -> Conv 3x3 (c2[1] channels, pad=1)
Branch 3: Conv 1x1 (c3[0] channels) -> Conv 5x5 (c3[1] channels, pad=2)
Branch 4: MaxPool 3x3 (stride=1, pad=1) -> Conv 1x1 (c4 channels)

Output = Concat(Branch1, Branch2, Branch3, Branch4) along channel dim
Output channels = c1 + c2[1] + c3[1] + c4
```

All branches use appropriate padding to maintain identical spatial dimensions. The 1x1 convolutions in branches 2 and 3 serve as **dimensionality reduction** (bottleneck) to control computational cost.

**GoogLeNet Full Architecture (input: 1x96x96 shown; designed for 224x224):**

| Module | Structure | Output Shape (96x96 input) |
|--------|-----------|---------------------------|
| **Stem (b1)** | Conv 7x7 (64ch, s=2, p=3) -> ReLU -> MaxPool 3x3 (s=2, p=1) | 64x24x24 |
| **b2** | Conv 1x1 (64ch) -> ReLU -> Conv 3x3 (192ch, p=1) -> ReLU -> MaxPool 3x3 (s=2, p=1) | 192x12x12 |
| **b3** | Inception(64,(96,128),(16,32),32) -> Inception(128,(128,192),(32,96),64) -> MaxPool 3x3 (s=2, p=1) | 480x6x6 |
| **b4** | Inception(192,(96,208),(16,48),64) -> Inception(160,(112,224),(24,64),64) -> Inception(128,(128,256),(24,64),64) -> Inception(112,(144,288),(32,64),64) -> Inception(256,(160,320),(32,128),128) -> MaxPool 3x3 (s=2, p=1) | 832x3x3 |
| **b5** | Inception(256,(160,320),(32,128),128) -> Inception(384,(192,384),(48,128),128) -> Global AvgPool -> Flatten | 1024 |
| **Head** | Linear(num_classes) | num_classes |

**Inception block channel breakdown (b3, first block):**
- Branch 1: 1x1 -> 64 channels
- Branch 2: 1x1 (96ch) -> 3x3 -> 128 channels
- Branch 3: 1x1 (16ch) -> 5x5 -> 32 channels
- Branch 4: MaxPool -> 1x1 -> 32 channels
- **Total: 64 + 128 + 32 + 32 = 256 channels**

**Total inception blocks: 9** (2 in b3, 5 in b4, 2 in b5)

**Key characteristics:**
- Cheaper to compute than predecessors despite higher accuracy
- 1x1 convolutions reduce channel dimensions before expensive 3x3 and 5x5 convolutions
- Uses Global Average Pooling (from NiN) instead of large FC layers
- Original paper included auxiliary classifiers for training stability (no longer needed with modern optimizers)

---

## 2.6 ResNet -- Residual Networks (He et al., 2016)

**Key innovation:** **Residual connections (skip connections)** -- instead of learning f(x) directly, learn the residual g(x) = f(x) - x, then compute f(x) = x + g(x). This ensures nested function classes (F1 subset F2 subset ... subset F6) so adding layers never degrades performance.

**Mathematical motivation:**

Given function class F, the optimal function is:
```
f*_F = argmin_f L(X, y, f)  subject to f in F
```

For nested function classes where F1 <= F2 <= ... <= F6, increasing capacity guarantees monotonic improvement. Residual connections achieve nesting because a new layer can always learn the identity: if g(x) = 0 (weights/biases pushed to zero), then f(x) = x.

**Residual Block (basic, ResNet-18/34):**
```
Input: X
    |
    +-----> Conv 3x3 (c channels, pad=1, stride=s) -> BatchNorm -> ReLU
    |       |
    |       Conv 3x3 (c channels, pad=1) -> BatchNorm
    |       |
    +-----> [1x1 Conv if dimensions change] -----> Addition
                                                      |
                                                    ReLU
                                                      |
                                                   Output: Y

Forward pass:
    Y = BN(Conv2(ReLU(BN(Conv1(X)))))
    If use_1x1conv: X = Conv3(X)   # match dimensions
    Output = ReLU(Y + X)
```

**When to use 1x1 convolution on skip path:**
- When stride > 1 (spatial dimensions change)
- When number of channels changes between input and output

**Bottleneck Residual Block (ResNet-50/101/152):**
```
Input: X
    |
    +-----> Conv 1x1 (b channels) -> BN -> ReLU      # reduce channels
    |       |
    |       Conv 3x3 (b channels, pad=1) -> BN -> ReLU  # spatial processing
    |       |
    |       Conv 1x1 (c channels) -> BN               # restore channels
    |       |
    +-----> [1x1 Conv if needed] -----> Addition -> ReLU
```

Where b < c (bottleneck). Reduces computation from O(c^2 * 9) to O(c*b*1 + b^2*9 + b*c*1).

**ResNet-18 Architecture (input: 1x96x96 shown):**

| Module | Blocks | Channels | Stride | Output Shape |
|--------|--------|----------|--------|-------------|
| Stem (b1) | Conv 7x7 (s=2, p=3) -> BN -> ReLU -> MaxPool 3x3 (s=2, p=1) | 64 | 2+2 | 64x24x24 |
| Stage 2 | 2 Residual blocks | 64 | 1 | 64x24x24 |
| Stage 3 | 2 Residual blocks (first: 1x1conv, s=2) | 128 | 2 | 128x12x12 |
| Stage 4 | 2 Residual blocks (first: 1x1conv, s=2) | 256 | 2 | 256x6x6 |
| Stage 5 | 2 Residual blocks (first: 1x1conv, s=2) | 512 | 2 | 512x3x3 |
| Head | Global AvgPool -> Flatten -> Linear | -- | -- | num_classes |

**ResNet Family:**

| Variant | Block Type | Layers per Stage [2,3,4,5] | Total Layers | Params (approx) |
|---------|-----------|---------------------------|-------------|-----------------|
| ResNet-18 | Basic | [2, 2, 2, 2] | 18 | ~11.7M |
| ResNet-34 | Basic | [3, 4, 6, 3] | 34 | ~21.8M |
| ResNet-50 | Bottleneck | [3, 4, 6, 3] | 50 | ~25.6M |
| ResNet-101 | Bottleneck | [3, 4, 23, 3] | 101 | ~44.5M |
| ResNet-152 | Bottleneck | [3, 8, 36, 3] | 152 | ~60.2M |

Layer count formula:
- Basic block: 2 conv layers per block
- Bottleneck block: 3 conv layers per block
- ResNet-18: 1 (stem) + 2*2*4 (stages) + 1 (FC) = 18 counting conv+FC layers

**Why ResNets work:**
1. Identity mapping is easy to learn (push residual weights to zero)
2. Gradient flows directly through skip connections (mitigates vanishing gradients)
3. Nested function classes guarantee monotonic improvement with depth
4. Default inductive bias shifts from f(x)=0 to f(x)=x
5. Layers can be initialized as identity during training, then gradually learn

---

## 2.7 ResNeXt (Xie et al., 2017)

**Key innovation:** **Grouped convolutions** within residual blocks. Splits channels into g independent groups, reducing computation by factor g while maintaining or improving accuracy.

**Grouped convolution cost:**
```
Standard:  O(c_i * c_o)
Grouped:   O(g * (c_i/g) * (c_o/g)) = O(c_i * c_o / g)
```

Where:
- `c_i` = input channels
- `c_o` = output channels
- `g` = number of groups

**ResNeXt Block:**
```
Input: X (c channels)
    |
    +-----> Conv 1x1 (b channels) -> BN -> ReLU            # reduce to bottleneck
    |       |
    |       Grouped Conv 3x3 (b channels, groups=b/g) -> BN -> ReLU  # grouped spatial
    |       |
    |       Conv 1x1 (c channels) -> BN                     # restore channels
    |       |
    +-----> [1x1 Conv if needed] -----> Addition -> ReLU
```

Where:
- `b` = bottleneck channels = `round(c * bot_mul)`
- `g` = group width
- Cost for 1x1: O(c * b), for grouped 3x3: O(b^2 / g)

---

## 2.8 DenseNet -- Densely Connected Networks (Huang et al., 2017)

**Key innovation:** Each layer connects to ALL preceding layers via **concatenation** (not addition). Inspired by Taylor expansion analogy.

**Mathematical formulation:**

ResNet decomposes as: `f(x) = x + g(x)`

DenseNet generalizes to:
```
x -> [x, f1(x), f2([x, f1(x)]), f3([x, f1(x), f2([x, f1(x)])]), ...]
```

Where `[,]` denotes concatenation along channel dimension.

**Dense Block:**
```
DenseBlock(num_convs, growth_rate):
    for each conv_block:
        Input: X (accumulated channels)
        Y = BN -> ReLU -> Conv 3x3 (growth_rate channels, pad=1)
        X = Concat(X, Y)  along channel dim

    Output channels = input_channels + num_convs * growth_rate
```

**Growth rate (k):** Each conv block within a dense block produces exactly k new channels. After n blocks with growth_rate k, total channels = initial_channels + n*k.

Example: Input 3 channels, 2 conv blocks, growth_rate=10 -> Output: 3+10+10 = 23 channels.

**Transition Layer (between dense blocks):**
```
TransitionBlock(num_channels):
    BatchNorm -> ReLU -> Conv 1x1 (num_channels) -> AvgPool 2x2 (stride=2)
```
Purpose: (1) Reduce channels via 1x1 conv (typically halve), (2) Halve spatial resolution via average pooling.

**DenseNet Architecture (default config: growth_rate=32, arch=(4,4,4,4)):**

| Module | Structure | Notes |
|--------|-----------|-------|
| Stem | Conv 7x7 (64ch, s=2, p=3) -> BN -> ReLU -> MaxPool 3x3 (s=2, p=1) | Same as ResNet |
| Dense Block 1 | 4 conv blocks, growth_rate=32 | 64 + 4*32 = 192 channels |
| Transition 1 | 1x1 Conv (96ch) -> AvgPool 2x2 | 192/2 = 96 channels |
| Dense Block 2 | 4 conv blocks, growth_rate=32 | 96 + 4*32 = 224 channels |
| Transition 2 | 1x1 Conv (112ch) -> AvgPool 2x2 | 224/2 = 112 channels |
| Dense Block 3 | 4 conv blocks, growth_rate=32 | 112 + 4*32 = 240 channels |
| Transition 3 | 1x1 Conv (120ch) -> AvgPool 2x2 | 240/2 = 120 channels |
| Dense Block 4 | 4 conv blocks, growth_rate=32 | 120 + 4*32 = 248 channels |
| Head | BN -> ReLU -> Global AvgPool -> Flatten -> Linear | No transition after last block |

**Channel growth formula:**
```
After dense block i:  channels = prev_channels + num_convs_i * growth_rate
After transition i:   channels = floor(channels / 2)
```

**Key tradeoff:** Feature reuse is computationally efficient but causes heavy GPU memory consumption due to stored intermediate concatenations.

**Conv block order (modified from ResNet):**
```
BN -> ReLU -> Conv  (pre-activation, not Conv -> BN -> ReLU)
```

---

## 2.9 Design Space Exploration: AnyNet / RegNet (Radosavovic et al., 2020)

**Key innovation:** Systematic exploration of **network design spaces** rather than searching for single optimal architectures. Produces families of performant networks (RegNetX/RegNetY) governed by simple rules.

**AnyNet template:**
```
Stem:  Conv 3x3 (c0 channels, stride=2) -> BN -> ReLU
       Input: RGB 224x224x3 -> Output: 112x112xc0

Body:  4 stages, each stage halves resolution
       Stage i: d_i ResNeXt blocks, c_i channels, g_i groups, k_i bottleneck ratio
       First block: stride=2 (halves spatial), 1x1 conv on skip
       Remaining blocks: stride=1

Head:  Global AvgPool -> Flatten -> Linear(num_classes)
```

**Design space parameters (17 total for AnyNet):**
- `c0, c1, c2, c3, c4` -- channel widths (5 params)
- `d1, d2, d3, d4` -- depths per stage (4 params)
- `k1, k2, k3, k4` -- bottleneck ratios (4 params)
- `g1, g2, g3, g4` -- group widths (4 params)

**Design space reduction (RegNet principles):**

| Constraint | Effect | Resulting Space |
|-----------|--------|----------------|
| Shared bottleneck: k_i = k for all i | Eliminates 3 params | AnyNetB |
| Shared group width: g_i = g for all i | Eliminates 3 more params | AnyNetC |
| Increasing widths: c_i <= c_{i+1} | Improves performance | AnyNetD |
| Increasing depths: d_i <= d_{i+1} | Improves performance | AnyNetE |
| Linear width growth: c_j ~ c_0 + c_a * j | Regularizes width | RegNet |
| Bottleneck ratio k = 1 (no bottleneck) | Simplest, works best | RegNetX |

**Empirical CDF for design space quality (Eq. 8.8.2):**
```
F_hat(e, Z) = (1/n) * sum_{i=1}^{n} 1(e_i <= e)
```
Where `Z = {net_1, ..., net_n}` is a sample of networks, `e_i` is the error of net_i.

A design space A is superior to B if `F_hat_A(e)` majorizes `F_hat_B(e)` -- i.e., for any error threshold, a larger fraction of networks from A achieve that threshold.

**RegNetX-32 example:**
```
k = 1 (no bottleneck), g = 16 (group width)
stem_channels = 32
Stage 1: depth=4, channels=32
Stage 2: depth=6, channels=80
```

**Output shapes (input 1x96x96):**
```
Stem:    32x48x48
Stage 1: 32x24x24
Stage 2: 80x12x12
Head:    num_classes
```

---

# 3. ARCHITECTURE COMPARISON TABLE

| Architecture | Year | Depth | Params | Key Innovation | Top-5 Error (ImageNet) |
|-------------|------|-------|--------|---------------|----------------------|
| LeNet-5 | 1995 | 5 | ~44K | First practical CNN | N/A (MNIST) |
| AlexNet | 2012 | 8 | ~62M | ReLU, dropout, GPU training | ~16.4% |
| VGG-16 | 2014 | 16 | ~138M | Repeated 3x3 conv blocks | ~7.3% |
| NiN | 2013 | -- | Much less | 1x1 conv, global avg pool | -- |
| GoogLeNet | 2014 | 22 | ~6.8M | Inception multi-branch blocks | ~6.7% |
| ResNet-152 | 2015 | 152 | ~60M | Skip connections, residual learning | ~3.6% |
| ResNeXt | 2017 | varies | varies | Grouped convolutions | improved over ResNet |
| DenseNet | 2017 | varies | fewer than ResNet | Dense concatenation, feature reuse | competitive with ResNet |
| RegNetX/Y | 2020 | varies | varies | Design space optimization | competitive SOTA |

---

# 4. RECURRING DESIGN PATTERNS

**Common network structure:**
```
Stem -> Body -> Head

Stem:  Large-kernel conv (5x5 or 7x7) + pooling for initial feature extraction
Body:  Repeated blocks, each stage doubles channels and halves spatial resolution
Head:  Global Average Pooling -> (Optional FC) -> Output
```

**Channel progression rule:** Double channels at each resolution halving:
```
64 -> 128 -> 256 -> 512
```

**Resolution progression (ImageNet 224x224):**
```
224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1 (GAP)
```

**Receptive field equivalences:**
- 1 layer of 5x5 ~ 2 layers of 3x3 (same receptive field)
- 1 layer of 7x7 ~ 3 layers of 3x3 (same receptive field)
- Deeper stacks of 3x3 preferred: fewer parameters, more nonlinearity

**Parameter count comparison (3x3 stacks vs single large kernel, c channels):**
```
Single 5x5: 25 * c^2
Two 3x3:    2 * 9 * c^2 = 18 * c^2   (28% fewer)

Single 7x7: 49 * c^2
Three 3x3:  3 * 9 * c^2 = 27 * c^2   (45% fewer)
```

**FC layer elimination timeline:**
- LeNet/AlexNet/VGG: Large FC layers (dominant parameter cost)
- NiN: Replaced FC with Global Average Pooling
- GoogLeNet onward: GAP standard; minimal or single FC at output

---

# 5. KEY EQUATIONS REFERENCE

| Equation | Formula | Context |
|----------|---------|---------|
| Conv output size | `floor((n - k + p + s) / s)` | Per spatial dimension |
| BN forward | `gamma * (x - mu_B) / sigma_B + beta` | Training mode |
| BN running avg | `mu_new = (1-m)*mu_old + m*mu_batch` | Momentum update |
| LN forward | `(x - mu) / sigma` | Per-observation |
| Residual | `f(x) = ReLU(g(x) + x)` | Skip connection |
| DenseNet | `x -> [x, f1(x), f2([x,f1(x)]), ...]` | Channel concat |
| Grouped conv cost | `O(c_i * c_o / g)` | g groups |
| Growth rate | `out_ch = in_ch + n * k` | DenseNet, n blocks, rate k |

---

## Raw Algorithms, Equations, and Architectures (Chapters 9-11)

---

## 1. Recurrent Neural Networks (RNNs)

### 1.1 Vanilla RNN Forward Pass

```
h_t = phi(X_t * W_xh + H_{t-1} * W_hh + b_h)
O_t = H_t * W_hq + b_q
```

**Variable definitions:**
- `X_t in R^{n x d}`: input at time step t (n = batch size, d = input dimension)
- `H_t in R^{n x h}`: hidden state at time step t (h = number of hidden units)
- `H_{t-1} in R^{n x h}`: hidden state from previous time step
- `W_xh in R^{d x h}`: input-to-hidden weight matrix
- `W_hh in R^{h x h}`: hidden-to-hidden weight matrix
- `b_h in R^{1 x h}`: hidden bias
- `W_hq in R^{h x q}`: hidden-to-output weight matrix (q = output dimension)
- `b_q in R^{1 x q}`: output bias
- `phi`: activation function (typically tanh)

### 1.2 Backpropagation Through Time (BPTT)

**Gradient computations (unrolled graph):**

```
dL/dW_xh = sum_{t=1}^{T} prod(dL/dh_t, dh_t/dW_xh) = sum_{t=1}^{T} (dL/dh_t) * x_t^T

dL/dW_hh = sum_{t=1}^{T} prod(dL/dh_t, dh_t/dW_hh) = sum_{t=1}^{T} (dL/dh_t) * h_{t-1}^T
```

**Variable definitions:**
- `dL/dh_t`: gradient of loss with respect to hidden state at time t (computed recursively)
- `x_t^T`: transpose of input at time t
- `h_{t-1}^T`: transpose of hidden state at time t-1

**Key property:** The gradient `dL/dh_t` is computed recursively backward through time. Stored intermediate values are reused to avoid duplicate calculations (e.g., `dL/dh_t` is used in computing both `dL/dW_xh` and `dL/dW_hh`).

**Truncation:** Regular or randomized truncation is applied for computational convenience and numerical stability.

### 1.3 Vanishing and Exploding Gradients

**Mechanism:** High powers of matrices lead to divergent or vanishing eigenvalues.

For a symmetric matrix `M in R^{n x n}` with eigenvalues `lambda_i` and eigenvectors `v_i`:
```
M^k has eigenvalues lambda_i^k
```

- If `|lambda_i| > 1`: gradients explode (eigenvalues grow exponentially)
- If `|lambda_i| < 1`: gradients vanish (eigenvalues decay exponentially)

**Gradient Clipping:** Project gradient `g` back to a ball of radius `theta`:
```
g <- min(1, theta / ||g||) * g
```

### 1.4 Language Models and Perplexity

**Perplexity** measures language model quality:
```
perplexity = exp(cross-entropy)
           = exp(-(1/n) * sum_{t=1}^{n} log P(x_t | x_{t-1}, ..., x_1))
```

**Variable definitions:**
- `n`: number of tokens in the sequence
- `P(x_t | x_{t-1}, ..., x_1)`: predicted probability of token x_t given all prior tokens
- Lower perplexity = better model (perplexity of 1 = perfect predictions)

---

## 2. Long Short-Term Memory (LSTM)

### 2.1 Gate Equations

**Input gate, Forget gate, Output gate:**
```
I_t = sigma(X_t * W_xi + H_{t-1} * W_hi + b_i)
F_t = sigma(X_t * W_xf + H_{t-1} * W_hf + b_f)
O_t = sigma(X_t * W_xo + H_{t-1} * W_ho + b_o)
```

**Variable definitions:**
- `I_t in R^{n x h}`: input gate activation
- `F_t in R^{n x h}`: forget gate activation
- `O_t in R^{n x h}`: output gate activation
- `sigma`: sigmoid function, forcing values into (0, 1)
- `X_t in R^{n x d}`: input at time step t
- `H_{t-1} in R^{n x h}`: previous hidden state
- `W_xi, W_xf, W_xo in R^{d x h}`: input weight matrices for each gate
- `W_hi, W_hf, W_ho in R^{h x h}`: hidden weight matrices for each gate
- `b_i, b_f, b_o in R^{1 x h}`: bias vectors for each gate

### 2.2 Candidate Memory Cell (Input Node)

```
C_tilde_t = tanh(X_t * W_xc + H_{t-1} * W_hc + b_c)
```

**Variable definitions:**
- `C_tilde_t in R^{n x h}`: candidate memory cell content
- `W_xc in R^{d x h}`, `W_hc in R^{h x h}`: weight matrices
- `b_c in R^{1 x h}`: bias
- tanh forces values into (-1, 1)

### 2.3 Cell State Update

```
C_t = F_t (hadamard) C_{t-1} + I_t (hadamard) C_tilde_t
```

**Variable definitions:**
- `C_t in R^{n x h}`: cell internal state at time t
- `C_{t-1} in R^{n x h}`: previous cell internal state
- `(hadamard)`: Hadamard (elementwise) product
- `F_t = 1, I_t = 0` => cell state persists unchanged (long-term memory)
- `F_t = 0` => previous state flushed entirely

### 2.4 Hidden State Output

```
H_t = O_t (hadamard) tanh(C_t)
```

**Variable definitions:**
- `H_t in R^{n x h}`: hidden state output (passed to other layers)
- `O_t`: output gate controls how much of cell state influences output
- tanh ensures hidden state values in (-1, 1)

**LSTM parameter count:** For input dimension d and hidden dimension h:
- 4 sets of (W_x, W_h, b): total = 4(dh + h^2 + h) parameters

**Key insight:** The cell state `C_t` can accrue information across many time steps without impacting the network (when output gate ~ 0), then suddenly influence output when the output gate flips to ~ 1. The self-connected recurrent edge with weight 1 (via forget gate = 1) allows gradients to flow without vanishing.

---

## 3. Gated Recurrent Units (GRU)

### 3.1 Reset Gate and Update Gate

```
R_t = sigma(X_t * W_xr + H_{t-1} * W_hr + b_r)
Z_t = sigma(X_t * W_xz + H_{t-1} * W_hz + b_z)
```

**Variable definitions:**
- `R_t in R^{n x h}`: reset gate (controls short-term dependencies)
- `Z_t in R^{n x h}`: update gate (controls long-term dependencies)
- `W_xr, W_xz in R^{d x h}`: input weight matrices
- `W_hr, W_hz in R^{h x h}`: hidden weight matrices
- `b_r, b_z in R^{1 x h}`: bias vectors

### 3.2 Candidate Hidden State

```
H_tilde_t = tanh(X_t * W_xh + (R_t (hadamard) H_{t-1}) * W_hh + b_h)
```

**Variable definitions:**
- `H_tilde_t in R^{n x h}`: candidate hidden state
- `R_t (hadamard) H_{t-1}`: reset gate modulates how much prior state influences candidate
- `R_t ~ 1`: reduces to vanilla RNN
- `R_t ~ 0`: candidate is MLP of X_t only (prior state reset to defaults)

### 3.3 Hidden State Update

```
H_t = Z_t (hadamard) H_{t-1} + (1 - Z_t) (hadamard) H_tilde_t
```

**Variable definitions:**
- Convex combination between old state `H_{t-1}` and candidate `H_tilde_t`
- `Z_t ~ 1`: retain old state, skip current input (long-term memory)
- `Z_t ~ 0`: fully adopt candidate state

**GRU parameter count:** 3 sets of (W_x, W_h, b): total = 3(dh + h^2 + h) parameters (lighter than LSTM's 4 sets).

---

## 4. Deep RNNs (Stacked Layers)

### 4.1 Multi-Layer Recurrence

```
H_t^(l) = phi_l(H_t^(l-1) * W_xh^(l) + H_{t-1}^(l) * W_hh^(l) + b_h^(l))
```

with `H_t^(0) = X_t` (input), and output:

```
O_t = H_t^(L) * W_hq + b_q
```

**Variable definitions:**
- `l = 1, ..., L`: layer index (L = total hidden layers)
- `H_t^(l) in R^{n x h}`: hidden state of layer l at time t
- `W_xh^(l) in R^{h x h}`: input-to-hidden weights for layer l (R^{d x h} for l=1)
- `W_hh^(l) in R^{h x h}`: hidden-to-hidden weights for layer l
- `b_h^(l) in R^{1 x h}`: bias for layer l
- `phi_l`: activation function for layer l
- `W_hq in R^{h x q}`, `b_q in R^{1 x q}`: output layer parameters

**Design:** Each layer receives the sequence output of the layer below. Any RNN cell (vanilla, LSTM, GRU) can be used at each layer. Common ranges: h in (64, 2048), L in (1, 8).

---

## 5. Bidirectional RNNs

### 5.1 Forward and Backward Hidden States

```
H_t_forward  = phi(X_t * W_xh^(f) + H_{t-1}_forward  * W_hh^(f) + b_h^(f))
H_t_backward = phi(X_t * W_xh^(b) + H_{t+1}_backward * W_hh^(b) + b_h^(b))
```

**Variable definitions:**
- `H_t_forward in R^{n x h}`: forward hidden state (processes x_1, ..., x_T)
- `H_t_backward in R^{n x h}`: backward hidden state (processes x_T, ..., x_1)
- `W_xh^(f), W_xh^(b) in R^{d x h}`: forward/backward input weights
- `W_hh^(f), W_hh^(b) in R^{h x h}`: forward/backward recurrent weights
- `b_h^(f), b_h^(b) in R^{1 x h}`: forward/backward biases

### 5.2 Concatenated Output

```
H_t = [H_t_forward ; H_t_backward]    (concatenation along feature dimension)
O_t = H_t * W_hq + b_q
```

**Variable definitions:**
- `H_t in R^{n x 2h}`: concatenated hidden state
- `W_hq in R^{2h x q}`: output weight matrix
- `b_q in R^{1 x q}`: output bias

**Usage:** Bidirectional RNNs are for sequence encoding/labeling where full context is available. NOT suitable for autoregressive generation (cannot look ahead).

---

## 6. Encoder-Decoder Architecture

### 6.1 General Framework

**Encoder:** Takes variable-length input X and transforms it into fixed-shape state.
```
(enc_outputs, enc_state) = Encoder(X)
```

**Decoder:** Generates variable-length output conditioned on encoded state.
```
dec_state = Decoder.init_state(enc_outputs)
output = Decoder(dec_input, dec_state)
```

**Combined:**
```
EncoderDecoder.forward(enc_X, dec_X):
    enc_all_outputs = encoder(enc_X)
    dec_state = decoder.init_state(enc_all_outputs)
    return decoder(dec_X, dec_state)
```

---

## 7. Sequence-to-Sequence (Seq2Seq) Learning

### 7.1 RNN Encoder

```
h_t = f(x_t, h_{t-1})           -- recurrent transformation at each step
c = q(h_1, ..., h_T)            -- context variable (typically c = h_T)
```

**Variable definitions:**
- `h_t`: encoder hidden state at time t
- `x_t`: input feature vector at time t (from embedding layer)
- `c`: context variable summarizing entire input

### 7.2 RNN Decoder

```
s_{t'} = g(y_{t'-1}, c, s_{t'-1})
P(y_{t'+1} | y_1, ..., y_{t'}, c) = softmax(output_layer(s_{t'}))
```

**Variable definitions:**
- `s_{t'}`: decoder hidden state at decoding step t'
- `y_{t'-1}`: previous target token
- `c`: context variable from encoder
- The context vector `c` is concatenated with decoder input at ALL time steps

### 7.3 Teacher Forcing

During training: feed ground-truth target tokens as decoder input (shifted by one).
```
Input:  <bos>, y_1, y_2, ..., y_{T'-1}
Target: y_1,   y_2, y_3, ..., y_{T'}
```

### 7.4 Loss Function with Masking

Cross-entropy loss computed only on non-padding tokens:
```
L = sum(l * mask) / sum(mask)
where mask = (Y != pad_token)
```

### 7.5 BLEU Score (Evaluation Metric)

```
BLEU = exp(min(0, 1 - len_label / len_pred)) * prod_{n=1}^{k} p_n^{1/2^n}
```

**Variable definitions:**
- `p_n`: precision of n-gram = (matched n-grams in predicted and target) / (n-grams in predicted)
- `len_label`: length of target sequence
- `len_pred`: length of predicted sequence
- `k`: longest n-gram considered for matching
- Brevity penalty: `exp(min(0, 1 - len_label/len_pred))` penalizes short predictions
- Longer n-gram matches get greater weight via `p_n^{1/2^n}` (increases as n grows)

### 7.6 Beam Search

At each time step, maintain top-k candidates (beam size = k).

**Scoring function for final candidates:**
```
score = (1 / L^alpha) * sum_{t'=1}^{L} log P(y_{t'} | y_1, ..., y_{t'-1}, c)
```

**Variable definitions:**
- `L`: length of candidate sequence
- `alpha`: length normalization exponent (typically 0.75)
- `L^alpha` in denominator penalizes long sequences
- Computational cost: `O(k * |Y| * T')` where |Y| = vocabulary size, T' = max output length
- `k = 1` reduces to greedy search
- `k = |Y|^{T'}` is exhaustive search

---

## 8. Attention Mechanisms

### 8.1 Attention Pooling (Query-Key-Value Framework)

```
Attention(q, D) = sum_{i=1}^{m} alpha(q, k_i) * v_i
```

**Variable definitions:**
- `q`: query vector
- `D = {(k_1, v_1), ..., (k_m, v_m)}`: database of m key-value pairs
- `alpha(q, k_i) in R`: scalar attention weight for key i
- Typically: weights are nonnegative, form convex combination (sum to 1)

**Softmax normalization:**
```
alpha(q, k_i) = exp(a(q, k_i)) / sum_j exp(a(q, k_j))
```

where `a(q, k)` is the attention scoring function.

### 8.2 Attention Scoring Functions

#### 8.2.1 Scaled Dot-Product Attention

```
a(q, k_i) = q^T * k_i / sqrt(d)
```

**Variable definitions:**
- `q in R^d`: query vector
- `k_i in R^d`: key vector (same dimension as query)
- `d`: dimension of query/key vectors
- `1/sqrt(d)`: scaling factor to keep variance of dot product at 1 (when elements have zero mean, unit variance, the dot product has variance d)

**Matrix form (batched):**
```
softmax(Q * K^T / sqrt(d)) * V in R^{n x v}
```

where `Q in R^{n x d}`, `K in R^{m x d}`, `V in R^{m x v}` (n queries, m key-value pairs).

#### 8.2.2 Additive Attention (Bahdanau)

```
a(q, k) = w_v^T * tanh(W_q * q + W_k * k) in R
```

**Variable definitions:**
- `W_q in R^{h x q_dim}`: learnable query projection
- `W_k in R^{h x k_dim}`: learnable key projection
- `w_v in R^h`: learnable weight vector
- `h`: hidden dimension of the attention MLP
- Allows queries and keys of DIFFERENT dimensions (unlike dot-product)
- Equivalent to: concatenate query and key, pass through single-hidden-layer MLP with tanh

#### 8.2.3 Masked Softmax Operation

For variable-length sequences with padding:
```
For positions i > valid_length: set attention score to -10^6 (effectively zero after softmax)
```

This ensures padding tokens receive zero attention weight.

### 8.3 Nadaraya-Watson Kernel Regression (Attention Precursor)

```
f(q) = sum_i v_i * alpha(q, k_i) / sum_j alpha(q, k_j)
```

**Common kernels:**
```
Gaussian:      alpha(q, k) = exp(-1/2 * ||q - k||^2)
Boxcar:        alpha(q, k) = 1 if ||q - k|| <= 1, else 0
Epanechnikov:  alpha(q, k) = max(0, 1 - ||q - k||)
```

**Parameterized Gaussian with bandwidth sigma:**
```
alpha(q, k) = exp(-||q - k||^2 / (2 * sigma^2))
```

Narrower kernel (smaller sigma) = less smooth estimate, better local adaptation.

---

## 9. Bahdanau Attention Mechanism

### 9.1 Dynamic Context Vector

Instead of fixed context `c = h_T`, compute dynamic context at each decoding step:

```
c_{t'} = sum_{t=1}^{T} alpha(s_{t'-1}, h_t) * h_t
```

**Variable definitions:**
- `c_{t'}`: context vector at decoding step t'
- `s_{t'-1}`: decoder hidden state from previous step (used as QUERY)
- `h_t`: encoder hidden state at position t (used as both KEY and VALUE)
- `alpha`: attention weight computed via additive attention scoring function
- `T`: length of input sequence

**Decoder state update with attention:**
```
s_{t'} = g(y_{t'-1}, c_{t'}, s_{t'-1})
```

The context vector is concatenated with the decoder input embedding before being fed to the RNN.

---

## 10. Multi-Head Attention

### 10.1 Single Attention Head

```
h_i = f(W_i^(q) * q, W_i^(k) * k, W_i^(v) * v) in R^{p_v}
```

**Variable definitions:**
- `h_i`: output of attention head i (i = 1, ..., h)
- `W_i^(q) in R^{p_q x d_q}`: query projection for head i
- `W_i^(k) in R^{p_k x d_k}`: key projection for head i
- `W_i^(v) in R^{p_v x d_v}`: value projection for head i
- `f`: attention pooling function (typically scaled dot-product)

### 10.2 Multi-Head Output

```
MultiHead(q, k, v) = W_o * [h_1; h_2; ...; h_h] in R^{p_o}
```

**Variable definitions:**
- `W_o in R^{p_o x (h * p_v)}`: output projection matrix
- `[h_1; ...; h_h]`: concatenation of all head outputs
- `h`: number of heads
- Common setting: `p_q = p_k = p_v = p_o / h` (each head operates on `d_model / h` dimensions)

### 10.3 Efficient Implementation

Reshape for parallel computation:
```
(batch_size, seq_len, num_hiddens)
  -> (batch_size, seq_len, num_heads, num_hiddens/num_heads)
  -> (batch_size, num_heads, seq_len, num_hiddens/num_heads)
  -> (batch_size * num_heads, seq_len, num_hiddens/num_heads)
```

All heads computed simultaneously via batch matrix multiplication. Output reversed via `transpose_output`.

---

## 11. Self-Attention

### 11.1 Definition

Given input sequence `x_1, ..., x_n` where `x_i in R^d`:

```
y_i = f(x_i, (x_1, x_1), ..., (x_n, x_n)) in R^d
```

**Q = K = V all come from the same sequence.** Each token attends to every other token (including itself).

### 11.2 Computational Complexity Comparison

| Architecture   | Complexity   | Sequential Ops | Max Path Length |
|---------------|-------------|---------------|----------------|
| CNN (kernel k) | O(k*n*d^2)  | O(1)          | O(n/k)         |
| RNN            | O(n*d^2)    | O(n)          | O(n)           |
| Self-Attention | O(n^2*d)    | O(1)          | O(1)           |

**Trade-off:** Self-attention has shortest path (O(1)) and full parallelism, but quadratic cost in sequence length n makes it prohibitively slow for very long sequences.

---

## 12. Positional Encoding

### 12.1 Sinusoidal Positional Encoding

```
p_{i, 2j}   = sin(i / 10000^{2j/d})
p_{i, 2j+1} = cos(i / 10000^{2j/d})
```

**Variable definitions:**
- `i`: position index in the sequence (row)
- `j`: dimension index / 2 (column pair)
- `d`: embedding dimension (model dimension)
- `P in R^{n x d}`: positional embedding matrix

**Applied as:** `X_out = X + P` (added to input embeddings)

### 12.2 Relative Position Property

For any fixed offset delta, the positional encoding at position `i + delta` is a linear projection of that at position `i`:

```
[cos(delta * omega_j)   sin(delta * omega_j) ] [p_{i,2j}  ]   [p_{i+delta,2j}  ]
[-sin(delta * omega_j)  cos(delta * omega_j) ] [p_{i,2j+1}] = [p_{i+delta,2j+1}]
```

where `omega_j = 1 / 10000^{2j/d}`.

**Key property:** The 2x2 rotation matrix depends only on the offset delta, NOT on position i. This allows the model to learn relative positional relationships.

### 12.3 Frequency Structure

- Lower dimensions (small j): high frequency oscillations
- Higher dimensions (large j): low frequency oscillations
- Analogous to binary representation where lower bits alternate faster

---

## 13. Transformer Architecture

### 13.1 Overall Structure

**Encoder-decoder architecture where both encoder and decoder are stacks of identical blocks. No recurrence or convolution.**

```
Input Embedding:  X_emb = Embedding(X) * sqrt(d_model) + PositionalEncoding
```

The `sqrt(d_model)` scaling ensures embedding magnitudes are comparable to positional encoding values (which are in [-1, 1]).

### 13.2 Transformer Encoder Block (Post-Norm -- Original)

```
SubLayer1: MultiHead Self-Attention
SubLayer2: Position-wise FFN

Y = LayerNorm(X + MultiHeadAttention(X, X, X, valid_lens))     -- self-attention + residual + norm
Output = LayerNorm(Y + FFN(Y))                                 -- FFN + residual + norm
```

**Post-norm pattern: residual connection THEN layer normalization.**

### 13.3 Transformer Decoder Block (Post-Norm -- Original)

```
SubLayer1: Masked Multi-Head Self-Attention (causal)
SubLayer2: Multi-Head Encoder-Decoder Cross-Attention
SubLayer3: Position-wise FFN

X2 = MultiHeadAttention(X, key_values, key_values, dec_valid_lens)  -- masked self-attention
Y  = LayerNorm(X + X2)                                              -- residual + norm
Y2 = MultiHeadAttention(Y, enc_outputs, enc_outputs, enc_valid_lens) -- cross-attention
Z  = LayerNorm(Y + Y2)                                              -- residual + norm
Output = LayerNorm(Z + FFN(Z))                                      -- FFN + residual + norm
```

**Masked self-attention:** Each position can only attend to positions at or before it.
```
dec_valid_lens[i] = [1, 2, ..., num_steps]    -- position i attends to positions 1..i
```

**Cross-attention:** Queries from decoder, keys and values from encoder output.

### 13.4 Position-Wise Feed-Forward Network (FFN)

```
FFN(X) = W_2 * ReLU(W_1 * X + b_1) + b_2
```

**Variable definitions:**
- `X in R^{n x d}`: input (same MLP applied independently at each position)
- `W_1 in R^{d x d_ff}`: first layer weights (d_ff = FFN hidden dimension, typically 4*d)
- `W_2 in R^{d_ff x d}`: second layer weights
- `b_1, b_2`: biases

**Key property:** Same MLP shared across ALL sequence positions (hence "position-wise"). Does NOT mix information across positions.

### 13.5 Layer Normalization (vs. Batch Normalization)

```
LayerNorm(x) = gamma * (x - mu) / sigma + beta
```

- **Batch Norm:** normalizes across batch dimension (examples within a minibatch)
- **Layer Norm:** normalizes across feature dimension (within each example)
- Layer norm is preferred in NLP because it is independent of batch size and handles variable-length sequences

### 13.6 Residual Connection with Layer Normalization (AddNorm)

```
AddNorm(X, SubLayer) = LayerNorm(Dropout(SubLayer(X)) + X)
```

Requires `SubLayer(X)` to have same shape as `X` (dimension d preserved throughout).

### 13.7 Full Encoder

```
class TransformerEncoder:
    embedding: vocab -> d_model
    pos_encoding: sinusoidal
    blks: N identical TransformerEncoderBlocks

    forward(X, valid_lens):
        X = PositionalEncoding(Embedding(X) * sqrt(d_model))
        for each block:
            X = block(X, valid_lens)
        return X
```

### 13.8 Full Decoder

```
class TransformerDecoder:
    embedding: vocab -> d_model
    pos_encoding: sinusoidal
    blks: N identical TransformerDecoderBlocks
    dense: d_model -> vocab_size

    init_state(enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None]*N]

    forward(X, state):
        X = PositionalEncoding(Embedding(X) * sqrt(d_model))
        for each block:
            X, state = block(X, state)
        return Linear(X), state           -- project to vocab logits
```

### 13.9 Typical Hyperparameters (from d2l experiments)

```
d_model (num_hiddens) = 256
d_ff (ffn_num_hiddens) = 64
num_heads = 4
num_layers (num_blks) = 2
dropout = 0.2
lr = 0.001
```

Production-scale (original Transformer paper):
```
d_model = 512
d_ff = 2048
num_heads = 8
num_layers = 6
```

---

## 14. Pre-Norm vs. Post-Norm Transformer

### 14.1 Post-Norm (Original Transformer)

```
Output = LayerNorm(X + SubLayer(X))
```

Normalization applied AFTER the residual connection.

### 14.2 Pre-Norm (Vision Transformer / GPT-2)

```
Output = X + SubLayer(LayerNorm(X))
```

Normalization applied BEFORE the sublayer. Pre-norm leads to more effective/efficient training for deep Transformers.

**ViT encoder block (pre-norm):**
```
X = X + MultiHeadAttention(LayerNorm(X), LayerNorm(X), LayerNorm(X))
X = X + MLP(LayerNorm(X))
```

---

## 15. Vision Transformer (ViT)

### 15.1 Patch Embedding

Given image of height h, width w, channels c, and patch size p:

```
num_patches = (h * w) / p^2
```

Each patch is flattened to a vector of length `c * p^2`, then linearly projected to dimension `d_model`.

**Implementation:** Single convolution with kernel_size = stride = patch_size.
```
PatchEmbedding(X) = Conv2d(X, out_channels=d_model, kernel_size=p, stride=p).flatten(2).transpose(1,2)
```

Output shape: `(batch_size, num_patches, d_model)`

### 15.2 ViT Architecture

```
1. Patch Embedding: image -> (batch, num_patches, d_model)
2. Prepend <cls> token: -> (batch, num_patches + 1, d_model)
3. Add LEARNABLE positional embeddings (not sinusoidal)
4. Apply dropout
5. Pass through N ViTBlocks (pre-norm Transformer encoder blocks with GELU activation)
6. Take <cls> token output: -> (batch, d_model)
7. LayerNorm + Linear -> class logits
```

### 15.3 ViT MLP Block

```
ViTMLP(x) = Dropout(Linear_2(Dropout(GELU(Linear_1(x)))))
```

Differences from original Transformer FFN:
- Uses GELU activation (smoother version of ReLU) instead of ReLU
- Dropout after each linear layer

### 15.4 ViT Block (Pre-Norm)

```
X = X + MultiHeadAttention(LN(X), LN(X), LN(X))
X = X + ViTMLP(LN(X))
```

### 15.5 ViT Hyperparameters (from d2l experiment)

```
img_size = 96, patch_size = 16
num_hiddens = 512, mlp_num_hiddens = 2048
num_heads = 8, num_blks = 2
emb_dropout = 0.1, blk_dropout = 0.1
```

**Scaling behavior:** ViTs underperform ResNets on small datasets (lack inductive biases of convolution: translation invariance, locality). On large datasets (300M+ images), ViTs significantly outperform ResNets.

---

## 16. Transformer Pretraining Paradigms

### 16.1 Encoder-Only (BERT)

**Architecture:** Transformer encoder only (bidirectional attention).

**Pretraining objective -- Masked Language Modeling (MLM):**
```
Input:  <cls> I <mask> this red car
Target: predict "love" at masked position
Loss:   cross-entropy on masked token predictions
```

- All tokens attend to all other tokens (no causal mask)
- <cls> token representation used for downstream classification
- 350M parameters, 250B training tokens

**Fine-tuning:** Add task-specific output layer on top of <cls> representation.

### 16.2 Encoder-Decoder (T5)

**Architecture:** Full Transformer encoder-decoder.

**Pretraining objective -- Span Corruption:**
```
Input:   I <X> this <Y>            (random spans replaced by sentinel tokens)
Target:  <X> love <Y> red car <Z>  (predict corrupted spans)
```

- Encoder: bidirectional attention (all tokens attend to all)
- Decoder: causal attention (each token attends only to past tokens)
- Cross-attention: decoder tokens attend to all encoder tokens
- 11B parameters (T5-11B), 1000B training tokens from C4 corpus
- All tasks framed as text-to-text with task description prefix

### 16.3 Decoder-Only (GPT Series)

**Architecture:** Transformer decoder only (causal/autoregressive attention).

**Pretraining objective -- Language Modeling:**
```
Input:  <bos> I love this red car
Target: I love this red car <eos>    (shifted by one token)
```

- Causal attention: each token attends only to itself and past tokens
- No encoder, no cross-attention

**GPT scaling:**
```
GPT:    117M parameters
GPT-2:  1.5B parameters, 40GB text
GPT-3:  175B parameters, 300B tokens
```

**GPT-2 modifications from original Transformer:**
- Pre-normalization (LayerNorm before sublayers)
- Improved initialization and weight scaling

**In-context learning (GPT-3):**
- Zero-shot: task description only, no examples
- One-shot: task description + 1 input-output example
- Few-shot: task description + several examples
- No parameter update required

### 16.4 Scaling Laws

Performance scales as power law with:
```
L(N) ~ N^{-alpha_N}     (model parameters, excluding embeddings)
L(D) ~ D^{-alpha_D}     (dataset size in tokens)
L(C) ~ C^{-alpha_C}     (training compute in PetaFLOP/s-days)
```

Key findings:
- All three factors must be scaled together for optimal performance
- Larger models achieve better sample efficiency (need fewer tokens for same performance)
- Performance improves smoothly and predictably with scale

---

## 17. Architectural Comparison Summary

| Component | Vanilla RNN | GRU | LSTM | Transformer |
|-----------|-------------|-----|------|-------------|
| Recurrence | Yes | Yes | Yes | No |
| Gates | 0 | 2 (reset, update) | 3 (input, forget, output) | 0 |
| Cell state | No | No | Yes | No |
| Parallelizable | No | No | No | Yes |
| Max path length | O(n) | O(n) | O(n) | O(1) |
| Parameters per layer | dh + h^2 + h | 3(dh + h^2 + h) | 4(dh + h^2 + h) | Attention + FFN |
| Handles long-range | Poor | Good | Good | Excellent |
| Key mechanism | Hidden state | Gating | Gating + cell | Self-attention |

---

## Raw Algorithms, Equations, and Technical Knowledge

---

# OPTIMIZATION ALGORITHMS (Chapter 12)

---

## 12.1 Optimization vs. Deep Learning

**Empirical risk** (training objective):
```
f(x) = (1/n) * sum_{i=1}^{n} f_i(x)
```
- `f_i(x)`: loss on training example i
- `x`: parameter vector
- `n`: number of training examples

**Key distinction**: Minimizing training error != minimizing generalization error. Optimization minimizes an objective; deep learning seeks a model that generalizes.

**Optimization challenges in deep learning**:

1. **Local minima**: f(x) < f(x') for all x' in vicinity of x, but not globally minimal. Noise from SGD can escape local minima.

2. **Saddle points**: All gradients vanish but point is neither local min nor local max.
   - Hessian eigenvalue test at zero-gradient position:
     - All eigenvalues > 0 --> local minimum
     - All eigenvalues < 0 --> local maximum
     - Mixed signs --> saddle point
   - In high dimensions, saddle points are far more common than local minima.

3. **Vanishing gradients**: e.g., f(x) = tanh(x), f'(4) = 1 - tanh^2(4) = 0.0013. Gradient nearly zero far from origin.

---

## 12.2 Convexity

### Definitions

**Convex set**: A set X is convex if for any a, b in X and all lambda in [0, 1]:
```
lambda * a + (1 - lambda) * b in X
```

**Convex function**: f: X -> R is convex if for all x, x' in X and all lambda in [0, 1]:
```
lambda * f(x) + (1 - lambda) * f(x') >= f(lambda * x + (1 - lambda) * x')
```
- `X`: convex set (domain)
- `lambda`: interpolation weight in [0, 1]

### Jensen's Inequality
```
sum_i alpha_i * f(x_i) >= f(sum_i alpha_i * x_i)
E[f(X)] >= f(E[X])
```
- `alpha_i`: nonneg reals summing to 1
- `X`: random variable

Application to log-likelihood:
```
E_{Y~P(Y)}[-log P(X|Y)] >= -log P(X)
```

### Properties

**Local minima of convex functions are global minima.** Proof by contradiction using the definition of convexity.

**Below sets of convex functions are convex**:
```
S_b = {x | x in X and f(x) <= b}
```

**Second derivative test**:
- 1D: f is convex iff f''(x) >= 0 for all x
- nD: f is convex iff Hessian H = nabla^2 f is positive semidefinite (x^T H x >= 0 for all x)

### Constraints

**Constrained optimization**:
```
minimize_x  f(x)
subject to  c_i(x) <= 0  for all i in {1, ..., n}
```

**Lagrangian**:
```
L(x, alpha_1, ..., alpha_n) = f(x) + sum_{i=1}^{n} alpha_i * c_i(x)
```
- `alpha_i >= 0`: Lagrange multipliers
- Solve via saddle point: maximize over alpha_i, minimize over x
- KKT condition: alpha_i = 0 when c_i(x) < 0 (inactive constraint)

**Penalty method**: Add lambda/2 * ||w||^2 to objective (weight decay is a special case).

**Projection onto convex set X**:
```
Proj_X(x) = argmin_{x' in X} ||x - x'||
```

**Gradient clipping as projection**:
```
g <- g * min(1, theta / ||g||)
```
- Projects g onto ball of radius theta

---

## 12.3 Gradient Descent

### One-Dimensional

**Taylor expansion**:
```
f(x + epsilon) = f(x) + epsilon * f'(x) + O(epsilon^2)
```

**Update rule**:
```
x <- x - eta * f'(x)
```
- `eta > 0`: learning rate (step size)
- `f'(x)`: derivative of objective at x
- Progress guaranteed when f'(x) != 0 and eta sufficiently small:
  `f(x - eta * f'(x)) = f(x) - eta * f'^2(x) + O(eta^2 * f'^2(x))`

### Multivariate

**Gradient**:
```
nabla f(x) = [df/dx_1, df/dx_2, ..., df/dx_d]^T
```

**Update rule**:
```
x <- x - eta * nabla f(x)
```
- `x in R^d`: parameter vector
- `eta`: learning rate

### Newton's Method

**Second-order Taylor expansion**:
```
f(x + epsilon) = f(x) + epsilon^T * nabla f(x) + (1/2) * epsilon^T * H * epsilon + O(||epsilon||^3)
```
- `H = nabla^2 f(x)`: Hessian matrix (d x d)

**Newton update**:
```
epsilon = -H^{-1} * nabla f(x)
x <- x + epsilon
```

**Convergence**: Quadratic convergence for convex functions:
```
e^{(k+1)} <= c * (e^{(k)})^2
```
- `e^{(k)}`: distance from optimality at step k

**Preconditioning** (diagonal Hessian approximation):
```
x <- x - eta * diag(H)^{-1} * nabla f(x)
```
- Per-coordinate learning rates; avoids O(d^2) cost of full Hessian

---

## 12.4 Stochastic Gradient Descent (SGD)

### Full Gradient vs. Stochastic Gradient

**Full gradient**:
```
nabla f(x) = (1/n) * sum_{i=1}^{n} nabla f_i(x)
```
- Cost per iteration: O(n)

**Stochastic gradient update** (sample single index i uniformly):
```
x <- x - eta * nabla f_i(x)
```
- Cost per iteration: O(1)
- Unbiased estimate: E_i[nabla f_i(x)] = nabla f(x)

### Dynamic Learning Rate Schedules

```
eta(t) = eta_i              if t_i <= t <= t_{i+1}    [piecewise constant]
eta(t) = eta_0 * e^{-lambda*t}                        [exponential decay]
eta(t) = eta_0 * (beta*t + 1)^{-alpha}                [polynomial decay]
```
- `eta_0`: initial learning rate
- `lambda, alpha, beta`: decay hyperparameters
- Polynomial with alpha = 0.5 is well-suited for convex problems

### Convergence Bound (Convex Case)

For convex f with stochastic gradients bounded by ||nabla f_i(x)|| <= L:
```
E[R(x_bar)] - R* <= (r^2 + L^2 * sum_{t=1}^{T} eta_t^2) / (2 * sum_{t=1}^{T} eta_t)
```
- `R(x)`: expected risk
- `R*`: minimum risk
- `r = ||x_1 - x*||`: initial distance from optimum
- `x_bar = (sum eta_t * x_t) / (sum eta_t)`: weighted average of iterates
- Optimal fixed rate: eta = r / (L * sqrt(T)), giving convergence rate O(1/sqrt(T))

### Sampling

Probability of choosing sample i at least once (with replacement, n draws):
```
P(choose i) = 1 - (1 - 1/n)^n ~ 1 - e^{-1} ~ 0.63
```
Sampling without replacement is preferred (lower variance, better data efficiency).

---

## 12.5 Minibatch Stochastic Gradient Descent

### Update Rule

```
g_t = nabla_w (1/|B_t|) * sum_{i in B_t} f(x_i, w)
w <- w - eta_t * g_t
```
- `B_t`: minibatch of size b = |B_t| drawn uniformly
- `g_t`: minibatch gradient (unbiased estimate of full gradient)
- Standard deviation of g_t reduced by factor b^{-1/2} compared to single-sample SGD

### Computational Efficiency

Matrix multiply performance (256x256 matrices):
- Element-wise: ~0.013 GFLOPS
- Column-wise: ~0.159 GFLOPS
- Full block: ~3.2 GFLOPS
- Minibatch (64 columns): ~1.258 GFLOPS

**Tradeoff**: Larger minibatches give better hardware utilization but diminishing statistical returns.

| Variant | Batch Size | Cost/Epoch | Convergence |
|---------|-----------|------------|-------------|
| Full GD | n (all data) | Low iterations, high per-iter cost | Smooth but slow |
| SGD | 1 | High iterations, low per-iter cost | Noisy |
| Minibatch SGD | b (e.g., 32-256) | Best of both | Good balance |

---

## 12.6 Momentum

### Update Equations
```
v_t = beta * v_{t-1} + g_{t,t-1}
x_t = x_{t-1} - eta_t * v_t
```
- `v_t`: velocity (momentum buffer), initialized v_0 = 0
- `beta in (0, 1)`: momentum coefficient
- `g_{t,t-1}`: gradient at step t using weights from t-1
- `eta_t`: learning rate

**Expanded form** (exponentially weighted sum of past gradients):
```
v_t = sum_{tau=0}^{t-1} beta^tau * g_{t-tau, t-tau-1}
```

### Hyperparameters and Defaults
| Parameter | Symbol | Typical Value | Notes |
|-----------|--------|---------------|-------|
| Momentum coefficient | beta | 0.9 | Higher = longer memory |
| Learning rate | eta | 0.01 | Often reduced when beta is large |

### Effective Step Size

Sum of weights: sum_{tau=0}^{inf} beta^tau = 1/(1-beta)

Effective step size = eta / (1 - beta). With beta=0.9, effective step is 10x the learning rate.

### Convergence (Quadratic Case)

For f(x) = (lambda/2) * x^2, momentum dynamics:
```
[v_{t+1}]   [  beta      lambda  ] [v_t]
[x_{t+1}] = [-eta*beta  1-eta*lambda] [x_t]
```
Convergence requires: 0 < eta*lambda < 2 + 2*beta (wider range than GD's 0 < eta*lambda < 2).

### When to Use
- **Advantages**: Accelerates convergence on ill-conditioned problems; smooths noisy gradients; escapes shallow local minima
- **Disadvantages**: Extra state variable per parameter; hyperparameter tuning (beta); can overshoot

---

## 12.7 AdaGrad

### Update Equations
```
g_t = nabla_w l(y_t, f(x_t, w))
s_t = s_{t-1} + g_t^2
w_t = w_{t-1} - (eta / sqrt(s_t + epsilon)) * g_t
```
- `s_t`: accumulated squared gradients (per-coordinate), initialized s_0 = 0
- `g_t^2`: element-wise square
- `eta`: global learning rate
- `epsilon`: small constant (~1e-6) preventing division by zero
- All operations are **coordinate-wise** (element-wise)

### Effective Per-Coordinate Learning Rate
```
eta_i^{eff} = eta / sqrt(s_{t,i} + epsilon)
```
- Coordinates with large cumulative gradients get smaller learning rates
- Coordinates with small cumulative gradients retain larger learning rates
- s_t grows ~linearly, giving O(t^{-1/2}) effective learning rate decay

### Hyperparameters and Defaults
| Parameter | Symbol | Typical Value | Notes |
|-----------|--------|---------------|-------|
| Learning rate | eta | 0.01-0.1 | Often larger than SGD due to adaptive scaling |
| Epsilon | epsilon | 1e-6 | Numerical stability |

### Preconditioning Interpretation

Condition number: kappa = Lambda_1 / Lambda_d (ratio of largest to smallest eigenvalue).

AdaGrad approximates diagonal preconditioning:
```
Q_tilde = diag^{-1/2}(Q) * Q * diag^{-1/2}(Q)
```
Uses gradient magnitude as proxy for diagonal of Hessian.

### When to Use
- **Advantages**: Excellent for sparse features (NLP, advertising); automatic per-parameter learning rates; no momentum tuning needed
- **Disadvantages**: Learning rate monotonically decreases (can become too small); poor for non-convex deep learning (premature convergence)

---

## 12.8 RMSProp

### Update Equations
```
s_t = gamma * s_{t-1} + (1 - gamma) * g_t^2
x_t = x_{t-1} - (eta / sqrt(s_t + epsilon)) * g_t
```
- `s_t`: exponential moving average of squared gradients
- `gamma in (0, 1)`: decay rate for squared gradient accumulator
- `eta`: learning rate (decoupled from adaptive scaling)
- `epsilon > 0`: typically 1e-6

**Key difference from AdaGrad**: Uses leaky average instead of cumulative sum, preventing learning rate from monotonically decaying to zero.

### Expanded Form
```
s_t = (1 - gamma) * (g_t^2 + gamma*g_{t-1}^2 + gamma^2*g_{t-2}^2 + ...)
```
Effective window: ~1/(1-gamma) steps. With gamma=0.9, averages over ~10 recent squared gradients.

### Hyperparameters and Defaults
| Parameter | Symbol | Typical Value | Notes |
|-----------|--------|---------------|-------|
| Decay rate | gamma | 0.9 | Controls memory length |
| Learning rate | eta | 0.01 | Independent of adaptive scaling |
| Epsilon | epsilon | 1e-6 | Numerical stability |

### When to Use
- **Advantages**: Fixes AdaGrad's decaying learning rate; works well for non-convex problems; good for RNNs
- **Disadvantages**: Two hyperparameters to tune; no bias correction (unlike Adam)

---

## 12.9 Adadelta

### Update Equations
```
s_t = rho * s_{t-1} + (1 - rho) * g_t^2                    [squared gradient EMA]
g'_t = (sqrt(Delta x_{t-1} + epsilon) / sqrt(s_t + epsilon)) * g_t   [rescaled gradient]
x_t = x_{t-1} - g'_t                                        [parameter update]
Delta x_t = rho * Delta x_{t-1} + (1 - rho) * g'_t^2       [squared update EMA]
```
- `s_t`: EMA of squared gradients
- `Delta x_t`: EMA of squared parameter updates, initialized Delta x_0 = 0
- `rho in (0, 1)`: decay rate (typically 0.9)
- `epsilon`: small constant (~1e-5)
- `g'_t`: rescaled gradient

### Key Property
**No explicit learning rate**. Uses ratio of RMS of past parameter updates to RMS of current gradient as implicit step size. Units match between numerator and denominator.

### Hyperparameters and Defaults
| Parameter | Symbol | Typical Value | Notes |
|-----------|--------|---------------|-------|
| Decay rate | rho | 0.9 | Half-life ~10 steps |
| Epsilon | epsilon | 1e-5 | Numerical stability |

### When to Use
- **Advantages**: No learning rate to tune; unit-corrected updates; robust
- **Disadvantages**: Can be slower than Adam; less popular/supported; still needs rho tuning

---

## 12.10 Adam (Adaptive Moment Estimation)

### Update Equations
```
v_t = beta_1 * v_{t-1} + (1 - beta_1) * g_t           [1st moment estimate (momentum)]
s_t = beta_2 * s_{t-1} + (1 - beta_2) * g_t^2         [2nd moment estimate (adaptive LR)]
v_hat_t = v_t / (1 - beta_1^t)                          [bias-corrected 1st moment]
s_hat_t = s_t / (1 - beta_2^t)                          [bias-corrected 2nd moment]
x_t = x_{t-1} - eta * v_hat_t / (sqrt(s_hat_t) + epsilon)   [parameter update]
```
- `v_t`: first moment (mean of gradients), initialized v_0 = 0
- `s_t`: second moment (mean of squared gradients), initialized s_0 = 0
- `g_t`: gradient at step t
- `beta_1`: decay for first moment
- `beta_2`: decay for second moment
- `eta`: learning rate
- `epsilon`: numerical stability constant
- `t`: timestep (starts at 1)
- Bias correction: needed because v_0 = s_0 = 0 causes underestimation early on

### Bias Correction Derivation

Since v_0 = 0: E[v_t] = (1 - beta_1^t) * E[g_t], so dividing by (1 - beta_1^t) removes bias.

### Hyperparameters and Defaults
| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Learning rate | eta | 0.001 | Step size |
| 1st moment decay | beta_1 | 0.9 | Momentum-like averaging |
| 2nd moment decay | beta_2 | 0.999 | Slow-moving variance estimate |
| Epsilon | epsilon | 1e-6 to 1e-8 | Numerical stability |

### Yogi Variant (fixes Adam divergence)

Standard Adam reformulated:
```
s_t = s_{t-1} + (1 - beta_2) * (g_t^2 - s_{t-1})
```

Yogi modification:
```
s_t = s_{t-1} + (1 - beta_2) * g_t^2 * sign(g_t^2 - s_{t-1})
```
- Magnitude of update to s_t no longer depends on deviation size
- More conservative: only increases s_t when gradient variance exceeds current estimate
- Prevents s_t from forgetting past values too quickly

### AdamW (Decoupled Weight Decay)

Standard Adam with L2 regularization applies weight decay through the gradient, which interacts with adaptive learning rates. AdamW decouples them:
```
v_t = beta_1 * v_{t-1} + (1 - beta_1) * g_t
s_t = beta_2 * s_{t-1} + (1 - beta_2) * g_t^2
v_hat_t = v_t / (1 - beta_1^t)
s_hat_t = s_t / (1 - beta_2^t)
x_t = x_{t-1} - eta * (v_hat_t / (sqrt(s_hat_t) + epsilon) + lambda * x_{t-1})
```
- `lambda`: weight decay coefficient (applied directly to parameters, NOT through gradient)
- `g_t` is the gradient of the loss only (no L2 term)
- Prevents adaptive scaling from diminishing the regularization effect

### When to Use Adam
- **Advantages**: Combines momentum + adaptive LR; works well out-of-the-box; bias correction handles early steps; widely used default optimizer
- **Disadvantages**: Can diverge in some convex settings (Reddi et al., 2019); generalization sometimes worse than SGD+momentum for vision tasks; more memory (two state variables per parameter)

---

## 12.11 Learning Rate Scheduling

### Square Root Decay
```
eta(t) = eta_0 * (t + 1)^{-0.5}
```
- `eta_0`: base learning rate
- `t`: current epoch/step

### Factor (Multiplicative) Decay
```
eta_{t+1} = max(eta_min, eta_t * alpha)
```
- `alpha in (0, 1)`: multiplicative factor per step
- `eta_min`: minimum learning rate floor

### Multi-Step (Piecewise Constant) Decay
```
eta_{t+1} = eta_t * alpha   whenever t in S
```
- `S`: set of milestone epochs (e.g., {15, 30})
- `alpha`: decay factor at each milestone (e.g., 0.5)
- Learning rate constant between milestones

### Cosine Annealing
```
eta_t = eta_T + (eta_0 - eta_T) / 2 * (1 + cos(pi * t / T))
```
- `eta_0`: initial (maximum) learning rate
- `eta_T`: final (minimum) learning rate
- `T`: total number of training steps
- For t > T: eta_t = eta_T (pinned)
- Smooth, gradual decay; less aggressive at start than polynomial

### Warmup

Linear warmup for first `W` steps:
```
eta(t) = eta_warmup_begin + (eta_0 - eta_warmup_begin) * t / W    for t < W
```
- `W`: number of warmup steps
- `eta_warmup_begin`: starting learning rate (often 0)
- `eta_0`: target learning rate after warmup
- Prevents divergence from random initialization in deep networks

### Warmup + Cosine Annealing (Combined)
```
eta(t) = { eta_warmup_begin + (eta_0 - eta_warmup_begin) * t / W,     if t < W
         { eta_T + (eta_0 - eta_T)/2 * (1 + cos(pi*(t-W)/(T-W))),    if W <= t <= T
         { eta_T,                                                        if t > T
```

### Practical Guidance
- Piecewise constant: simple, effective when you know training dynamics
- Cosine: good default for computer vision; smooth decay prevents sharp transitions
- Warmup: essential for large batch training, Transformers, and deep networks
- Polynomial decay: theoretically grounded for convex problems

---

## Optimizer Comparison Summary

| Optimizer | State Variables | Learning Rate | Key Feature |
|-----------|----------------|---------------|-------------|
| SGD | None | Global, fixed or scheduled | Baseline |
| Momentum | v (velocity) | Global | Accelerates via gradient averaging |
| AdaGrad | s (sum of sq. gradients) | Per-parameter, decreasing | Good for sparse features |
| RMSProp | s (EMA of sq. gradients) | Per-parameter, stable | Fixes AdaGrad decay |
| Adadelta | s, Delta_x | Per-parameter, implicit | No explicit learning rate |
| Adam | v (1st moment), s (2nd moment) | Per-parameter, bias-corrected | Combines momentum + RMSProp |
| AdamW | v, s | Per-parameter + decoupled decay | Proper weight decay with Adam |

---

# COMPUTATIONAL PERFORMANCE (Chapter 13)

---

## 13.1 Compilers and Interpreters

### Imperative vs. Symbolic Programming

| Aspect | Imperative | Symbolic |
|--------|-----------|----------|
| Execution | Line-by-line | After full graph compilation |
| Debugging | Easy (print, breakpoints) | Harder |
| Performance | Slower (Python overhead) | Faster (compiler optimizations) |
| Portability | Python-dependent | Language-independent |

### Hybrid Programming (PyTorch)
- `torch.jit.script(model)`: Converts imperative PyTorch to optimized TorchScript
- Enables: graph optimization, constant folding, dead code elimination
- Benefit: Eliminates Python interpreter bottleneck on multi-GPU setups

---

## 13.2 Asynchronous Computation

### GPU Asynchrony Model
- PyTorch GPU operations are **asynchronous by default**
- Frontend (Python) enqueues operations; backend (C++) executes them
- `torch.cuda.synchronize(device)`: Forces completion of all queued GPU operations

### Architecture
```
Python Frontend --> C++ Backend --> GPU Execution Queue
```
- Backend tracks dependencies via computational graph
- Independent operations can execute in parallel without explicit coding
- Synchronize per minibatch to prevent excessive memory consumption from task queue overflow

---

## 13.3 Automatic Parallelism

### GPU-GPU Parallelism
- Independent operations on different GPUs execute simultaneously without explicit scheduling
- Total time for parallel ops ~ max(time_gpu1, time_gpu2) instead of sum

### Computation-Communication Overlap
- `non_blocking=True` in `.to()` and `.copy_()` bypasses synchronization
- Allows GPU computation to overlap with CPU<->GPU data transfer
- Can copy gradient y[i-1] to CPU while computing y[i] on GPU

### Key Interconnect Bandwidths
| Connection | Bandwidth |
|-----------|-----------|
| PCIe 4.0 x16 | ~32 GB/s (16 GB/s per direction) |
| NVLink (V100, per link) | ~300 Gbit/s bidirectional (~18 GB/s per direction) |
| 100 GbE Ethernet | ~10 GB/s |
| CPU Memory (DDR4) | 20-100 GB/s |

---

## 13.4 Hardware

### Memory Hierarchy

| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| CPU Registers | ~tens | 0 cycles | At clock speed |
| L1 Cache | 32-64 KB | 1.5 ns (4 cycles) | Very high |
| L2 Cache | 256-512 KB/core | 5 ns (12-17 cycles) | High |
| L3 Cache | 4-256 MB (shared) | 16-40 ns | Moderate |
| Main RAM (DDR4) | GBs | 46-120 ns | 20-100 GB/s |
| GPU Shared Memory | Per SM | 30 ns (30-90 cycles) | Very high |
| GPU Global Memory (HBM/GDDR6) | GBs | 200 ns (200-800 cycles) | 500+ GB/s |

### Latency Reference Table

| Action | Time |
|--------|------|
| L1 cache hit | 1.5 ns |
| Floating-point add/mult/FMA | 1.5 ns |
| L2 cache hit | 5 ns |
| Branch mispredict | 6 ns |
| L3 cache hit (unshared) | 16 ns |
| L3 cache hit (shared, other core) | 25 ns |
| Mutex lock/unlock | 25 ns |
| 64MB memory ref (local CPU) | 46 ns |
| 64MB memory ref (remote CPU) | 70 ns |
| Transfer 1MB via NVLink | 30 us |
| Transfer 1MB via PCIe | 80 us |
| Round trip same datacenter | 500 us |
| Read 1MB sequential from NVMe SSD | 208 us |
| Read 1MB sequential from SATA SSD | 2 ms |
| Read 1MB sequential from HDD | 5 ms |
| Random disk access (seek+rotation) | 10 ms |

### GPU Architecture (NVIDIA Turing)

- **Processing block**: 16 integer units + 16 FP units + 2 tensor cores
- **Streaming Multiprocessor (SM)**: 4 processing blocks
- **Graphics Processing Cluster**: 12 SMs
- **Tensor cores**: Optimized for 4x4 to 16x16 matrix operations (FP16, INT8, INT4)
- RTX 2080 Ti: 4,352 CUDA cores, 352-bit memory bus, GDDR6

### Precision Tradeoffs
| Precision | Use Case | Notes |
|-----------|----------|-------|
| FP32 | Training (gradient accumulation) | Default, avoids underflow |
| FP16 | Training (mixed precision) | 2x throughput, needs loss scaling |
| BF16 | Training (mixed precision) | Same range as FP32, less precision |
| INT8 | Inference | 4x throughput vs FP32 |
| INT4 | Inference | 8x throughput vs FP32 |

### Storage Performance

| Device | IOPs | Sequential BW | Latency |
|--------|------|---------------|---------|
| HDD (7200 RPM) | ~100 | 100-200 MB/s | ~8 ms (rotation) |
| SATA SSD | 100K-500K | ~550 MB/s | ~500 us |
| NVMe SSD | 100K-500K | 1-8 GB/s | ~120 us (read), ~30 us (write) |

### Design Principles
1. Prefer large bulk transfers over many small ones (amortize overhead)
2. Align data structures to memory boundaries (64-bit for CPU, match tensor core sizes for GPU)
3. Traverse memory sequentially (forward direction for cache prefetching)
4. Match algorithm to hardware (fit working set in cache for order-of-magnitude speedup)

---

## 13.5 Data Parallelism (Multi-GPU Training)

### Three Parallelism Strategies

| Strategy | Description | Communication | Recommended |
|----------|-------------|---------------|-------------|
| **Model parallelism** | Split network layers across GPUs | Activations + gradients between layers | Only for very large models |
| **Layer-wise parallelism** | Split channels/units within layers | All-to-all after each layer | Not recommended (high BW cost) |
| **Data parallelism** | Same model on each GPU, split data | Aggregate gradients after each batch | Yes (simplest, most practical) |

### Data Parallelism Algorithm

Given k GPUs and minibatch of size B:
```
1. Split minibatch into k parts of size B/k
2. Each GPU i:
   a. Forward pass on its data shard
   b. Compute local loss and gradient g_i
3. AllReduce: aggregate gradients g = sum(g_i) across all GPUs
4. Each GPU updates its local copy of parameters using g
```

**Scaling rule**: When training on k GPUs, increase minibatch size to k*B and scale learning rate accordingly.

### AllReduce Operation

Naive implementation:
```python
def allreduce(data):   # data = list of tensors on different GPUs
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

### Gradient Aggregation Across Workers
```
g_i = sum_{k in workers} sum_{j in GPUs} g_{ijk}
```
- Commutative reduction: order doesn't matter
- Independent across different parameter groups i

### PyTorch API
```python
net = nn.DataParallel(net, device_ids=devices)
```

---

## 13.6 Concise Multi-GPU (nn.DataParallel)

### Key Steps
1. Initialize network parameters on all devices
2. For each minibatch: scatter data across devices
3. Each device computes forward + backward independently
4. Aggregate gradients (handled automatically by DataParallel)
5. Update parameters

### Batch Normalization Note
When using data parallelism, batch normalization statistics are computed per-GPU on the local sub-batch. For synchronized batch norm across GPUs, use `nn.SyncBatchNorm`.

---

## 13.7 Parameter Servers and Distributed Training

### Ring Synchronization (Ring-AllReduce)

For n GPUs connected in a ring:
1. Split gradients into n chunks
2. Each GPU i starts sending chunk i to neighbor
3. Each receiving GPU adds received chunk to its local chunk, forwards sum
4. After n-1 steps: each GPU has one fully-reduced chunk
5. Another n-1 steps to broadcast all reduced chunks

**Time complexity**: O((n-1)/n) ~ O(1) independent of ring size (asymptotically).

**Example**: 160 MB across 8 V100 GPUs via NVLink:
```
Time ~ 2 * 160MB / (3 * 18 GB/s) ~ 6 ms
```

### NVLink Topology (8-GPU DGX)
- Each GPU: 6 NVLink connections
- Network decomposes into two rings:
  - Ring 1 (double NVLink BW): 1-2-3-4-5-6-7-8-1
  - Ring 2 (single NVLink BW): 1-4-6-3-5-8-2-7-1

### Multi-Machine Training

**Distributed data parallel with parameter servers**:
```
1. Each machine reads different batch, split across local GPUs
2. Local gradients aggregated on one GPU per machine
3. Gradients sent to CPU, then to parameter server(s)
4. Parameter server aggregates all machine gradients
5. Updated parameters broadcast back to all machines/GPUs
```

**Scaling**: Single parameter server is bottleneck at O(m) for m workers. With n parameter servers, each storing O(1/n) parameters, total time is O(m/n). Setting m = n gives constant scaling.

### Key-Value Store Abstraction
- **push(key, value)**: Send gradient to parameter server (accumulated by summing)
- **pull(key, value)**: Retrieve aggregated gradient from parameter server

### Communication Optimization

For 160 MB gradients across 4 GPUs via PCIe (16 GB/s per link):
| Strategy | Time |
|----------|------|
| All-to-one GPU | 30 ms send + 30 ms broadcast = 60 ms |
| All-to-CPU | 4 * 10 ms = 40 ms + 40 ms back = 80 ms |
| Split 4 ways, aggregate in parallel | 7.5 ms + 7.5 ms = 15 ms |

### Overlapping Computation and Communication
- Begin synchronizing gradients of earlier layers while later layers still computing backprop
- Implemented by frameworks like Horovod (Sergeev and Del Balso, 2018)

---

## Mixed Precision Training

### Principle
- Forward pass: FP16 (or BF16) for speed
- Gradient accumulation: FP32 for numerical stability
- Master weights: FP32 copy maintained

### Loss Scaling
- FP16 has limited range: gradients can underflow
- Multiply loss by scale factor S before backward pass
- Divide gradients by S after backward pass, before optimizer step
- Dynamic loss scaling: increase S when no overflow; decrease when overflow detected

### Memory Savings
| Data | FP32 | FP16 |
|------|------|------|
| Model parameters | 4 bytes/param | 2 bytes/param |
| Activations | 4 bytes/element | 2 bytes/element |
| Gradients | 4 bytes/param | 2 bytes/param |

---

## Hardware Design Implications for Deep Learning

### Key Bottlenecks
1. **Memory bandwidth**: GPU compute >> memory BW; must maximize data reuse
2. **Communication bandwidth**: PCIe << NVLink << on-chip; minimize cross-device transfers
3. **Python overhead**: Single-threaded interpreter; use compiled graphs, large batch operations

### Performance Rules
- Arithmetic intensity = FLOPs / bytes transferred. Higher = better hardware utilization.
- Target: keep data in cache/registers. Working set should fit in L1/L2 for CPU, shared memory for GPU.
- Prefer fewer large operations over many small ones (amortize kernel launch overhead ~10 us on GPU).
- Align convolution dimensions to tensor core sizes for optimal GPU utilization.

---

## 1. COMPUTER VISION (End of Chapter 14)

> **Note**: This source file begins at the tail end of Chapter 14. The bulk of Computer Vision content (augmentation, fine-tuning, object detection, anchor boxes, IoU, NMS, R-CNN variants, SSD, FCN, neural style transfer) was covered in an earlier part of the book. Only the Kaggle dog-breed classification tail is present here.

### 1.1 ImageNet Dog Breed Classification (Kaggle)

**Transfer Learning Pipeline**:
1. Use a pre-trained model (e.g., ResNet) on full ImageNet as feature extractor
2. Replace final fully connected layer with custom output network matching target classes
3. Only train the custom output network (freeze backbone or fine-tune with small LR)
4. Use image augmentation adapted to target dataset dimensions

**Key Insight**: For ImageNet subsets, leveraging pre-trained models to extract features and training only a small output network reduces computational time and memory cost significantly.

---

## 2. NLP PRETRAINING (Chapter 15)

### 2.1 Word2Vec

#### 2.1.1 Skip-Gram Model
- **Objective**: Predict context words given a center word
- Center word vector: `v_c in R^d`
- Context word vector: `u_o in R^d`

**Conditional Probability (Softmax)**:
```
P(w_o | w_c) = exp(u_o^T v_c) / sum_{i in V} exp(u_i^T v_c)
```
- `w_c` = center word, `w_o` = context word
- `V` = vocabulary
- Denominator sums over entire vocabulary (expensive)

**Log-Likelihood for corpus**:
```
sum_{t=1}^{T} sum_{-m <= j <= m, j != 0} log P(w^(t+j) | w^(t))
```
- `T` = total words in corpus
- `m` = context window size

**Gradient for center word vector**:
```
d/d(v_c) log P(w_o | w_c) = u_o - sum_{j in V} P(w_j | w_c) u_j
```

#### 2.1.2 CBOW (Continuous Bag of Words)
- **Objective**: Predict center word given context words
- Context vector: `v_bar_o = (1/2m) sum_{-m<=j<=m, j!=0} v_{i_{t+j}}`

**Conditional Probability**:
```
P(w_c | W_o) = exp(u_c^T v_bar_o) / sum_{i in V} exp(u_i^T v_bar_o)
```
- `W_o` = set of context words
- `v_bar_o` = averaged context word vectors

#### 2.1.3 Negative Sampling
- Approximates the expensive softmax denominator
- For each positive pair, sample `K` negative examples

**Loss (per center-context pair)**:
```
-log sigma(u_o^T v_c) - sum_{k=1}^{K} log sigma(-u_{h_k}^T v_c)
```
- `sigma(x) = 1/(1 + exp(-x))` = sigmoid function
- `h_1, ..., h_K` = indices of K noise words
- `K` = number of negative samples (typically 5-20)

**Noise distribution**: `P(w) proportional to f(w)^{3/4}`, where `f(w)` = word frequency

#### 2.1.4 Hierarchical Softmax
- Uses a binary tree (e.g., Huffman tree) over vocabulary
- Path from root to leaf represents a word
- Reduces softmax from `O(|V|)` to `O(log|V|)` per word
- Each internal node has a parameter vector; probability is product of sigmoid decisions along path

#### 2.1.5 Subword Sampling (Subsampling)
- Frequent words (e.g., "the", "a") provide less information
- Discard word `w_i` with probability:
```
P(discard w_i) = max(1 - sqrt(t / f(w_i)), 0)
```
- `t` = threshold (e.g., 1e-4), `f(w_i)` = relative frequency of word `w_i`

### 2.2 GloVe (Global Vectors for Word Representation)

**Co-occurrence matrix**: `X` where `x_{ij}` = count of word `j` in context of word `i`

**Loss Function**:
```
L = sum_{i,j} h(x_{ij}) (u_j^T v_i + b_i + c_j - log x_{ij})^2
```
- `v_i` = center word vector for word `i`
- `u_j` = context word vector for word `j`
- `b_i, c_j` = scalar bias terms
- `h(x)` = weight function

**Weight Function**:
```
h(x) = (x/c)^alpha   if x < c
h(x) = 1             otherwise
```
- `c` = clipping threshold (e.g., 100)
- `alpha` = power parameter (e.g., 0.75)

**Key Property**: Word vector `w_i = v_i + u_i` (sum of center and context vectors) used as final embedding.

### 2.3 Subword Tokenization

#### 2.3.1 Byte Pair Encoding (BPE)
**Algorithm**:
1. Initialize vocabulary with all individual characters + end-of-word symbol
2. Repeat for desired number of merges:
   a. Count all consecutive symbol pairs in corpus
   b. Find the most frequent pair
   c. Merge that pair into a new symbol
   d. Add new symbol to vocabulary
3. Final vocabulary = initial characters + all merged symbols

**Properties**:
- Balances between character-level and word-level tokenization
- Handles OOV (out-of-vocabulary) words by decomposing into subword units
- Merge operations are deterministic given training corpus

#### 2.3.2 WordPiece
- Similar to BPE but uses likelihood-based criterion instead of frequency
- Selects merge that maximizes likelihood of training data
- Used in BERT tokenization

### 2.4 Word Similarity and Analogy

**Cosine Similarity**:
```
cos(w_A, w_B) = (w_A . w_B) / (||w_A|| * ||w_B||)
```

**Analogy Task**: "A is to B as C is to ?"
```
argmax_{d in V, d != A,B,C} cos(w_d, w_B - w_A + w_C)
```

### 2.5 BERT (Bidirectional Encoder Representations from Transformers)

#### 2.5.1 Architecture
- Based on Transformer **encoder** only (bidirectional)
- **BERT_BASE**: L=12 layers, H=768 hidden, A=12 attention heads, 110M parameters
- **BERT_LARGE**: L=24 layers, H=1024 hidden, A=16 attention heads, 340M parameters

#### 2.5.2 Input Representation
```
Input = TokenEmbedding(x) + SegmentEmbedding(s) + PositionalEmbedding(p)
```
- `TokenEmbedding`: maps token IDs to vectors (includes special tokens `<cls>`, `<sep>`)
- `SegmentEmbedding`: learned embedding for sentence A vs sentence B (binary)
- `PositionalEmbedding`: learned positional embeddings (up to 512 positions)

**Input Format**: `<cls> sentence_A <sep> sentence_B <sep>`

#### 2.5.3 Pre-training Task 1: Masked Language Model (MLM)
- Randomly select 15% of tokens for prediction
- Of the selected tokens:
  - 80% replaced with `<mask>` token
  - 10% replaced with random token
  - 10% kept unchanged
- Predict original token using cross-entropy loss

**MLM Loss**:
```
L_MLM = -sum_{i in masked} log P(x_i | x_masked)
```

#### 2.5.4 Pre-training Task 2: Next Sentence Prediction (NSP)
- Binary classification: Is sentence B the actual next sentence after A?
- 50% positive pairs (consecutive sentences), 50% negative (random sentence)
- Uses `<cls>` token representation for classification

**NSP Loss**:
```
L_NSP = -[y log(p) + (1-y) log(1-p)]
```
- `y in {0, 1}` = binary label (IsNext / NotNext)
- `p` = predicted probability from MLP on `<cls>` output

**Total Pre-training Loss**:
```
L = L_MLM + L_NSP
```

#### 2.5.5 BERTEncoder Implementation
```python
class BERTEncoder(nn.Module):
    # token_embedding: Embedding(vocab_size, hidden_size)
    # segment_embedding: Embedding(2, hidden_size)
    # pos_embedding: Parameter(1, max_len, hidden_size)
    # blks: nn.Sequential of TransformerEncoderBlock layers

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

#### 2.5.6 MaskLM Head
```python
class MaskLM(nn.Module):
    # mlp: nn.Sequential(nn.LazyLinear(hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        # Gather hidden states at masked positions
        # Apply MLP to predict vocab distribution
```

#### 2.5.7 NextSentencePred Head
```python
class NextSentencePred(nn.Module):
    # output: nn.LazyLinear(2)

    def forward(self, X):
        return self.output(X)  # X = <cls> hidden state
```

### 2.6 GPT (Generative Pre-trained Transformer)

- Uses Transformer **decoder** only (autoregressive, left-to-right)
- Unidirectional language model: predicts next token
- Pre-training objective:
```
L_GPT = -sum_{t=1}^{T} log P(x_t | x_1, ..., x_{t-1}; theta)
```
- Fine-tuning: add task-specific linear layer on top

**Key Difference from BERT**: GPT is autoregressive (left-to-right), BERT is bidirectional (masked). GPT uses decoder with causal masking; BERT uses encoder without causal masking.

---

## 3. NLP APPLICATIONS (Chapter 16)

### 3.1 Sentiment Analysis

#### 3.1.1 BiRNN with Pre-trained Embeddings
**Architecture**:
1. Embedding layer initialized with pre-trained GloVe vectors
2. Bidirectional RNN (e.g., LSTM or GRU)
3. Concatenate final hidden states from both directions: `h = [h_forward; h_backward]`
4. Fully connected layer for classification

**Key**: Freeze pre-trained embeddings or fine-tune with small learning rate.

#### 3.1.2 TextCNN (Convolutional Neural Network for Text)
**Architecture**:
1. Embedding layer (1D representation: sequence_length x embedding_dim)
2. Multiple 1D convolutional layers with different kernel sizes (e.g., 3, 4, 5)
3. **Max-over-time pooling**: take max value from each feature map
   ```
   c_i = max(conv_i(x))  for each filter i
   ```
4. Concatenate pooled features from all kernel sizes
5. Fully connected layer + dropout for classification

**1D Convolution for Text**:
- Input: `(batch_size, sequence_length, embedding_dim)`
- Kernel of size `k` slides over `k` consecutive words
- Each kernel produces one feature map
- Multiple kernels capture n-gram patterns of different sizes

**Max-over-time Pooling**:
```
c = max_{j} h_j    (over all time steps j)
```
- Captures the most important feature regardless of position
- Produces fixed-size output regardless of input length

### 3.2 Natural Language Inference (NLI)

#### 3.2.1 Decomposable Attention Model
Three-step architecture for textual entailment:

**Step 1 -- Attend**:
- Compute soft-alignment between premise tokens `a_i` and hypothesis tokens `b_j`
- Attention weights: `e_{ij} = F(a_i)^T F(b_j)` where `F` is a feedforward network
- Aligned representations:
```
beta_i = sum_j [exp(e_{ij}) / sum_k exp(e_{ik})] * b_j     (for each premise token i)
alpha_j = sum_i [exp(e_{ij}) / sum_k exp(e_{kj})] * a_i     (for each hypothesis token j)
```

**Step 2 -- Compare**:
- Compare each token with its aligned counterpart:
```
v_{A,i} = G([a_i, beta_i])    (feedforward on concatenation)
v_{B,j} = G([b_j, alpha_j])
```

**Step 3 -- Aggregate**:
- Sum over compared representations:
```
v_A = sum_i v_{A,i}
v_B = sum_j v_{B,j}
```
- Final classification: `y_hat = H([v_A, v_B])` (feedforward + softmax)

**Output classes**: entailment, contradiction, neutral

#### 3.2.2 Fine-tuning BERT for NLI
- Input: `<cls> premise <sep> hypothesis <sep>`
- Use `<cls>` token output for classification
- Add MLP head: `hidden -> num_classes`
- Fine-tune entire BERT model

**Fine-tuning BERT for Other Tasks**:
- **Single text classification**: `<cls> text <sep>` -> classify `<cls>` output
- **Text pair classification**: `<cls> text_A <sep> text_B <sep>` -> classify `<cls>` output
- **Tagging**: use each token's output for per-token classification
- **Question answering**: predict start/end span positions

---

## 4. REINFORCEMENT LEARNING (Chapter 17)

### 4.1 Markov Decision Process (MDP)

**Definition**: Tuple `(S, A, T, r)` where:
- `S` = state space (set of all possible states)
- `A` = action space (set of all possible actions)
- `T: S x A x S -> [0, 1]` = transition probability function
  - `T(s, a, s') = P(s' | s, a)` = probability of transitioning to state `s'` given state `s` and action `a`
- `r: S x A -> R` = reward function
  - `r(s, a)` = expected immediate reward for taking action `a` in state `s`

**Policy**: `pi: S x A -> [0, 1]` where `pi(a | s)` = probability of taking action `a` in state `s`

**Trajectory**: `tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)`

### 4.2 Discounted Return

```
R(tau) = sum_{t=0}^{infinity} gamma^t r_t
```
- `gamma in [0, 1)` = discount factor
- `r_t` = reward at time step `t`
- Ensures finite sum for bounded rewards: `R(tau) <= r_max / (1 - gamma)`

### 4.3 Value Functions

**State-Value Function** (expected return starting from state `s` under policy `pi`):
```
V^pi(s) = E_pi[R(tau) | s_0 = s]
```

**Action-Value Function** (Q-function):
```
Q^pi(s, a) = E_pi[R(tau) | s_0 = s, a_0 = a]
```

**Relationship**:
```
V^pi(s) = sum_a pi(a|s) Q^pi(s, a)
```

### 4.4 Bellman Equations

**Bellman Equation for V**:
```
V^pi(s) = sum_a pi(a|s) [r(s,a) + gamma * sum_{s'} P(s'|s,a) V^pi(s')]
```

**Bellman Equation for Q**:
```
Q^pi(s,a) = r(s,a) + gamma * sum_{s'} P(s'|s,a) sum_{a'} pi(a'|s') Q^pi(s',a')
```

**Bellman Optimality Equations**:
```
V*(s) = max_a [r(s,a) + gamma * sum_{s'} P(s'|s,a) V*(s')]
Q*(s,a) = r(s,a) + gamma * sum_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

### 4.5 Value Iteration Algorithm

```
Initialize V_0(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V_{k+1}(s) = max_a { r(s,a) + gamma * sum_{s'} P(s'|s,a) V_k(s') }
```
- Converges to optimal value function `V*`
- Requires known transition probabilities (model-based)

**Policy Extraction**:
```
pi*(s) = argmax_a { r(s,a) + gamma * sum_{s'} P(s'|s,a) V*(s') }
```

### 4.6 Q-Learning (Model-Free)

**Update Rule**:
```
Q(s_t, a_t) <- (1 - alpha) Q(s_t, a_t) + alpha [r_t + gamma * (1 - done) * max_{a'} Q(s_{t+1}, a')]
```
- `alpha in (0, 1]` = learning rate
- `done` = 1 if `s_{t+1}` is terminal, 0 otherwise
- Model-free: does not require `P(s'|s,a)`

**Exploration Strategies**:

**Epsilon-Greedy**:
```
a = argmax_a Q(s, a)    with probability 1 - epsilon
a = random action         with probability epsilon
```
- `epsilon` typically decayed over time

**Softmax (Boltzmann) Exploration**:
```
pi(a | s) = exp(Q(s,a) / tau) / sum_{a'} exp(Q(s,a') / tau)
```
- `tau` = temperature parameter (high tau -> more uniform, low tau -> more greedy)

---

## 5. GENERATIVE ADVERSARIAL NETWORKS (Chapter 20)

### 5.1 GAN Framework

**Components**:
- **Generator** `G(z)`: maps latent variable `z ~ N(0, I)` to fake data `x' = G(z)`
- **Discriminator** `D(x)`: outputs probability that `x` is real data

**Discriminator Objective** (minimize cross-entropy):
```
min_D { -y log D(x) - (1-y) log(1 - D(x)) }
```
- `y = 1` for real data, `y = 0` for fake data

**Generator Objective** (practical form):
```
min_G { -log(D(G(z))) }
```
- Equivalent to maximizing `log(D(G(z)))` -- fool D into classifying fakes as real

### 5.2 Minimax Objective (Theoretical)

```
min_G max_D { E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))] }
```
- `p_data` = true data distribution
- `p_z` = prior noise distribution (e.g., `N(0, I)`)

**Optimal Discriminator** (for fixed G):
```
D*(x) = p_data(x) / (p_data(x) + p_G(x))
```

**At optimality**: `D*(x) = 1/2` everywhere, meaning generator perfectly matches data distribution.

### 5.3 GAN Training Procedure

```
For each training iteration:
    1. Sample minibatch of real data {x_1, ..., x_m} from training set
    2. Sample minibatch of noise {z_1, ..., z_m} from p_z

    3. Update Discriminator:
       - Compute D loss on real: L_real = -mean(log D(x_i))
       - Generate fakes: x'_i = G(z_i)
       - Compute D loss on fakes: L_fake = -mean(log(1 - D(x'_i)))
       - L_D = (L_real + L_fake) / 2
       - Backprop and update D parameters (detach G)

    4. Update Generator:
       - Sample new noise {z_1, ..., z_m}
       - Generate fakes: x'_i = G(z_i)
       - L_G = -mean(log D(x'_i))
       - Backprop and update G parameters
```

**Key Implementation Detail**: When updating G, detach the discriminator computation; when updating D, detach the generator computation (`fake_X.detach()`).

### 5.4 DCGAN (Deep Convolutional GAN)

#### 5.4.1 Generator Architecture
- Input: latent vector `z in R^{100}` reshaped as `(100, 1, 1)`
- Architecture (progressive upsampling):
```
z (100 x 1 x 1)
  -> ConvTranspose2d + BN + ReLU  -> (512, 4, 4)
  -> ConvTranspose2d + BN + ReLU  -> (256, 8, 8)
  -> ConvTranspose2d + BN + ReLU  -> (128, 16, 16)
  -> ConvTranspose2d + BN + ReLU  -> (64, 32, 32)
  -> ConvTranspose2d + Tanh       -> (3, 64, 64)
```
- Each `ConvTranspose2d`: kernel=4, stride=2, padding=1 (doubles spatial dims)
- Output activation: `Tanh` (maps to [-1, 1])

**Transposed Convolution Output Size**:
```
n'_h = k_h + s_h * (n_h - 1) - 2 * p_h
```

#### 5.4.2 Discriminator Architecture (Mirror of Generator)
- Input: image `(3, 64, 64)`
- Architecture (progressive downsampling):
```
Input (3, 64, 64)
  -> Conv2d + BN + LeakyReLU(0.2)  -> (64, 32, 32)
  -> Conv2d + BN + LeakyReLU(0.2)  -> (128, 16, 16)
  -> Conv2d + BN + LeakyReLU(0.2)  -> (256, 8, 8)
  -> Conv2d + BN + LeakyReLU(0.2)  -> (512, 4, 4)
  -> Conv2d                         -> (1, 1, 1)
```
- Each `Conv2d`: kernel=4, stride=2, padding=1 (halves spatial dims)
- No BN on first discriminator layer

**Leaky ReLU**:
```
LeakyReLU(x) = x         if x > 0
LeakyReLU(x) = alpha * x  otherwise
```
- `alpha = 0.2` typically

#### 5.4.3 DCGAN Training Details
- **Loss**: `BCEWithLogitsLoss` (binary cross-entropy)
- **Optimizer**: Adam with `lr = 0.005`, `betas = (0.5, 0.999)`
  - `beta_1 = 0.5` (not default 0.9) to reduce momentum smoothness for adversarial training
- **Weight initialization**: `Normal(0, 0.02)` for all weights
- **Data normalization**: pixel values to `[-1, 1]` (mean=0.5, std=0.5)

---

## 6. GAUSSIAN PROCESSES (Chapter 18)

### 6.1 GP Definition

A Gaussian Process is a collection of random variables, any finite subset of which follows a multivariate Gaussian distribution:
```
f(x) ~ GP(m(x), k(x, x'))
```
- `m(x) = E[f(x)]` = mean function (often set to 0)
- `k(x, x') = Cov(f(x), f(x'))` = covariance/kernel function

### 6.2 Kernel Functions

#### 6.2.1 RBF (Radial Basis Function) / Squared Exponential Kernel
```
k(x, x') = a^2 * exp(-||x - x'||^2 / (2 * l^2))
```
- `a` = amplitude (output scale) -- controls vertical variation
- `l` = length-scale -- controls how quickly correlation decays with distance
- Produces infinitely differentiable (very smooth) functions

#### 6.2.2 Neural Network Kernel
Derived from infinite-width single-layer neural network:
```
k_{NN}(x, x') = (2/pi) * arcsin( (2 x^T Sigma x') / sqrt((1 + 2 x^T Sigma x)(1 + 2 x'^T Sigma x')) )
```
- `Sigma` = parameter matrix related to weight prior

### 6.3 GP Prior

Given input locations `X* = {x*_1, ..., x*_n}`:
```
f* ~ N(m(X*), K(X*, X*))
```
- `K(X*, X*)` = `n x n` covariance matrix where `K_{ij} = k(x*_i, x*_j)`
- Samples from prior are random functions

### 6.4 GP Posterior (Regression with Noise)

**Observation model**: `y = f(x) + epsilon` where `epsilon ~ N(0, sigma^2)`

Given training data `(X, y)` and test inputs `X*`:

**Posterior Mean** (predictive mean):
```
m* = K(X*, X) [K(X, X) + sigma^2 I]^{-1} y
```

**Posterior Covariance** (predictive uncertainty):
```
S* = K(X*, X*) - K(X*, X) [K(X, X) + sigma^2 I]^{-1} K(X, X*)
```

Where:
- `K(X, X)` = `n x n` training covariance matrix
- `K(X*, X)` = `n* x n` cross-covariance matrix
- `K(X*, X*)` = `n* x n*` test covariance matrix
- `sigma^2` = observation noise variance
- `I` = identity matrix

**Predictive distribution**: `f* | X, y, X* ~ N(m*, S*)`

### 6.5 Marginal Likelihood (Model Selection)

**Log Marginal Likelihood** (for hyperparameter optimization):
```
log p(y | X, theta) = -1/2 y^T [K_theta(X,X) + sigma^2 I]^{-1} y
                      - 1/2 log |K_theta(X,X) + sigma^2 I|
                      - n/2 log(2 pi)
```
- `theta` = kernel hyperparameters (e.g., `a`, `l`, `sigma`)
- **Term 1**: Data fit (how well model explains observations)
- **Term 2**: Complexity penalty (penalizes complex models)
- **Term 3**: Normalization constant

**Hyperparameter optimization**: Maximize log marginal likelihood w.r.t. `theta` using gradient-based methods. Gradients are available in closed form.

### 6.6 GP with GPyTorch (Scalable Implementation)

```python
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```
- Training: minimize negative log marginal likelihood
- Prediction: compute posterior mean and variance

---

## 7. HYPERPARAMETER OPTIMIZATION (Chapter 19)

### 7.1 Problem Formulation

```
lambda* = argmin_{lambda in Lambda} E[L(lambda)]
```
- `lambda` = hyperparameter configuration (e.g., learning rate, batch size, architecture choices)
- `Lambda` = search space (continuous, discrete, or mixed)
- `L(lambda)` = validation loss for configuration `lambda`

### 7.2 Grid Search
- Enumerate all combinations of discretized hyperparameter values
- Cost: `O(n^d)` evaluations for `d` hyperparameters with `n` values each
- Curse of dimensionality: exponential in number of hyperparameters

### 7.3 Random Search
- Sample configurations uniformly at random from search space
- **Advantage over grid search**: More efficient in high dimensions because each trial tests a different value for every hyperparameter
- With `n` trials, each hyperparameter is tested at `n` distinct values

### 7.4 Multi-Fidelity HPO

#### 7.4.1 Successive Halving Algorithm
```
Input: initial set of n configs, budget B, reduction factor eta
    1. Allocate r_0 = B / (n * ceil(log_eta(n))) resources to each config
    2. Evaluate all configs with r_0 resources (e.g., epochs)
    3. Keep top 1/eta fraction of configs
    4. Double resources for surviving configs
    5. Repeat steps 2-4 until one config remains
```
- `eta` = halving rate (typically 3)
- `r_min` = minimum resource (e.g., 1 epoch)
- `r_max` = maximum resource (e.g., full training)

**Rung levels**: `r_min, eta * r_min, eta^2 * r_min, ..., r_max`

**Key Idea**: Spend few resources evaluating many configs; spend many resources on promising configs only.

#### 7.4.2 ASHA (Asynchronous Successive Halving)
- Asynchronous variant of successive halving
- **Key difference**: Does not synchronize workers at rung levels
- Promotes configurations to next rung as soon as enough results are available
- More efficient in distributed/parallel settings (no idle workers waiting)
- Trades off slightly suboptimal promotion decisions for much better wall-clock time

**Algorithm**:
```
Repeat:
    1. If a worker is free:
       - If any config at rung r has enough peers evaluated and is in top 1/eta:
           promote it to rung r+1
       - Else: start evaluating a new random config at rung 0
```

### 7.5 Bayesian Optimization (Conceptual)

- Model the objective function with a **surrogate model** (often a Gaussian Process)
- Use an **acquisition function** to decide which config to evaluate next
- Balances **exploration** (uncertain regions) vs **exploitation** (promising regions)

**Common Acquisition Functions**:
- **Expected Improvement (EI)**: `EI(x) = E[max(f(x) - f(x_best), 0)]`
- **Upper Confidence Bound (UCB)**: `UCB(x) = mu(x) + kappa * sigma(x)`

---

## 8. MATHEMATICS APPENDIX

### 8.1 Linear Algebra

#### 8.1.1 Dot Product and Angle
```
u . v = sum_i u_i * v_i = ||u|| ||v|| cos(theta)
```
- `theta = arccos(u . v / (||u|| ||v||))` = angle between vectors

**Cosine Similarity**:
```
cos(theta) = (u . v) / (||u|| * ||v||)
```
- Range: `[-1, 1]`; `1` = same direction, `-1` = opposite, `0` = orthogonal

#### 8.1.2 Orthogonality
Two vectors are orthogonal if and only if `u . v = 0`

#### 8.1.3 Hyperplanes
Set of points `{v : w . v = b}` defines a hyperplane in `R^d` with normal vector `w`.
- Divides space into two half-spaces: `w . v > b` and `w . v < b`
- Used as decision boundaries in linear classifiers

#### 8.1.4 Linear Dependence and Rank
- Vectors `{v_1, ..., v_k}` are **linearly dependent** if there exist `a_1, ..., a_k` (not all zero) such that `sum_i a_i v_i = 0`
- **Rank** of matrix `A` = largest number of linearly independent columns = number of non-zero eigenvalues
- Full rank: `rank(A) = min(n, m)` for an `n x m` matrix

#### 8.1.5 Matrix Inverse
```
A^{-1} A = A A^{-1} = I
```
- Exists if and only if `det(A) != 0` (equivalently, full rank)
- For 2x2 matrix: `A^{-1} = (1/(ad-bc)) * [[d, -b], [-c, a]]`
- **Numerical caution**: Avoid explicit inversion; use `Ax = b` solvers instead

#### 8.1.6 Determinant
- `det(A)` = signed volume scaling factor of the linear transformation
- `det(A) = product of eigenvalues = lambda_1 * lambda_2 * ... * lambda_n`
- `det(A) = 0` iff `A` is singular (not invertible)

#### 8.1.7 Einstein Summation (einsum)
Compact notation for tensor contractions:
- Matrix multiply: `c_{ik} = a_{ij} b_{jk}` -> `torch.einsum("ij,jk->ik", A, B)`
- Dot product: `c = v_i w_i` -> `torch.einsum("i,i->", v, w)`
- Trace: `c = a_{ii}` -> `torch.einsum("ii->", A)`
- General contraction: `y_{il} = x_{ijkl} a_{jk}` -> `torch.einsum("ijkl,jk->il", X, A)`

### 8.2 Eigendecompositions

**Eigenvalue equation**: `A v = lambda v`
- `v` = eigenvector (direction unchanged by `A`)
- `lambda` = eigenvalue (scaling factor)

**Finding eigenvalues**: Solve `det(A - lambda I) = 0` (characteristic polynomial)

**Eigendecomposition**:
```
A = W Sigma W^{-1}
```
- `W` = matrix of eigenvectors (columns)
- `Sigma` = diagonal matrix of eigenvalues
- Requires `n` linearly independent eigenvectors

**Properties**:
- `A^n = W Sigma^n W^{-1}` (matrix powers)
- `A^{-1} = W Sigma^{-1} W^{-1}` (inverse via eigenvalues)
- `det(A) = lambda_1 * ... * lambda_n`
- `rank(A)` = number of non-zero eigenvalues

**Symmetric Matrices** (`A = A^T`):
- All eigenvalues are real
- Eigenvectors can be chosen orthogonal: `A = W Sigma W^T` where `W^T W = I`

**Gershgorin Circle Theorem**:
Every eigenvalue lies in at least one disc `D_i` centered at `a_{ii}` with radius `r_i = sum_{j != i} |a_{ij}|`:
```
|lambda - a_{ii}| <= r_i
```
- Useful for quick eigenvalue estimation when diagonal dominates

**Principal Eigenvalue and Neural Network Initialization**:
- Largest eigenvalue governs long-term behavior of iterated matrix multiplication `A^N v`
- For random Gaussian `n x n` matrix: largest eigenvalue ~ `sqrt(n)`
- To prevent exploding/vanishing: rescale weight matrices so principal eigenvalue ~ 1

### 8.3 Calculus

#### 8.3.1 Derivative Definition
```
df/dx(x) = lim_{epsilon -> 0} [f(x + epsilon) - f(x)] / epsilon
```

**Linear Approximation**:
```
f(x + epsilon) ~ f(x) + epsilon * df/dx(x)
```

#### 8.3.2 Common Derivatives
| Function | Derivative |
|----------|-----------|
| `c` (constant) | `0` |
| `ax` | `a` |
| `x^n` | `n x^{n-1}` |
| `e^x` | `e^x` |
| `log(x)` | `1/x` |

#### 8.3.3 Derivative Rules
- **Sum**: `d/dx [g(x) + h(x)] = g'(x) + h'(x)`
- **Product**: `d/dx [g(x) h(x)] = g(x) h'(x) + g'(x) h(x)`
- **Chain**: `d/dx g(h(x)) = g'(h(x)) * h'(x)`

#### 8.3.4 Taylor Series
```
f(x) = sum_{n=0}^{infinity} f^{(n)}(x_0) / n! * (x - x_0)^n
```
- `f^{(n)}` = n-th derivative of `f`
- `n!` = factorial

**Common Series**:
```
e^x = 1 + x + x^2/2! + x^3/3! + ...
```

#### 8.3.5 Gradient (Multivariable)
```
nabla_w L = [dL/dw_1, dL/dw_2, ..., dL/dw_N]^T
```

**Gradient Descent Update**:
```
w <- w - eta * nabla_w L(w)
```
- `eta` = learning rate
- Direction of steepest descent = `-nabla_w L(w)`

#### 8.3.6 Multivariate Chain Rule
```
df/da = (df/du)(du/da) + (df/dv)(dv/da)
```
- Sum over all paths from `a` to `f` in computational graph

**Backpropagation**: Compute gradients from output to inputs:
1. **Forward pass**: Compute values and single-step partial derivatives
2. **Backward pass**: Propagate `df/d(node)` from output backwards using chain rule

#### 8.3.7 Hessian Matrix
```
H_f = [[d^2f/dx_1^2, ..., d^2f/dx_1 dx_n],
       [...,          ...,  ...            ],
       [d^2f/dx_n dx_1, ..., d^2f/dx_n^2  ]]
```
- Symmetric: `d^2f/(dx_i dx_j) = d^2f/(dx_j dx_i)` (for continuous second derivatives)

**Quadratic Approximation**:
```
f(x) ~ f(x_0) + nabla f(x_0)^T (x - x_0) + 1/2 (x - x_0)^T H_f(x_0) (x - x_0)
```

#### 8.3.8 Key Matrix Calculus Identities
```
d/dx (beta^T x) = beta
d/dx (x^T A x) = (A + A^T) x
d/dV ||X - UV||^2_F = -2 U^T (X - UV)
```
- `beta` = constant vector, `A` = constant matrix, `U, V, X` = matrices
- `||.||_F` = Frobenius norm

### 8.4 Probability and Random Variables

#### 8.4.1 Probability Density Function (PDF)
For continuous random variable `X`:
```
P(X in [a, b]) = integral_a^b p(x) dx
```
- `p(x) >= 0` for all `x`
- `integral_{-inf}^{inf} p(x) dx = 1`

#### 8.4.2 Cumulative Distribution Function (CDF)
```
F(x) = P(X <= x) = integral_{-inf}^{x} p(t) dt
```
- `F(-inf) = 0`, `F(inf) = 1`
- Non-decreasing

#### 8.4.3 Mean (Expected Value)
- Discrete: `mu_X = E[X] = sum_i x_i p_i`
- Continuous: `mu_X = integral x p(x) dx`
- Linearity: `E[aX + b] = a E[X] + b`
- `E[X + Y] = E[X] + E[Y]` (always, even if dependent)

#### 8.4.4 Variance and Standard Deviation
```
Var(X) = E[(X - mu_X)^2] = E[X^2] - mu_X^2
sigma_X = sqrt(Var(X))
```
- `Var(aX + b) = a^2 Var(X)`
- For independent X, Y: `Var(X + Y) = Var(X) + Var(Y)`

**Chebyshev's Inequality**:
```
P(|X - mu| >= alpha * sigma) <= 1/alpha^2
```
- 99% of samples within 10 standard deviations of mean

### 8.5 Common Distributions

| Distribution | Notation | PMF/PDF | Mean | Variance |
|---|---|---|---|---|
| **Bernoulli** | `X ~ Bernoulli(p)` | `P(X=1)=p, P(X=0)=1-p` | `p` | `p(1-p)` |
| **Discrete Uniform** | `X ~ U(n)` | `P(X=i) = 1/n` for `i=1,...,n` | `(1+n)/2` | `(n^2-1)/12` |
| **Continuous Uniform** | `X ~ U(a,b)` | `p(x) = 1/(b-a)` for `x in [a,b]` | `(a+b)/2` | `(b-a)^2/12` |
| **Binomial** | `X ~ Binomial(n,p)` | `P(X=k) = C(n,k) p^k (1-p)^{n-k}` | `np` | `np(1-p)` |
| **Poisson** | `X ~ Poisson(lambda)` | `P(X=k) = lambda^k e^{-lambda} / k!` | `lambda` | `lambda` |
| **Gaussian** | `X ~ N(mu, sigma^2)` | `p(x) = (1/sqrt(2pi sigma^2)) exp(-(x-mu)^2/(2sigma^2))` | `mu` | `sigma^2` |

**Exponential Family** (general form):
```
p(x | eta) = h(x) exp(eta^T T(x) - A(eta))
```
- `eta` = natural parameters
- `T(x)` = sufficient statistics
- `A(eta)` = log-partition function (normalizer)
- `h(x)` = base measure
- Bernoulli, Gaussian, Poisson, Binomial all belong to exponential family

**Central Limit Theorem**: Sum of `N` i.i.d. random variables, properly normalized, converges to Gaussian as `N -> infinity`:
```
(X^{(N)} - mu_{X^{(N)}}) / sigma_{X^{(N)}} -> N(0, 1)
```

### 8.6 Maximum Likelihood Estimation (MLE)

**Principle**: Find parameters that maximize probability of observed data:
```
theta_hat = argmax_theta P(X | theta)
```

**Negative Log-Likelihood** (minimize this in practice):
```
NLL = -log P(X | theta) = -sum_i log P(x_i | theta)
```
- Converts products to sums (numerically stable)
- Derivative is linear time: `O(n)`

**For independent observations**:
```
P(X | theta) = product_i P(x_i | theta)
-log P(X | theta) = -sum_i log P(x_i | theta)
```

**Connection to Cross-Entropy**: Average NLL over dataset approximates cross-entropy between true distribution and model distribution.

### 8.7 Statistics

#### 8.7.1 Bias-Variance of Estimators
- **Bias**: `B(theta_hat) = E[theta_hat] - theta`
- **Variance**: `Var(theta_hat) = E[(theta_hat - E[theta_hat])^2]`
- **MSE**: `MSE = Bias^2 + Variance`

#### 8.7.2 Confidence Intervals
- 95% CI for Gaussian: `[x_bar - 1.96 * sigma/sqrt(n), x_bar + 1.96 * sigma/sqrt(n)]`

### 8.8 Information Theory

#### 8.8.1 Self-Information
```
I(x) = -log_2(p(x))    [bits]
```
- Rare events carry more information

#### 8.8.2 Entropy (Shannon Entropy)
```
H(X) = -E[log p(x)] = -sum_i p_i log p_i     (discrete)
H(X) = -integral p(x) log p(x) dx              (continuous, differential entropy)
```
- Measures average information / uncertainty
- `H(X) >= 0` for discrete; can be negative for continuous
- Maximum entropy for discrete: `H(X) <= log(k)` when `p_i = 1/k` (uniform)

#### 8.8.3 Joint Entropy
```
H(X, Y) = -E[log p(x, y)]
```
- `H(X), H(Y) <= H(X, Y) <= H(X) + H(Y)`
- Equality `H(X,Y) = H(X) + H(Y)` iff X, Y independent

#### 8.8.4 Conditional Entropy
```
H(Y | X) = -E[log p(y | x)] = H(X, Y) - H(X)
```
- Information in Y not already in X

#### 8.8.5 Mutual Information
```
I(X, Y) = H(X) + H(Y) - H(X, Y)
         = H(X) - H(X | Y)
         = H(Y) - H(Y | X)
         = E[log(p(x,y) / (p(x)p(y)))]
```
- `I(X, Y) >= 0`
- `I(X, Y) = 0` iff X, Y independent
- Symmetric: `I(X, Y) = I(Y, X)`

**Pointwise Mutual Information**:
```
PMI(x, y) = log(p(x,y) / (p(x)p(y)))
```

#### 8.8.6 KL Divergence (Relative Entropy)
```
D_KL(P || Q) = E_{x~P}[log(p(x)/q(x))] = sum_i p_i log(p_i / q_i)
```
- `D_KL(P || Q) >= 0` (Gibbs' inequality)
- `D_KL(P || Q) = 0` iff `P = Q`
- **NOT symmetric**: `D_KL(P || Q) != D_KL(Q || P)` in general
- Undefined when `q(x) = 0` and `p(x) > 0`

**Relationship to Mutual Information**:
```
I(X, Y) = D_KL(P(X,Y) || P(X)P(Y))
```

#### 8.8.7 Cross-Entropy
```
CE(P, Q) = -E_{x~P}[log q(x)] = H(P) + D_KL(P || Q)
```
- Always `>= H(P)`, with equality when `Q = P`

**Cross-Entropy Loss for Classification**:
- Binary: `CE = -[y log(p) + (1-y) log(1-p)]`
- Multi-class: `CE = -sum_{j=1}^{k} y_j log(p_j)`

**Equivalence**:
1. Maximizing log-likelihood = Minimizing cross-entropy = Minimizing KL divergence (when `H(P)` is constant)

---

*End of Curated Summary -- Dive into Deep Learning Part 5 (Chapters 14-20 + Appendix)*

---

