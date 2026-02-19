# Artificial Neural Networks: An Introduction
## Comprehensive Mathematical & Algorithmic Summary
### Priddy & Keller (SPIE Tutorial Text TT68, 2005)

---

# Chapter 1: Introduction

## Key Concepts
Artificial neural networks are mathematical inventions inspired by biological neural systems that map input spaces to output spaces. The fundamental building block is the artificial neuron, which takes weighted inputs, sums them, applies a transfer function, and produces an output. A single neuron with a hard-limiter transfer function (the perceptron) can form linear decision boundaries (hyperplanes) to separate linearly separable classes. Combining neurons into layers permits the formation of arbitrarily complex, nonlinear decision regions. The credit-assignment problem -- determining which weights to modify and by how much -- was solved by Werbos through gradient descent on the mean squared error using differentiable sigmoid transfer functions (backpropagation).

## Mathematical Equations

### Eq 1.1 -- Neuron Model Output
The output of a neuron with weighted inputs and a transfer function:

```
z = f( SUM_{i=0}^{3} w_i * x_i )
```

### Eq 1.2 -- Logistic Sigmoid Transfer Function
The most common transfer function, with nice mathematical properties (monotonicity, continuity, differentiability):

```
output = 1 / (1 + e^{-(SUM_i w_i*x_i + w_0)})
```

where:
- `i` = index on the inputs to the neuron
- `x_i` = input to the neuron
- `w_i` = weighting factor attached to input i
- `w_0` = bias to the neuron

### Eq 1.3 -- Net Stimulus
```
Net Stimulus = SUM_{i=0}^{2} w_i*f_i = w_2*f_2 + w_1*f_1 + w_0
```

### Eq 1.4 -- Decision Line Equation
Rearranging the net stimulus to form an equation of a line:
```
f_2 = (w_1 * f_1) / w_2 + (w_0 - Net Stimulus) / w_2
```
This is a line with slope `w_1/w_2` and intercept `(w_0 - Net Stimulus)/w_2` on the f_2 axis.

## Algorithm: Perceptron Learning Algorithm (Table 1.1)

```
1. Assign desired output {-1, 1} to each data point in the training set
2. Augment each feature vector with 1 to form x = {1, x_1, x_2, ..., x_n}
3. Choose update step size eta in (0,1), usually eta = 0.1
4. Randomly initialize weights w = {w_0, w_1, w_2, ..., w_n}
5. Execute perceptron weight adjustment:

   error_flag = 1
   while error_flag == 1
       error = 0
       for ii = 1 to total_number_of_data_points
           grab feature vector (x) and its desired output {1, -1}
           output = signum(w * x^T)
           if output != desired
               w = w - eta * x
               error = error + (output - desired)
           end
       end
       if error == 0
           error_flag = 0
       end
   end
```

---

# Chapter 2: Learning Methods

## Key Concepts
Neural network learning methods fall into two broad categories: supervised and unsupervised. Supervised training employs a "teacher" that provides desired responses for each input; the error between desired and actual output adapts the weights. Unsupervised training has no teacher; instead, adaptation rules govern how weights are adjusted based on the system response alone. The behavior of unsupervised systems depends heavily on the adaptation rule used (e.g., neighborhood and learning rate for SOMs, vigilance parameter for ART).

## Key Principles
- **Supervised**: Each input-feature vector has an associated desired-output vector
- **Unsupervised**: No desired output is required; adaptation rules perform error-signal generation

---

# Chapter 3: Data Normalization

## Key Concepts
Data normalization minimizes bias within a neural network for one feature over another by ensuring the same range of values for each input feature. It can also speed up training by starting each feature within the same scale. Normalization is especially important when inputs are on widely different scales. The choice of normalization technique depends on the application: statistical norm reduces outlier effects, min-max preserves all data relationships, sigmoidal reduces extreme value influence, energy norm works on each vector independently, and PCA projects data onto maximum-variance directions.

## Mathematical Equations

### Eq 3.1 -- Z-Score (Statistical) Normalization
Uses mean (mu) and standard deviation (sigma) for each feature across the training set. Produces data with zero mean and unit variance:

```
x'_i = (x_i - mu_i) / sigma_i
```

**Advantage**: Reduces the effects of outliers in the data.

### Eq 3.2 -- Min-Max Normalization
Rescales features to lie within a target range using linear interpolation:

```
x'_i = (max_target - min_target) * ((x_i - min_value) / (max_value - min_value)) + min_target
```

where `(max_value - min_value) != 0`. When this equals 0 for a feature, it indicates a constant value and the feature should be removed.

**Advantage**: Preserves exactly all relationships in the data; introduces no bias.

### Eq 3.3 -- Sigmoidal Normalization (Logistic Sigmoid)
Nonlinear transformation that reduces influence of extreme values. Output range [0, 1]:

```
x'_i = 1 / (1 + e^{-(x_i - mu_i) / sigma_i})
```

### Eq 3.4 -- Sigmoidal Normalization (Hyperbolic Tangent)
Output range [-1, 1]:

```
x'_i = (1 - e^{-(x_i - mu_i) / sigma_i}) / (1 + e^{-(x_i - mu_i) / sigma_i})
```

**Properties**: Almost linear near the mean; smooth nonlinearity at extremes; maintains resolution within one standard deviation of the mean.

### Eq 3.5 -- Minkowski Norm (General)
The energy normalization transformation using any desired Minkowski norm (1, 2, ...):

```
M_n(x) = (SUM_{i=1}^{N} |x_i|^n)^{1/n}
```

Popular norms: L1 (n=1, taxicab norm), L2 (n=2, Euclidean norm).

### Eq 3.6 -- Energy Normalization
The normalized feature vector is computed by dividing each input by its Minkowski norm:

```
x'_i = x_i / (SUM_{i=1}^{N} |x_i|^n)^{1/n}
```

When the dot product of the normalized vector with itself is computed, the length is 1.

### Principal Components Analysis (PCA) / Hotelling Transform
- Based on premise that salient information lies in features with the largest variance
- Uses eigenvector analysis on the covariance or correlation matrix
- Projects data onto eigenvectors sorted largest to smallest eigenvalue
- New features provide best separation starting with maximum variance
- Covariance matrix requires features be in similar ranges (apply stat norm or min-max first)
- PCA on covariance matrix differs from PCA on correlation matrix

---

# Chapter 4: Data Collection, Preparation, Labeling, and Input Coding

## Key Concepts
Neural networks are data driven ("garbage in, garbage out"), so proper data collection, preparation, labeling, and coding are essential. The curse of dimensionality means the number of data points needed grows exponentially with dimensionality, so feature selection/reduction is critical. Feature distance metrics provide quantitative measures of similarity between feature vectors in multidimensional space. The data-collection plan must define the problem, identify quantities, define methodology, identify expected ranges, ensure repeatability, and protect from unauthorized access.

## Distance Metrics

### Eq 4.1-4.3 -- Distance Metric Properties
A distance metric d(x,y) must satisfy:
```
d(x,x) = 0                              (identity)
d(x,y) = d(y,x)                         (symmetry)
d(x,z) <= d(x,y) + d(y,z)               (triangle inequality)
```

### Eq 4.4 -- Euclidean Distance
```
d(x,y) = sqrt( SUM_{i=1}^{N} (x_i - y_i)^2 )
```

### Eq 4.5 -- Sum of Squared Difference (SSD)
```
d(x,y) = SUM_{i=1}^{N} (x_i - y_i)^2
```

### Eq 4.6 -- Taxicab (Manhattan / City Block) Distance
```
d(x,y) = SUM_{i=1}^{N} |x_i - y_i|
```

### Eq 4.7 -- Mahalanobis Distance
Takes into account the distribution of feature vectors. C is the covariance matrix:
```
d(x,y) = sqrt( (x - y) * C^{-1} * (x - y)^T )
```

### Eq 4.8 -- Hamming Distance
For binary-valued vectors:
```
d(x,y) = SUM_{i=1}^{N} |x_i - y_i|
```
(Identical to taxicab metric with binary constraint)

---

# Chapter 5: Output Coding

## Key Concepts
Neural network outputs typically range from -1 to 1 or 0 to 1. For classifiers, outputs are coded with 1 for class membership and 0/-1 for absence. Target output values are often pulled back from sigmoid extremes (e.g., 0.9/0.1 instead of 1/0). For estimators, the target output must be scaled within the neuron's output range, with care to prevent data squashing. The designer must decide between continuous (single neuron) or binary (multiple neurons) output coding, considering whether a predetermined ranking exists among output categories.

### Eq 5.1 -- Estimator Output Rescaling
```
t' = (max_target - min_target) * f((t - min_value) / (max_value - min_value)) + min_target
```
where f is the neuron's activation function.

---

# Chapter 6: Post-processing

## Key Concepts
Neural network outputs must be interpreted through post-processing. For classifiers, outputs are thresholded to determine class membership. Dual thresholds can create an "indeterminate" region for uncertain classifications. For estimators, outputs must be rescaled from the neuron's unitless range back to real-world quantities. The choice of threshold affects the detection-to-false-alarm ratio and can be optimized for the application.

### Eq 6.1 -- Linear Output Rescaling
```
y' = (Max_value - Min_value) * ((y - Min_target) / (Max_target - Min_target)) + Min_value
```

### Eq 6.2 -- Output Rescaling with Inverse Sigmoid
```
y' = (Max_value - Min_value) * f^{-1}((y - Min_target) / (Max_target - Min_target)) + Min_value
```

---

# Chapter 7: Supervised Training Methods

## Key Concepts
Supervised training uses a "knowledge expert" to provide desired responses. The error between the learning system output and desired response adapts the network weights. The most common supervised neural network is the feedforward network with sigmoid transfer functions trained by backpropagation. The way training data are organized (multiclass vs. modular, augmented with zeros, etc.) dramatically affects decision regions and classifier performance. The VC dimension, Foley's rule, Cover's rule, and empirical guidelines help determine network architecture and training set requirements.

## Training Rules and Rules of Thumb

### Foley's Rule
```
S/N > 3
```
The ratio of number of samples per class (S) to the number of features (N) should exceed 3 for optimal performance. When S/N > 3, training-set error approximates test-set error and approaches Bayes optimal.

### Cover's Rule
When the ratio of training samples to total degrees of freedom for a two-class classifier is less than 2, the classifier will find a solution even if classes are drawn from the same distribution (i.e., it memorizes/overfits).

### VC Dimension (Vapnik-Chervonenkis)
The VC dimension is the size of the largest set S of training samples for which the system can partition all 2^S dichotomies on S.
- **Lower bound**: Number of weights between input and hidden layer
- **Upper bound**: Approximately 2x total number of weights in network
- **Rule of thumb**: Total training samples should be ~10x the VC dimension

### Number of Hidden Layers
- Cybenko (1989): A single hidden layer, given enough neurons, can form any mapping
- In practice, two hidden layers often speed convergence
- One or two hidden layers should solve almost any problem

### Number of Hidden Neurons
- For < 5 inputs: approximately 2x as many hidden neurons as inputs
- As inputs increase, the ratio of hidden neurons to inputs decreases
- Minimize hidden neurons to keep free variables small
- Use validation-set error to determine optimal number empirically

### Transfer Functions
- Must be differentiable for backpropagation
- Most common: logistic sigmoid, hyperbolic tangent, Gaussian, linear
- Linear neurons require very small learning rates
- Gaussian used in radial basis function (RBF) networks
- Nonlinear functions work best for classification problems

## Training and Testing Procedure (Table 7.3)
```
Step 1a. Randomly sample population 3 times for independent training, validation, test sets
Step 1b. If single dataset: divide into training, validation, test via random selection
Step 2.  Choose neural network, configure architecture, set training parameters
Step 3.  Train with training data, monitor with validation set
Step 4.  Evaluate using validation set
Step 5.  Repeat Steps 2-4 with different architectures and parameters
Step 6.  Select best network (smallest validation error)
Step 7.  Train chosen network with training data, monitoring with validation set
Step 8.  Assess with test set and report performance
```

### Validation Error for Stopping Training
- Training error decreases monotonically with gradient descent
- Validation error initially decreases, then rises when overfitting begins
- Stop training at minimum validation error
- Use those weights for the operational network

---

# Chapter 8: Unsupervised Training Methods

## Key Concepts
Unsupervised methods require no teacher; adaptation rules govern weight adjustment. The Self-Organizing Map (SOM) maps high-dimensional input relationships to lower-dimensional output while preserving topological relationships. The Adaptive Resonance Theory (ART) network dynamically creates new cluster neurons as patterns are presented, controlled by a vigilance parameter that determines cluster granularity. Both methods are used for clustering, data visualization, and discovering structure in unlabeled data.

## SOM Training Algorithm (Table 8.1)

```
Step 1: Choose number of neurons, total iterations (loops), eta, alpha, etc.
Step 2: Initialize weights randomly
Step 3: Loop until total iterations met
Step 4:   Get data example and find winner
          Winner = node closest to input vector by distance metric (L1, L2, etc.)
Step 5:   Update winner and all neurons in the neighborhood:
              w+ = w- + alpha * (x - w-)
Step 6:   Update neighborhood size and lower eta, alpha as needed
Step 7: End loop
```

**Key Parameters**:
- `alpha` (neighborhood decay): controls shrinkage of neighborhood over time
- `eta` (weight-update): controls how far each weight is adjusted toward input
- Neighborhood starts large (entire SOM) and reduces to single neuron

## ART Network Algorithm

The ART network dynamically assigns neurons to cluster centers during training:

```
1. Normalize input feature vector using energy norm (project onto unit hypersphere)
2. Pass normalized input to F1 layer
3. Find closest match among stored neurons on F2 layer
4. Compute similarity using dot product: -1 <= S <= 1
   (S=1: identical, S=-1: antipodal)
5. Compare similarity to vigilance parameter:
   - If similarity < vigilance: assign new neuron (if available)
     New neuron weights = input vector
   - If similarity >= vigilance: adapt winning neuron weights
     Weight update uses weighted average based on hit count:
     neuron = input/(hits) + neuron*((hits-1)/hits)
     Then renormalize: neuron = neuron / ||neuron||
6. Larger vigilance = easier to add neurons (match harder to achieve)
```

**Key insight**: In the limit, stored F2 neurons converge to the means of the cluster centers.

---

# Chapter 9: Recurrent Neural Networks

## Key Concepts
Recurrent neural networks feed outputs from neurons back to adjacent neurons, themselves, or preceding layers. The Hopfield network is an auto-associative memory that retrieves stored patterns from noisy/corrupted inputs using energy minimization. The Bidirectional Associative Memory (BAM) generalizes Hopfield to hetero-associative mappings with different input/output sizes. The Generalized Linear Neural Network (GLNN) uses the Hopfield architecture with linear activation functions to compute matrix pseudo-inverses. The Elman network adds a recurrent layer to capture temporal dynamics through single time-delay feedback.

## Hopfield Network

### Eq 9.1 -- Hopfield Weight Encoding (Outer Product Rule)
Patterns encoded as M_{pi} in {-1, 1}:
```
W = M^T * M - p*I
```
where p = total number of training patterns, I = identity matrix.

### Eq 9.2 -- Hopfield Output
```
output = f( SUM_{j!=i} w_ij * input_i )
```
Original work used step function as activation function.

### Eq 9.3 -- Maximum Storage Capacity
```
P_max = N / (2 * ln(N))
```
for a network of N neurons.

### Hopfield Configuration Process (Table 9.1)
```
Step 1. Encode each pattern as a binary vector
Step 2. Form N (inputs) x P (patterns) matrix; each vector becomes a row
Step 3. Encode weights via outer product: W = M^T*M - p*I
        Zero out diagonal terms (w_ii = 0)
```

## Bidirectional Associative Memory (BAM)

### Eq 9.4 -- BAM Weight Encoding
```
W = A^T * B
```
where A = set of P input patterns, B = set of P output patterns, encoded as {-1, 1}.

### Eq 9.5 -- BAM Forward Pass (Output)
```
output = f(W * input)
```

### Eq 9.6 -- BAM Backward Pass (Input)
```
input = f(W^T * output)
```

### BAM Configuration Process (Table 9.2)
```
Step 1. Encode each input pattern as a binary vector
Step 2. Form N x P matrix A (input patterns as rows)
Step 3. Encode each output pattern as a binary vector
Step 4. Form M x P matrix B (output patterns as rows)
Step 5. Encode weights: W = A^T * B
```

## Generalized Linear Neural Network (GLNN)

The GLNN uses the Hopfield network with linear activation to compute matrix pseudo-inverses.

### Eq 9.7-9.10 -- GLNN Derivation
Starting from Hopfield iteration:
```
x(n+1) = W * x(n) + u
```
In the limit as n -> infinity: `x(n+1) ~= x(n)`, so:
```
x ~= W*x + u
x*(I - W) ~= u
x ~= (I - W)^{-1} * u
```

### Overdetermined / Full-Rank Case
Substitutions:
```
u = alpha * A^T * y
0 < alpha < 2 / trace(A^T * A)
W = I - alpha * A^T * A
```
Result (Moore-Penrose pseudo-inverse):
```
x = (A^T * A)^{-1} * y                    (Eq 9.11)
```

### Underdetermined Case
Substitutions:
```
u = y
W = I - alpha * A * A^T
0 < alpha < 2 / trace(A * A^T)
```
Result:
```
x = (A * A^T)^{-1} * y                    (Eq 9.12)
```

## Elman Recurrent Network
- Each hidden neuron connects to one neuron in a recurrent layer
- Copies hidden layer output and cycles back through a weighting factor on next sample
- Adds delay loop: information from time t-1 combined with time t
- Indirectly captures decreasing portions from t-2, t-3, t-4, etc.
- Enables modeling of temporal dynamics

---

# Chapter 10: A Plethora of Applications

## Key Concepts
Neural networks excel at three main application areas: function approximation (mapping inputs to continuous outputs), pattern recognition (mapping inputs to class labels), and database mining / self-organization (clustering). Feedforward networks are universal approximators that perform well for interpolation but poorly for extrapolation. Network complexity (number of neurons) should match problem complexity to avoid underfit or overfit. The Elman recurrent network captures time-varying dynamics for temporal modeling applications. SOMs can be used for data mining to discover hidden structure and relationships.

### Eq 10.1 -- Example Function Approximation Target
```
f(x) = 0.02 * (12 + 3x - 3.5x^2 + 7.2x^3) * [1 + cos(4*pi*x)] * [1 + 0.8*sin(3*pi*x)]
```
on interval [0, 1].

### Key Application Examples
| Application | Architecture | Training | Inputs | Outputs |
|---|---|---|---|---|
| Boston Housing Estimator | FF 13-10-5-1 | Backpropagation | 13 attributes | Median price |
| Cardiopulmonary Modeler | Elman Network | Modified Backprop | 2 (workload, temp) | 4 (HR, RR, SBP, DBP) |
| Tree Classifier | FF with sigmoid | Backpropagation | 2 (needle, cone length) | 4 species |
| Handwritten Numbers | FF network | Backpropagation | Image features | 10 digits |
| Electronic Nose | FF network | Backpropagation | Sensor responses | Chemical classes |
| Serial Killer Data Mining | SOM | Unsupervised | Crime attributes | Cluster map |

---

# Chapter 11: Dealing with Limited Amounts of Data

## Key Concepts
When data is limited, statistical resampling techniques help maximize data use for training and testing. K-fold cross-validation partitions data into k mutually exclusive subsets; k-1 train and 1 tests, repeated k times. Leave-one-out is k-fold where k equals sample size, useful for continuous error functions. Jackknifing uses the same structure but estimates bias rather than generalization error. Bootstrap resampling creates new datasets by randomly sampling with replacement, preserving a priori probabilities and providing more realistic simulation of real-world conditions.

### Methods Summary

| Method | Purpose | Process |
|---|---|---|
| K-fold Cross-Validation | Estimate generalization error | Partition into k subsets; train on k-1, test on 1; repeat k times |
| Leave-One-Out | Estimate generalization error | Remove one sample at a time for testing; n total runs |
| Jackknife | Estimate bias | Same structure as leave-one-out; compare statistics of subsets |
| Bootstrap | Accurate estimates with small samples | Random sampling WITH replacement to create b new datasets of size n |

### Eq (B.21) -- Feature Ranking Formula (from Appendix B context)
```
Feature_rank = SUM_{n=1}^{N} K_n * (N - n - 1)
```

---

# Appendix A: The Feedforward Neural Network -- FULL Backpropagation Derivation

## Notation (Table A.1)

| Symbol | Definition |
|---|---|
| E | Total output error when all patterns are presented |
| E_p | Total output error when pattern p is presented |
| x_{pi} | Input to ith neuron in input (zeroth) layer for pattern p |
| y_{pj} | Output of jth neuron in final layer for pattern p |
| t_{pj} | Target output for jth neuron in final layer for pattern p |
| f^l_j(u) | Activation function for jth neuron in lth layer |
| f-dot^l_j(u) | Derivative of the activation function |
| o^l_{pi} | Output of ith neuron in lth layer for pattern p |
| w^l_{ki} | Weights linking ith neuron in (l-1)th layer to kth neuron in lth layer |
| theta | Bias term for activation function |
| w^l_{j0} | Bias term written as weight to unitary input |
| net | Net stimulus to ith neuron in lth layer for pattern p |
| N_l | Number of neurons in lth layer |
| L | Number of layers excluding input |
| P | Number of patterns in training set |
| delta^l_{ik} | Delta term for weight update |
| eta | Learning rate |
| alpha | Momentum |

## A.1 Mathematics of the Feedforward Process

### Eq A.1 -- Input Layer (Zeroth Layer)
```
o^0_i = x_i,    where i = 1, ..., N_0
```

### Eq A.2 -- Net Stimulus
```
net stimulus = SUM_{i=1}^{N} w_i * x_i + theta
```

### Eq A.3 -- Neuron Output
```
output = f(net) = f( SUM_{i=1}^{N} w_i * x_i + theta ) = f( SUM_{i=1}^{N_0} w_i * o^0_i + w_0 )
```

### Eq A.4 -- Net Stimulus for jth Neuron in lth Layer
```
net^l_j = SUM_{i=1}^{N_{l-1}} w^l_{ji} * o^{l-1}_i + theta^l_j = SUM_{i=1}^{N_{l-1}} w^l_{ji} * o^{l-1}_i + w^l_{j0}
```

### Eq A.5 -- Output of jth Neuron in lth Layer
```
o^l_j = f^l_j(net^l_j) = f^l_j( SUM_{i=1}^{N_{l-1}} w^l_{ji} * o^{l-1}_i + w^l_{j0} )
```

### Eq A.6 -- Final Layer Output
```
y_j = o^L_j,    where j = 1, ..., N_L
```

### Eq A.7 -- Network Topology Notation
```
N_0 - N_1 - N_2 - ... - N_L
```
Example: 3-10-2 = 3 inputs, 10 hidden neurons, 2 outputs.

## A.2 The Backpropagation Algorithm -- Generalized Delta Rule

### Eq A.8 -- Squared Error for Pattern p
```
E_p = (1/2) * SUM_{j=1}^{N_L} (t_{pj} - y_{pj})^2
```

### Eq A.9 -- Total Error over All Patterns
```
E = SUM_{p=1}^{P} E_p = (1/2) * SUM_{p=1}^{P} SUM_{j=1}^{N_L} (t_{pj} - y_{pj})^2
```

### Eq A.10 -- Gradient Descent Condition
```
dE / dw^l_{jk} = 0
```

### Eq A.11 -- Error Decomposition
```
dE / dw^l_{ji} = SUM_{p=1}^{P} dE_p / dw^l_{ji}
```

### Eq A.12 -- Chain Rule Application
```
dE_p / dw^l_{ji} = (dE_p / dnet^l_{pj}) * (dnet^l_{pj} / dw^l_{ji})
```

### Eq A.13 -- Net Stimulus Derivative w.r.t. Weights
```
dnet^l_{pj} / dw^l_{ji} = o^{l-1}_{pk}
```
(The output of the neuron on the layer below)

### Eq A.14 -- Combined
```
dE_p / dw^l_{ji} = (dE_p / dnet^l_{pj}) * o^{l-1}_{pi}
```

### Eq A.15 -- Delta Term Definition
```
delta^l_{pj} = -(dE_p / dnet^l_{pj})
```

### Eq A.16 -- Generalized Delta Rule
```
dE_p / dw^l_{ji} = -delta^l_{pj} * o^{l-1}_{pi}
```

### Eq A.17 -- Further Chain Rule Decomposition
```
dE_p / dnet^l_{pj} = (dE_p / do^l_{pj}) * (do^l_{pj} / dnet^l_{pj})
```

### Eq A.18 -- Activation Function Derivative
```
do^l_{pj} / dnet^l_{pj} = f-dot^l_j(net^l_{pj})
```

### Eq A.19 -- Output Layer Error Derivative
```
dE_p / do^l_{pj} = -(t_{pj} - o_{pj})    when l = L (output layer)
```

### Eq A.20 -- Hidden Layer Error Derivative
```
dE_p / do^l_{pj} = SUM_{k=1}^{N_{l+1}} (dE_p / dnet^{l+1}_{pk}) * (dnet^{l+1}_{pk} / do^l_{pj})    when l < L
```

### Eq A.21-A.23 -- Solving Hidden Layer Terms
```
dE_p / dnet^{l+1}_{pk} = -delta^{l+1}_{pk}

dnet^{l+1}_{pk} / do^l_{pj} = w^{l+1}_{kj}

Therefore: dE_p / do^l_{pj} = -SUM_{k=1}^{N_{l+1}} delta^{l+1}_{pk} * w^{l+1}_{kj}    when l < L
```

### Eq A.24 -- Delta for Output Layer
```
delta^L_{pj} = f-dot^L_j(net^L_{pj}) * (t_{pj} - y_{pj})    when l = L
```

### Eq A.25 -- Delta for Hidden Layers (BACKPROPAGATION OF ERROR)
```
delta^l_{pj} = f-dot^l_j(net^l_{pj}) * SUM_{k=1}^{N_{l+1}} delta^{l+1}_{pk} * w^{l+1}_{kj}    when l < L
```

**Key insight**: Calculate delta for output layer first, then propagate backwards through hidden layers. This gives rise to the name "backpropagation of error."

### Eq A.26 -- Weight Update Equation
```
Delta_w^l_{pji} = -eta * (dE_p / dw^l_{pji})
```

### Eq A.27 -- Weight Update (Generalized Delta Rule)
```
Delta_w^l_{pji} = eta * delta^l_{pj} * o^{l-1}_{pi}
```

### Eq A.28 -- Batch Weight Update (Summed over all patterns)
```
Delta_w^l_{ji} = SUM_{p=1}^{P} Delta_w^l_{pji}
```

### Eq A.29 -- Logistic Sigmoid Derivative
```
f-dot(u) = d/du logistic(u) = e^{-u} / (1 + e^{-u})^2 = f(u) * [1 - f(u)]
```

### Eq A.30 -- Hyperbolic Tangent Derivative
```
f-dot(u) = d/du tanh(u) = 1 - tanh^2(u) = 1 - f^2(u)
```

Both derivatives contain the original function, so feedforward values can be reused in backpropagation.

### Eq A.31 -- Weight Update with Momentum
```
Delta_w^l_{pij}(n) = eta * delta^l_{pj} * o^l_{pij} + alpha * Delta_w^l_{pij}(n-1)
```
where alpha = momentum (range 0 to 0.9, typically 0.5-0.9).

### Eq A.32 -- Weight Update with Exponential Smoothing
```
Delta_w^l_{pij}(n) = (1 - sigma) * eta * delta^l_{pj} * o^l_{pij} + sigma * Delta_w^l_{pij}(n-1)
```

## Backpropagation Training Procedure (Table A.2)
```
Step 1.  Initialize weights: w^l_{ij} = Uniform Random over [-epsilon, epsilon]
Step 2.  Pick labeled pattern (input x_{pi}, target t_{pj}) from training set
Step 3.  Propagate forward: generate output y_{pj}
Step 4.  Calculate error using Eq A.8
Step 5.  Propagate error backwards; calculate weight changes using Eq A.27 or A.31
  Step 5a. If batch mode: do NOT apply weight changes yet
  Step 5b. If incremental mode: apply weight changes
Step 6.  If more patterns (p < P): loop to Step 2
Step 7a. If batch mode: update weights using sum from Eq A.28
Step 8.  If error high or max iterations not met: loop to Step 2
```

## A.3 Alternatives to Backpropagation

### Conjugate Gradient Descent
- Performs series of line searches across the error surface
- Determines steepest descent direction, projects line to locate minimum
- Updates weights once per epoch
- Searches along conjugate direction (ensures prior minimized directions stay minimized)
- Assumes quadratic error surface; falls back to steepest descent if assumption fails

### Cascade Correlation (Table A.5)
```
Step 1. Initialize feedforward network with no hidden neurons
Step 2. Train until minimum MSE reached
Step 3. Add candidate hidden neuron, initialize weights
Step 4. Train candidate: maximize correlation between its output and network error
Step 5. Add candidate to network: freeze its weights, connect to hidden and output neurons
Step 6. Train full network including new neuron to minimum MSE
Step 7. Repeat Steps 3-6 until desired error reached
```

### Quick Propagation (Eq A.34-A.35)
Assumes quadratic error surface:
```
Delta_w(n) = S(n) / (S(n-1) - S(n)) * Delta_w(n-1)
```
where S(n) = gradient dE/dw at current iteration. If slope gets steeper:
```
Delta_w(n) = a * Delta_w(n-1)
```
where a is an acceleration coefficient.

### Quasi-Newton (Eq A.36)
Uses inverse Hessian matrix H and steepest descent direction g:
```
Delta_w = -H^{-1} * g
```
Memory requirements scale as O(weights^2).

### Levenberg-Marquardt (Eq A.37-A.39)
Approximates Hessian with Jacobian:
```
H ~= J^T * J
g = J^T * e
Delta_w = -(J^T*J + mu*I)^{-1} * J^T * e
```
- mu = 0: Newton method
- Large mu: gradient descent
- When error decreases: reduce mu (reinforce linear assumption)
- When error increases: increase mu (use gradient descent)

### Levenberg-Marquardt Training Procedure (Table A.6)
```
Step 1.  Initialize weights randomly
Step 2.  Present each pattern to network
Step 3.  Propagate forward; calculate error
Step 4.  If more patterns: loop to Step 2
Step 5.  Calculate error vector e using summed squared error
Step 6.  Compute Jacobian matrix J
Step 7.  Compute weight update via Eq A.39
Step 8.  Recalculate SSE: if lower, reduce mu by beta, update weights, go to Step 9
         If higher, increase mu by beta, go to Step 7
Step 9.  If norm(g) < desired: stop; else loop to Step 1
```

### Evolutionary Computation (Table A.7)
```
Step 1. Select neural network architecture (inputs, outputs)
Step 2. Train several networks to produce a population
Step 3. Evaluate population with test data (MSE fitness function)
Step 4. Create offspring by randomly varying neurons, layers, weights, biases
Step 5. Evaluate offspring with test data
Step 6. Conduct pairwise competitions; winners become next-generation parents
Step 7. If desired result obtained, stop; otherwise return to Step 4
```

---

# Appendix B: Feature Saliency

## Key Concepts
Feature saliency uses trained network weights to determine which input features are most important. The intuitive metric -- the L1 norm of weights emanating from a feature -- is mathematically related to the Bayesian probability of error for the network used as a classifier.

### Eq B.1 -- Simple Saliency Metric
```
Lambda_i = SUM_{j=1}^{N_l} |w^l_{ji}|
```
Sum of absolute weights from feature i to all neurons in the layer above.

### Eq B.2-B.4 -- Bayesian Foundation
Network outputs approximate posterior probabilities:
```
z_j = P(C_j | x)
SUM_j z_j = SUM_j P(C_j | x) = 1
P_error(j, x) = 1 - SUM_j P(C_j | x) = SUM_{k!=j} z_k
```

### Eq B.13 -- Full Saliency Derivative (3-layer network)
```
dP_error(j,x) / dx_i = SUM_{k!=j} delta^3_k * SUM_m w^3_{km} * delta^2_m * SUM_n w^2_{mn} * delta^1_n * w^1_{ni}
```
where `delta^l_k = z_k * (1 - z_k)` for sigmoid neurons.

### Eq B.20 -- Saliency Proportionality
```
Omega_i is proportional to SUM_n |w^1_{ni}| = Lambda_i
```

### Eq B.21 -- Feature Ranking Formula
```
Feature_rank = SUM_{n=1}^{N} K_n * (N - n - 1)
```
where K_n = count for importance n, N = total features.

### Practical Procedure
1. Train multiple networks (e.g., 100) with same topology, different initial weights
2. For each network, rank features by Lambda_i
3. Build histogram of rankings across all networks
4. Apply Feature_rank formula to determine overall feature importance
5. Remove low-ranked features; test performance

---

# Appendix C: Matlab Code -- ALL Code Verbatim

## C.1 -- Principal Components Normalization (pca.m)

```matlab
function [C, eigvecs, lambdas] = pca(A,N)
%
%usage:
% [C, eigvecs, lambdas] = pca(A,N)
%
%
% A - The input data array of feature vectors
% N - the number of eigenvalues/eigenvectors to return and can be left
%     blank.
% C - The transformed data array obtained from a projection of A onto the
%     N eigenvectors of the cov(A).
% eigvecs - The N eigenvectors of the cov(A) sorted from largest to smallest in
%           column-major order
% lambdas - The sorted N eigenvalues from largest to smallest.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% pca.m
%
% Simple Principal Components Analysis
%
% Written By Kevin L. Priddy
% Copyright 2004
% All Rights Reserved
%
% This instantiation returns the transformed data array obtained from a
% projection of A onto the N eigenvectors of the cov(A). If N is missing, N
% is set to min(size(A)).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B = cov(A)                       % Compute the covariance of the input array
if exist('N','var')
    if (N > size(B,1))           % Make sure that we don't have too many eigenvalues
        N = size(B,1);
    end
else
    N = size(B,1);               % Make sure that we have the right number of eigenvalues
end

[V D] = eig(B);
[rows cols] = size(D);

% Check to see if the result is in descending order and reorder N eigenvectors
% in descending order
if D(1,1) < D(rows,rows)        % If true sort the array in descending order
    for ii = 1:N
        eigvecs(:,ii) = V(:,rows-(ii-1));
        lambdas(ii,1) = D(rows-(ii-1),rows-(ii-1));
    end
else
    eigvecs = V(:,1:N);
    for ii = 1:N
        lambdas(ii,1) = D(ii,ii);
    end
end

% Project the original data onto the first N eigenvectors of the covariance matrix.
C = A*eigvecs;
```

## C.2 -- Hopfield Network (hop.m)

```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hop.m
%
% usage: [result, x] = hop(n,u,W)
%
%
% result is the predicted solution
% x is the predicted x values for each iteration through the Hopfield network
% n is the number of allowed iterations
% u is the initial input to the network
% W is the Weight matrix
%
% called by glnn.m
%
% Author: Kevin L. Priddy
% Copyright 2004
% All rights reserved
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [result, x] = hop(n,u,W)
x(:,1) = W*u;
for ii = 2:n
    x(:,ii) = W*x(:,ii-1) + u;
end
result = x(:,n);
```

## C.3 -- Generalized Neural Network (glnn.m)

```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% usage: [result, x] = glnn(A,y,n)
%
% glnn.m
%
% A is the data matrix
% y is the set of desired outputs
% x is the solution to the problem
% n is the number of allowed iterations
%
% calls the lin_hop function included in glnn.m
%
% Author: Kevin L. Priddy
% Copyright 2004
% All rights reserved
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [result, x] = glnn(A,y,n)
% full rank and overdetermined cases
[rows cols] = size(A);
if rows >= cols
    alpha = 1/(trace(A'*A));       % 0 < alpha < 2/(trace(A'*A)) to ensure convergence
    W = eye(size(A'*A,2))-alpha*(A'*A);
    u = alpha*A'*y;
    [result, x] = lin_hop(n,u,W);
else
    % underdetermined case
    alpha = 1/(trace(A*A'));       % 0 < alpha < 2/(trace(A*A')) to ensure convergence
    W = eye(size(A*A',2))-alpha*(A*A');
    Wff = alpha*A'*y;
    [result, xp] = lin_hop(n,y,W);
    x = Wff*xp;
end

function [result, x] = lin_hop(n,u,W)
x(:,1) = W*u;
for ii = 2:n
    x(:,ii) = W*x(:,ii-1) + u;
end
result = x(:,n);
```

## C.4 -- Generalized Neural Network Example (glnn_example.m)

```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% glnn_example.m
%
% Demonstrates how the GLNN can solve a set of simultaneous equations
% A is the data matrix
% y is the set of desired outputs
% result is the glnn solution to the problem
% n is the number of allowed iterations
% x is the predicted x values for each iteration through the Hopfield network
%
% calls the glnn.m function
%
% Author: Kevin L. Priddy
% Copyright 2004
% All rights reserved
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
n = 500;                          % Set iterations to 500
A = [-4.207  1.410  0.451 -0.910;
     -0.344 -3.473  2.380  3.267;
      3.733 -1.999 -3.728  2.850;
      1.096 -4.277  1.538 -3.952];
y = [0.902; -27.498; -0.480; 3.734];
x_act = [1; 2; -3; -4];

[result, x] = glnn(A,y,n);

% plot the mean squared error for x as a function of n
for ii = 1:n
    mse(ii) = ((x(:,ii)-x_act)'*(x(:,ii)-x_act))/size(x,1);
end

close all
figure(1)
semilogy(mse, 'r', 'linewidth', 2);
grid on
ylabel('Mean Squared Error');
xlabel('Iterations through Hopfield Network');
title('Mean Squared Error for the Estimated Values of x');
x_act
result
delta = x_act-result
```

## C.5 -- ART-like Network (artl.m)

```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% usage neuron = artl(outfile, A, num_nodes, vigilence);
%
% ARTL(outfile, A, num_nodes, vigilence)
% Computes an ART-Like network using floating point numbers
% A is the data matrix
% nodes is the total number of neurons in the F2 Layer
% vigilence is the fitness of a normalized vector to the winning node.
%
% Author: Kevin L. Priddy
% Copyright 2004
% All rights reserved
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function neuron = artl(outfile, A, num_nodes, vigilence);

% First Set up some default variables
[numvecs vecsize] = size(A);
neuron = zeros(num_nodes,vecsize);
node_wins = zeros(num_nodes,1);
count = 1;
flag = 0;

for ii = 1:numvecs
    temp = A(ii,:);
    temp = temp/norm(temp);

    if(flag == 0)               % Set the first neuron to be the first normalized vector
        neuron(1,:) = temp;
        flag = 1;
        node_wins(1) = 1;
        win_node(ii,1) = 1;
    else
        % Find the winning neuron
        % Set the initial winner to be the first F2 neuron
        winner = 1;
        win_val = temp*neuron(1,:)';
        % Now compare the input to the rest of the F2 neurons
        for jj = 2:count
            dotprod = temp*neuron(jj,:)';
            if(dotprod > win_val)
                winner = jj;
                win_val = dotprod;
            end
        end
        node_wins(winner) = node_wins(winner) + 1;

        % Compare with Vigilence - small number means need a new neuron
        if(win_val < vigilence)
            if(count < num_nodes)
                count = count + 1;
                neuron(count,:) = temp;
                node_wins(count) = 1;
                win_node(ii,1) = count;
            else
                error('You are out of neurons. Decrease vigilence and try again.');
            end
        else
            % adjust centroid of winning neuron. Weight shift by previous
            % number of hits and then renormalize.
            neuron(winner,:) = temp/(node_wins(winner)) + ...
                neuron(winner,:)*((node_wins(winner)-1)/(node_wins(winner)));
            % renormalize the winner
            neuron(winner,:) = neuron(winner,:)/norm(neuron(winner,:));
            win_node(ii,1) = winner;
        end
    end
end

result = [A win_node];
% Save the result into a file
save(outfile, 'neuron', 'result', '-ascii');

% Note that the norm(X) function is simply the L2 or Euclidean norm of X.
% In C it would be:
%
% float norm(float *x, int length)
% {
%   int i;
%   float tempval;
%   tempval = 0.0;
%   for (ii = 0; ii < length-1; ii++)
%   {
%       tempval += x[ii]*x[ii];
%   }
%   tempval = sqrt(tempval);
%   float *temp;
%   for (jj = 0; jj < length-1; jj++)
%   {
%       temp = x[ii]/tempval;
%   }
%   return(temp);
% }
```

## C.6 -- Simple Perceptron Algorithm (perceptron.m)

```matlab
function [W, count] = perceptron(InArray, Desired_Output, eta)
%
%usage:
% [W, count] = perceptron(InArray, Desired_Output, eta)
%
%   InArray - the set of input feature vectors used for training
%             (N vectors * M input features)
%   Desired_Output - The associated output value (-1,1) for a given
%                    feature vector (N vectors x 1 output)
%
%   eta - The desired weight update step size (usually set in range (0.1,1))
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% perceptron.m
%
% Simple Perceptron Program
%
% Written By Kevin L. Priddy
% Copyright 2004
% All Rights Reserved
%
% This instantiation updates the weight vector whenever a feature vector
% presentation produces an error
% Execution stops when there is a hyperplane that produces no errors over
% the training set
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the bias term by augmenting each input with a 1
InArray = [InArray ones(size(InArray,1),1)];

% note we've included the bias term in the number of features
[num_vecs num_features] = size(InArray);
bias = 0;
W = zeros(1,num_features);        % Include the bias term as a weight
error_flag = 1;
count = 1;

while error_flag == 1
    % Keep adjusting the weights until there are no errors
    error = 0;
    for ii = 1:num_vecs
        % Run through the set of input vectors
        invec = [InArray(ii,:)'];  % Get an input vector
        out(ii,1) = W*invec;
        % Compute the perceptron output and apply the signum transfer function
        if out(ii,1) >= 0
            out(ii,1) = 1;
        else
            out(ii,1) = -1;
        end
        if out(ii,1) ~= Desired(ii,1)  % Update weights each time an error occurs
            W = W - eta*invec';
            error = error + (out(ii,1) - Desired(ii,1));
        end
    end
    total_error(count) = error;    % Update error for plotting when finished
    if total_error(count) == 0
        % Check for NO ERROR case
        error_flag = 0;            % breaks the while loop
    end
    count = count + 1;             % update the counter
end

figure(1)
plot(total_error)                  % Plot the total error versus count
```

## C.7 -- Kohonen Self-Organizing Feature Map (sofm.m)

```matlab
% SOFM(outfile, A, Wrows, Wcols, nh_size, eta, iterations, num_maps)
% Computes a two-dimensional Kohonen Self-organizing Map using square neighborhoods
%
% Written by: Kevin L. Priddy
% Copyright 2004
% All rights reserved
%
% A is the data matrix (M rows by N features) where M is the total number
% of feature vectors
% Wrows is the number of rows in the SOFM
% Wcols is the number of cols in the SOFM
% nh_size is the starting neighborhood size (nh_size x nh_size)
% eta is the stepsize for the weight update
% iterations is the total number of iterations to be performed
% num_maps is the number of maps to be stored for the total number of iterations

function W = sofm(outfile, A, Wrows, Wcols, nh_size, eta, iterations, num_maps);

% First Set up some default variables
[numvecs vecsize] = size(A);
A_max = max(max(A));
A_min = min(min(A));
map = zeros(10*Wrows*Wcols, vecsize);

% set up initial weight ranges for SOM. Note that we will be
% using the row and column indexing scheme to index each
% row vector that corresponds to the weights from a given node in the
% SOM to the input array

% Spread weights throughout data cloud
W = rand(Wrows*Wcols, vecsize)*(A_max-A_min) + A_min;

% Make sure neighborhood size is odd
if((nh_size > 1) & (mod(nh_size,2) == 0))
    nh_size = nh_size - 1;
end

% Calculate until iterations are used up
count = 1;
map_count = 1;

for m = 1:5
    for n = 1:floor((iterations/5) + 0.5)
        % grab an input exemplar from the data set
        temp = A(ceil(rand*(numvecs-1) + 0.5), :);

        % now find closest neuron in SOM to input
        min_D = (W(1,:)-temp)*(W(1,:)-temp)';
        win_row = 1;
        win_col = 1;

        for i = 1:Wrows
            for j = 1:Wcols
                % Compute Euclidean Distance
                D = (W((i-1)*Wcols+j,:) - temp) * (W((i-1)*Wcols+j,:) - temp).';
                if (min_D > D)       % Check if D is new winner for W(i,j)
                    win_row = i;
                    win_col = j;
                    min_D = D;
                end
            end
        end

        % now we update the winner and everyone in the neighborhood
        row_min = win_row - floor(nh_size*0.5);
        if row_min < 1
            row_min = 1;
        end
        col_min = win_col - floor(nh_size*0.5);
        if col_min < 1
            col_min = 1;
        end
        row_max = win_row + floor(nh_size*0.5);
        if row_max > Wrows
            row_max = Wrows;
        end
        col_max = win_col + floor(nh_size*0.5);
        if col_max > Wcols
            col_max = Wcols;
        end

        for i = row_min:row_max
            for j = col_min:col_max
                delta = eta * (temp - W((i-1)*Wcols+j,:));
                % update the node in the neighborhood
                W((i-1)*Wcols+j,:) = W((i-1)*Wcols+j,:) + delta;
            end
        end

        if(mod(count, floor(iterations/num_maps)) == 0)
            for ii = 1:Wrows
                for jj = 1:Wcols
                    map((map_count-1)*Wrows*Wcols + (ii-1)*Wcols + jj,:) = ...
                        W((ii-1)*Wcols + jj,:);
                end
            end
            map_count = map_count + 1;
        end
        count = count + 1;
    end

    % decrement eta and neighborhood size
    eta = eta*0.8;
    if nh_size > 1
        nh_size = nh_size - 2;
    end
end

save(outfile, 'map', '-ascii');
```

---

# Quick Reference: Equation Index

| Equation | Description | Chapter |
|---|---|---|
| 1.1 | Neuron model output | Ch 1 |
| 1.2 | Logistic sigmoid function | Ch 1 |
| 1.3-1.4 | Net stimulus and decision line | Ch 1 |
| 3.1 | Z-score normalization | Ch 3 |
| 3.2 | Min-Max normalization | Ch 3 |
| 3.3 | Sigmoidal norm (logistic) | Ch 3 |
| 3.4 | Sigmoidal norm (tanh) | Ch 3 |
| 3.5 | Minkowski norm | Ch 3 |
| 3.6 | Energy normalization | Ch 3 |
| 4.4-4.8 | Distance metrics (Euclidean, SSD, Taxicab, Mahalanobis, Hamming) | Ch 4 |
| 5.1 | Estimator output rescaling | Ch 5 |
| 6.1-6.2 | Post-processing output rescaling | Ch 6 |
| 9.1 | Hopfield weight encoding | Ch 9 |
| 9.2 | Hopfield output | Ch 9 |
| 9.3 | Hopfield max storage capacity | Ch 9 |
| 9.4-9.6 | BAM weight encoding and I/O | Ch 9 |
| 9.7-9.12 | GLNN derivation and pseudo-inverse | Ch 9 |
| A.1-A.7 | Feedforward math (layers, net stimulus, output) | App A |
| A.8-A.9 | Squared error (per-pattern and total) | App A |
| A.10-A.16 | Generalized delta rule derivation | App A |
| A.17-A.25 | Backpropagation delta terms (output and hidden) | App A |
| A.26-A.28 | Weight update equations | App A |
| A.29-A.30 | Sigmoid and tanh derivatives | App A |
| A.31-A.32 | Momentum and exponential smoothing | App A |
| A.34-A.35 | Quick propagation | App A |
| A.36 | Quasi-Newton weight update | App A |
| A.37-A.39 | Levenberg-Marquardt | App A |
| B.1 | Simple saliency metric | App B |
| B.13 | Full saliency derivative | App B |
| B.21 | Feature ranking formula | App B |

---

*Source: Priddy, K.L. and Keller, P.E., "Artificial Neural Networks: An Introduction," SPIE Tutorial Text TT68, 2005.*
