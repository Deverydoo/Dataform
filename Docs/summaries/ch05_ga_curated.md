# Chapter 5 -- Genetic Algorithms (Colin R. Reeves)
## Curated Technical Extraction

---

## 1. GA Framework

### 1.1 Problem Formulation

Discrete search space **X**, objective function:

```
f : X -> IR
```

Goal: find `arg min_{x in X} f`

where **x** is a vector of decision variables.

### 1.2 Genotype-Phenotype Mapping

The vector **x** is represented by a string **s** of length **l**, composed of symbols from alphabet **A**, via mapping:

```
c : A^l -> X
```

Search space constraint:

```
S <= A^l    (subset, reflecting that some strings may be invalid)
```

Optimization becomes:

```
arg min_{s in S} g,    where g(s) = f(c(s))
```

**Fitness function** (general form):

```
fitness = h(f(c(s)))
```

where `h : IR -> IR+` is a suitable monotonic function (eliminates negative fitness).

**Requirement**: `c` should ideally be a bijection (unique x for every s, and unique s for every x).

### 1.3 GA Template (Pseudocode)

```
Choose an initial population of chromosomes;
while termination condition not satisfied do
    repeat
        if crossover condition satisfied then
            {select parent chromosomes;
             choose crossover parameters;
             perform crossover};
        if mutation condition satisfied then
            {choose mutation points;
             perform mutation};
        evaluate fitness of offspring
    until sufficient offspring created;
    select new population;
endwhile
```

### 1.4 Schema Theorem

A schema **S** is a subset of `A^l` where all strings share defined values at certain positions.
Notation uses alphabet `A union {*}` (wildcard). Example for binary, l=4:

```
1 * * 1  represents {1001, 1011, 1101, 1111}
```

**Schema Theorem** (Holland):

```
E[N(S, t+1)] >= (F(S,t) / F_avg(t)) * N(S,t) * {1 - epsilon(S,t)}
```

Where:
- `N(S,t)` = number of instances of schema S in population at time t
- `F(S,t)` = fitness of schema S
- `F_avg(t)` = average fitness of population
- `epsilon(S,t)` = potential for genetic operators to destroy instances of S

**Intrinsic Parallelism**: A population of size M contains information on O(M^3) schemata (under certain conditions).

**No Free Lunch Theorem (NFLT)**: On average, no algorithm is better than random search over all possible problems. Success requires adapting technique to the problem.

---

## 2. Representation

### 2.1 Binary Encoding

Standard: `A = {0, 1}`

**Example -- 0-1 Knapsack Problem**:

```
maximize  sum_{i=1}^{n} x_i * v_i
subject to sum_{i=1}^{n} x_i * c_i <= C

where x_i in {0, 1}
```

String of length n; no genotype-phenotype distinction needed.

**Warning**: Natural binary encodings nearly always bring substantial problems for simple GAs. Two feasible parents may not produce feasible offspring.

### 2.2 Discrete (Non-Binary) Encoding

Alphabet of cardinality k.

**Example -- Rotor-Stacking Problem**:
- n rotors, k holes each
- String of length n (or n-1 if one rotor orientation is fixed)
- Each gene = rotor, alleles from {1,...,k} = orientation
- Example: string (1322) means rotor 1 at orientation 1, rotor 2 at orientation 3, etc.

**Competing conventions problem**: When `c(.)` is not injective, multiple genotypes map to same phenotype. Fix by anchoring one element.

### 2.3 Permutation Encoding

Search space = permutation space Pi_l of numbers 1,...,l.

**Example -- Permutation Flowshop Sequencing Problem (PFSP)**:
- n jobs, m machines
- Processing time `p(i,j)` for job i on machine j
- For permutation {pi_1, pi_2, ..., pi_n}, completion times:

```
C(pi_1, 1) = p(pi_1, 1)
C(pi_i, 1) = C(pi_{i-1}, 1) + p(pi_i, 1)          for i = 2,...,n
C(pi_1, j) = C(pi_1, j-1) + p(pi_1, j)             for j = 2,...,m
C(pi_i, j) = max{C(pi_{i-1}, j), C(pi_i, j-1)} + p(pi_i, j)
                                                      for i = 2,...,n; j = 2,...,m
```

Goal: find permutation pi* such that `f(pi*) <= f(pi)` for all pi in Pi.

Standard crossover operators FAIL to preserve permutation structure.

### 2.4 Non-Binary (Integer/Real) Problems

Transform to binary string. Requires:
1. Known domain for each decision variable
2. Known precision requirement

**Example**:

```
maximize f(x) = x^3 - 60x^2 + 900x + 100
over X = {x : x in Z, x in {0,...,31}}
```

Encoding: 5 binary digits, standard binary-to-integer mapping.
`(0,0,0,0,0) = 0, ..., (1,1,1,1,1) = 31`

---

## 3. Selection Methods

### 3.1 Roulette-Wheel Selection (RWS) -- Fitness-Proportional

Selection probability of string i is proportional to its fitness:

```
P(select i) = fitness_i / sum(all fitnesses)
```

**Example**: Population of 5 strings with fitnesses {32, 9, 17, 17, 25}:

| String | Fitness | Cumulative Prob |
|--------|---------|-----------------|
| 1      | 32      | 0.00 -- 0.32    |
| 2      | 9       | 0.32 -- 0.41    |
| 3      | 17      | 0.41 -- 0.58    |
| 4      | 17      | 0.58 -- 0.75    |
| 5      | 25      | 0.75 -- 1.00    |

Finding the appropriate string for pseudo-random number r: **O(log M)** time (binary search on cumulative array).

**Problems**: High stochastic variability; actual selection count N_C may differ greatly from E[N_C].

### 3.2 Stochastic Universal Selection (SUS) -- Baker

Instead of single pointer spun M times, use M equally-spaced connected pointers on roulette wheel. One spin produces all M selections simultaneously.

Equivalent to **systematic sampling**. Experimentally demonstrated superior to basic RWS.

### 3.3 Linear Ranking Selection

Rank chromosomes in fitness order. Selection probability for rank k:

```
P[k] = alpha + beta * k
```

Constraint (must sum to 1):

```
sum_{k=1}^{M} (alpha + beta * k) = 1
```

**Selection pressure** definition:

```
phi = P[selecting fittest string] / P[selecting average string]
    = (alpha + beta*M) / (alpha + beta*(M+1)/2)
```

Solving for parameters:

```
beta = 2*(phi - 1) / (M*(M - 1))
alpha = (2*M - phi*(M + 1)) / (M*(M - 1))
```

**Valid range**: `1 <= phi <= 2`

Finding rank k for random number r -- solve quadratic:

```
alpha*k + beta*k*(k+1)/2 = r
```

Solution:

```
k = ( -(2*alpha + beta) +/- sqrt((2*alpha + beta)^2 + 4*beta*r) ) / (2*beta)
```

**Time complexity**: O(1) -- versus O(log M) for fitness-proportional.

### 3.4 Tournament Selection

- Choose tau chromosomes at random, select the best
- Best string selected every time it is compared
- Median string chosen with probability `2^{-(tau-1)}`
- Selection pressure: `phi = 2^{tau-1}`
- For tau=2: similar to linear ranking when alpha -> 0

**Probability of a given string NOT appearing** (sampling with replacement):

```
P(not appearing) ~ e^{-1} ~ 0.368
```

**Variance reduction** (Saliby's method): Construct tau random permutations of {1,...,M}, concatenate into one sequence, chop into M pieces of tau indices each for consecutive tournaments.

**Unique advantage**: Only needs preference ordering (no formal objective function required -- handles subjective fitness).

---

## 4. Crossover Operators

### 4.1 One-Point Crossover (1X)

Choose random crossover point. Swap tails.

```
P1: a1 a2 a3 | a4 a5 a6     O1: a1 a2 a3 b4 b5 b6
P2: b1 b2 b3 | b4 b5 b6     O2: b1 b2 b3 a4 a5 a6
```

**Binary example** (crosspoint = 3):

```
P1: 1 0 1 | 0 0 1 0    ->    O1: 1 0 1 1 0 0 1
P2: 0 1 1 | 1 0 0 1    ->    O2: 0 1 1 0 0 1 0
```

**Has positional bias** (relies on building-block hypothesis). **No distributional bias**.

### 4.2 Two-Point Crossover (2X)

Choose two random crosspoints from {1,...,l-1}. Swap middle segment.

```
P1: a1 a2 | b3 b4 | a5 a6
P2: b1 b2 | a3 a4 | b5 b6
```

**Example** (crosspoints 2 and 4):

```
O1: (a1, a2, b3, b4, a5, a6)
O2: (b1, b2, a3, a4, b5, b6)
```

### 4.3 Multi-Point Crossover (mX)

Generalization of 2X to m crosspoints.

**Empirical finding** (Eshelman et al.): 8-point crossover was the best overall in terms of function evaluations needed to reach global optimum, averaged over a range of problem types.

**Two sources of bias**:
1. **Positional bias** -- reliance on building-block hypothesis
2. **Distributional bias** -- limits information exchange between parents

### 4.4 Uniform Crossover (UX)

Generate binary mask stochastically using Bernoulli distribution with parameter p.

```
Mask:  1 0 1 0 0 1
```

- 1 = take allele from parent 1
- 0 = take allele from parent 2

Default: p = 0.5 (Syswerda). Can bias toward one parent by choosing p != 0.5.

**Disruption tuning**: Amount of disruption in UX can be tuned by choosing different values of p (De Jong & Spears formal analysis).

### 4.5 Crossover as Binary Mask (General)

Any crossover operator can be written as a binary mask string:

```
Mask: 1 1 0 0 1 1    (equivalent to 2X with crosspoints 2 and 4)
```

### 4.6 Non-Linear Crossover (Permutation Problems)

Standard crossover on permutations produces INFEASIBLE solutions.

**Failure example** (1X, crosspoint=2, TSP):

```
P1: 1 6 | 3 4 5 2    ->    O1: 1 6 1 2 6 5  (INFEASIBLE: repeats 1,6; misses 3,4)
P2: 4 3 | 1 2 6 5    ->    O2: 4 3 3 4 5 2  (INFEASIBLE: repeats 3,4; misses 1,6)
```

#### 4.6.1 PMX (Partially Mapped Crossover)

1. Choose two crosspoints uniformly at random
2. Section between crosspoints defines an interchange mapping
3. Swap cut blocks; apply mapping to resolve conflicts

**Example** (crosspoints X=2, Y=5):

```
P1: 1 6 | 3 4 5 | 2
P2: 4 3 | 1 2 6 | 5

Interchange mapping:  3<->1,  4<->2,  5<->6

O1: 3 5 1 2 6 4    (feasible permutation)
O2: 2 1 3 4 5 6    (feasible permutation)
```

#### 4.6.2 Order-Based Crossover (via Binary Mask)

Apply binary mask with different semantics:
1. Positions with 1s: copy alleles from one parent
2. Positions with 0s: fill gaps with remaining elements in order from other parent

**Example** (mask = 1 0 1 0 0 1):

```
P1: 1 6 3 4 5 2  ->  copy 1s: 1 _ 3 _ _ 2  ->  fill from P2 order: O1 = 1 4 3 6 5 2
P2: 4 3 1 2 6 5  ->  copy 1s: 4 _ 1 _ _ 5  ->  fill from P1 order: O2 = 4 6 1 3 2 5
```

### 4.7 Reduced Surrogate (Booker)

Before applying crossover, compute XOR of parents. Only use crossover points between the outermost 1s of the XOR string to guarantee non-clone offspring.

```
P1: 1 1 0 1 0 0 1
P2: 1 1 0 0 0 1 0
XOR: 0 0 0 1 0 1 1
```

Only crossover points at positions 4, 5, 6 (between outermost 1s) produce different offspring. First three positions always yield clones.

### 4.8 Crossover Rate (chi)

- Crossover applied with probability chi < 1 (stochastic approach)
- Some implementations always apply crossover (chi = 1)
- **Adaptive crossover rate** (Booker): Vary according to "percent involvement" -- percentage of current population producing offspring. Low value = loss of diversity = premature convergence.

### 4.9 Crossover-AND-Mutation vs. Crossover-OR-Mutation

- **AND**: First attempt crossover, then attempt mutation on offspring
- **OR**: Apply either crossover or mutation, but not both
- **Davis recommendation**: High crossover at start, high mutation as population converges. Operator proportions adapted online based on track record.

---

## 5. Mutation Operators

### 5.1 Binary Mutation

Complement the chosen bit(s).

```
Mask: 0 1 0 0 0 1    (Bernoulli with small p at each locus)
```

Example: string `1 0 1 1 0 0 1` with genes 3 and 5 mutated -> `1 0 0 1 1 0 1`

### 5.2 Mutation Rate (mu)

- Bit-wise mutation rate: **mu = 1/l** (shown to be "optimal" as early as 1966, Bremermann et al.)
- Equivalent to lambda = 1 average mutation per chromosome

### 5.3 Efficient Implementation via Poisson Distribution

Instead of drawing a random number for every gene:

```
1. Draw m ~ Poisson(lambda), where lambda = mu * l
2. If m > 0, draw m positions uniformly without replacement from {1,...,l}
3. Mutate those positions
```

Common: lambda = 1 (one mutation per chromosome on average).

### 5.4 K-ary Mutation

When alleles have ordinal relation: restrict new value to alleles close to current value, or bias probability distribution in their favor.

### 5.5 Adaptive Mutation

- **Fogarty**: Different mutation rates at different loci
- **Reeves**: Mutation probability varied according to population diversity (coefficient of variation of fitnesses)
- General: Diversity-responsive mutation (increase mutation when diversity drops)

### 5.6 Parameter Interaction

High selection pressure requires high mutation rate to avoid premature convergence.

---

## 6. Diversity Maintenance

### 6.1 No-Duplicates Policy

Offspring not admitted to population if they are clones of existing individuals.
- Downside: O(M) comparison per insertion for large populations
- Possible improvement: hashing

### 6.2 Reduced Surrogate Crossover

Ensure crossover points only in regions where parents differ (see Section 4.7).

### 6.3 Termination-Based Diversity Tracking

Track population diversity. Stop when at every locus, proportion of one allele > 90%.

Diversity measures:
- Genotype statistics (most common)
- Phenotype statistics
- Fitness statistics (coefficient of variation)

### 6.4 Population Overlap / Incremental Replacement

Strategies to maintain diversity through replacement policies:
- **Elitism**: Preserve best individual; replace remaining M-1
- **Overlapping populations**: Replace only fraction G (generation gap) per generation
- **Steady-state / incremental**: Generate only 1 (or 2) new chromosomes per step

### 6.5 Deletion Strategies for Incremental GA

| Strategy | Description |
|----------|-------------|
| Replace parents | Children replace their parents |
| Delete worst | Remove worst member(s) -- very strong selection pressure (GENITOR) |
| Delete from worst p% | Milder; e.g., p=50 means select for deletion from worse-than-median |
| Age-based | Delete oldest strings |

**Warning** (Goldberg & Deb): Delete-worst exerts very strong selective pressure; may need large populations and high mutation to prevent rapid diversity loss.

---

## 7. Parameter Settings and Recommendations

### 7.1 Population Size

| Approach | Recommendation |
|----------|----------------|
| Goldberg (schema theory) | M grows exponentially with string length l (impractical) |
| Experimental (Grefenstette; Schaffer et al.) | Much smaller than Goldberg's theory suggests |
| Reeves (coverage theory) | M grows as O(log l) -- sufficient for search space coverage |

**Minimum population size for binary strings** (probability of having all alleles at each locus):

```
P*_2 = (1 - (1/2)^{M-1})^l
```

**Example**: M = 17 ensures P*_2 > 99.9% for l = 50.

For q-ary alphabets: see Reeves [98] for numerical expressions.

### 7.2 Latin Hypercube Initialization

For alphabet size k, population size M = multiple of k:
1. For each gene column: generate independent random permutation of {0,...,M-1}
2. Take values modulo k

**Table -- Latin Hypercube Example** (l=6, |A|=5, M=10):

| Individual | Gene 1 | Gene 2 | Gene 3 | Gene 4 | Gene 5 | Gene 6 |
|------------|--------|--------|--------|--------|--------|--------|
| 1          | 0      | 1      | 3      | 0      | 2      | 4      |
| 2          | 1      | 4      | 4      | 2      | 3      | 0      |
| 3          | 0      | 0      | 1      | 2      | 4      | 3      |
| 4          | 2      | 4      | 0      | 3      | 1      | 4      |
| 5          | 3      | 3      | 0      | 4      | 4      | 2      |
| 6          | 4      | 1      | 2      | 4      | 3      | 0      |
| 7          | 2      | 0      | 1      | 3      | 0      | 1      |
| 8          | 1      | 3      | 3      | 1      | 2      | 2      |
| 9          | 4      | 2      | 2      | 1      | 1      | 3      |
| 10         | 3      | 2      | 4      | 0      | 0      | 1      |

Each allele occurs exactly twice (M/k = 10/5 = 2) per gene.

### 7.3 Seeding

Include known good solutions (from other heuristics) in initial population.
- Pro: Faster convergence to high-quality solutions
- Con: Risk of premature convergence

### 7.4 Practitioner's Recommendations (Levine + Community Additions)

| # | Recommendation |
|---|----------------|
| 1 | Steady-state (incremental) generally more effective than generational |
| 2 | Do NOT use simple roulette-wheel selection. Use tournament or SUS |
| 3 | Do NOT use one-point crossover. Prefer UX or 2X |
| 4 | Use adaptive mutation rate -- fixed rate (even 1/l) is too inflexible |
| 5 | Hybridize wherever possible -- use problem-specific information |
| 6 | Make diversity maintenance a priority |
| 7 | Run the GA multiple times (attractors are typically a subset of local optima) |

**Caveat**: Points 1 and 2 conflict slightly -- SUS functions best in generational setting.

---

## 8. Constraint Handling

### 8.1 Problem

Two feasible parents may produce infeasible offspring. This is the fundamental challenge for GAs in constrained combinatorial optimization.

The search space `S <= A^l` reflects that some strings in the image of `A^l` under mapping `c` may represent invalid solutions.

### 8.2 Approaches Mentioned

| Approach | Description |
|----------|-------------|
| Special crossover operators | Construct crossover operators that preserve feasibility (e.g., PMX for permutations) |
| Subset selection | For problems like knapsack, treat as subset selection problems rather than naive binary GA |
| Problem-specific encoding | Choose representation that naturally avoids infeasibility |

*(Note: This chapter does not provide detailed coverage of penalty functions, repair operators, or decoder methods -- these are covered in reference [11].)*

---

## 9. Multi-Objective GA

This chapter does **not** cover multi-objective optimization (no Pareto, NSGA, SPEA content).

---

## 10. All Equations and Formulas (Consolidated)

### Mapping and Objective

```
c : A^l -> X                        (genotype-phenotype mapping)
g(s) = f(c(s))                      (objective in genotype space)
fitness = h(f(c(s)))                 (h : IR -> IR+, monotonic)
```

### Schema Theorem

```
E[N(S, t+1)] >= (F(S,t) / F_avg(t)) * N(S,t) * {1 - epsilon(S,t)}
```

### Population Size -- Minimum for Binary (All-Alleles Coverage)

```
P*_2 = (1 - (1/2)^{M-1})^l
```

M=17 => P*_2 > 0.999 for l=50. Growth: O(log l).

### Roulette-Wheel Selection

```
P(select i) = f_i / sum_{j=1}^{M} f_j
```

### Linear Ranking Selection

```
P[k] = alpha + beta * k

beta = 2*(phi - 1) / (M*(M-1))
alpha = (2*M - phi*(M+1)) / (M*(M-1))

Constraint: 1 <= phi <= 2

CDF inversion (find k given r):
alpha*k + beta*k*(k+1)/2 = r

k = ( -(2*alpha + beta) +/- sqrt((2*alpha + beta)^2 + 4*beta*r) ) / (2*beta)
```

### Tournament Selection

```
phi = 2^{tau-1}

P(median selected) = 2^{-(tau-1)}
P(string not appearing, sampling w/ replacement) ~ e^{-1} ~ 0.368
```

### Mutation Rate

```
mu_optimal = 1/l    (one mutation per chromosome on average)
lambda = mu * l = 1  (Poisson parameter)
```

### Flowshop Completion Time Recurrence

```
C(pi_1, 1) = p(pi_1, 1)
C(pi_i, 1) = C(pi_{i-1}, 1) + p(pi_i, 1)            for i = 2,...,n
C(pi_1, j) = C(pi_1, j-1) + p(pi_1, j)               for j = 2,...,m
C(pi_i, j) = max{C(pi_{i-1}, j), C(pi_i, j-1)} + p(pi_i, j)   for i=2,...,n; j=2,...,m
```

### Knapsack Problem

```
maximize sum_{i=1}^{n} x_i * v_i
subject to sum_{i=1}^{n} x_i * c_i <= C
x_i in {0, 1}
```

### Example Continuous Optimization (Binary-Encoded)

```
f(x) = x^3 - 60*x^2 + 900*x + 100
x in {0, 1, ..., 31}
Encoding: 5-bit binary
```

---

## 11. Population Management Strategies (ES Notation)

| Strategy | Description |
|----------|-------------|
| (lambda, mu) | Generate mu (> lambda) offspring from lambda parents; select best lambda offspring for next generation |
| (lambda + mu) | Generate mu offspring; select best lambda from combined parents + offspring |
| Generational | Replace entire population each generation |
| Elitist | Preserve best individual; replace M-1 others |
| Steady-state | Replace 1 or 2 individuals per generation (G -> 0) |
| Generation gap G | Replace fraction G of population per generation |

---

## 12. Termination Criteria

| Method | Description |
|--------|-------------|
| Evaluation limit | Fixed number of fitness evaluations |
| Time limit | Fixed CPU clock time |
| Diversity threshold | Stop when allele proportion at every locus exceeds 90% for one allele |

---

## 13. Crossover Bias Classification

| Crossover Type | Positional Bias | Distributional Bias |
|----------------|-----------------|---------------------|
| 1X             | High            | None                |
| 2X             | Moderate        | Some                |
| mX (m=8 best)  | Reduced         | Increased           |
| UX (p=0.5)     | None            | None                |
| UX (p != 0.5)  | None            | Biased toward one parent |

---

## 14. Key Theoretical Concepts

### Epistasis
Non-linear interaction between genes: the expression of alleles at some loci depends on alleles at other loci. Equivalent to interaction terms in the fitness function. If known, could guide algorithm choice.

### Building Block Hypothesis (BBH)
Recombination of small good schemata into larger good schemata leads to optimal solutions. Negative evidence: "royal road" functions designed with building blocks of increasing size were more efficiently solved by non-genetic methods.

### Landscape Theory
Connections between GAs and neighbourhood search through problem landscape analysis. Rigorous mathematical foundation unifying different metaheuristics.

### Competing Conventions
When encoding mapping `c(.)` is not injective, multiple genotypes represent the same phenotype. Fix by anchoring/fixing one element of the encoding.

---

*Source: Reeves, C.R. (2010). Genetic Algorithms. In: Gendreau, M., Potvin, J.-Y. (eds) Handbook of Metaheuristics. International Series in Operations Research & Management Science, vol 146. Springer.*
