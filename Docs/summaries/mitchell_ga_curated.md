# Genetic Algorithms: Technical Reference
## Extracted from "An Introduction to Genetic Algorithms" by Melanie Mitchell (MIT Press, 1996)

---

## 1. Core GA Algorithm

### 1.1 Simple Genetic Algorithm (SGA)

```
SIMPLE-GA(n, l, pc, pm):
  1. Generate random population of n chromosomes, each of length l bits
  2. Calculate fitness f(x) for each chromosome x in population
  3. Repeat until n offspring created:
     a. Select pair of parents from population
        (probability of selection is increasing function of fitness)
        Selection is WITH REPLACEMENT (same chromosome can be selected multiple times)
     b. With probability pc, apply single-point crossover at random position
        - If crossover: produce two offspring from recombined segments
        - If no crossover: offspring are exact copies of parents
     c. Mutate each bit in both offspring with probability pm
     d. Place offspring in new population
  4. Replace current population with new population
  5. Go to step 2
```

**Typical parameter values:**
- String length l: 50-1000
- Population size n: 50-1000 (commonly 100)
- Crossover rate pc: 0.6-0.95 (commonly 0.7)
- Mutation rate pm: 0.001-0.01 per bit

**Iterations:** 50-500+ generations per run. Multiple runs with different random seeds are standard practice.

### 1.2 Steady-State GA

```
STEADY-STATE-GA(n, l):
  1. Generate random population of n chromosomes
  2. Calculate fitness for all chromosomes
  3. Repeat:
     a. Select fittest individuals as parents
     b. Apply crossover and mutation to produce small number of offspring
     c. Replace least-fit individuals in population with new offspring
     d. Evaluate fitness of new offspring
  4. Go to step 3
```

- Only a few individuals replaced each generation (small "generation gap")
- Useful for incremental learning and classifier systems
- Population members collectively solve the problem

### 1.3 Rank-Selection Elitist GA (used in cellular automata experiments)

```
RANK-ELITIST-GA(n, E, l, generations):
  1. Generate random population of n chromosomes
  2. For each generation:
     a. Calculate fitness for each chromosome
     b. Rank population by fitness
     c. Copy top E chromosomes ("elite") to next generation unchanged
     d. Generate (n - E) offspring by:
        - Randomly choosing pairs from elite (with replacement)
        - Applying single-point crossover
        - Mutating offspring
     e. Merge elite and offspring to form new population
  3. Go to step 2
```

Parameters used in CA experiments: n=100, E=20, generations=100, 2 mutations per offspring.

### 1.4 Genetic Programming (GP) Algorithm (Koza)

```
GP-ALGORITHM(pop_size, functions, terminals, fitness_cases):
  1. Choose function set and terminal set for the problem
  2. Generate initial population of random parse trees (syntactically correct)
     - Maximum tree size restricted
  3. Calculate fitness of each program on fitness cases
  4. Form new population:
     a. 10% of population: copy (probabilistically by fitness) without modification
     b. 90% of population: crossover between fitness-proportionate parents
        - Choose random point in each parent tree
        - Exchange subtrees beneath those points
        - Tree size can increase or decrease
  5. Repeat steps 3-4 for desired number of generations
```

### 1.5 Messy GA

```
MESSY-GA(l, k):
  Phase 1 - PRIMORDIAL PHASE:
    1. Initialize population with all schemas of order k
       (each bit tagged with its "real" locus)
    2. Evaluate fitness using "competitive templates":
       - Find local optimum via hill climbing
       - Fill missing bits from local optimum
       - Apply fitness function
    3. Selection only (no crossover/mutation)
    4. Cull population by half at regular intervals

  Phase 2 - JUXTAPOSITIONAL PHASE:
    1. Population size fixed; selection continues
    2. Apply CUT operator: cut string at random point into two strings
    3. Apply SPLICE operator: concatenate two strings
    4. Overspecification resolved: first-come-first-served (left to right)
    5. Underspecification resolved: fill from competitive template
```

Number of order-k schemas in length-l string:
```
n = 2^k * C(l, k)
```

---

## 2. Representation Schemes

### 2.1 Binary Encoding
- Most common encoding; each locus has alleles {0, 1}
- Holland's justification: length-l binary string contains 2^l schemas vs. 2^(l/log2(a)) for alphabet size a
- Supports standard crossover and mutation operators directly

### 2.2 Gray Coding
- Alternative binary encoding where adjacent integers differ by exactly one bit
- Reduces Hamming cliffs in parameter spaces
- Referenced: Bethke 1980; Caruana and Schaffer 1988

### 2.3 Real-Valued Encoding
- Each gene is a real number (e.g., neural network weights, torsion angles)
- Natural for continuous optimization problems
- Requires modified crossover and mutation operators
- Empirical studies show better performance than binary on some problems (Janikow and Michalewicz 1991; Wright 1991)

### 2.4 Tree Encoding (Genetic Programming)
- Parse trees representing programs (Lisp S-expressions)
- Open-ended: tree size can grow/shrink under crossover
- Nodes = functions (operators); leaves = terminals (variables, constants)
- Crossover: exchange subtrees at random points
- Potential problem: uncontrolled growth ("bloat")

### 2.5 Tagged/Messy Encoding
- Each allele tagged with its position index: {(position, value), ...}
- Allows underspecification (missing loci) and overspecification (duplicate loci)
- Supports variable-length chromosomes
- Compatible with cut and splice operators

### 2.6 Diploid Encoding (Hillis)
- Chromosomes in pairs (15 pairs of 32-bit chromosomes for sorting networks)
- Each chromosome: 8 codons of 4 bits each
- Homozygous positions: one comparison inserted in phenotype
- Heterozygous positions: both comparisons inserted
- More homozygous = smaller network

### 2.7 Encoding for Ordering Problems
- Inversion operator: reverse segment between two random points
- Each allele carries index for its "real" position
- Fitness evaluated using index-ordered string

---

## 3. Selection Methods

### 3.1 Fitness-Proportionate Selection (Roulette Wheel)

**Expected number of offspring for individual i:**
```
ExpVal(i, t) = f(i) / f_bar(t)
```
- `f(i)` = fitness of individual i
- `f_bar(t)` = mean fitness of population at time t

**Roulette Wheel Sampling:**
```
ROULETTE-WHEEL(population, N):
  1. T = sum of all expected values in population
  2. Repeat N times:
     a. r = random integer in [0, T]
     b. Loop through individuals, summing expected values
     c. Select individual whose cumulative sum >= r
```

**Problem:** High fitness variance early causes premature convergence; low variance later causes stagnation.

### 3.2 Stochastic Universal Sampling (SUS) - Baker 1987

```c
ptr = Rand();  // uniform in [0, 1]
for (sum = i = 0; i < N; i++)
    for (sum += ExpVal(i, t); sum > ptr; ptr++)
        Select(i);
```

- Single spin with N equally-spaced pointers
- **Guarantee:** Individual i reproduces at least floor(ExpVal(i,t)) times and at most ceil(ExpVal(i,t)) times
- Minimizes spread (range of possible actual offspring counts)

### 3.3 Sigma Scaling

**Expected value formula:**
```
ExpVal(i, t) = 1 + (f(i) - f_bar(t)) / (2 * sigma(t))
```
- `f(i)` = fitness of individual i
- `f_bar(t)` = mean population fitness at time t
- `sigma(t)` = standard deviation of population fitnesses at time t
- If ExpVal < 0, reset to small positive value (e.g., 0.1)

**When to use:** Keeps selection pressure relatively constant throughout the run. Individual with fitness one standard deviation above mean gets 1.5 expected offspring.

**Variant (used in Royal Road experiments):**
```
ExpVal(i) = (f(i) - f_bar) / sigma
Capped at maximum of 1.5
```

### 3.4 Boltzmann Selection

**Expected value formula:**
```
ExpVal(i, t) = e^(f(i)/T) / <e^(f(i)/T)>_t
```
- `T` = temperature parameter (decreases over time)
- `<...>_t` = average over population at time t
- High T: low selection pressure (all individuals roughly equal probability)
- Low T: high selection pressure (fittest individuals dominate)

**When to use:** When you want to control exploration/exploitation balance over time (similar to simulated annealing). Start with high T for exploration, decrease for exploitation.

**Alternative formulation (Prugel-Bennett and Shapiro):**
```
p_alpha = e^(-beta * E_alpha) / sum_over_population(e^(-beta * E_i))
```
- `E_alpha` = energy (negative fitness) of individual alpha
- `P` = population size
- `beta` = selection strength parameter

### 3.5 Rank Selection (Baker 1985)

**Linear ranking formula:**
```
ExpVal(i, t) = Min + (Max - Min) * (rank(i) - 1) / (N - 1)
```
- `rank(i)` = rank of individual i (1 = least fit, N = most fit)
- `Max` = expected value of best individual (constraint: 1 <= Max <= 2)
- `Min` = 2 - Max (expected value of worst individual)
- Recommended: Max = 1.1

**When to use:** Prevents premature convergence by obscuring absolute fitness differences. Maintains selection pressure when fitness variance is low. No need for fitness scaling.

### 3.6 Tournament Selection

```
TOURNAMENT-SELECT(population, k):
  1. Choose 2 individuals at random from population
  2. Generate random number r in [0, 1]
  3. If r < k (e.g., k = 0.75):
     - Select the FITTER individual
  4. Else:
     - Select the LESS FIT individual
  5. Return both to population (can be selected again)
```

**When to use:** Computationally efficient (no need to compute population statistics); easily parallelizable; selection pressure similar to rank selection.

### 3.7 (mu + lambda) Selection (Evolution Strategies style)

```
(MU+LAMBDA)-SELECT(population, mu_frac):
  1. Rank population by fitness
  2. Select top mu fraction as parents (e.g., top 20%)
  3. Copy all parents to next generation
  4. Generate offspring from parents via crossover/mutation
  5. Replace bottom (1 - mu) fraction with offspring
```

**When to use:** Noisy fitness functions; preserves elite individuals for repeated testing across generations.

### 3.8 Elitism (De Jong 1975)

- Retain the best N_e individuals from each generation unchanged
- Prevents loss of best solutions through crossover or mutation
- Can be added to any selection method
- Significantly improves GA performance in many applications

---

## 4. Crossover Operators

### 4.1 Single-Point Crossover

```
SINGLE-POINT-CROSSOVER(parent1, parent2, pc):
  1. With probability pc:
     a. Choose random position k in [1, l-1]
     b. offspring1 = parent1[1..k] + parent2[k+1..l]
     c. offspring2 = parent2[1..k] + parent1[k+1..l]
  2. Otherwise:
     offspring1 = copy(parent1)
     offspring2 = copy(parent2)
```

**Limitations:**
- Positional bias: schemas near string endpoints treated differently
- Cannot combine all schema pairs (e.g., cannot combine 11*****1 and ****11**)
- Tends to destroy long-defining-length schemas
- Preserves hitchhikers near beneficial schemas

### 4.2 Two-Point Crossover

```
TWO-POINT-CROSSOVER(parent1, parent2, pc):
  1. With probability pc:
     a. Choose two random positions k1, k2 (k1 < k2)
     b. offspring1 = parent1[1..k1] + parent2[k1+1..k2] + parent1[k2+1..l]
     c. offspring2 = parent2[1..k1] + parent1[k1+1..k2] + parent2[k2+1..l]
  2. Otherwise: copy parents
```

- Less likely to disrupt schemas with large defining lengths
- Can combine more schemas than single-point
- Exchanged segments do not necessarily contain string endpoints

### 4.3 Uniform Crossover (Parameterized)

```
UNIFORM-CROSSOVER(parent1, parent2, p):
  For each bit position i from 1 to l:
    With probability p:
      offspring1[i] = parent1[i]; offspring2[i] = parent2[i]
    With probability (1-p):
      offspring1[i] = parent2[i]; offspring2[i] = parent1[i]
```

- Typical p: 0.5 to 0.8
- No positional bias: any schemas can potentially be recombined
- Can be highly disruptive of coadapted alleles
- Common in recent applications with p ~ 0.7-0.8

### 4.4 GP Crossover (Tree Crossover)

```
GP-CROSSOVER(tree1, tree2):
  1. Choose random node in tree1 as crossover point
  2. Choose random node in tree2 as crossover point
  3. Exchange subtrees beneath the two crossover points
  4. Result: two offspring trees (size may change)
```

- Allows variable-size offspring
- Subtree exchange preserves syntactic correctness

### 4.5 Diploid Crossover (Hillis)

```
DIPLOID-CROSSOVER(individual1, individual2):
  For each of 15 chromosome pairs:
    1. Choose random crossover point within chromosome pair
    2. Form gamete from individual1:
       - Codons before crossover point from chromosome A
       - Codons after crossover point from chromosome B
    3. Form gamete from individual2 (same procedure)
    4. Pair gametes to form one chromosome pair in offspring
```

### 4.6 Montana-Davis Crossover (Neural Network Weights)

```
NN-CROSSOVER(parent1, parent2):
  For each non-input unit in offspring:
    1. Choose one parent at random
    2. Copy ALL incoming link weights from that parent
  Result: single offspring (not two)
```

**Rationale:** Incoming weights to a unit form a functional building block.

### 4.7 Crossover Hot Spots (Schaffer and Morishima 1987)

```
HOT-SPOT-CROSSOVER(parent1, parent2):
  Each chromosome has attached crossover template (binary string)
  1. Template 1s mark allowed crossover positions
  2. Multi-point crossover at marked positions
  3. Templates are inherited alongside chromosome bits
  4. Mutation acts on both chromosome and template
```

---

## 5. Mutation Operators

### 5.1 Bit-Flip Mutation

```
BIT-FLIP-MUTATION(chromosome, pm):
  For each bit position i:
    With probability pm:
      chromosome[i] = 1 - chromosome[i]
```
- Standard for binary encodings
- Typical pm: 0.001 to 0.01 per bit
- Primary role: prevent permanent fixation at any locus

### 5.2 Real-Valued Mutation (Schulze-Kremer)

**Type 1 - Value Replacement:**
```
Replace randomly chosen gene with value from set of
most frequently occurring values for that parameter
```

**Type 2 - Incremental:**
```
gene[i] = gene[i] + small_random_delta
```

### 5.3 Neural Network Weight Mutation (Montana and Davis)

```
NN-MUTATION(weight_vector, n_units):
  1. Select n non-input units at random
  2. For each selected unit:
     For each incoming link weight:
       weight += random_value_in[-1.0, +1.0]
```

### 5.4 Condition Set Mutations (Meyer and Packard)

Four mutation operators for evolving condition sets:
```
1. ADD: Insert a new condition to the set
2. DELETE: Remove a condition from the set
3. BROADEN/SHRINK: Widen or narrow a range condition
4. SHIFT: Move a range condition up or down
```

### 5.5 Adaptive Mutation (Kitano)

```
pm(offspring) = f(hamming_distance(parent1, parent2))
  - High distance between parents -> Low mutation rate
  - Low distance between parents -> High mutation rate
```

**Rationale:** Responds to loss of population diversity by increasing mutation.

### 5.6 GP Mutation

```
GP-MUTATION(tree):
  1. Choose random node in tree
  2. Replace entire subtree beneath that node with randomly generated subtree
```

---

## 6. Schema Theorem and Building Blocks

### 6.1 Definitions

- **Schema:** Template over {0, 1, *} where * = "don't care"
  - Example: H = 1****1 represents all 6-bit strings starting and ending with 1
- **Order o(H):** Number of defined (non-*) bits in schema H
- **Defining length d(H):** Distance between outermost defined bits
- **Instance:** String matching a schema template
- **A string of length l is an instance of exactly 2^l different schemas**
- **Total schemas for length l:** 3^l
- **Schemas in a population of n strings:** between 2^l and n * 2^l

### 6.2 Schema Theorem (Holland 1975)

**Without crossover and mutation (selection only):**
```
E[m(H, t+1)] = m(H, t) * u_hat(H, t) / f_bar(t)
```

**Variables:**
- `m(H, t)` = number of instances of schema H at time t
- `u_hat(H, t)` = observed average fitness of instances of H at time t
- `f_bar(t)` = average fitness of entire population at time t
- `E[...]` = expected value

**Full Schema Theorem (with destructive effects of crossover and mutation):**
```
E[m(H, t+1)] >= m(H, t) * [u_hat(H, t) / f_bar(t)] * Sc(H) * Sm(H)
```

**Where:**

Survival probability under single-point crossover:
```
Sc(H) >= 1 - pc * d(H) / (l - 1)
```
- `pc` = crossover probability
- `d(H)` = defining length of schema H
- `l` = string length

Survival probability under mutation:
```
Sm(H) = (1 - pm)^o(H)
```
- `pm` = per-bit mutation probability
- `o(H)` = order of schema H

**Key implication:** Short, low-order schemas with above-average fitness receive exponentially increasing samples over time.

### 6.3 Building Block Hypothesis

Crossover combines short, low-order, high-fitness schemas ("building blocks") into progressively higher-order, higher-fitness candidate solutions.

### 6.4 Implicit Parallelism

- A population of n strings implicitly estimates average fitnesses of between 2^l and n * 2^l schemas
- No additional memory or computation time beyond processing the n individuals
- The GA implements a near-optimal multi-armed bandit strategy across schema partitions

### 6.5 Schema Partitions

A partition of order k divides the search space into 2^k directly competing schemas.

**Correct interpretation (Holland):** The GA plays a 2^k-armed bandit in each order-k schema partition, proceeding from low-order partitions early to higher-order partitions later.

### 6.6 Limitations of Static Schema Analysis

**Collateral convergence:** Once population converges at some loci, sampling of other schemas becomes biased (non-independent).

**High fitness variance:** If a schema's static average fitness has high variance, the GA's estimate may be inaccurate.

**Static Building Block Hypothesis (SBBH):** "GA converges to schema with best static average fitness in each partition" -- this has never been proved or validated, and deceptive problems are defined in terms of it.

**Deception** is neither necessary nor sufficient to cause difficulties for GAs (Grefenstette 1993).

---

## 7. Fitness Functions and Scaling

### 7.1 Sigma Scaling (Sigma Truncation)

```
ExpVal(i, t) = 1 + (f(i) - f_bar(t)) / (2 * sigma(t))
```
If ExpVal(i, t) < 0, set to small value (e.g., 0.1)

**Purpose:** Maintain relatively constant selection pressure regardless of fitness variance in population.

### 7.2 Boltzmann Scaling

```
ExpVal(i, t) = e^(f(i)/T) / <e^(f(j)/T)>_t
```

- T starts high (low selection pressure) and decreases over time
- Similar to simulated annealing temperature schedule

### 7.3 Linear Rank Scaling (Baker 1985)

```
ExpVal(i, t) = Min + (Max - Min) * (rank(i) - 1) / (N - 1)
```

Constraints: 1 <= Max <= 2, Min = 2 - Max, recommended Max = 1.1

### 7.4 Fitness Sharing (Goldberg and Richardson 1987)

```
f_shared(i) = f(i) / sum_j(sh(d(i,j)))
```

- `sh(d)` = sharing function (decreasing function of distance d)
- Penalizes individuals similar to many others
- Induces speciation -- population converges on multiple peaks

### 7.5 Competitive Templates (Messy GA)

```
EVALUATE-MESSY(chromosome, local_optimum):
  1. For underspecified positions: fill from local_optimum
  2. For overspecified positions: use first (leftmost) value
  3. Apply standard fitness function to completed string
```

### 7.6 Meyer-Packard Fitness (Prediction)

```
fitness(C) = (1 - sigma_C/sigma_0) - alpha/N_C
```

- `sigma_C` = std deviation of y-values for data points satisfying condition C
- `sigma_0` = std deviation of y-values over entire dataset
- `N_C` = number of data points satisfying condition C
- `alpha` = penalty constant for poor statistics
- First term: information content; second term: statistical reliability penalty

### 7.7 Hinton-Nowlan Fitness (Baldwin Effect)

```
fitness = 1 + (19 * n) / 1000
```
- n = number of remaining trials (out of 1000) after correct solution found
- Correct from start: fitness = 20
- Never found: fitness = 1

---

## 8. Population Management

### 8.1 Elitism
- Retain top N_e individuals unchanged each generation
- Prevents loss of best solutions
- Significantly improves performance (De Jong 1975)

### 8.2 Crowding (De Jong 1975)
- New offspring replaces the existing individual most similar to it
- Prevents "crowds" of identical individuals
- Maintains population diversity

### 8.3 Fitness Sharing (Goldberg and Richardson 1987)
- Each individual's fitness decreased by presence of similar individuals
- Amount of decrease is increasing function of similarity
- Induces speciation: population converges on multiple peaks

### 8.4 Mating Restrictions
- **Similarity-based mating:** Only sufficiently similar individuals mate (forms species) -- Deb and Goldberg 1989
- **Incest prevention:** Disallow mating between sufficiently similar individuals (maintains diversity) -- Eshelman and Schaffer 1991
- **Mating tags:** Parts of chromosome identify prospective mates -- Holland 1975; Booker 1985

### 8.5 Spatial Structure
- Population on 2D lattice; mating restricted to spatial neighbors
- Fosters speciation at different spatial locations
- Innovations occur at boundaries between species (Hillis 1992)

### 8.6 Host-Parasite Coevolution (Hillis)
- Two populations on same grid: "hosts" (solutions) and "parasites" (test cases)
- Host fitness = percentage of parasite test cases solved correctly
- Parasite fitness = percentage of test cases that stump the host
- Parasites evolve to target weaknesses; hosts forced to keep improving
- Prevented getting stuck at local optima in sorting network evolution

---

## 9. GA Theory

### 9.1 Two-Armed Bandit Problem (Holland)

**Setup:** N coins, two arms with unknown mean payoffs mu_1, mu_2 and variances sigma_1^2, sigma_2^2.

**Optimal allocation:** Give n* trials to observed worse arm, N - n* to observed better arm.

```
N - n* ~ e^(c * n*)
```

The optimal number of trials to the observed better arm increases **exponentially** with trials given to the observed worse arm.

**GA connection:** Under the Schema Theorem, schemas with above-average observed fitness receive exponentially increasing samples -- approximating the optimal multi-armed bandit strategy.

### 9.2 Exploration vs. Exploitation

- **Exploration:** Testing new, previously unseen schemas
- **Exploitation:** Using and propagating known good schemas
- GA achieves near-optimal balance (Holland's claim)
- On-line performance: every trial counts (GA strength)
- Off-line performance: only best result matters (hybrid methods often better)
- For true optimization: GA + hill climber hybrid recommended

### 9.3 Deception

A fitness function is **fully deceptive** if every low-order schema partition's "winner" (highest static average fitness) points toward the complement of the global optimum.

Example: If 111...1 is global optimum, any schema containing all 1s in its defined positions is a winner -- except 111...1 itself. And 000...0 is a winner.

**Deception is neither necessary nor sufficient for GA difficulty** (Grefenstette 1993).

### 9.4 Exact Mathematical Models (Vose and Liepins 1991)

**Population representation:**
- Population at time t: vector p(t) of length 2^l
- p_i(t) = proportion of string i in population at time t
- Selection probability: s_i(t) = F * p(t) / |F * p(t)|
- F = diagonal fitness matrix: F_{i,i} = f(i)

**GA as operator:**
```
p(t+1) ~ G(p(t))
```
where G = F composed with recombination operator R.

**Recombination probability r_{i,j}(0):**
Probability that string 0 (all zeros) is produced from parents i and j:

```
r_{i,j}(0) = (1/2)(1-pc) * [theta^|i| * (1-pm)^(l-|i|) + theta^|j| * (1-pm)^(l-|j|)]
            + (1/2) * pc/(l-1) * sum_{c=1}^{l-1} [theta^|h| * (1-pm)^(l-|h|) + theta^|k| * (1-pm)^(l-|k|)]
```
where theta = pm/(1-pm), |i| = number of ones in string i, and h, k are the two offspring from crossover at point c.

**Fixed points:**
- **Selection alone (F):** Stable fixed points are populations consisting entirely of maximally fit strings
- **Recombination alone (R):** Single fixed point is uniform distribution over all strings

### 9.5 Finite-Population Markov Chain Model (Nix and Vose 1991)

**State space:** All possible populations of size n from strings of length l.

Number of possible populations:
```
C(n + 2^l - 1, 2^l - 1)
```

**Transition probability** from population P_i to P_j:
```
Q_{i,j} = [n! / (Z_{0,j}! * Z_{1,j}! * ... * Z_{2^l-1,j}!)] * product_{y=0}^{2^l-1} p_i(y)^{Z_{y,j}}
```

where Z_{y,j} = number of occurrences of string y in population j, and p_i(y) = probability of generating string y from population P_i.

**Key results:**
- As n -> infinity, Markov chain trajectories converge to iterates of infinite-population operator G
- If G_p has single fixed point, GA asymptotically spends all time there
- Long-term behavior determined by structure of "GA surface" (basins of attraction)

### 9.6 Statistical Mechanics Approach (Prugel-Bennett and Shapiro 1994)

**Strategy:** Predict macroscopic quantities (mean energy, variance) rather than track every individual.

**Iteration scheme:**
```
rho_0(E) -> [selection] -> rho_s_t(E) -> [crossover] -> rho_sc_t(E) = rho_{t+1}(E)
```

**Cumulant equations (selection):**
```
k1_s ~ k1 - beta * k2          (mean shifts toward lower energy)
k2_s ~ (1 - 1/P) * k2 - beta * k3   (variance decreases)
```
- k1 = mean energy, k2 = variance, k3 = skew
- beta = selection strength, P = population size

**When to use:** For predicting population-level statistics and choosing optimal parameters.

---

## 10. GA + Neural Networks

### 10.1 Evolving Weights (Montana and Davis 1989)

**Encoding:** Chromosome = ordered list of all 126 real-valued weights
**Network:** 4 input, 7+10 hidden (2 layers), 1 output, fully connected feedforward
**Fitness:** Negative sum of squared errors over 236 training examples
**Population:** 50 weight vectors, rank selection, 200 generations

**Crossover:** For each non-input unit in offspring, randomly choose one parent and copy all incoming weights from that parent.

**Mutation:** Select n non-input units; add random value in [-1, +1] to each incoming weight.

**Result:** GA significantly outperformed back-propagation on sonar classification task.

### 10.2 Evolving Network Architecture - Direct Encoding (Miller, Todd, and Hegde 1989)

**Encoding:** N x N connectivity matrix where entries are 0 (no connection) or L (learnable connection)
**Chromosome:** Rows of matrix concatenated into bit string (0 -> 0, L -> 1)
**Crossover:** Choose random row index, swap corresponding rows between parents
**Fitness:** Sum of squared errors after back-propagation training for fixed number of epochs
**Parameters:** Population 50, crossover rate 0.6, mutation rate 0.005

### 10.3 Evolving Network Architecture - Grammatical Encoding (Kitano 1990)

**Encoding:** Graph-generation grammar rules encoded in chromosomes
- Each rule: 5 loci (1 left-hand side + 4 symbols for 2x2 right-hand matrix)
- Alleles: {A-Z} (nonterminals), {a-p} (terminals representing 2x2 binary matrices)
- Grammar develops into connectivity matrix -> neural network

**Fitness:** Sum of squared errors after back-propagation training
**Selection:** Fitness-proportionate, multi-point crossover, adaptive mutation

**Result:** Grammatical encoding consistently outperformed direct encoding on encoder/decoder problems (lower error, faster convergence, better scaling with network size).

### 10.4 Evolving Learning Rules (Chalmers 1990)

**Learning rule form:**
```
delta_w_ij = k0 * (k1*a_i + k2*o_j + k3*t_j + k4*w_ij + k5*a_i*o_j + k6*a_i*t_j + k7*a_i*w_ij + k8*o_j*t_j + k9*o_j*w_ij + k10*t_j*w_ij)
```

**Variables:**
- a_i = activation of input unit i
- o_j = activation of output unit j
- t_j = training signal on output unit j
- w_ij = current weight from i to j
- k0 = scale parameter (learning rate)
- k1...k10 = coefficients to evolve

**Encoding:** k0 = 5 bits (sign + 4-bit exponent), k1-k10 = 3 bits each (sign + 2-bit magnitude)
**Target:** Delta rule: delta_w_ij = eta * a_i * (t_j - o_j)

**Parameters:** Population 40, two-point crossover (rate 0.8), mutation rate 0.01, 1000 generations
**Environment:** 30 linearly separable mappings, 20 used for training

**Results:**
- Mean best fitness: ~92% (delta rule = 98%)
- GA discovered delta rule on 1/10 runs
- Test set performance: 91.9% (excellent generalization)
- Diverse environments (10-20 tasks) required for general rules

---

## 11. Royal Road Functions

### 11.1 Royal Road R1

**Definition:**
```
R1(x) = sum over schemas s_i: c_s if x is instance of s_i
```

**Structure:** 8 adjacent blocks of 8 ones each in a 64-bit string.
- c_s = order(s) = 8 for each block
- R1(11111111 00...0) = 8
- R1(11111111 00...0 11111111) = 16
- R1(111...1) = 64 (optimum)

### 11.2 Experimental Results on R1

| Algorithm | Mean Evaluations | Median Evaluations |
|-----------|-----------------|-------------------|
| GA (sigma scaling) | 61,334 (SE: 2,304) | 54,208 |
| SAHC | > 256,000 | > 256,000 |
| NAHC | > 256,000 | > 256,000 |
| RMHC | 6,179 (SE: 186) | 5,775 |

**GA parameters:** Population 128, sigma truncation selection (capped at 1.5 offspring), pc=0.7, pm=0.005.

**RMHC outperformed GA by factor of ~10** on R1 -- a function designed as a "royal road" for the GA.

### 11.3 Analysis of RMHC on R1

**Expected time for RMHC to find optimum (N blocks of K ones):**
```
E(K, N) = E(K, 1) * N * sum_{j=1}^{N} (1/j)
         ~ E(K, 1) * N * (ln(N) + 0.5772)
```

- `E(K, 1)` ~ 2^K (from Markov chain analysis; for K=8, E(K,1) = 301.2)
- For K=8, N=8: E = 6549 (observed: 6179)

The factor `sum_{j=1}^{N} (1/j)` arises because finding the j-th block wastes fraction K(j-1)/(KN) of mutations on already-completed blocks.

### 11.4 Idealized Genetic Algorithm (IGA)

```
IGA:
  1. On each step, choose new string completely at random
  2. First time a string contains a desired schema, sequester it
  3. When string with new desired schema found, instantly cross over
     with sequestered string to combine all discovered schemas
```

**Expected time for IGA (N blocks of K ones):**
```
E_N ~ 2^K * sum_{n=1}^{N} (-1)^(n+1) * C(N,n) * (1/n)
    = 2^K * [ln(2)] * ... (simplification)
```

For K=8, N=8: E_N ~ 696 (observed: 696, SE: 19.7)

**Speedup: IGA ~ O(2^K * ln N) vs. RMHC ~ O(2^K * N * ln N) -- factor of N faster.**

### 11.5 Hitchhiking Problem

Once instances of a highly fit schema spread in the population, zeros at other positions "hitchhike" along, preventing independent sampling in other schema partitions. This is the primary reason the GA underperformed RMHC on R1.

**Conditions for GA to approximate IGA:**
1. Population large enough for independent sampling
2. Selection slow enough to prevent hitchhiking
3. Mutation rate high enough to prevent fixation
4. Crossover rate such that combination time << schema discovery time
5. String long enough for factor-N speedup to be significant

---

## 12. NK Landscapes

Not explicitly covered in this book. The book references fitness landscapes and Walsh transforms but does not present the NK landscape model.

---

## 13. Key Parameters -- Recommended Ranges

### 13.1 De Jong (1975) -- Systematic Study

Optimized for on-line and off-line performance on test suite:

| Parameter | Recommended Value |
|-----------|------------------|
| Population size | 50-100 |
| Crossover rate (single-point) | ~0.6 |
| Mutation rate (per bit) | 0.001 |

### 13.2 Grefenstette (1986) -- Meta-GA Optimization

Optimized for on-line performance via meta-level GA:

| Parameter | Value |
|-----------|-------|
| Population size | 30 |
| Crossover rate | 0.95 |
| Mutation rate | 0.01 |
| Generation gap | 1.0 |
| Selection | Elitist |

### 13.3 Schaffer, Caruana, Eshelman, and Das (1989) -- Systematic Testing

Over 1 year of CPU time; optimized for on-line performance:

| Parameter | Range |
|-----------|-------|
| Population size | 20-30 |
| Crossover rate | 0.75-0.95 |
| Mutation rate | 0.005-0.01 |

### 13.4 Typical Ranges from Literature

| Parameter | Common Range | Notes |
|-----------|-------------|-------|
| Population size | 20-1000 | Small for on-line; larger for off-line |
| Crossover rate | 0.6-0.95 | Two-point or uniform common in modern usage |
| Mutation rate | 0.001-0.01 per bit | Higher for small populations |
| String length | 50-1000 | Problem dependent |
| Generations | 50-500+ | Until convergence |
| Elitism | 1-5 individuals | Widely recommended |

### 13.5 Parameter Interactions

- Parameters interact nonlinearly; cannot be optimized independently
- Optimal settings change over course of a run
- Strong selection requires higher mutation to maintain diversity
- Larger populations can tolerate lower mutation rates
- Small populations benefit from higher crossover and mutation rates

### 13.6 Self-Adaptation of Operator Rates (Davis 1989, 1991)

```
ADAPTIVE-OPERATOR-RATES:
  1. Each operator starts with equal fitness
  2. At each step, choose operator probabilistically by operator fitness
  3. Apply operator to create new individual
  4. Track which operator created each individual
  5. When individual achieves new best fitness:
     - Credit the operator that created it
     - Credit operators that created its ancestors (up to specified depth)
  6. Periodically update operator fitnesses based on accumulated credits
```

---

## 14. All Tables and Data

### 14.1 Hill Climbing Algorithms (for comparison with GA)

**Steepest-Ascent Hill Climbing (SAHC):**
```
1. Choose random string (current-hilltop)
2. Systematically flip each bit left-to-right, record fitnesses
3. If any one-bit mutant increases fitness, set current-hilltop to best mutant
4. If no increase, save hilltop and go to step 1
5. Return highest hilltop found within evaluation budget
```

**Next-Ascent Hill Climbing (NAHC):**
```
1. Choose random string (current-hilltop)
2. For i = 1 to l: flip bit i; if fitness increases, keep new string immediately
   and restart scanning from position i+1
3. If no increase found in full pass, save hilltop and restart
4. Return highest hilltop found within evaluation budget
```

**Random-Mutation Hill Climbing (RMHC):**
```
1. Choose random string (best-evaluated)
2. Flip random bit; if fitness >= current, keep new string
3. Repeat step 2 until optimum found or budget exhausted
4. Return best-evaluated
```

### 14.2 Cellular Automata Performance Comparison

Performance of r=3 CA rules on density classification task (fraction correct on unbiased distribution of 10,000 ICs):

| CA Rule | Symbol | N=149 | N=599 | N=999 |
|---------|--------|-------|-------|-------|
| Majority rule | phi_maj | 0.000 | 0.000 | 0.000 |
| Expand 1-blocks (GA) | phi_a | 0.652 | 0.515 | 0.503 |
| Particle-based (GA) | phi_b | 0.697 | 0.580 | 0.522 |
| Particle-based (GA) | phi_c | 0.742 | 0.718 | 0.701 |
| Particle-based (GA) | phi_d | 0.769 | 0.725 | 0.714 |
| GKL (hand-designed) | phi_GKL | 0.816 | 0.766 | 0.757 |

Standard deviation of p_149 ~ 0.004 when calculated 100 times for same rule.

### 14.3 Messy GA -- Deceptive Function Scores

Three-bit segment scoring for 30-bit deceptive problem:

| Segment | Score |
|---------|-------|
| 000 | 28 (local optimum) |
| 001 | 26 |
| 010 | 22 |
| 011 | 0 |
| 100 | 14 |
| 101 | 0 |
| 110 | 0 |
| 111 | 30 (global optimum) |

Fitness of 30-bit string = sum of 10 segment scores.

### 14.4 Sorting Network Results (Hillis)

| Method | Best Comparisons Found |
|--------|----------------------|
| Bose-Nelson (1962) | 65 |
| Batcher / Floyd-Knuth (1964) | 63 |
| Shapiro (1969) | 62 |
| Green (1969) | 60 |
| GA without coevolution | 65 |
| GA with host-parasite coevolution | 61 |

GA parameters: Population 512 to ~1,000,000; ~5000 generations; pm=0.001.

### 14.5 Prisoner's Dilemma Payoff Matrix

|  | B cooperates | B defects |
|--|-------------|-----------|
| A cooperates | (3, 3) | (0, 5) |
| A defects | (5, 0) | (1, 1) |

Axelrod's GA: Population 20, 70-letter strings (64 strategy + 6 initial), 50 generations, 40 runs.

### 14.6 GP Block Stacking Results

| Generation | Best Program | Fitness (out of 166) |
|------------|-------------|---------------------|
| 0 | (EQ (MT CS) NN) | 0 |
| 0 | (MS TB) | 1 |
| 0 | (EQ (MS NN) (EQ (MS NN) (MS NN))) | 4 |
| 5 | (DU (MS NN) (NOT NN)) | 10 |
| 10 | (EQ (DU (MT CS) (NOT CS)) (DU (MS NN) (NOT NN))) | 166 (perfect) |

Population: 300, 10% copy, 90% crossover.

### 14.7 When to Use a GA

**Good conditions:**
- Large search space
- Not smooth/unimodal
- Space not well understood
- Noisy fitness function
- Satisficing (good enough) rather than optimizing
- Parallel evaluation possible

**Poor conditions:**
- Small search space (exhaustive search better)
- Smooth/unimodal (gradient methods better)
- Well-understood space (domain-specific heuristics better)
- Global optimum required (hybrid methods better)

### 14.8 Cumulant Predictions (Prugel-Bennett and Shapiro)

For 1D spin glass with P=50, N+1=64, beta=0.05:
- Predictions for k1 (mean energy) and k2 (variance) matched observations extremely closely over 300 generations averaged over 500 runs
- Demonstrates feasibility of statistical-mechanics approach to GA prediction

---

## Quick Reference: Algorithm Selection Guide

| Problem Type | Recommended Approach |
|-------------|---------------------|
| Binary optimization | Standard SGA with sigma scaling or rank selection |
| Continuous parameter optimization | Real-valued encoding with Gaussian mutation |
| Combinatorial ordering | Permutation encoding with specialized crossover (PMX, OX) |
| Program synthesis | GP with tree encoding |
| Neural network weights | Real-valued GA (Montana-Davis style) |
| Neural network topology | Grammatical encoding (Kitano style) |
| Multimodal optimization | GA with fitness sharing or spatial structure |
| Noisy fitness | Rank selection with elitism; (mu+lambda) strategy |
| Real-time/online tasks | Boltzmann selection with adaptive temperature |
| Deceptive landscapes | Messy GA or larger populations with higher mutation |
