# Introduction to Genetic Algorithms
## Comprehensive Algorithmic & Mathematical Summary
### Sivanandam & Deepa (Springer, 2008) -- ISBN 978-3-540-73189-4

---

# 1. GA Algorithm -- Complete Pseudocode

## 1.1 Simple Genetic Algorithm (SGA) -- Core Loop

```
ALGORITHM: Simple_Genetic_Algorithm
INPUT:  fitness_function, population_size, chromosome_length,
        crossover_rate (Pc), mutation_rate (Pm), max_generations

1.  t = 0
2.  INITIALIZE population P(t) of N individuals randomly
3.  EVALUATE fitness of each individual in P(t)
4.  WHILE termination_condition NOT met DO:
5.      t = t + 1
6.      SELECT parents from P(t-1) using selection_method
7.      APPLY CROSSOVER to selected parents with probability Pc
8.          --> produce offspring
9.      APPLY MUTATION to offspring with probability Pm
10.     EVALUATE fitness of offspring
11.     REPLACE P(t-1) with new population P(t) using replacement_strategy
12. END WHILE
13. RETURN best individual found

OUTPUT: best solution (highest fitness individual)
```

## 1.2 Evolutionary Algorithm Flowchart (Fig. 1.7)

```
START
  |
  v
Initialize Population (random)
  |
  v
Evaluate Fitness
  |
  v
[Termination met?] --YES--> STOP
  |
  NO
  v
Apply Selection
  |
  v
Randomly Vary Individuals (crossover + mutation)
  |
  v
(loop back to Evaluate Fitness)
```

## 1.3 Key GA Parameters

| Parameter               | Typical Range       | Notes                                    |
|--------------------------|---------------------|------------------------------------------|
| Population size (N)      | 20--200             | Problem-dependent                        |
| Crossover rate (Pc)      | 0.6--0.9            | Probability of applying crossover        |
| Mutation rate (Pm)       | 0.001--0.01         | Probability per bit (binary encoding)    |
| Max generations          | 100--10000          | Upper bound on iterations                |
| Chromosome length (L)    | Problem-dependent   | Number of genes per individual           |
| Elitism count            | 1--5                | Top individuals copied unchanged         |

---

# 2. Encoding Schemes (Chapter 3, Sec. 3.9)

## 2.1 Binary Encoding
Each chromosome is a fixed-length bit string of 0s and 1s.

```
Chromosome A: 1 0 1 1 0 0 1 0
Chromosome B: 0 1 1 0 1 1 0 1
```

- Most common and historically standard representation.
- Each bit position represents a feature or parameter.
- Decoding: for a variable x in [a, b] encoded with L bits:

```
x = a + (decimal_value_of_bitstring / (2^L - 1)) * (b - a)
```

- Precision: (b - a) / (2^L - 1)

## 2.2 Octal Encoding
Chromosomes encoded using digits 0--7.

```
Chromosome A: 3 5 7 1 2 6
Chromosome B: 4 0 2 7 5 3
```

## 2.3 Hexadecimal Encoding
Chromosomes encoded using digits 0--9 and A--F.

```
Chromosome A: 3 A F 2 C 9
Chromosome B: B 0 7 E 1 D
```

## 2.4 Permutation Encoding (Real Number Coding)
Each chromosome is a permutation of integers, used for ordering problems (e.g., TSP).

```
Chromosome A: [3, 1, 5, 2, 4, 7, 6, 8]
Chromosome B: [2, 5, 1, 8, 3, 6, 4, 7]
```

- Every gene is a unique integer; no duplicates allowed.
- Special crossover/mutation operators required (e.g., PMX, OX, CX).

## 2.5 Value Encoding
Genes can be any value: real numbers, characters, or complex objects.

```
Chromosome A: [1.23, 5.43, 0.21, 7.99]     (real-valued)
Chromosome B: [A, B, C, D, E, F]            (character-valued)
Chromosome C: [back, left, forward, right]   (instruction-valued)
```

- Used when complicated values needed (e.g., weights in neural networks).

## 2.6 Tree Encoding
Chromosomes are tree structures (used in genetic programming).

```
       +
      / \
     *   3
    / \
   x   2
```

- Leaves = terminals (variables, constants)
- Internal nodes = functions (+, -, *, /, sin, cos, etc.)
- Tree evaluation: leftmost depth-first traversal
- Closure principle: all functions and terminals return same type

---

# 3. Selection Methods (Chapter 3, Sec. 3.10.1)

## 3.1 Roulette Wheel Selection (Fitness-Proportionate)

```
ALGORITHM: Roulette_Wheel_Selection
INPUT: population P of N individuals with fitness f_i

1.  Compute total fitness: F = SUM_{i=1}^{N} f_i
2.  Compute selection probability for each individual:
        p_i = f_i / F
3.  Compute cumulative probability:
        c_i = SUM_{j=1}^{i} p_j
4.  FOR each selection needed:
        a. Generate random number r in [0, 1]
        b. Select individual i such that c_{i-1} < r <= c_i
5.  RETURN selected individuals
```

**Selection probability formula:**
```
p_i = f_i / SUM_{j=1}^{N} f_j
```

**Expected count of individual i:**
```
E_i = (f_i / f_avg) = (N * f_i) / SUM_{j=1}^{N} f_j
```

## 3.2 Rank Selection

```
ALGORITHM: Rank_Selection
INPUT: population P of N individuals with fitness f_i

1.  Sort population by fitness (worst = rank 1, best = rank N)
2.  Assign selection probability based on rank:
        p_i = rank_i / SUM_{j=1}^{N} rank_j
    where SUM of ranks = N*(N+1)/2
3.  Apply roulette wheel on rank-based probabilities
```

- Prevents premature convergence by reducing dominance of super-fit individuals.
- Selection pressure is constant regardless of fitness distribution.

**Linear ranking formula:**
```
p_i = (1/N) * [eta_min + (eta_max - eta_min) * (rank_i - 1) / (N - 1)]
```
where eta_min + eta_max = 2, typically eta_max in [1.1, 2.0].

## 3.3 Tournament Selection

```
ALGORITHM: Tournament_Selection
INPUT: population P, tournament size k

1.  FOR each selection needed:
        a. Randomly choose k individuals from P
        b. Select the one with highest fitness from the k
2.  RETURN selected individual

Variant (probabilistic):
    - With probability p, select best; with (1-p), select second best, etc.
    - Typical p = 0.75--1.0
```

- k = 2 is most common (binary tournament).
- Larger k --> higher selection pressure.
- Does NOT require global fitness comparison; easily parallelizable.

## 3.4 Boltzmann Selection

```
Selection probability using temperature T:

p_i = exp(f_i / T) / SUM_{j=1}^{N} exp(f_j / T)
```

- T starts high (low selection pressure, exploration).
- T decreases over generations (high selection pressure, exploitation).
- Analogy to simulated annealing.

## 3.5 Stochastic Universal Sampling (SUS)

```
ALGORITHM: Stochastic_Universal_Sampling
INPUT: population P of N individuals, number to select n

1.  Compute cumulative fitness as in roulette wheel
2.  Set pointer distance: d = F / n   (F = total fitness)
3.  Generate single random start: r in [0, d)
4.  Set pointers: ptr_i = r + (i-1) * d, for i = 1..n
5.  Walk through cumulative distribution:
        FOR each pointer, select the individual it lands on
```

- Single spin, equally spaced pointers; zero bias; minimum spread.

## 3.6 Elitism

```
STRATEGY: Elitism
1.  Copy the top E individuals directly to next generation (unchanged).
2.  Fill remaining (N - E) slots via normal selection + crossover + mutation.
```

- Guarantees best solution found is never lost.
- E = 1 or 2 is typical.

## 3.7 Steady-State Selection

```
STRATEGY: Steady_State
1.  Select a few (typically 2) parents.
2.  Produce offspring via crossover and mutation.
3.  Replace worst individuals in current population with offspring.
4.  (No generational replacement; population changes incrementally.)
```

---

# 4. Crossover Operators (Chapter 3, Sec. 3.10.2 & Chapter 4)

## 4.1 Single-Point Crossover

```
OPERATOR: Single_Point_Crossover
INPUT: Parent1, Parent2, crossover_point k

Parent1:  [a1 a2 a3 | a4 a5 a6]    crossover at position k=3
Parent2:  [b1 b2 b3 | b4 b5 b6]

Child1:   [a1 a2 a3 | b4 b5 b6]
Child2:   [b1 b2 b3 | a4 a5 a6]
```

- k is chosen uniformly at random from {1, 2, ..., L-1}.

## 4.2 Two-Point Crossover

```
OPERATOR: Two_Point_Crossover
INPUT: Parent1, Parent2, crossover_points k1 and k2 (k1 < k2)

Parent1:  [a1 | a2 a3 a4 | a5 a6]    points at k1=1, k2=4
Parent2:  [b1 | b2 b3 b4 | b5 b6]

Child1:   [a1 | b2 b3 b4 | a5 a6]
Child2:   [b1 | a2 a3 a4 | b5 b6]
```

- Segment between the two points is swapped.

## 4.3 Multi-Point Crossover (N-Point)

```
OPERATOR: N_Point_Crossover
INPUT: Parent1, Parent2, n crossover points

1.  Select n random crossover points
2.  Sort them in increasing order
3.  Alternate segments between parents:
    - Odd segments from Parent1 to Child1 (Parent2 to Child2)
    - Even segments swapped
```

## 4.4 Uniform Crossover

```
OPERATOR: Uniform_Crossover
INPUT: Parent1, Parent2, mixing ratio p (typically 0.5)

FOR each gene position i:
    Generate random number r in [0,1]
    IF r < p:
        Child1[i] = Parent1[i]
        Child2[i] = Parent2[i]
    ELSE:
        Child1[i] = Parent2[i]
        Child2[i] = Parent1[i]
```

- Each bit independently chosen from either parent.
- Mixing ratio p = 0.5 gives equal contribution.

## 4.5 Arithmetic Crossover (Real-Valued)

```
OPERATOR: Arithmetic_Crossover
INPUT: Parent1 = [x1, x2, ..., xn], Parent2 = [y1, y2, ..., yn]
       alpha in [0, 1] (typically 0.5 or random)

Child1[i] = alpha * Parent1[i] + (1 - alpha) * Parent2[i]
Child2[i] = (1 - alpha) * Parent1[i] + alpha * Parent2[i]
```

- Produces children within the convex hull of parents.

## 4.6 Partially Matched Crossover (PMX) -- for Permutation Encoding (Sec. 4.4.1)

```
OPERATOR: PMX
INPUT: Parent1, Parent2, two crossover points i, j

1.  Copy segment [i..j] from Parent1 to Child1
2.  For positions outside [i..j] in Child1:
    a. Take gene from Parent2 at that position
    b. If gene already exists in Child1[i..j]:
       - Find where it maps in the matched segment
       - Follow the mapping chain until an unused value is found
       - Place that value
3.  Repeat symmetrically for Child2

Example:
  Parent1: [1 2 | 3 4 5 | 6 7 8]   segment [3,4,5]
  Parent2: [3 7 | 5 1 6 | 8 2 4]   segment [5,1,6]

  Mapping: 3<->5, 4<->1, 5<->6

  Child1:  [6 7 | 3 4 5 | 8 2 1]
  Child2:  [3 2 | 5 1 6 | 4 7 8]  (after resolving conflicts)
```

## 4.7 Order Crossover (OX) -- for Permutation Encoding (Sec. 4.4.2)

```
OPERATOR: OX
INPUT: Parent1, Parent2, two crossover points i, j

1.  Copy segment [i..j] from Parent1 to Child1
2.  Starting from position (j+1) in Parent2, fill remaining
    positions of Child1 in order, skipping values already present
3.  Wrap around when reaching end of chromosome
4.  Repeat symmetrically for Child2

Example:
  Parent1: [1 2 3 | 4 5 6 | 7 8 9]
  Parent2: [5 4 6 | 9 2 1 | 7 8 3]

  Child1:  [2 1 3 | 4 5 6 | 9 7 8]  (fill from Parent2 order: 7,8,3,5,4,6,9,2,1)
```

## 4.8 Cycle Crossover (CX) -- for Permutation Encoding (Sec. 4.4.3)

```
OPERATOR: CX
INPUT: Parent1, Parent2

1.  Start at position 1 of Parent1
2.  Identify the cycle:
    a. Note value in Parent1[1]
    b. Find same value's position in Parent2
    c. Go to that position in Parent1, note value
    d. Repeat until cycle returns to starting value
3.  Copy cycle positions from Parent1 to Child1
4.  Copy all remaining positions from Parent2 to Child1
5.  Repeat symmetrically for Child2

Example:
  Parent1: [1 2 3 4 5 6 7 8]
  Parent2: [8 5 2 1 3 6 4 7]

  Cycle: 1->8->7->4->1 (positions 1,4,7,8)
  Child1: [1 5 2 4 3 6 7 8]
  Child2: [8 2 3 1 5 6 4 7]
```

## 4.9 Subtree Crossover (Genetic Programming, Sec. 6.3)

```
OPERATOR: Subtree_Crossover
INPUT: Tree1, Tree2

1.  Randomly select a node in Tree1 (crossover point 1)
2.  Randomly select a node in Tree2 (crossover point 2)
3.  Swap the subtrees rooted at these nodes

Result: Two new offspring trees
```

- Closure principle ensures legal offspring (all subtrees return same type).

## 4.10 Intermediate Recombination (Evolutionary Strategies)

```
Child[i] = (Parent1[i] + Parent2[i]) / 2    for each element i
```

- Element-by-element averaging of real-valued vectors.

---

# 5. Mutation Operators (Chapter 3, Sec. 3.10.3 & Chapter 4)

## 5.1 Bit-Flip Mutation (Binary Encoding)

```
OPERATOR: Bit_Flip_Mutation
INPUT: chromosome C, mutation rate Pm

FOR each bit position i in C:
    Generate random r in [0,1]
    IF r < Pm:
        C[i] = 1 - C[i]     (flip: 0->1 or 1->0)
```

## 5.2 Swap Mutation (Permutation Encoding)

```
OPERATOR: Swap_Mutation
INPUT: chromosome C

1.  Randomly select two positions i, j
2.  Swap C[i] and C[j]

Before: [1 2 3 4 5 6 7 8]    (swap positions 3 and 6)
After:  [1 2 6 4 5 3 7 8]
```

## 5.3 Inversion Mutation (Permutation Encoding, Sec. 4.4)

```
OPERATOR: Inversion_Mutation
INPUT: chromosome C

1.  Randomly select two positions i, j (i < j)
2.  Reverse the segment C[i..j]

Before: [1 2 3 4 5 6 7 8]    (invert positions 3 to 6)
After:  [1 2 6 5 4 3 7 8]
```

## 5.4 Insert Mutation (Permutation Encoding)

```
OPERATOR: Insert_Mutation
INPUT: chromosome C

1.  Randomly select a gene at position j
2.  Randomly select target position i
3.  Remove gene from j, insert at i, shift others
```

## 5.5 Scramble Mutation (Permutation Encoding)

```
OPERATOR: Scramble_Mutation
INPUT: chromosome C

1.  Randomly select two positions i, j (i < j)
2.  Randomly shuffle genes in segment C[i..j]
```

## 5.6 Gaussian Mutation (Real-Valued Encoding, Sec. 1.2.3)

```
OPERATOR: Gaussian_Mutation
INPUT: chromosome C (real-valued vector), sigma

FOR each gene i:
    C[i] = C[i] + N(0, sigma)

where N(0, sigma) is a random value from Gaussian distribution
with mean 0 and standard deviation sigma.
```

## 5.7 Creep Mutation (Real-Valued)

```
C[i] = C[i] + delta
where delta is a small random value (uniform or Gaussian with small sigma)
```

## 5.8 Segregation and Translocation (Sec. 4.6.1)

```
Segregation: split chromosome into sub-chromosomes at random points
Translocation: exchange sub-chromosomes between individuals
```

## 5.9 Duplication and Deletion (Sec. 4.6.2)

```
Duplication: copy a gene segment and insert elsewhere in chromosome
Deletion: remove a gene segment from chromosome
(used for variable-length representations)
```

---

# 6. Fitness Functions (Chapter 3, Sec. 3.5, 3.13, 3.16)

## 6.1 Fitness Evaluation

```
fitness(individual) = evaluation_function(decode(chromosome))
```

- **Maximization**: fitness = f(x) directly
- **Minimization**: fitness = 1 / (1 + f(x))  OR  fitness = C_max - f(x)
  where C_max is a large constant or the maximum f value in the population

## 6.2 Fitness Scaling (Sec. 3.16)

### 6.2.1 Linear Scaling (Sec. 3.16.1)

```
f_scaled = a * f_raw + b
```

**Conditions to determine a and b:**
```
f_scaled_avg = f_raw_avg              (preserves average)
f_scaled_max = C_mult * f_raw_avg     (C_mult typically 1.2 to 2.0)
```

**Solving for a, b:**
```
a = (C_mult - 1) * f_avg / (f_max - f_avg)
b = f_avg * (f_max - C_mult * f_avg) / (f_max - f_avg)
```

- If negative fitness values result, truncate to zero and re-scale.

### 6.2.2 Sigma Truncation (Sec. 3.16.2)

```
f_scaled = f_raw - (f_avg - c * sigma)
```

where:
- f_avg = population mean fitness
- sigma = population standard deviation of fitness
- c = small integer, typically 1 to 3
- If f_scaled < 0, set f_scaled = 0

### 6.2.3 Power Law Scaling (Sec. 3.16.3)

```
f_scaled = (f_raw)^k
```

where k is a problem-specific constant close to 1 (e.g., k = 1.005).

---

# 7. Convergence Criteria / Termination Conditions (Chapter 3, Sec. 3.11)

## 7.1 Termination Methods

| Method             | Condition                                                        |
|--------------------|------------------------------------------------------------------|
| Best Individual    | Fitness of best individual does not change for G generations     |
| Worst Individual   | Fitness of worst individual does not change for G generations    |
| Sum of Fitness     | Total fitness of population does not change for G generations    |
| Median Fitness     | Median fitness of population does not change for G generations   |
| Max Generations    | Predetermined number of generations reached                      |
| Fitness Threshold  | Best fitness meets or exceeds target value                       |
| Population Diversity | Population diversity falls below threshold (convergence)       |

## 7.2 Convergence Detection

```
IF |best_fitness(t) - best_fitness(t-G)| < epsilon:
    TERMINATE  (G = patience window, epsilon = tolerance)

IF |avg_fitness(t) - avg_fitness(t-G)| < epsilon:
    TERMINATE
```

---

# 8. Replacement Strategies (Chapter 3, Sec. 3.10.4)

## 8.1 Generational Replacement
```
Entire population P(t) is replaced by offspring to form P(t+1).
Population size remains constant at N.
```

## 8.2 Steady-State Replacement
```
Only 1 or 2 individuals replaced per generation.
Worst individuals in population replaced by new offspring.
```

## 8.3 Elitist Replacement
```
Best E individuals from P(t) are carried over to P(t+1).
Remaining N-E slots filled by offspring.
```

---

# 9. Schema Theorem (Chapter 3, Sec. 3.12.4)

## 9.1 Definitions

| Term               | Symbol          | Definition                                            |
|--------------------|-----------------|-------------------------------------------------------|
| Schema             | H               | Template with {0, 1, *}, where * = wildcard           |
| Order              | o(H)            | Number of defined (non-*) positions                   |
| Defining length    | delta(H)        | Distance between first and last defined positions     |
| Schema fitness     | f(H)            | Average fitness of all individuals matching schema H  |

## 9.2 Schema Theorem (Holland's Fundamental Theorem)

```
m(H, t+1) >= m(H, t) * [f(H) / f_avg] * [1 - Pc * delta(H)/(L-1)] * [1 - Pm]^o(H)
```

where:
- m(H, t) = number of instances of schema H at generation t
- f(H) = average fitness of strings matching H
- f_avg = average fitness of entire population
- Pc = crossover probability
- Pm = mutation probability per bit
- delta(H) = defining length of H
- o(H) = order of H
- L = chromosome length

**Interpretation:**
- Short, low-order, above-average schemata receive exponentially increasing trials.
- These are called "building blocks."

## 9.3 Building Block Hypothesis (Sec. 3.12.1)

```
GAs work by combining short, low-order, high-fitness schemata
(building blocks) to form complete high-fitness solutions.
```

## 9.4 Implicit Parallelism (Sec. 3.12.6)

```
A population of N individuals of length L simultaneously processes
approximately O(N^3) schemata per generation.
```

## 9.5 No Free Lunch Theorem (Sec. 3.12.7)

```
For any search algorithm A, averaged over ALL possible objective
functions, A performs identically to any other algorithm B
(including random search).

Implication: GAs are not universally superior; they excel when their
inductive bias matches the problem structure.
```

---

# 10. Advanced Operators and Techniques (Chapter 4)

## 10.1 Diploidy and Dominance (Sec. 4.2)

```
Standard GA: haploid (single chromosome per individual)
Diploid GA: two chromosomes per individual (homologous pairs)

Phenotype determined by dominance map:
  - Dominant allele expressed when heterozygous
  - Recessive allele expressed only when homozygous
  - Enables "genetic memory" for changing environments
```

## 10.2 Multiploid (Sec. 4.3)

```
Extension beyond diploid: individuals carry multiple (>2)
copies of each chromosome. More complex dominance interactions.
```

## 10.3 Niche and Speciation (Sec. 4.5)

### 10.3.1 Fitness Sharing (Multimodal Problems, Sec. 4.5.1)

```
f_shared(i) = f(i) / niche_count(i)

niche_count(i) = SUM_{j=1}^{N} sh(d(i,j))

sh(d) = { 1 - (d / sigma_share)^alpha,   if d < sigma_share
         { 0,                               otherwise

where:
  d(i,j) = distance between individuals i and j
  sigma_share = niche radius (sharing distance)
  alpha = shape parameter (typically alpha = 1)
```

### 10.3.2 Crowding (Unimodal/Multimodal, Sec. 4.5.2)

```
ALGORITHM: Deterministic_Crowding
1.  Shuffle population, pair adjacent individuals as parents
2.  Apply crossover and mutation to produce offspring
3.  Each offspring competes with its MOST SIMILAR parent
4.  Winner survives to next generation
```

### 10.3.3 Restricted Mating (Sec. 4.5.3)

```
Only individuals within distance threshold sigma_mate may mate.
Prevents crossover between individuals from different niches.
```

---

# 11. GA Classifications (Chapter 5)

## 11.1 Simple GA (SGA) -- Sec. 5.2
```
Standard generational GA:
- Binary encoding
- Roulette wheel selection
- Single-point crossover
- Bit-flip mutation
- Generational replacement
```

## 11.2 Parallel and Distributed GA (PGA/DGA) -- Sec. 5.3

### 11.2.1 Master-Slave Parallelization (Sec. 5.3.1)
```
- Single population; master manages selection & crossover
- Fitness evaluations distributed to slave processors
- Speedup ~ linear with number of processors
- Functionally equivalent to sequential SGA
```

### 11.2.2 Fine-Grained (Cellular) GA (Sec. 5.3.2)
```
- One individual per processor on 2D grid
- Selection and crossover only with neighboring individuals
- Overlapping neighborhoods create diffusion of solutions
- Encourages diversity; niches form naturally
```

### 11.2.3 Coarse-Grained (Island Model) GA (Sec. 5.3.3)
```
ALGORITHM: Island_Model_GA
1.  Divide population into D subpopulations (demes)
2.  Run independent GA on each deme
3.  Every M generations, MIGRATE best individuals between demes
4.  Migration topology: ring, fully connected, random, etc.

Parameters:
  - D = number of demes (islands)
  - M = migration interval
  - migration_rate = fraction of individuals migrated
  - topology = connectivity pattern
```

### 11.2.4 Hierarchical Parallel GA (Sec. 5.3.4)
```
Combines coarse-grained (inter-deme migration) with
fine-grained (intra-deme cellular structure).
Two-level parallelism.
```

## 11.3 Hybrid GA (HGA) -- Sec. 5.4

```
ALGORITHM: Hybrid_GA
1.  Run standard GA operators (selection, crossover, mutation)
2.  Apply LOCAL SEARCH (hill climbing) to each offspring
3.  Evaluate improved offspring
4.  Replace population

Variants:
  - Lamarckian: improved solution AND its genotype replace original
  - Baldwinian: only improved fitness assigned; genotype unchanged
```

### RemoveSharp Algorithm (Sec. 5.4.3)
```
Local search heuristic to remove "sharp" infeasible features
from solutions (constraint-specific).
```

### LocalOpt Algorithm (Sec. 5.4.4)
```
Apply deterministic local optimization (e.g., gradient descent,
2-opt, 3-opt) to each GA-produced solution.
```

## 11.4 Adaptive GA (AGA) -- Sec. 5.5

```
Crossover and mutation rates adapt during the run:

Pc = { k1 * (f_max - f') / (f_max - f_avg),   if f' >= f_avg
     { k3,                                       if f' < f_avg

Pm = { k2 * (f_max - f) / (f_max - f_avg),     if f >= f_avg
     { k4,                                       if f < f_avg

where:
  f_max = maximum fitness in population
  f_avg = average fitness in population
  f' = larger fitness of the two parents (for Pc)
  f = fitness of individual to be mutated (for Pm)
  k1, k2 in (0, 1)  and  k3, k4 in (0, 1)
```

**Behavior:**
- High-fitness individuals: low Pc, low Pm (preserved)
- Low-fitness individuals: high Pc, high Pm (disrupted more)

## 11.5 Fast Messy GA (fmGA) -- Sec. 5.6

```
ALGORITHM: Fast_Messy_GA
1.  PRIMORDIAL PHASE: Generate partial solutions (building blocks)
    - Variable-length chromosomes with (gene_locus, allele) pairs
2.  COMPETITIVE TEMPLATE: Use best from SGA as template
3.  JUXTAPOSITIONAL PHASE: Combine partial solutions via
    cut-and-splice operators
4.  Under-specified genes filled from competitive template
5.  Over-specified genes resolved by leftmost wins
```

## 11.6 Independent Sampling GA (ISGA) -- Sec. 5.7

```
ALGORITHM: ISGA
Phase 1 - Independent Sampling:
    Generate random solutions; identify good building blocks
Phase 2 - Breeding:
    Combine building blocks using crossover
    Focused exploitation of discovered structure
```

---

# 12. Genetic Programming (Chapter 6)

## 12.1 GP Representation (Sec. 6.3)

```
Individual = parse tree (S-expression)
Terminal set T = {variables, constants}      (leaves)
Function set F = {+, -, *, /, sin, cos, IF, ...}  (internal nodes)

Requirements:
  - Closure: every function returns same type
  - Sufficiency: F and T can express solution
```

## 12.2 GP Algorithm (Sec. 6.5)

```
ALGORITHM: Genetic_Programming
1.  Define terminal set T and function set F
2.  Define fitness measure
3.  Set control parameters (population size, max depth, etc.)
4.  Set termination criterion
5.  Generate initial population of random trees:
    a. FULL method: all branches reach max depth
    b. GROW method: branches have variable length
    c. RAMPED HALF-AND-HALF: 50% full, 50% grow at each depth level
6.  WHILE termination NOT met:
    a. Evaluate fitness of each tree
    b. Select parents (tournament or fitness-proportionate)
    c. Apply genetic operators:
       - Subtree crossover (dominant, ~90%)
       - Subtree mutation (~1%)
       - Reproduction (copy, ~9%)
       - Point mutation
       - Permutation
       - Encapsulation
    d. Form new generation
7.  RETURN best program tree
```

## 12.3 GP Initialization Methods

| Method             | Description                                              |
|--------------------|----------------------------------------------------------|
| Full               | All branches have exactly max_depth                      |
| Grow               | Branches terminate randomly between 1 and max_depth      |
| Ramped Half-and-Half | Population split across depths 2..max_depth; 50/50 full/grow at each depth |

## 12.4 Bloat Control

```
Methods to prevent uncontrolled tree growth:
- Maximum depth limit (hard constraint)
- Parsimony pressure: fitness = raw_fitness - c * tree_size
- Tarpeian method: randomly remove oversized individuals
```

---

# 13. Optimization Problem Formulations (Chapter 7)

## 13.1 Multi-Objective Optimization (Sec. 4.8, 7.2)

```
Minimize/Maximize:  [f_1(x), f_2(x), ..., f_k(x)]
Subject to:         g_j(x) <= 0,  j = 1, ..., m
                    h_l(x) = 0,   l = 1, ..., p

Pareto Dominance:
  Solution A dominates B if:
    - A is no worse than B in all objectives, AND
    - A is strictly better than B in at least one objective

Pareto Front: Set of all non-dominated solutions
```

## 13.2 Weighted Sum Method

```
F(x) = SUM_{i=1}^{k} w_i * f_i(x)

where: SUM w_i = 1, w_i > 0
```

## 13.3 Constraint Handling

```
Penalty Function Method:
  fitness = f(x) - P * violation(x)

where:
  P = penalty coefficient
  violation(x) = SUM of constraint violation magnitudes

Death Penalty:
  If x violates any constraint, fitness = 0 (or very low)

Repair:
  Map infeasible solution to nearest feasible solution
```

## 13.4 Combinatorial Optimization (Sec. 7.4)

### Traveling Salesman Problem (TSP) -- Sec. 3.17.2, 9.2
```
Encoding: permutation of cities [c1, c2, ..., cn]
Fitness: 1 / total_tour_distance
Operators: PMX, OX, CX crossover; swap, inversion mutation
```

### Job Shop Scheduling (JSSP) -- Sec. 7.5
```
Encoding: permutation-based (operation sequence)
Fitness: minimize makespan (total completion time)
Decoding: map chromosome to feasible schedule via dispatching rules
```

---

# 14. Comparison: GA vs. Other Methods (Chapter 2, Sec. 2.4, 2.6)

## 14.1 Methods Comparison Table

| Feature                | GA            | Gradient Descent | Simulated Annealing | Random Search |
|------------------------|---------------|------------------|---------------------|---------------|
| Population-based       | Yes           | No               | No                  | No            |
| Gradient required      | No            | Yes              | No                  | No            |
| Escapes local optima   | Yes           | No               | Yes                 | Yes (slow)    |
| Search type            | Parallel      | Serial           | Serial              | Serial        |
| Representation         | Encoded       | Direct           | Direct              | Direct        |
| Stochastic             | Yes           | No               | Yes                 | Yes           |
| Uses fitness function  | Yes           | Uses gradient    | Uses objective      | Uses objective|
| Handles discrete vars  | Yes           | No               | Yes                 | Yes           |

## 14.2 GA vs. Traditional Optimization

```
GA properties distinguishing it from classical methods:
1. Works with encoded parameters, not parameters directly
2. Searches from a POPULATION, not a single point
3. Uses FITNESS (objective function), not derivatives
4. Uses PROBABILISTIC transition rules, not deterministic
```

---

# 15. Simulated Annealing (Sec. 2.4.4)

```
ALGORITHM: Simulated_Annealing
INPUT: initial_solution S, initial_temperature T, cooling_rate alpha

1.  S_best = S
2.  WHILE T > T_min:
3.      S_new = NEIGHBOR(S)
4.      delta_E = f(S_new) - f(S)
5.      IF delta_E > 0:   (for maximization)
6.          S = S_new
7.      ELSE:
8.          Accept S_new with probability exp(delta_E / T)
9.      IF f(S) > f(S_best):
10.         S_best = S
11.     T = alpha * T     (alpha typically 0.9 to 0.99)
12. RETURN S_best
```

---

# 16. Particle Swarm Optimization (Chapter 11, Sec. 11.2)

## 16.1 PSO Algorithm

```
ALGORITHM: Particle_Swarm_Optimization
INPUT: num_particles, num_dimensions, max_iterations

1.  FOR each particle i:
        Initialize position x_i randomly in search space
        Initialize velocity v_i randomly
        personal_best_i = x_i
2.  global_best = argmin f(personal_best_i) over all i

3.  FOR t = 1 to max_iterations:
    FOR each particle i:
        FOR each dimension d:
            r1 = random(0, 1)
            r2 = random(0, 1)
            v_i[d] = w * v_i[d]
                   + c1 * r1 * (personal_best_i[d] - x_i[d])
                   + c2 * r2 * (global_best[d] - x_i[d])
            x_i[d] = x_i[d] + v_i[d]

        IF f(x_i) < f(personal_best_i):
            personal_best_i = x_i
        IF f(x_i) < f(global_best):
            global_best = x_i

4.  RETURN global_best
```

## 16.2 PSO Parameters

| Parameter          | Symbol | Typical Value | Role                           |
|--------------------|--------|---------------|--------------------------------|
| Inertia weight     | w      | 0.4--0.9      | Momentum / exploration-exploitation balance |
| Cognitive coeff.   | c1     | 2.0           | Pull toward personal best      |
| Social coeff.      | c2     | 2.0           | Pull toward global best        |
| Num particles      | N      | 20--50        | Swarm size                     |
| Max velocity       | v_max  | Problem-dep.  | Prevents explosion             |

## 16.3 PSO Velocity Update Equation

```
v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
```

## 16.4 PSO vs GA Comparison (Sec. 11.2.4)

| Aspect              | PSO                          | GA                            |
|---------------------|------------------------------|-------------------------------|
| Encoding            | Real-valued positions        | Binary or real-valued         |
| Operators           | Velocity update              | Crossover + mutation          |
| Memory              | Personal + global best       | Population only (unless elitism) |
| Selection           | Implicit (follow best)       | Explicit (roulette, tournament) |
| Convergence         | Often faster initially       | More robust diversity         |
| Tuning parameters   | w, c1, c2                    | Pc, Pm, selection scheme      |

---

# 17. Ant Colony Optimization (Chapter 11, Sec. 11.3)

## 17.1 ACO Algorithm (for TSP)

```
ALGORITHM: Ant_Colony_Optimization
INPUT: distance_matrix, num_ants, alpha, beta, rho, Q, max_iterations

1.  Initialize pheromone trails tau_ij = tau_0 for all edges (i,j)
2.  FOR t = 1 to max_iterations:
    a. FOR each ant k = 1 to num_ants:
        Construct solution (tour) using probabilistic transition rule:

        p_ij^k = [tau_ij^alpha * eta_ij^beta] / SUM_{l in allowed} [tau_il^alpha * eta_il^beta]

        where:
          tau_ij = pheromone on edge (i,j)
          eta_ij = 1 / d_ij = heuristic desirability (inverse distance)
          alpha = pheromone influence exponent
          beta = heuristic influence exponent
          allowed = set of unvisited cities

    b. Update pheromone trails:
        tau_ij = (1 - rho) * tau_ij + SUM_{k=1}^{num_ants} delta_tau_ij^k

        delta_tau_ij^k = { Q / L_k,   if ant k used edge (i,j)
                         { 0,          otherwise

        where:
          rho = evaporation rate (0 < rho < 1)
          Q = pheromone deposit constant
          L_k = tour length of ant k

3.  RETURN best tour found
```

## 17.2 ACO Parameters

| Parameter         | Symbol  | Typical Value | Role                              |
|-------------------|---------|---------------|-----------------------------------|
| Pheromone influence | alpha | 1.0           | Weight of trail information       |
| Heuristic influence | beta  | 2.0--5.0      | Weight of greedy heuristic        |
| Evaporation rate  | rho     | 0.1--0.5      | Rate of pheromone decay           |
| Pheromone constant | Q      | Problem-dep.  | Amount of pheromone deposited     |
| Number of ants    | m       | N (num cities)| Swarm size                        |
| Initial pheromone | tau_0   | Small positive| Starting trail intensity          |

## 17.3 ACO Variants (Sec. 11.3.4)

| Variant            | Key Modification                                           |
|--------------------|------------------------------------------------------------|
| Ant System (AS)    | Original; all ants deposit pheromone                       |
| Ant Colony System  | Only best ant deposits; local pheromone update             |
| Max-Min AS (MMAS) | Pheromone bounded in [tau_min, tau_max]; best-only deposit  |
| Rank-Based AS     | Pheromone weighted by ant rank                              |

---

# 18. Key Figures and Tables Referenced

## 18.1 Figure Descriptions

| Figure | Description                                                          |
|--------|----------------------------------------------------------------------|
| 1.1    | Bit-string crossover: parents a & b produce offspring c & d          |
| 1.2    | Bit-flipping mutation: parent a produces offspring b                 |
| 1.3    | Subtree crossover of GP trees: parents a & b to offspring c & d      |
| 1.4    | Gaussian mutation of real-valued vector: parent a to offspring b      |
| 1.5    | Intermediate recombination: averaging parents a & b to form child c  |
| 1.6    | Fitness landscape for 2 phenotypic traits with isoclines             |
| 1.7    | Flowchart of evolutionary algorithm (Init -> Eval -> Select -> Vary) |
| 1.8    | Cart-pole balancing system (two poles, angles theta1, theta2)        |

## 18.2 Key Tables from Book (13 Total per Title Page)

| Table/Section  | Content                                                           |
|----------------|-------------------------------------------------------------------|
| Table 2.x      | Comparison of GA with conventional optimization techniques        |
| Table 3.x      | Roulette wheel selection probabilities example                    |
| Table 3.x      | Schema theorem components and example calculations                |
| Table 3.x      | Fitness scaling parameters and effects                            |
| Table 5.x      | PGA migration parameters and topologies                           |
| Table 8.x      | MATLAB GA Toolbox function reference                              |
| Table 11.x     | PSO vs GA comparative features                                   |

---

# 19. Key Equations Summary

| #  | Equation                                                                       | Context                    |
|----|--------------------------------------------------------------------------------|----------------------------|
| 1  | p_i = f_i / SUM f_j                                                           | Roulette wheel probability |
| 2  | E_i = N * f_i / SUM f_j                                                       | Expected selection count   |
| 3  | m(H,t+1) >= m(H,t) * [f(H)/f_avg] * [1 - Pc*delta(H)/(L-1)] * [1-Pm]^o(H)  | Schema Theorem             |
| 4  | f_scaled = a * f_raw + b                                                       | Linear fitness scaling     |
| 5  | f_scaled = f_raw - (f_avg - c*sigma)                                           | Sigma truncation           |
| 6  | f_scaled = f_raw^k                                                             | Power law scaling          |
| 7  | sh(d) = 1 - (d/sigma_share)^alpha                                              | Fitness sharing function   |
| 8  | Pc = k1*(f_max - f')/(f_max - f_avg)                                           | Adaptive crossover rate    |
| 9  | Pm = k2*(f_max - f)/(f_max - f_avg)                                            | Adaptive mutation rate     |
| 10 | v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)                               | PSO velocity update        |
| 11 | p_ij = [tau_ij^a * eta_ij^b] / SUM [tau_il^a * eta_il^b]                      | ACO transition probability |
| 12 | tau_ij = (1-rho)*tau_ij + SUM delta_tau_ij^k                                   | ACO pheromone update       |
| 13 | x = a + (decimal/(2^L - 1)) * (b - a)                                         | Binary decoding formula    |
| 14 | Child = alpha*P1 + (1-alpha)*P2                                                | Arithmetic crossover       |
| 15 | C[i] = C[i] + N(0, sigma)                                                     | Gaussian mutation          |

---

# 20. GA Design Checklist

```
Step 1: REPRESENTATION
  - Choose encoding (binary, real, permutation, tree)
  - Determine chromosome length and gene structure

Step 2: FITNESS FUNCTION
  - Map decoded chromosome to objective function
  - Apply scaling if needed (linear, sigma, power law)
  - Handle constraints (penalty, repair, death)

Step 3: INITIALIZATION
  - Random generation of N individuals
  - Optionally seed with heuristic solutions

Step 4: SELECTION
  - Choose method (roulette, tournament, rank, SUS, Boltzmann)
  - Set selection pressure parameters

Step 5: CROSSOVER
  - Choose operator matching representation
  - Set crossover rate Pc (0.6--0.9)

Step 6: MUTATION
  - Choose operator matching representation
  - Set mutation rate Pm (0.001--0.01 for binary; higher for real)

Step 7: REPLACEMENT
  - Generational, steady-state, or elitist
  - Set elitism count if applicable

Step 8: TERMINATION
  - Max generations AND/OR fitness threshold AND/OR convergence detection

Step 9: PARAMETER TUNING
  - Population size, Pc, Pm, selection pressure
  - Consider adaptive parameters (AGA)
```

---

*Source: Sivanandam, S.N. & Deepa, S.N. "Introduction to Genetic Algorithms." Springer, 2008. 442 pages, 193 figures, 13 tables.*
