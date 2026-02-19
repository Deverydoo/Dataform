# Temporal Graph Networks (TGN) -- Curated Technical Extraction

Source: "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (Rossi et al., 2020)

---

## TGN Architecture Overview

### Problem Formulation

TGN operates on **continuous-time dynamic graphs (CTDG)** represented as a timed sequence of events.

A temporal (multi-)graph is modeled as:

```
G = {x(t_1), x(t_2), ...}
```

where events are ordered: `0 <= t_1 <= t_2 <= ...`

**Two event types:**

1. **Node-wise event** `v_i(t)`: creates node `i` (if new) or updates its features. `i` = node index, `v` = vector attribute.
2. **Interaction event** `e_ij(t)`: a directed temporal edge between nodes `i` and `j` (multigraph -- multiple edges allowed between a pair).

**Temporal notation:**
- `V(T) = {i : exists v_i(t) in G, t in T}` -- temporal vertex set
- `E(T) = {(i,j) : exists e_ij(t) in G, t in T}` -- temporal edge set
- `N_i(T) = {j : (i,j) in E(T)}` -- neighborhood of node `i` in time interval `T`
- `N_i^k(T)` -- k-hop neighborhood
- `G(t) = (V[0,t], E[0,t])` -- snapshot at time `t` with `n(t)` nodes

### Core Architecture

TGN is an **encoder-decoder** pair:
- **Encoder**: maps dynamic graph to node embeddings `Z(t) = (z_1(t), ..., z_{n(t)}(t))`
- **Decoder**: takes one or more node embeddings, makes task-specific prediction (e.g., edge prediction, node classification)

**Five core modules** (in data-flow order):
1. **Message Function** -- computes messages from events
2. **Message Aggregator** -- aggregates multiple messages for same node
3. **Memory Updater** -- updates node memory state from aggregated messages
4. **Embedding Module** -- computes node embeddings from memory + graph structure
5. **Decoder** -- makes predictions from embeddings

---

## Memory Module

### Node Memory

Each node `i` maintains a state vector `s_i(t)` representing its history in compressed format.

- **Initialization**: `s_i = 0` (zero vector) when node first encountered
- **Update trigger**: memory updates upon each event involving the node
- **Persistence**: memory persists and updates even after training is complete
- **Dimension**: 172 (in experiments)

### Message Function

For an **interaction event** `e_ij(t)` between source `i` and target `j`:

```
m_i(t) = msg_s(s_i(t-), s_j(t-), delta_t, e_ij(t))     -- source message
m_j(t) = msg_d(s_j(t-), s_i(t-), delta_t, e_ij(t))     -- destination message
```

**Equation (1)**

Where:
- `s_i(t-)` = memory of node `i` just before time `t` (from its previous event)
- `s_j(t-)` = memory of node `j` just before time `t`
- `delta_t` = time elapsed since last interaction
- `e_ij(t)` = edge features of the interaction
- `msg_s`, `msg_d` = learnable message functions (e.g., MLPs)

For a **node-wise event** `v_i(t)`:

```
m_i(t) = msg_n(s_i(t-), t, v_i(t))
```

**Equation (2)**

Where:
- `v_i(t)` = node feature vector associated with the event

For **edge deletion events** `(i, j, t', t)` where edge created at `t'` is deleted at `t`:

```
m_i(t) = msg_s'(s_i(t-), s_j(t-), delta_t, e_ij(t))
m_j(t) = msg_d'(s_j(t-), s_i(t-), delta_t, e_ij(t))
```

**Equation (12)**

**In all experiments**: the message function is **identity (id)**, i.e., simple concatenation of inputs.

### Message Aggregator

When multiple events involve the same node `i` in a single batch, producing messages `m_i(t_1), ..., m_i(t_b)`:

```
m_bar_i(t) = agg(m_i(t_1), ..., m_i(t_b))
```

**Equation (3)**

Where `agg` is an aggregation function. Two non-learnable implementations tested:
- **Most recent message (last)**: keep only the most recent message for a given node
- **Mean message (mean)**: average all messages for a given node

### Memory Updater

Memory is updated upon each event involving the node:

```
s_i(t) = mem(m_bar_i(t), s_i(t-))
```

**Equation (4)**

Where:
- `m_bar_i(t)` = aggregated message
- `s_i(t-)` = previous memory state
- `mem` = learnable memory update function

**Implementation**: Recurrent neural network -- either **GRU** (used in TGN-attn) or **LSTM** or vanilla **RNN** (used for Jodie/DyRep baselines).

For interaction events, memories of **both** nodes `i` and `j` are updated. For node-wise events, only the involved node's memory is updated.

---

## Embedding Module

The embedding module generates temporal embedding `z_i(t)` for node `i` at time `t`. Its primary purpose is to mitigate the **memory staleness problem** -- when a node has no events for a long time, its memory becomes outdated.

General form:

```
z_i(t) = emb(i, t) = SUM_{j in N_i^k([0,t])} h(s_i(t), s_j(t), e_ij, v_i(t), v_j(t))
```

Where `h` is a learnable function.

### Variant 1: Identity (id)

```
emb(i, t) = s_i(t)
```

Uses memory directly as embedding. No graph aggregation.

### Variant 2: Time Projection (time)

```
emb(i, t) = (1 + delta_t * w) . s_i(t)
```

Where:
- `w` = learnable parameters (same dimension as `s_i`)
- `delta_t` = time since last interaction
- `.` = element-wise (Hadamard) product

This is the method used in **Jodie** (Kumar et al., 2019).

### Variant 3: Temporal Graph Attention (attn)

A series of `L` graph attention layers. The final embedding is `z_i(t) = h_i^{(L)}(t)`.

**Input representation** at layer 0:

```
h_j^{(0)}(t) = s_j(t) + v_j(t)
```

Where `s_j(t)` = node memory, `v_j(t)` = temporal node features.

**Layer l computation (Equations 5-9):**

```
h_i^{(l)}(t) = MLP^{(l)}(h_i^{(l-1)}(t) || h_tilde_i^{(l)}(t))                    -- Eq. (5)
```

```
h_tilde_i^{(l)}(t) = MultiHeadAttention^{(l)}(q^{(l)}(t), K^{(l)}(t), V^{(l)}(t))  -- Eq. (6)
```

**Query:**
```
q^{(l)}(t) = h_i^{(l-1)}(t) || phi(0)                                               -- Eq. (7)
```

**Keys and Values:**
```
K^{(l)}(t) = V^{(l)}(t) = C^{(l)}(t)                                                -- Eq. (8)
```

**Context matrix:**
```
C^{(l)}(t) = [h_1^{(l-1)}(t) || e_i1(t_1) || phi(t - t_1),  ...,  h_N^{(l-1)}(t) || e_iN(t_N) || phi(t - t_N)]   -- Eq. (9)
```

Where:
- `||` = concatenation operator
- `phi(.)` = time encoding function (Time2Vec)
- `h_i^{(l-1)}(t)` = node `i` representation at previous layer
- `h_1^{(l-1)}(t), ..., h_N^{(l-1)}(t)` = representations of `i`'s neighbors
- `t_1, ..., t_N` = timestamps of neighbor interactions
- `e_i1(t_1), ..., e_iN(t_N)` = edge features of those interactions
- `MLP^{(l)}` = multi-layer perceptron at layer `l`
- `MultiHeadAttention^{(l)}` = standard multi-head attention (Vaswani et al., 2017)
- Query is the reference node (target or L-1 hop neighbor)
- Keys/Values are its neighbors (context)

### Variant 4: Temporal Graph Sum (sum)

A simpler, faster graph aggregation (Equations 10-11):

```
h_i^{(l)}(t) = W_2^{(l)} * (h_i^{(l-1)}(t) || h_tilde_i^{(l)}(t))                        -- Eq. (10)
```

```
h_tilde_i^{(l)}(t) = ReLU( SUM_{j in N_i([0,t])} W_1^{(l)} * (h_j^{(l-1)}(t) || e_ij || phi(t - t_j)) )   -- Eq. (11)
```

Where:
- `W_1^{(l)}`, `W_2^{(l)}` = learnable weight matrices at layer `l`
- `phi(.)` = time encoding (Time2Vec)
- `z_i(t) = h_i^{(L)}(t)` = final embedding

### Time Encoding

Both temporal graph attention and temporal graph sum modules use the **Time2Vec** encoding (Kazemi et al., 2019), also used in TGAT (Xu et al., 2020). This maps scalar time differences to a vector representation `phi(t)`.

---

## Training

### Training Algorithm (Algorithm 1 -- Pseudocode)

```
Initialize: s <- 0               // Memory to zeros for all nodes
Initialize: m_raw <- {}           // Empty raw message store

FOR EACH batch (i, j, e, t) in training data (chronological order):
    n <- sample_negatives                              // Sample negative nodes
    m <- msg(m_raw)                                    // Compute messages from stored raw features
    m_bar <- agg(m)                                    // Aggregate messages for same nodes
    s_hat <- mem(m_bar, s)                             // Get updated memory
    z_i, z_j, z_n <- emb_{s_hat}(i,t), emb_{s_hat}(j,t), emb_{s_hat}(n,t)   // Compute embeddings
    p_pos, p_neg <- dec(z_i, z_j), dec(z_i, z_n)      // Compute interaction probabilities
    l = BCE(p_pos, p_neg)                              // Binary cross-entropy loss
    m_raw_i, m_raw_j <- (s_hat_i, s_hat_j, t, e), (s_hat_j, s_hat_i, t, e)  // Compute raw messages
    m_raw <- store_raw_messages(m_raw, m_raw_i, m_raw_j)   // Store for next batch
    s_i, s_j <- s_hat_i, s_hat_j                       // Store updated memory
END FOR
```

### Temporal Batching Strategy

**The core problem**: Memory-related modules (message function, message aggregator, memory updater) do not directly influence the loss and would not receive a gradient in a naive implementation.

**Solution -- Raw Message Store**:
1. Before predicting batch interactions, update memory using messages from **previous** batches (stored in Raw Message Store)
2. Predict interactions using the just-updated memory
3. After prediction, store raw messages from **current** batch interactions in the Raw Message Store for use by the next batch

This avoids **information leakage** (using event `e_ij(t)` to update memory before predicting that same event) while ensuring memory-related modules receive gradients.

**Key property**: All predictions in a given batch see the **same memory state**. The memory is up-to-date from the perspective of the first interaction in the batch, but outdated from the perspective of the last. This creates a trade-off:
- **Large batch size** = faster but staler memory (extreme: batch = full dataset means all predictions use initial zero memory)
- **Small batch size** = more up-to-date memory but slower
- **Optimal batch size** = 200 (empirically determined trade-off)

### Link Prediction Objective

**Decoder**: MLP mapping from concatenation of two node embeddings to edge probability.

**Loss function**: Binary Cross-Entropy (BCE)

```
l = BCE(p_pos, p_neg)
```

Where:
- `p_pos = dec(z_i, z_j)` = predicted probability for actual interaction
- `p_neg = dec(z_i, z_n)` = predicted probability for negative (non-interaction)

### Negative Sampling

- Sample an **equal number** of negatives to positive interactions
- Negatives are sampled for each batch

### Temporal Ordering

- Training data is processed in **chronological order** (batches follow temporal sequence)
- 70%-15%-15% **chronological split** for train-validation-test (not random)
- Early stopping with patience of 5 epochs

---

## Key Equations -- Consolidated Reference

### Message Computation

**Interaction event (source and destination):**
```
m_i(t) = msg_s(s_i(t-), s_j(t-), delta_t, e_ij(t))
m_j(t) = msg_d(s_j(t-), s_i(t-), delta_t, e_ij(t))
```

**Node-wise event:**
```
m_i(t) = msg_n(s_i(t-), t, v_i(t))
```

### Message Aggregation
```
m_bar_i(t) = agg(m_i(t_1), ..., m_i(t_b))
```

### Memory Update
```
s_i(t) = mem(m_bar_i(t), s_i(t-))
```
Implementation: GRU (primary), LSTM, or vanilla RNN.

### Temporal Graph Attention Embedding (layer l)
```
h_i^{(l)}(t)       = MLP^{(l)}( h_i^{(l-1)}(t) || h_tilde_i^{(l)}(t) )
h_tilde_i^{(l)}(t) = MultiHeadAttention^{(l)}( q^{(l)}(t), K^{(l)}(t), V^{(l)}(t) )
q^{(l)}(t)         = h_i^{(l-1)}(t) || phi(0)
K^{(l)}(t) = V^{(l)}(t) = C^{(l)}(t)
C^{(l)}(t)         = [ h_j^{(l-1)}(t) || e_ij(t_j) || phi(t - t_j) ]  for each neighbor j
```

Input: `h_j^{(0)}(t) = s_j(t) + v_j(t)`

Output: `z_i(t) = h_i^{(L)}(t)`

### Temporal Graph Sum Embedding (layer l)
```
h_i^{(l)}(t)       = W_2^{(l)} * ( h_i^{(l-1)}(t) || h_tilde_i^{(l)}(t) )
h_tilde_i^{(l)}(t) = ReLU( SUM_j W_1^{(l)} * ( h_j^{(l-1)}(t) || e_ij || phi(t - t_j) ) )
```

### Time Projection Embedding
```
emb(i, t) = (1 + delta_t * w) . s_i(t)
```

### Static GNN Baseline (for reference)
```
z_i = SUM_{j in N_i} h(m_ij, v_i)
m_ij = msg(v_i, v_j, e_ij)
```

---

## Experimental Results

### Datasets

| Property | Wikipedia | Reddit | Twitter |
|---|---|---|---|
| Nodes | 9,227 | 11,000 | 8,861 |
| Edges | 157,474 | 672,447 | 119,872 |
| Edge feature dim | 172 | 172 | 768 |
| Edge feature type | LIWC | LIWC | BERT |
| Timespan | 30 days | 30 days | 7 days |
| Split | 70-15-15% | 70-15-15% | 70-15-15% |
| Nodes with dynamic labels | 217 | 366 | -- |
| Graph type | Bipartite (user-page) | Bipartite (user-subreddit) | Non-bipartite (user-user retweets) |

- **Wikipedia**: users editing pages; interaction features from edit text; labels = user banned
- **Reddit**: users posting to subreddits; interaction features from post text; labels = user banned
- **Twitter**: retweet graph from 2020 RecSys Challenge; BERT-encoded text features
- **No node features** in any dataset (zero vector assigned to all nodes)

### Future Edge Prediction -- Average Precision (%)

**Transductive:**

| Method | Wikipedia | Reddit | Twitter |
|---|---|---|---|
| GAE (static) | 91.44 +/- 0.1 | 93.23 +/- 0.3 | -- |
| VGAE (static) | 91.34 +/- 0.3 | 92.92 +/- 0.2 | -- |
| DeepWalk (static) | 90.71 +/- 0.6 | 83.10 +/- 0.5 | -- |
| Node2Vec (static) | 91.48 +/- 0.3 | 84.58 +/- 0.5 | -- |
| GAT (static) | 94.73 +/- 0.2 | 97.33 +/- 0.2 | 67.57 +/- 0.4 |
| GraphSAGE (static) | 93.56 +/- 0.3 | 97.65 +/- 0.2 | 65.79 +/- 0.6 |
| CTDNE | 92.17 +/- 0.5 | 91.41 +/- 0.3 | -- |
| Jodie | 94.62 +/- 0.5 | 97.11 +/- 0.3 | 85.20 +/- 2.4 |
| TGAT | 95.34 +/- 0.1 | 98.12 +/- 0.2 | 70.02 +/- 0.6 |
| DyRep | 94.59 +/- 0.2 | 97.98 +/- 0.1 | 83.52 +/- 3.0 |
| **TGN-attn** | **98.46 +/- 0.1** | **98.70 +/- 0.1** | **94.52 +/- 0.5** |

**Inductive:**

| Method | Wikipedia | Reddit | Twitter |
|---|---|---|---|
| GAT (static) | 91.27 +/- 0.4 | 95.37 +/- 0.3 | 62.32 +/- 0.5 |
| GraphSAGE (static) | 91.09 +/- 0.3 | 96.27 +/- 0.2 | 60.13 +/- 0.6 |
| Jodie | 93.11 +/- 0.4 | 94.36 +/- 1.1 | 79.83 +/- 2.5 |
| TGAT | 93.99 +/- 0.3 | 96.62 +/- 0.3 | 66.35 +/- 0.8 |
| DyRep | 92.05 +/- 0.3 | 95.68 +/- 0.2 | 78.38 +/- 4.0 |
| **TGN-attn** | **97.81 +/- 0.1** | **97.55 +/- 0.1** | **91.37 +/- 1.1** |

Note: Static methods (GAE, VGAE, DeepWalk, Node2Vec) and CTDNE do not support inductive setting.

### Dynamic Node Classification -- ROC AUC (%)

| Method | Wikipedia | Reddit |
|---|---|---|
| GAE (static) | 74.85 +/- 0.6 | 58.39 +/- 0.5 |
| VGAE (static) | 73.67 +/- 0.8 | 57.98 +/- 0.6 |
| GAT (static) | 82.34 +/- 0.8 | 64.52 +/- 0.5 |
| GraphSAGE (static) | 82.42 +/- 0.7 | 61.24 +/- 0.6 |
| CTDNE | 75.89 +/- 0.5 | 59.43 +/- 0.6 |
| Jodie | 84.84 +/- 1.2 | 61.83 +/- 2.7 |
| TGAT | 83.69 +/- 0.7 | 65.56 +/- 0.7 |
| DyRep | 84.59 +/- 2.2 | 62.91 +/- 2.4 |
| **TGN-attn** | **87.81 +/- 0.3** | **67.06 +/- 0.9** |

### Ablation Studies

**TGN Variants Tested (Table 1):**

| Variant | Memory | Mem. Updater | Embedding | Msg. Agg. | Msg. Func. |
|---|---|---|---|---|---|
| Jodie | node | RNN | time | -- (t-batches) | id |
| TGAT | -- | -- | attn (2l, 20n, uniform) | -- | -- |
| DyRep | node | RNN | id | -- | attn (graph attn on dest neighbors) |
| TGN-attn | node | GRU | attn (1l, 10n) | last | id |
| TGN-2l | node | GRU | attn (2l, 10n) | last | id |
| TGN-no-mem | -- | -- | attn (1l, 10n) | -- | -- |
| TGN-time | node | GRU | time | last | id |
| TGN-id | node | GRU | id | last | id |
| TGN-sum | node | GRU | sum (1l, 10n) | last | id |
| TGN-mean | node | GRU | attn (1l, 10n) | mean | id |

**Key ablation findings (Wikipedia transductive edge prediction):**

1. **Memory vs no memory**: TGN-attn ~4% higher precision than TGN-no-mem; models with memory consistently outperform those without across all neighbor sampling counts.

2. **Embedding module comparison**: Graph-based (TGN-attn, TGN-sum) >> graph-less (TGN-id) by a large margin. TGN-time slightly hurts vs TGN-id. TGN-attn is top performer, only slightly slower than TGN-sum.

3. **Message aggregator**: TGN-mean (mean aggregator) performs slightly better than TGN-attn (last message aggregator), but is more than 3x slower.

4. **Number of layers**: TGN-attn (1 layer) achieves nearly identical performance to TGN-2l (2 layers). Memory makes 2nd layer redundant because 1-hop neighbor memories already encode multi-hop information. In contrast, TGAT without memory loses >10% AP going from 2 to 1 layer.

5. **Neighbor sampling strategy**: Most recent neighbors >> uniform sampling (at 10 neighbors sampled). Memory + most-recent sampling reduces the number of neighbors needed for best performance.

### Scalability

- TGN-attn is up to **30x faster** than TGAT per epoch
- Requires similar number of epochs to converge
- The 1-layer advantage (enabled by memory) is the main speed driver
- Speed-accuracy tradeoff: TGN-attn and TGN-sum occupy the Pareto frontier (best accuracy for given speed)

---

## Hyperparameters

### All Reported Settings

| Parameter | Value |
|---|---|
| Memory Dimension | 172 |
| Node Embedding Dimension | 100 |
| Time Embedding Dimension | 100 |
| Number of Attention Heads | 2 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 200 (train, validation, and test) |
| Early Stopping Patience | 5 epochs |
| Negative Samples | Equal to positive interactions |
| Reference Metric | Average Precision |
| Number of Graph Attention Layers | 1 (TGN-attn), 2 (TGN-2l) |
| Neighbors Sampled per Layer | 10 (TGN variants), 20 (TGAT baseline) |
| Neighbor Sampling Strategy | Most recent (default), uniform (ablation) |
| Memory Updater | GRU (TGN variants), vanilla RNN (Jodie/DyRep) |
| Message Function | Identity (concatenation of inputs) |
| Message Aggregator | Last message (default), Mean (ablation) |
| Hardware | AWS p3.16xlarge |
| Number of Runs | 10 (for all reported results) |
| Data Split | 70% train, 15% validation, 15% test (chronological) |

---

## Module Interconnection Map

```
EVENT e_ij(t)
    |
    v
MESSAGE FUNCTION  -->  m_i(t), m_j(t) = concat(s_i(t-), s_j(t-), delta_t, e_ij(t))
    |
    v
MESSAGE AGGREGATOR  -->  m_bar_i(t) = last_or_mean(messages for node i in batch)
    |
    v
MEMORY UPDATER  -->  s_i(t) = GRU(m_bar_i(t), s_i(t-))
    |
    v
EMBEDDING MODULE  -->  z_i(t) = graph_attention(s_i(t), neighbors' s_j, edge features, time encoding)
    |
    v
DECODER  -->  p = MLP(z_i || z_j)  -->  BCE Loss
```

### Previous Models as TGN Special Cases

- **Jodie** = TGN with RNN memory updater + time projection embedding + identity message + t-batch training
- **TGAT** = TGN without memory + 2-layer graph attention embedding (20 uniform neighbors) + no messages
- **DyRep** = TGN with RNN memory updater + identity embedding + graph-attention-augmented message function
