# Perceiver IO: A General Architecture for Structured Inputs & Outputs

**Source:** Jaegle et al., ICLR 2022, DeepMind
**Core claim:** Single architecture handles arbitrary input/output modalities, scaling linearly in both input and output size.

---

## 1. Architecture Overview: Encode-Process-Decode

Three-phase fully attentional read-process-write pipeline:

```
Input array  -->  [ENCODE]  -->  Latent array  -->  [PROCESS]  -->  Latent array  -->  [DECODE]  -->  Output array
x in R^{M x C}               z in R^{N x D}                     z in R^{N x D}                  y in R^{O x E}
```

- **M** = number of input elements (can be very large: up to 365,056 for flow, 50,657 for multimodal)
- **C** = input feature/channel dimension
- **N** = number of latent vectors (hyperparameter, typically 256-2048)
- **D** = latent feature dimension (hyperparameter, typically 512-1536)
- **O** = number of output elements (task-dependent, can be very large: up to 803,297 for multimodal autoencoding)
- **E** = output feature dimension

M, C, O, E are data properties. N and D are free hyperparameters chosen for tractability.

### 1.1 Encode (Cross-Attention: Input -> Latent)

Maps input array `x in R^{M x C}` to latent array `z in R^{N x D}`.

- Key-value input: input array `x` (M elements)
- Query input: latent array `z` (N elements, initialized as learned parameters)
- Output shape: `R^{N x D}` (same as query)

The latent array has fewer elements than the input (N << M), creating an information bottleneck.

### 1.2 Process (Self-Attention on Latent)

Series of L self-attention modules operating entirely within the latent space `R^{N x D}`.

- Key-value input: latent array (N elements)
- Query input: same latent array (N elements)
- Output shape: `R^{N x D}`
- Operates independently of input/output size
- Deep processing: up to 48 self-attention layers

### 1.3 Decode (Cross-Attention: Latent -> Output)

Maps latent array `z in R^{N x D}` to output array `y in R^{O x E}`.

- Key-value input: latent array (N elements)
- Query input: output query array (O elements)
- Output shape: `R^{O x E}` (same index dimension as query)

Each output point depends only on its query and the latent array -- outputs are decoded in parallel.

---

## 2. Attention Module Equations

Built from GPT-2-style Transformer attention modules: QKV attention followed by MLP.

### 2.1 QKV Attention

Given query input `X_Q in R^{N x D}` and key-value input `X_KV in R^{M x C}`:

**Equation 1 -- Linear projections:**
```
Q = f_Q(X_Q)       # R^{N x F}
K = f_K(X_KV)      # R^{M x F}
V = f_V(X_KV)      # R^{M x F}
```
where `f_Q`, `f_K`, `f_V` are linear layers mapping to shared feature dimension F.

**Equation 2 -- Attention scores:**
```
X_QK = softmax(Q K^T / sqrt(F))     # R^{N x M}
```

**Equation 3 -- Weighted value aggregation + output projection:**
```
X_QKV = X_QK * V                    # R^{N x F}
Attn(X_Q, X_KV) = f_O(X_QKV)       # R^{N x D}
```
where `f_O` is a linear layer projecting output to target channel dimension (typically same as X_Q's).

All linear layers are applied convolutionally over the index dimension (first dimension).
Multi-headed attention is used (head dimension omitted from equations for readability).

### 2.2 Full Module with Residuals and LayerNorm

**Equation 4 -- Attention with normalization:**
```
X_QKV = Attn(layerNorm(X_Q), layerNorm(X_KV))
```

**Equation 5 -- Residual connection:**
```
X_QKV = X_QKV + X_Q
```

**Equation 6 -- MLP with residual:**
```
X_QKV = X_QKV + MLP(layerNorm(X_QKV))
```

MLP is 2-layer with GELU nonlinearity after first layer, applied independently per element of the index dimension.

**Decoder residual note:** For decode attention, the residual `X_QKV = X_QKV + X_Q` (Eq. 5) is sometimes omitted. When queries contain input-space features (e.g., RGB + Fourier for optical flow), this residual forces the network to add those features to its output, making learning unnecessarily difficult.

---

## 3. Computational Complexity

### Per-module complexity

Each attention module is dominated by two matrix multiplications in QKV attention:
- `Q K^T`: shapes `(N x F)` and `(M x F)^T` --> `O(MNF)`
- `X_QK * V`: shapes `(N x M)` and `(M x F)` --> `O(MNF)`

Per-module complexity: **O(MNF)**

### Full model complexity

| Module | KV shape | Q shape | Complexity |
|--------|----------|---------|------------|
| Encoder (1 layer) | M x F | N x F | O(MNF) |
| Latent self-attention (L layers) | N x F | N x F | O(L * N^2 * F) |
| Decoder (1 layer) | N x F | O x F | O(ONF) |

**Total complexity: O([M + O + LN] * N * F)**

Key properties:
- **Linear in input size M** (not quadratic like standard Transformers)
- **Linear in output size O** (not quadratic)
- **Depth L of latent transformer is decoupled from input/output size**
- Standard Transformer: O(L * M^2 * F) -- quadratic in input at every layer

This is what enables Perceiver IO to process inputs with hundreds of thousands of dimensions, while Transformers require preprocessing to at most a few thousand.

---

## 4. Output Query Design

The output query array (O x E) controls what information is decoded. Query construction varies by task:

### Query types by domain

| Domain | Query composition | Query channels |
|--------|-------------------|----------------|
| Language (MLM) | Learned position embeddings | 1280 |
| Language (GLUE single-task) | Single learned class query (per task) | 1280 |
| Language (GLUE multitask) | Per-task learned query embedding | 1280 |
| Optical flow | [Linear(RGB), 2D Fourier features] from query pixel | 322 |
| Multimodal autoencoding (video) | [3D Fourier features, learned modality embedding] | 1026 |
| Multimodal autoencoding (audio) | [1D Fourier features, learned modality embedding] | 1026 |
| Multimodal autoencoding (label) | Learned modality embedding | 1026 |
| ImageNet classification | Single learned class query | 1024 |
| StarCraft II | Entity features (from input) | 128 |
| AudioSet classification | Single learned class query | 1024 |

### Query construction rules

1. **Classification:** Single learned embedding, reusable across examples.
2. **Spatial/sequence outputs:** Position encoding (Fourier features or learned) for each output position.
3. **Multi-task/multimodal:** One learned embedding per task/modality, distinguishing output types.
4. **Content-dependent:** Input features at the query point (e.g., flow uses RGB at queried pixel; StarCraft uses unit features).
5. **Heterogeneous outputs:** Combine position-specific features with modality embeddings; pad to fixed length.

---

## 5. Positional Encodings: Fourier Features

Used for image, audio, and video experiments. Sine and cosine bands with frequencies spaced linearly from minimum to maximum frequency.

### Construction

For each spatial dimension:
- **Number of bands:** 64 sine/cosine bands per dimension
- **Minimum frequency:** single full oscillation over the input dimension
- **Maximum frequency:** Nyquist frequency of the input (e.g., 112 cycles for 224-pixel dimension)
- **Input position normalization:** scaled to [-1, 1] per dimension

Dimensionality examples:
- 2D Fourier features (images): 64 bands x 2 (sin/cos) x 2 dims + 2 raw coords = 258 features
- 3D Fourier features (video): 64 bands x 2 x 3 dims + 3 raw coords = 387 features
- 1D Fourier features (audio): 64 bands x 2 x 1 dim + 1 raw coord = 129 features (padded to 385 in multimodal)

### Alternatives

- **Learned positional encodings:** Array of shape (num_positions x encoding_dim), randomly initialized (truncated Gaussian, scale 0.02). Used for language and some ImageNet experiments.
- When using learned positions on ImageNet (no 2D structure information): 50,176 x 256 array.

---

## 6. Input Handling by Modality

All inputs are serialized into a 2D array `R^{M x C}` regardless of original structure.

| Domain | Raw input | Input representation | M x C |
|--------|-----------|---------------------|-------|
| Language (tokens) | SentencePiece tokens | token embedding + learned position | 512 x 768 |
| Language (bytes) | UTF-8 bytes | byte embedding + learned position | 2048 x 768 |
| Optical flow (raw) | Two RGB frames concatenated | 3x3 patch per pixel, concat channels + 2D Fourier | 182,528 x 64 |
| Optical flow (separate) | Two RGB frames separately | 3x3 patch per pixel + 3D Fourier | 365,056 x 64 |
| Optical flow (conv) | Conv+maxpool preprocessed | Conv features + 2D Fourier | 22,816 x 64 |
| Multimodal (Kinetics) | Video patches + audio patches + one-hot label | Patched + Fourier + modality embedding, padded to same C | 50,657 x 704 |
| ImageNet (raw pixels) | 224x224 RGB image | [RGB, 2D Fourier features] | 50,176 x 261 |
| ImageNet (conv) | Conv+maxpool preprocessed | [Conv features, 2D Fourier] | 3,136 x 322 |
| StarCraft II | Unit entity set | Entity feature vectors | 512 x 256 |
| AudioSet | Video patches + audio patches | Patched + Fourier + modality embedding | 13,024-17,344 x 487 |

### Multimodal serialization

For multimodal inputs (video + audio + label):
1. Pad each modality's features with a learned modality-specific embedding vector
2. Pad all elements to the same feature dimension C
3. Concatenate along the index dimension to form a single 2D array

---

## 7. Comparison: Perceiver IO vs. Standard Transformer

| Property | Standard Transformer | Perceiver IO |
|----------|---------------------|-------------|
| Per-layer complexity | O(M^2 * F) | Encode: O(MNF), Process: O(N^2 * F), Decode: O(ONF) |
| Total complexity (L layers) | O(L * M^2 * F) | O([M + O + LN] * N * F) |
| Scaling with input size | Quadratic per layer | Linear (encoder only) |
| Max practical input size | ~few thousand tokens | Hundreds of thousands of elements |
| Output flexibility | Fixed (same as input, or pooled) | Arbitrary size/structure via query array |
| Depth vs. input coupling | Coupled: every layer sees full input | Decoupled: deep processing in latent space |
| Tokenization required | Yes (for language) | No (works on raw bytes) |
| Modality-specific architecture | Required | Not required |

### Transformer speed comparison

At matched FLOPs (~113B vs 109B), byte-level Perceiver IO (2048 inputs) is 2.6x faster than byte-level BERT (2048 inputs) in training steps/sec: 7.6 vs 2.9.

---

## 8. Experimental Results

### 8.1 Language -- GLUE Benchmark

| Model | Tokenization | M (inputs) | N (latents) | Depth | Params | FLOPs | Steps/sec | GLUE Avg |
|-------|-------------|------------|-------------|-------|--------|-------|-----------|----------|
| BERT Base (test) | SentencePiece | 512 | 512 | 12 | 110M | 109B | - | 81.0 |
| BERT Base (ours) | SentencePiece | 512 | 512 | 12 | 110M | 109B | 7.3 | 81.1 |
| Perceiver IO Base | SentencePiece | 512 | 256 | 26 | 223M | 119B | 7.4 | 81.2 |
| BERT (matching FLOPs) | UTF-8 bytes | 2048 | 2048 | 6 | 20M | 130B | 2.9 | 71.5 |
| Perceiver IO | UTF-8 bytes | 2048 | 256 | 26 | 201M | 113B | 7.6 | 81.0 |
| Perceiver IO++ | UTF-8 bytes | 2048 | 256 | 40 | 425M | 241B | 4.2 | 81.8 |

Key findings:
- Perceiver IO on raw bytes matches BERT on SentencePiece tokens (81.0 vs 81.1)
- Perceiver IO++ on bytes outperforms BERT Base (81.8 vs 81.0)
- Byte-level BERT at matched FLOPs collapses to 71.5 -- Transformer cannot handle 2048-length sequences efficiently
- Perceiver IO is 2.6x faster than byte-level BERT despite being much deeper (26 vs 6 layers)

### Multitask GLUE

| Method | GLUE Avg |
|--------|----------|
| Single-task query | 81.0 |
| Shared input token (CLS-style) | 81.5 |
| Task-specific input tokens | 81.8 |
| Multitask query (Perceiver IO) | 81.8 |

Multitask queries match or outperform single-task approaches, without needing CLS tokens.

### Latent dimension ablation (UTF-8 bytes Perceiver IO)

| N (latents) | D (latent width) | FLOPs | GLUE Avg |
|-------------|-----------------|-------|----------|
| 128 | 1920 | 120B | 75.84 |
| 256 | 1280 | 113B | 80.95 |
| 512 | 896 | 125B | 80.92 |

Sweet spot at N=256, D=1280 for language.

### 8.2 Optical Flow -- Sintel and KITTI

| Network | Sintel.clean (EPE) | Sintel.final (EPE) | KITTI (EPE) |
|---------|-------------------|-------------------|-------------|
| PWCNet | 2.17 | 2.91 | 5.76 |
| RAFT | 1.95 | 2.57 | 4.23 |
| **Perceiver IO** | **1.81** | **2.42** | 4.98 |

- State-of-the-art on Sintel.final at time of publication
- No cost volumes, no explicit warping, no hierarchical structure, no 2D layout in latent
- Input: two frames concatenated along channel dim, 3x3 patches per pixel (54 values per pixel)
- Query: input encoding at each pixel
- Architecture: 2048 latents, 512 channels, 24 self-attention modules, 16 heads each
- Parameters: ~27.9M; FLOPs per 368x496 forward pass: ~987B
- Training: AutoFlow dataset, 480 epochs, cosine LR from 4e-4, batch size 512, LAMB optimizer

### Flow ablations

| Patch size | Concat frames | Downsample | Depth | Latents | Sintel.clean | Sintel.final | KITTI |
|-----------|--------------|-----------|-------|---------|-------------|-------------|-------|
| 3x3 | Yes | No | 24 | 2048 | 1.81 | 2.42 | 4.98 |
| 3x3 | No | No | 24 | 2048 | 1.78 | 2.70 | 6.19 |
| 1x1 | Yes | No | 24 | 2048 | 1.91 | 2.56 | 5.39 |
| 1x1 | No | No | 24 | 2048 | 1.72 | 2.63 | 5.93 |
| N/A | Yes | Yes (conv) | 24 | 2048 | 1.84 | 2.52 | 4.83 |
| N/A | Yes | Yes (conv) | 16 | 1024 | 2.06 | 2.67 | 6.12 |

Concatenating frames helps on harder datasets (Sintel.final, KITTI). Spatial context matters.

### Inference speed (1088 x 436 images)

| Model | GPU (TITAN Xp) | TPU v3 |
|-------|----------------|--------|
| Perceiver IO (full) | 0.8 fps | 4.4 fps |
| Perceiver IO (lightweight, conv+RAFT upsample) | 3.3 fps | 17.8 fps |
| RAFT (TF implementation) | - | 1.6 fps |

Perceiver IO is 2.75-11x faster on TPU v3 than RAFT due to absence of gather operations.

### 8.3 Multimodal Autoencoding -- Kinetics-700-2020

Input: 16 frames at 224x224 (50k 4x4 patches) + 30k raw audio samples + 700-class one-hot label = 50,657 input elements with 704 features.

Output: 803,297 elements (pixels + audio samples + label).

Training: subsampled decoding (512 audio + 512 pixels + label per step). Full decoding at test time.

| Compression Ratio | Latents | Audio PSNR | Video PSNR | Top-1 Accuracy |
|-------------------|---------|------------|------------|----------------|
| 88x | 784 | 26.97 | 24.37 | 10.2% |
| 176x | 392 | 25.33 | 24.27 | 8.6% |
| 352x | 196 | 14.15 | 23.21 | 11.5% |

With stronger class loss weighting: 45% top-1 accuracy while maintaining 20.7 video PSNR.

Latent array: 512 channels. Loss: weighted sum of L1 (video, weight 0.03), L1 (audio, weight 1.0), cross-entropy (label, weight 0.0001).

Model: 20.0M params, 310B FLOPs (train), 6.85T FLOPs (eval).

### 8.4 ImageNet Classification

| Model | Pretrained | Top-1 Acc | FLOPs | Params |
|-------|-----------|-----------|-------|--------|
| ResNet-50 | No | 78.6 | 4.1B | 26M |
| NFNet-F6+SAM | No | 86.5 | 377.3B | 438.4M |
| ViT-B/16 | No | 77.9 | 55.4B | 86M |
| ViT-H/14 | Yes (JFT) | 88.6 | - | 632M |
| DeiT (1000 ep) | No | 85.2 | - | 87M |
| Perceiver (2D FF) | No | 78.6 | 404B | 42.1M |
| Perceiver IO (config A, 2D FF) | No | 79.0 | 407B | 48.4M |
| Perceiver IO (config B, 2D FF, JFT pretrained) | Yes | 84.5 | 213B | 212M |
| Perceiver (learned pos) | No | 67.6 | 404B | 55.9M |
| Perceiver IO (config A, learned pos) | No | 72.7 | 407B | 62.3M |
| Perceiver (conv) | No | 77.4 | 367B | 42.1M |
| Perceiver IO (config A, conv) | No | 82.1 | 369B | 48.6M |
| Perceiver IO (config B, conv, JFT pretrained) | Yes | 86.4 | 176B | 212M |

Key findings:
- Perceiver IO consistently outperforms original Perceiver (attention decoder > average+project decoder)
- 84.5% top-1 without 2D convolutions (with JFT pretraining)
- 86.4% with conv preprocessing + JFT pretraining
- 72.7% with fully learned positions (no 2D information at all) -- best known result without any 2D features

Architecture: 8 blocks of 6 attention modules, weights shared between corresponding modules in each block.
Training: 110 epochs, batch size 1024, 64 TPUs, LAMB optimizer, LR 2e-3 flat for 55 epochs then cosine decay to 0.

### 8.5 StarCraft II

Replaced AlphaStar's entity Transformer with Perceiver IO.

| Entity Encoder | Win Rate | Params | FLOPs | Steps/sec |
|----------------|----------|--------|-------|-----------|
| Transformer (AlphaStar) | 0.87 | 144M | 3.3B | 2.9 |
| Perceiver IO | 0.87 | 140M | 0.93B | 2.9 |

- 3.5x reduction in FLOPs with identical win rate
- Latent index dimension: 32 (vs 512 input entities)
- Only hyperparameter swept: latent index dimension (32 vs 64)
- 3-layer Perceiver IO, 2 heads, feature dimension 128

### 8.6 AudioSet Classification

| Model | Input | mAP | Latent channels | Params | FLOPs |
|-------|-------|-----|-----------------|--------|-------|
| Perceiver | Raw audio + video | 42.4 | 512 | 21.0M | 52.3B |
| Perceiver IO | Raw audio + video | 43.3 | 512 | 25.0M | 52.9B |
| Perceiver | mel-spectrogram + video | 43.6 | 512 | 21.0M | 60.7B |
| Perceiver IO | mel-spectrogram + video | 44.9 | 1024 | 88.2M | 129.5B |

Attention-based decoder consistently outperforms average+project decoder across all settings.

---

## 9. Architecture Hyperparameters by Domain

### Language

| Parameter | BERT Base | Perceiver IO | Perceiver IO++ |
|-----------|-----------|-------------|----------------|
| Tokenizer | SentencePiece | UTF-8 bytes | UTF-8 bytes |
| Number of inputs (M) | 512 | 2048 | 2048 |
| Input embedding size (C) | 768 | 768 | 768 |
| Number of process layers | 12 | 26 | 40 |
| Number of latents (N) | - | 256 | 256 |
| Latent size (D) | - | 1280 | 1536 |
| FFW hidden dim (latents) | - | 1280 | 1536 |
| Output queries (O) during pretrain | - | 2048 | 2048 |
| Query dimension (E) | - | 768 | 768 |
| FFW hidden dim (outputs) | - | 768 | 768 |

Vocabulary: SentencePiece = 32,000 tokens. Bytes = 256 + 4 special tokens ([PAD], [MASK], [CLS], [SEP]).

### Training hyperparameters

**MLM pretraining:** 500K steps, batch size 2048, LAMB optimizer, LR 0.00125, 1000 warmup steps, cosine decay over 500K steps, weight decay 0.01. Data: 70% C4 + 30% English Wikipedia.

**GLUE finetuning:** 10 epochs, batch size {16, 32, 64}, LAMB optimizer, LR {1e-4, 5e-5, 2e-5, 1e-5}, 200 warmup steps, weight decay 0.01.

### Optical Flow

- Latent array: 2048 elements, 512 channels
- 24 self-attention modules, 16 heads each
- Training: AutoFlow, 480 epochs, cosine LR from 4e-4, batch 512, LAMB
- Position encoding: 64 sine/cosine bands per spatial dimension + raw coordinates = 258 extra features
- Input/query features projected to 64 dims before entering transformer

### Multimodal Autoencoding (Kinetics)

- Latent array: 784 latents (88x compression) or 392/196, 512 channels
- Patch sizes: 1x4x4 (video), 16 (audio)
- Position encoding: 387-dim 3D Fourier (video), 385-dim 1D Fourier (audio)
- Modality embeddings: 317-dim (video), 319-dim (audio), 4-dim (label)
- All inputs padded to 704 features
- Decoder queries: 1026 features (Fourier + modality embeddings)
- Loss: 0.03 * L1(video) + 1.0 * L1(audio) + 0.0001 * CE(label)
- Training: batch 1024, LR 1e-3
- Class label masked 50% during training

### ImageNet

- 8 blocks of 6 attention modules with weight sharing across blocks
- Training: 110 epochs, batch 1024, 64 TPUs, LAMB, LR 2e-3
- LR schedule: flat for 55 epochs, cosine decay to 0 over final 55 epochs
- Weight decay: 0.1, gradient clip: max global norm 10
- Augmentation: RandAugment (4 layers, magnitude 5), CutMix (ratio 0.2), MixUp
- No dropout
- JFT pretraining: 14 epochs, 256 TPUs, batch 8192, LR 3e-4 cosine to 0, 16-layer latent (no weight sharing)

---

## 10. Decoder Comparison: Attention vs. Average+Project

### Average+Project (original Perceiver)
1. Uniform averaging: `z_avg = (1/N) * sum(z_i)`
2. Linear projection to output dimension

### Attention Decoder (Perceiver IO)
1. Data-dependent weighted average via attention scores (learned, input-dependent weights)
2. Value projection + MLP processing

The attention decoder is strictly more expressive: it learns which latents to attend to per query (vs. uniform averaging) and uses a richer projection pipeline. Perceiver IO consistently outperforms Perceiver across all domains tested (ImageNet, AudioSet, flow).

---

## 11. Output Subsampling for Large Outputs

For very large output spaces (e.g., 803,297 elements for Kinetics multimodal autoencoding):
- Each output depends only on its query + latent array (no inter-output dependencies)
- **Training:** Subsample output array, compute loss on affordable subset (e.g., 512 audio + 512 pixels + label)
- **Inference:** Generate outputs in sequential batches to produce full output array
- This amortizes memory cost while maintaining full coverage

---

## 12. Key Design Principles

1. **Domain agnosticism:** All inputs serialized to 2D byte arrays. No modality-specific processing in the core architecture.
2. **Bottleneck via latent:** N << M decouples computational depth from input size.
3. **Query-based decoding:** Output structure encoded entirely in queries, not architecture.
4. **Linear scaling:** No quadratic dependence on input or output size.
5. **Parallel decoding:** Each output point independent -- enables subsampled training.
6. **Weight sharing in depth:** Optional sharing across blocks reduces parameters while maintaining depth.
7. **No tokenization required:** Operates on raw bytes/pixels with no performance loss vs. tokenized input.
