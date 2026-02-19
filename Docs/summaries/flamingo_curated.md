# Flamingo: Visual Language Model for Few-Shot Learning -- Raw Technical Extraction

**Source:** Alayrac, Donahue, Luc, Miech et al., DeepMind, NeurIPS 2022

---

## 1. CORE FORMULATION

Flamingo models the likelihood of text y conditioned on interleaved images/videos x:

```
p(y | x) = PRODUCT(l=1..L) p(y_l | y_<l, x_<=l)
```

- `y_l` = l-th language token
- `y_<l` = all preceding tokens
- `x_<=l` = all images/videos preceding token y_l in the interleaved sequence
- p is parameterized by the Flamingo model

---

## 2. ARCHITECTURE

### 2.1 Overall Pipeline

```
Input: interleaved [images/videos + text] --> Vision Encoder (frozen NFNet-F6)
  --> Perceiver Resampler --> visual tokens (fixed 64 per image/video)
  --> injected via GATED XATTN-DENSE layers into frozen LM (Chinchilla)
  --> Output: autoregressive text generation
```

Three frozen/trainable boundaries:
- **Frozen:** Vision Encoder (NFNet-F6), Language Model (Chinchilla)
- **Trained from scratch:** Perceiver Resampler, GATED XATTN-DENSE layers

### 2.2 Vision Encoder

- Architecture: **NFNet-F6** (Normalizer-Free ResNet)
- Pretrained with **contrastive objective** (two-term contrastive loss from CLIP/Radford et al.)
- Contrastive pretraining datasets: ALIGN + LTIP
- Contrastive training resolution: **288 x 288**
- Flamingo training resolution: **320 x 320** (increased for improved CNN test-time performance)
- Fine-tuning resolution: **480 x 480**
- Output: 2D spatial grid of features, flattened to 1D sequence
- For video: frames sampled at **1 FPS**, encoded independently, producing 3D spatio-temporal grid; learned temporal embeddings added, then flattened to 1D
- **Frozen** during Flamingo training

**Contrastive Loss (text-to-image):**
```
L_contrastive:txt2im = -(1/N) * SUM_i log( exp(L_i^T * V_i * beta) / SUM_j exp(L_i^T * V_j * beta) )
```

**Contrastive Loss (image-to-text):**
```
L_contrastive:im2txt = -(1/N) * SUM_i log( exp(V_i^T * L_i * beta) / SUM_j exp(V_i^T * L_j * beta) )
```

- `V_i`, `L_i` = normalized vision/language embeddings of i-th batch element
- `beta` = trainable inverse temperature parameter
- `N` = batch size
- Total loss = sum of both terms
- Language encoder for contrastive pretraining: BERT architecture
- Joint embedding space size: 1376
- Contrastive batch size: 16,384
- Contrastive training: 1.2M parameter update steps on 512 TPUv4 chips

### 2.3 Perceiver Resampler

Converts variable-size vision encoder output to **fixed 64 visual tokens**.

**Pseudo-code:**
```python
def perceiver_resampler(
    x_f,               # [T, S, d] visual features (T=time, S=space)
    time_embeddings,    # [T, 1, d] time positional embeddings
    x,                  # R learned latents of shape [R, d]  (R=64)
    num_layers,         # Number of layers (6 for all models)
):
    x_f = x_f + time_embeddings
    x_f = flatten(x_f)           # [T, S, d] -> [T*S, d]
    for i in range(num_layers):
        x = x + attention_i(q=x, kv=concat([x_f, x]))   # cross-attend + self-attend
        x = x + ffw_i(x)
    return x   # [R, d] = 64 visual tokens
```

Key design choices:
- R = 64 learned latent queries (output tokens)
- Keys/values = concatenation of visual features x_f AND learned latents x (differs from DETR/Perceiver)
- Only temporal positional encodings, NO spatial grid encodings (CNNs implicitly encode spatial info)
- ~194M parameters, **same across all model sizes**
- 6 layers, hidden dim 1536, 16 heads, Squared ReLU activation
- Key/value dim per head = 1536/16 = 96

### 2.4 GATED XATTN-DENSE Layers

Inserted between frozen LM layers. Conditions frozen LM on visual representations.

**Pseudo-code:**
```python
def gated_xattn_dense(
    y,              # input language features
    x,              # input visual features (from Perceiver Resampler)
    alpha_xattn,    # cross-attention gating parameter, initialized to 0
    alpha_dense,    # FFW gating parameter, initialized to 0
):
    # 1. Gated Cross Attention
    y = y + tanh(alpha_xattn) * attention(q=y, kv=x)
    # 2. Gated Feed Forward (dense) Layer
    y = y + tanh(alpha_dense) * ffw(y)
    # 3. Regular frozen self-attention + FFW
    y = y + frozen_attention(q=y, kv=y)
    y = y + frozen_ffw(y)
    return y   # output: visually informed language features
```

Key design choices:
- **tanh gating**: multiplies new layer output by `tanh(alpha)` where alpha is learnable scalar initialized to 0
- At initialization: tanh(0) = 0, so model output = original pretrained LM output (stability)
- alpha values grow during training; deeper layers tend to have larger |tanh(alpha)| values
- Cross-attention: queries from language, keys/values from vision
- Activation: **Squared ReLU** (newly trained layers), vs GeLU (frozen LM)
- FFW hidden dimension = 4D

### 2.5 Multi-Visual Input: Per-Image Attention Masking

At each text token, the model **cross-attends only to the visual tokens of the most recent preceding image** (not all previous images).

Formally: function phi: [1, L] -> [0, N] assigns to each text position the index of the last image/video appearing before that position (or 0 if none).

```
x_<=l = { x_i | i <= phi(l) }
```

- Masking applied to cross-attention matrix
- Dependency on ALL previous images maintained implicitly through LM self-attention
- Training: up to N=5 images per sequence on M3W
- Inference: generalizes to up to N=32 image/video shots
- Ablation showed single-image cross-attention is 7.2% better than attending to all previous images

### 2.6 Frozen LM Backbone

- Architecture: **Chinchilla** decoder-only Transformer
- Three sizes: 1.4B, 7B, 70B parameters
- Pretrained on **MassiveText** dataset
- **All LM layers remain frozen** during Flamingo training
- Unfreezing LM (pretrained): -8.0% performance (catastrophic forgetting)
- Training LM from scratch: -12.9% performance
- Co-training on MassiveText (unfrozen LM): still worse than keeping frozen

---

## 3. MODEL CONFIGURATIONS

### 3.1 Transformer Hyperparameters

| Component | Model | Layers (L) | Hidden Dim (D) | Heads (H) | Activation | FFW Hidden |
|---|---|---|---|---|---|---|
| Perceiver Resampler | All | 6 | 1536 | 16 | Sq. ReLU | 6144 |
| GATED XATTN-DENSE | Flamingo-3B | 24 | 2048 | 16 | Sq. ReLU | 8192 |
| GATED XATTN-DENSE | Flamingo-9B | 10 | 4096 | 32 | Sq. ReLU | 16384 |
| GATED XATTN-DENSE | Flamingo-80B | 12 | 8192 | 64 | Sq. ReLU | 32768 |
| Frozen LM | Flamingo-3B | 24 | 2048 | 16 | GeLU | 8192 |
| Frozen LM | Flamingo-9B | 40 | 4096 | 32 | GeLU | 16384 |
| Frozen LM | Flamingo-80B | 80 | 8192 | 64 | GeLU | 32768 |

- Key/Value dim per head: 96 (Perceiver Resampler), 128 (GATED XATTN-DENSE + LM)

### 3.2 Parameter Counts

| Model | Frozen LM | Frozen Vision | Trainable GATED XATTN-DENSE | Trainable Resampler | Total | XATTN Frequency |
|---|---|---|---|---|---|---|
| Flamingo-3B | 1.4B | 435M | 1.2B | 194M | 3.2B | Every layer |
| Flamingo-9B | 7.1B | 435M | 1.6B | 194M | 9.3B | Every 4th layer |
| Flamingo-80B | 70B | 435M | 10B | 194M | 80B | Every 7th layer |

---

## 4. TRAINING

### 4.1 Training Objective

Weighted sum of per-dataset expected negative log-likelihoods:

```
Loss = SUM(m=1..M) lambda_m * E_{(x,y)~D_m} [ -SUM(l=1..L) log p(y_l | y_<l, x_<=l) ]
```

- D_m = m-th dataset
- lambda_m = dataset weight
- Gradients accumulated over all datasets simultaneously (NOT round-robin)

### 4.2 Training Data

| Dataset | Type | Size | Weight (lambda) |
|---|---|---|---|
| M3W (MultiModal MassiveWeb) | Interleaved image+text from webpages | ~43M webpages | 1.0 |
| ALIGN | Image-text pairs (alt-text) | 1.8B pairs | 0.2 |
| LTIP (Long Text & Image Pairs) | Image-text pairs (longer descriptions) | 312M pairs | 0.2 |
| VTP (Video & Text Pairs) | Video-text pairs (~22s avg) | 27M pairs | 0.03 |

**M3W Processing:**
- Extract text + images from HTML DOM; insert `<image>` tags at image positions
- Insert `<EOC>` (end of chunk) token before each image and at document end
- Sample random subsequence of L=256 tokens, take up to first N=5 images
- Image placement augmentation: p_next=0.5 (randomly assign text to previous or next image)

**Paired Dataset Processing:**
- Prepend `<image>` tag, append `<EOC>` token to each caption
- ALIGN avg text length: 12.4 tokens; LTIP avg text length: 20.5 tokens

**Deduplication:** Training images deduplicated against evaluation benchmark images using visual encoder embeddings + approximate nearest neighbor search.

### 4.3 Optimization Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0 -> 1e-4 (linear warmup over 5000 steps, then constant) |
| Weight decay | 0.1 (except Perceiver Resampler: 0) |
| Gradient clipping | Global norm clipping = 1 |
| Training steps | 500,000 |
| Precision | Params stored/updated in float32; activations/gradients in bfloat16; frozen params in bfloat16 |
| Infrastructure | JAX + Haiku, TPUv4 |
| Flamingo-80B | 1536 TPUv4 chips, 15 days, 16-way Megatron model parallelism, ZeRO stage 1 |

**Data Augmentation:**
- Visual inputs resized to 320x320, preserving aspect ratio, padding with mean value
- Random left/right flips, color augmentation
- Text: stochastically prepend single space character with probability 0.5
- Video training: 8 frames at 1 FPS per video clip
- Video inference: 30 frames at 3 FPS (temporal position embeddings linearly interpolated)

**Contrastive Pretraining (Vision Encoder):**
- Optimizer: Adam
- Learning rate: 1e-3 linear decay to 0
- Label smoothing: 0.1
- Adaptive gradient clipping (AGC): 1e-2 for NFNet, global norm 10 for BERT encoder
- Training: 1.2M steps, 512 TPUv4 chips

### 4.4 Fine-tuning Protocol

- Keep LM layers frozen; train same Flamingo layers as during pretraining
- Additionally unfreeze vision backbone
- Resolution: 480x480
- Learning rate: 3e-8 to 1e-5 (grid search)
- LR schedule: exponential decay by factors of 10x
- Batch size: 8 or 16
- Data augmentation: color augmentation, random horizontal flips (task-dependent)

---

## 5. FEW-SHOT PROMPTING MECHANISM

### 5.1 Prompt Construction

Support examples interleaved as (image, text) or (video, text) pairs, followed by query visual input:

**Visual QA format:**
```
<BOS><image>Question: {q1} Answer: {a1}<EOC><image>Question: {q2} Answer: {a2}<EOC><image>Question: {query_q} Answer:
```

**Vision-to-text format:**
```
<BOS><image>Output: {caption1}<EOC><image>Output: {caption2}<EOC><image>Output:
```

### 5.2 Evaluation Modes

- **Open-ended:** Beam search (beam size = 3), stop at first `<EOC>` token
- **Close-ended:** Score each possible answer using log-likelihood, rank by decreasing confidence

### 5.3 Zero-shot Protocol

- Prompt with 2 text-only examples (no images) from the task
- 1 text example is highly detrimental (model biased to produce similar text)
- More than 2 helps only marginally

### 5.4 RICES (Retrieval In-Context Example Selection)

- Retrieve support examples similar to query image using frozen vision encoder features
- Build prompt with top-N most similar examples
- Order by increasing similarity (most similar appears right before query, due to LM recency bias)
- +9.2% improvement on ImageNet over random selection

### 5.5 Prompt Ensembling

- Close-ended setting: average log-likelihoods over 6 random permutations of selected few-shot examples

---

## 6. BENCHMARK RESULTS

### 6.1 Few-Shot Results (Flamingo-80B, 32-shot)

| Benchmark | Type | Metric | Flamingo 32-shot | Fine-tuned SotA | FT SotA Data |
|---|---|---|---|---|---|
| OKVQA | Image VQA | VQA acc | **57.8** | 54.4 | 10K |
| VQAv2 | Image VQA | VQA acc | 67.6 | 80.2 | 444K |
| COCO | Image Caption | CIDEr | **113.8** | 143.3 | 500K |
| MSVDQA | Video VQA | Top-1 | **52.3** | 47.9 | 27K |
| VATEX | Video Caption | CIDEr | **65.1** | 76.3 | 500K |
| VizWiz | Image VQA | VQA acc | 49.8 | 57.2 | 20K |
| Flickr30K | Image Caption | CIDEr | **75.4** | 67.4 | 30K |
| MSRVTTQA | Video VQA | Top-1 | 31.0 | 46.8 | 130K |
| iVQA | Video VQA | iVQA acc | **45.3** | 35.4 | 6K |
| YouCook2 | Video Caption | CIDEr | 86.8 | 138.7 | 10K |
| STAR | Video MC QA | Top-1 | **42.2** | 36.7 | 46K |
| VisDial | Image Dialogue | NDCG | 55.6 | 75.2 | 123K |
| TextVQA | Image VQA | VQA acc | 37.9 | 54.7 | 20K |
| NextQA | Video QA | WUPS | **33.5** | 25.2 | 38K |
| HatefulMemes | Image Classif. | ROC AUC | 70.0 | 79.1 | 9K |
| RareAct | Video Retrieval | mWAP | 60.8 (0-shot) | - | - |

**Bold** = Flamingo 32-shot exceeds FT SotA. 6 of 16 tasks surpassed with only 32 examples.

### 6.2 Scaling: Few-Shot by Model Size (4-shot)

| Benchmark | Flamingo-3B | Flamingo-9B | Flamingo-80B |
|---|---|---|---|
| OKVQA | 43.3 | 49.3 | 57.4 |
| VQAv2 | 53.2 | 56.3 | 63.1 |
| COCO CIDEr | 85.0 | 93.1 | 103.2 |
| MSVDQA | 33.0 | 36.2 | 41.7 |
| VATEX CIDEr | 50.0 | 51.7 | 56.0 |

### 6.3 Fine-tuned Flamingo-80B (New SotA on 5 tasks)

| Benchmark | 32-shot | Fine-tuned | Previous SotA |
|---|---|---|---|
| VQAv2 (test-dev) | 67.6 | **82.0** | 81.3 |
| VATEX (test) | 65.1 | **84.2** | 76.3 |
| VizWiz (test-dev) | 49.8 | **65.7** | 57.2 |
| MSRVTTQA (test) | 31.0 | **47.4** | 46.8 |
| HatefulMemes (test seen) | 70.0 | **86.6** | 84.6 |
| COCO (test) | 113.8 | 138.1 | 149.6 |
| VisDial (valid) | 56.8 | 61.8 | 75.2 |
| YouCook2 (valid) | 86.8 | 118.6 | 138.7 |
| TextVQA (valid) | 36.0 | 57.1 | 54.7 |

### 6.4 Classification (Flamingo-80B)

| Method | Shots/class | ImageNet Top-1 | Kinetics700 Avg |
|---|---|---|---|
| Random prompt | ~0.02 | 66.4 | 51.2 |
| RICES, 16 prompt, 1/class | 1 | 71.7 | 62.7 |
| RICES, 16 prompt, 5/class | 5 | 76.0 | 63.5 |
| RICES + ensembling, 5/class | 5 | 77.3 | 64.2 |
| Contrastive SotA (zero-shot) | 0 | 85.7 (BASIC) | 69.6 (CLIP) |
| Fine-tuned SotA | full | 90.9 | 89.0 |

### 6.5 Contrastive Model Retrieval (Zero-Shot)

| Model | Flickr30K i2t R@1 | Flickr30K t2i R@1 | COCO i2t R@1 | COCO t2i R@1 |
|---|---|---|---|---|
| Flamingo NFNet-F6 | **89.3** | **79.5** | **65.9** | **48.0** |
| Florence | 90.9 | 76.7 | 64.7 | 47.2 |
| ALIGN | 88.6 | 75.7 | 58.6 | 45.6 |
| CLIP | 88.0 | 68.7 | 58.4 | 37.7 |

---

## 7. ABLATION STUDIES (Flamingo-3B, 4-shot, DEV benchmarks)

### 7.1 Training Data

| Setting | Overall Score |
|---|---|
| Full (M3W + ITP + VTP) | 70.7 |
| w/o Video-Text pairs | 67.3 (-3.4) |
| w/o Image-Text pairs | 60.9 (-9.8) |
| Image-Text pairs -> LAION-400M | 66.4 (-4.3) |
| w/o M3W | 53.4 (-17.3) |

M3W is the most critical dataset; removing it causes -17.3% drop.

### 7.2 Optimization Strategy

| Setting | Overall Score |
|---|---|
| Gradient Accumulation (default) | 70.7 |
| Round Robin | 62.9 (-7.8) |

### 7.3 Tanh Gating

| Setting | Overall Score |
|---|---|
| With tanh gating (default) | 70.7 |
| Without tanh gating | 66.5 (-4.2) |

Disabling also leads to training instabilities.

### 7.4 Cross-Attention Architecture

| Architecture | Params | Overall Score |
|---|---|---|
| GATED XATTN-DENSE (default) | 3.2B | 70.7 |
| VANILLA XATTN | 2.4B | 66.9 |
| GRAFTING | 3.3B | 63.1 |

### 7.5 Cross-Attention Frequency

| Frequency | Params | Step Time | Overall Score |
|---|---|---|---|
| Every layer | 3.2B | 1.74s | 70.7 |
| Every 2nd | 2.6B | 1.24s | 68.2 |
| Every 4th | 2.3B | 1.02s | 68.8 |
| Single in middle | 2.0B | 0.87s | 59.8 |

Every 4th: 66% faster training, only -1.9% score drop.

### 7.6 Resampler Architecture

| Architecture | Overall Score |
|---|---|
| Perceiver Resampler (default) | 70.7 |
| Transformer | 66.7 |
| MLP | 66.6 |

### 7.7 Vision Encoder

| Encoder | Overall Score |
|---|---|
| NFNet-F6 (default) | 70.7 |
| CLIP ViT-L/14 | 64.9 |
| NFNet-F0 | 62.7 |

### 7.8 Freezing LM

| Setting | Overall Score |
|---|---|
| Frozen pretrained LM (default) | 70.7 |
| Fine-tuned pretrained LM | 62.7 (-8.0) |
| Trained from scratch | 57.8 (-12.9) |

### 7.9 Multi-Image Attention

| Strategy | Overall Score |
|---|---|
| Attend to last preceding image only (default) | 70.7 |
| Attend to all previous images | 63.5 (-7.2) |

### 7.10 LM Pretraining Data

| LM Pretrained On | Overall Score |
|---|---|
| MassiveText (default) | 70.7 |
| C4 | 62.8 (-7.9) |

### 7.11 Freezing Vision Encoder

| Setting | Overall Score |
|---|---|
| Frozen pretrained (default) | 70.7 |
| Fine-tuned pretrained | 68.1 (-2.6) |
| Trained from scratch | 61.4 (-9.3) |

---

## 8. KEY DESIGN PRINCIPLES (Summary)

1. **Freeze both vision encoder and LM** -- prevents catastrophic forgetting, reduces compute
2. **Bridge via Perceiver Resampler** -- variable input to fixed 64 tokens, better than MLP/Transformer
3. **GATED XATTN-DENSE with tanh(0) init** -- preserves pretrained LM at initialization, improves stability
4. **Single-image cross-attention masking** -- better generalization to variable numbers of images
5. **Gradient accumulation across datasets** -- outperforms round-robin
6. **M3W interleaved data is essential** -- removing it causes the largest performance drop
7. **Squared ReLU** in newly trained layers outperforms GeLU
8. **Larger frozen LM = better few-shot** -- performance scales with LM size and number of shots
9. **Quality over quantity** for paired data (LTIP > ALIGN despite 6x smaller)
10. **Train with N=5 images, evaluate with N=32** -- architecture generalizes to more shots than training
