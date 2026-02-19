# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

**Source:** Li, Li, Savarese, Hoi (Salesforce Research, 2023)
**arXiv:** 2301.12597v3

---

## 1. Core Idea

Bridge frozen pre-trained image encoders and frozen large language models using a lightweight Querying Transformer (Q-Former) trained in two stages. Only the Q-Former (188M parameters) is trainable. Everything else stays frozen.

---

## 2. Architecture: Q-Former

### 2.1 Structure

Q-Former consists of two transformer submodules that **share the same self-attention layers**:

1. **Image Transformer** -- interacts with frozen image encoder via cross-attention
2. **Text Transformer** -- functions as both text encoder and text decoder

Key design details:
- Cross-attention layers inserted **every other transformer block**
- Initialized from **BERT_base** pre-trained weights
- Cross-attention layers are **randomly initialized**
- Total parameters: **188M** (including learned queries)

### 2.2 Learned Queries

- **32 learnable query embeddings**
- Each query dimension: **768** (matches Q-Former hidden dimension)
- Output query representation Z has shape: **32 x 768**
- Queries are treated as model parameters
- Queries interact with each other via self-attention
- Queries interact with frozen image features via cross-attention
- Queries can interact with text via shared self-attention layers

### 2.3 Information Bottleneck

Output Z (32 x 768) is **much smaller** than frozen image features:
- ViT-L/14 output: 257 x 1024
- ViT-g/14 output: larger still

This bottleneck forces queries to extract only the visual information most relevant to the text.

### 2.4 Frozen Components

| Component | Role | Status |
|-----------|------|--------|
| Image Encoder (ViT-L/14 or ViT-g/14) | Visual feature extraction | **Frozen** |
| LLM (OPT or FlanT5) | Language generation | **Frozen** |
| Q-Former | Bridge between modalities | **Trainable** |
| FC projection layer | Dimension adaptation to LLM | **Trainable** |

Image encoder detail: Remove the **last layer** of the ViT; use **second-to-last layer** output features (slightly better performance).

---

## 3. Two-Stage Pre-training

### 3.1 Stage 1: Vision-Language Representation Learning

**Goal:** Train Q-Former to extract visual representation most relevant to text, using a frozen image encoder.

Three jointly optimized objectives, each with a **different self-attention masking strategy**:

#### 3.1.1 Image-Text Contrastive Learning (ITC)

- Aligns image representation and text representation by maximizing mutual information
- Contrasts positive image-text pairs against negative pairs
- Image side: output query representation Z from image transformer
- Text side: output embedding t of [CLS] token from text transformer
- Similarity computation: pairwise similarity between each query output and t; **select the highest** as image-text similarity
- **Attention mask:** Uni-modal self-attention -- queries and text **cannot see each other**
- Uses **in-batch negatives** (not momentum queue, since frozen encoder allows larger batch sizes)

#### 3.1.2 Image-Text Matching (ITM)

- Binary classification: positive (matched) vs. negative (unmatched) image-text pair
- **Attention mask:** Bi-directional self-attention -- all queries and text tokens **can attend to each other**
- Output query embeddings Z capture multimodal information
- Each output query embedding fed into a **two-class linear classifier** to get a logit
- Final matching score: **average of logits across all queries**
- Uses **hard negative mining** strategy (from Li et al., 2021; 2022)

#### 3.1.3 Image-Grounded Text Generation (ITG)

- Generates text conditioned on image input
- Information must flow: frozen image encoder -> queries (via cross-attention) -> text tokens (via self-attention)
- Forces queries to capture **all** information about the text from the image
- **Attention mask:** Multimodal causal self-attention (similar to UniLM):
  - Queries can attend to each other but **NOT** to text tokens
  - Each text token can attend to **all queries** and its **previous text tokens**
- Replaces [CLS] token with **[DEC]** token as first text token to signal decoding

#### Self-Attention Mask Summary (Q = query positions, T = text positions)

| Objective | Q -> Q | Q -> T | T -> Q | T -> T |
|-----------|--------|--------|--------|--------|
| ITC (Uni-modal) | Yes | **No** | **No** | Yes |
| ITM (Bi-directional) | Yes | Yes | Yes | Yes |
| ITG (Multimodal Causal) | Yes | **No** | Yes | Causal (left-to-right) |

### 3.2 Stage 2: Vision-to-Language Generative Learning

**Goal:** Connect Q-Former output to frozen LLM to harvest generative language capability.

Architecture addition:
- **Fully-connected (FC) layer** linearly projects output query embeddings Z from Q-Former dimension to LLM text embedding dimension
- Projected query embeddings are **prepended** to input text embeddings
- They function as **soft visual prompts** conditioning the LLM on visual representation

#### Decoder-based LLMs (OPT)

- Loss: **Language modeling loss**
- Frozen LLM generates text conditioned on visual representation from Q-Former

#### Encoder-decoder-based LLMs (FlanT5)

- Loss: **Prefix language modeling loss**
- Text split into two parts:
  - **Prefix text:** concatenated with visual representation as input to LLM encoder
  - **Suffix text:** generation target for LLM decoder

---

## 4. Loss Functions

### Stage 1 (all three optimized jointly)

```
L_stage1 = L_ITC + L_ITM + L_ITG
```

**L_ITC (Image-Text Contrastive):**
- Contrastive loss over image-text similarity
- sim(I, T) = max_i (q_i^T * t) for i in {1, ..., 32}
- Where q_i = i-th query output, t = [CLS] output embedding
- In-batch negatives

**L_ITM (Image-Text Matching):**
- Binary cross-entropy
- score = mean_i(linear_classifier(z_i)) for i in {1, ..., 32}
- Hard negative mining for informative negatives

**L_ITG (Image-Grounded Text Generation):**
- Autoregressive language modeling loss (cross-entropy over tokens)
- Conditioned on image features extracted by queries

### Stage 2

**For decoder-based LLM (OPT):**
```
L_stage2 = L_LM = -sum_t log P(y_t | y_{<t}, Z_proj)
```

**For encoder-decoder LLM (FlanT5):**
```
L_stage2 = L_prefix_LM = -sum_t log P(y_t^suffix | y_{<t}^suffix, [Z_proj; y^prefix])
```

Where Z_proj = FC(Z), the linearly projected query embeddings.

---

## 5. Pre-training Configuration

### Data

- **129M images total**
- Sources: COCO, Visual Genome, CC3M, CC12M, SBU, 115M images from LAION-400M
- Synthetic captions via CapFilt: generate 10 captions with BLIP_large, rank by CLIP ViT-L/14 similarity, keep top-2 per image, randomly sample 1 per step

### Image Encoders (Frozen)

| Encoder | Source |
|---------|--------|
| ViT-L/14 | CLIP |
| ViT-g/14 | EVA-CLIP |

### LLMs (Frozen)

| LLM | Type | Source |
|-----|------|--------|
| OPT-2.7B | Decoder | Unsupervised pre-trained |
| OPT-6.7B | Decoder | Unsupervised pre-trained |
| FlanT5-XL | Encoder-Decoder | Instruction-tuned |
| FlanT5-XXL | Encoder-Decoder | Instruction-tuned |

### Pre-training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Stage 1 steps | 250k |
| Stage 2 steps | 80k |
| Stage 1 batch size (ViT-L) | 2320 |
| Stage 1 batch size (ViT-g) | 1680 |
| Stage 2 batch size (OPT) | 1920 |
| Stage 2 batch size (FlanT5) | 1520 |
| Optimizer | AdamW |
| beta_1 | 0.9 |
| beta_2 | 0.98 |
| Weight decay | 0.05 |
| LR schedule | Cosine decay |
| Peak learning rate | 1e-4 |
| Linear warmup steps | 2000 |
| Min LR (Stage 2) | 5e-5 |
| Image size | 224 x 224 |
| Augmentation | Random resized crop + horizontal flip |
| Frozen model precision | FP16 (BFloat16 for FlanT5) |

### Compute

- Hardware: single 16-A100 (40GB) machine
- Largest model (ViT-g + FlanT5-XXL): < 6 days (Stage 1) + < 3 days (Stage 2)

---

## 6. Fine-tuning Hyperparameters

### COCO Captioning (ViT-g)

| Parameter | FlanT5-XL | OPT-2.7B | OPT-6.7B |
|-----------|-----------|----------|----------|
| Epochs | 5 | 5 | 5 |
| Warmup steps | 1000 | 1000 | 1000 |
| Learning rate | 1e-5 | 1e-5 | 1e-5 |
| Batch size | 256 | 256 | 256 |
| AdamW beta | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) |
| Weight decay | 0.05 | 0.05 | 0.05 |
| Drop path | 0 | 0 | 0 |
| Image resolution | 364 | 364 | 364 |
| Prompt | "a photo of" | "a photo of" | "a photo of" |
| Beam size | 5 | 5 | 5 |
| ViT layer-wise LR decay | 1 | 1 | 0.95 |

### VQA (ViT-g)

| Parameter | FlanT5-XL | OPT-2.7B | OPT-6.7B |
|-----------|-----------|----------|----------|
| Epochs | 5 | 5 | 5 |
| Warmup steps | 1000 | 1000 | 1000 |
| Learning rate | 1e-5 | 1e-5 | 1e-5 |
| Batch size | 128 | 128 | 128 |
| AdamW beta | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) |
| Weight decay | 0.05 | 0.05 | 0.05 |
| Drop path | 0 | 0 | 0 |
| Image resolution | 490 | 490 | 490 |
| Prompt | "Question: {} Answer:" | "Question: {} Answer:" | "Question: {} Answer:" |
| Beam size | 5 | 5 | 5 |
| ViT layer-wise LR decay | 0.95 | 0.95 | 0.9 |

### Image-Text Retrieval (COCO)

| Parameter | ViT-L/14 | ViT-g/14 |
|-----------|----------|----------|
| Epochs | 5 | 5 |
| Warmup steps | 1000 | 1000 |
| Learning rate | 5e-6 | 1e-5 |
| Batch size | 224 | 224 |
| AdamW beta | (0.9, 0.98) | (0.9, 0.999) |
| Weight decay | 0.05 | 0.05 |
| Drop path | 0 | 0 |
| Image resolution | 364 | 364 |
| ViT layer-wise LR decay | 1 | 0.95 |

### VQA Fine-tuning Details

- Q-Former receives the question as additional input (question tokens interact with queries via self-attention)
- This guides cross-attention to focus on more informative image regions
- LLM receives Q-Former output + question, generates answer
- VQA data: VQAv2 train+val splits + Visual Genome training samples
- LLM stays frozen; Q-Former + image encoder updated

### Retrieval Inference

- Two-stage retrieval: first select k=128 candidates by image-text feature similarity, then re-rank by pairwise ITM scores
- No LLM involved (uses Stage 1 model directly)

---

## 7. Trainable Parameter Counts

| Configuration | Trainable Params | Total Params |
|---------------|-----------------|--------------|
| Q-Former alone (pre-training) | 188M | -- |
| BLIP-2 ViT-L OPT-2.7B | 104M | 3.1B |
| BLIP-2 ViT-g OPT-2.7B | 107M | 3.8B |
| BLIP-2 ViT-g OPT-6.7B | 108M | 7.8B |
| BLIP-2 ViT-L FlanT5-XL | 103M | 3.4B |
| BLIP-2 ViT-g FlanT5-XL | 107M | 4.1B |
| BLIP-2 ViT-g FlanT5-XXL | 108M | 12.1B |
| Flamingo-80B (comparison) | 10.2B | 80B |

---

## 8. Benchmark Results

### 8.1 Zero-Shot VQA

| Model | Trainable | VQAv2 (test-dev) | OK-VQA | GQA (test-dev) |
|-------|-----------|-------------------|--------|----------------|
| Frozen | 40M | 29.6 | 5.9 | -- |
| Flamingo-3B | 1.4B | 49.2 | 41.2 | -- |
| Flamingo-9B | 1.8B | 51.8 | 44.7 | -- |
| Flamingo-80B | 10.2B | 56.3 | 50.6 | -- |
| **BLIP-2 ViT-L OPT-2.7B** | 104M | 49.7 | 30.2 | 33.9 |
| **BLIP-2 ViT-g OPT-2.7B** | 107M | 52.3 | 31.7 | 34.6 |
| **BLIP-2 ViT-g OPT-6.7B** | 108M | 52.6 | 36.4 | 36.4 |
| **BLIP-2 ViT-L FlanT5-XL** | 103M | 62.3 | 39.4 | 44.4 |
| **BLIP-2 ViT-g FlanT5-XL** | 107M | 63.0 | 40.7 | 44.2 |
| **BLIP-2 ViT-g FlanT5-XXL** | 108M | **65.0** | **45.9** | **44.7** |

Key result: BLIP-2 (ViT-g + FlanT5-XXL) outperforms Flamingo-80B by **8.7%** on VQAv2 with **54x fewer** trainable parameters.

### 8.2 Zero-Shot Image Captioning (NoCaps val)

| Model | Trainable | CIDEr | SPICE |
|-------|-----------|-------|-------|
| BLIP | 583M | 113.2 | 14.8 |
| SimVLM | ~1.4B | 112.2 | -- |
| **BLIP-2 (ViT-g + FlanT5-XL)** | 188M | **121.6** | **15.8** |

### 8.3 Fine-tuned Image Captioning

#### NoCaps (val, zero-shot transfer from COCO fine-tune)

| Model | Overall CIDEr | Overall SPICE |
|-------|--------------|---------------|
| BLIP | 113.2 | 14.8 |
| SimVLM | 112.2 | -- |
| **BLIP-2 ViT-g OPT-2.7B** | 119.7 | 15.4 |
| **BLIP-2 ViT-g OPT-6.7B** | 121.0 | 15.3 |
| **BLIP-2 ViT-g FlanT5-XL** | **121.6** | **15.8** |

#### COCO Captioning (Karpathy test)

| Model | B@4 | CIDEr |
|-------|-----|-------|
| OFA | 43.9 | 145.3 |
| Flamingo | -- | 138.1 |
| SimVLM | 40.6 | 143.3 |
| **BLIP-2 ViT-g OPT-2.7B** | 43.7 | **145.8** |
| BLIP-2 ViT-g OPT-6.7B | 43.5 | 145.2 |
| BLIP-2 ViT-g FlanT5-XL | 42.4 | 144.5 |

### 8.4 Fine-tuned VQA (VQAv2)

| Model | Type | Trainable | test-dev | test-std |
|-------|------|-----------|----------|----------|
| BLIP | Open-ended | 385M | 78.25 | 78.32 |
| OFA | Open-ended | 930M | 82.00 | 82.00 |
| Flamingo-80B | Open-ended | 10.6B | 82.00 | 82.10 |
| **BLIP-2 ViT-g OPT-6.7B** | Open-ended | 1.2B | **82.19** | **82.30** |
| BLIP-2 ViT-g OPT-2.7B | Open-ended | 1.2B | 81.59 | 81.74 |
| BLIP-2 ViT-g FlanT5-XL | Open-ended | 1.2B | 81.55 | 81.66 |
| BEIT-3 | Closed-ended | 1.9B | 84.19 | 84.03 |

### 8.5 Image-Text Retrieval

#### Flickr30K Zero-shot (1K test)

| Model | TR@1 | IR@1 |
|-------|------|------|
| CLIP | 88.0 | 68.7 |
| ALIGN | 88.6 | 75.7 |
| BEIT-3 | 94.9 | 81.5 |
| BLIP | 96.7 | 86.7 |
| **BLIP-2 ViT-L** | 96.9 | 88.6 |
| **BLIP-2 ViT-g** | **97.6** | **89.7** |

#### COCO Fine-tuned (5K test)

| Model | TR@1 | IR@1 |
|-------|------|------|
| BEIT-3 | 84.8 | 67.2 |
| BLIP | 82.4 | 65.1 |
| **BLIP-2 ViT-g** | **85.4** | **68.3** |

### 8.6 Effect of ITG on Retrieval

| Finetuning Objectives | TR@1 | IR@1 |
|-----------------------|------|------|
| ITC + ITM | 84.5 | 67.2 |
| ITC + ITM + ITG | **85.4** | **68.3** |

ITG improves retrieval by forcing queries to extract language-relevant visual features.

---

## 9. Ablation: Stage 1 Representation Learning

Without Stage 1 representation learning, performance on zero-shot VQA drops substantially:
- OPT models suffer **catastrophic forgetting** (performance degrades as training proceeds)
- FlanT5 models show significantly lower convergence
- Stage 1 is critical for bridging the modality gap before generative learning

---

## 10. Scaling Observations

Three validated scaling axes:

1. **Stronger image encoder helps:** ViT-g > ViT-L for both OPT and FlanT5
2. **Larger LLM helps:** Within same family, larger models outperform smaller ones
3. **Instruction-tuned LLM helps:** FlanT5 (instruction-tuned) > OPT (unsupervised) on VQA

---

## 11. Key Comparisons with Prior Work

### vs. Flamingo

| Property | Flamingo-80B | BLIP-2 (best) |
|----------|-------------|---------------|
| Trainable params | 10.2B | 108M |
| Total params | 80B | 12.1B |
| Zero-shot VQAv2 | 56.3 | **65.0** |
| Approach to alignment | New cross-attention in LLM + LM loss only | Q-Former with ITC+ITM+ITG then LM loss |
| In-context learning | Yes (uses M3W interleaved data) | No (single image-text pairs only) |

### vs. Frozen (Tsimpoukelli et al., 2021)

| Property | Frozen | BLIP-2 |
|----------|--------|--------|
| Image encoder | Finetuned | Frozen |
| LLM | Frozen | Frozen |
| Bridge mechanism | Image encoder outputs as soft prompts | Q-Former with two-stage pre-training |
| Zero-shot VQAv2 | 29.6 | **65.0** |

### vs. CLIP

BLIP-2 uses CLIP's ViT-L/14 as one of its frozen image encoders but achieves higher retrieval performance through the Q-Former's additional alignment training.

---

## 12. Inference Prompts

| Task | LLM Type | Prompt |
|------|----------|--------|
| Zero-shot VQA | OPT | "Question: {} Answer:" |
| Zero-shot VQA | FlanT5 | "Question: {} Short answer:" |
| Image Captioning | All | "a photo of" |
| VQA Fine-tune | All | "Question: {} Answer:" |

Generation settings: beam search with beam width 5, length penalty -1 (encourages shorter answers).

---

## 13. Limitations

1. **No in-context learning:** Pre-training uses single image-text pairs, so the model cannot learn correlations among multiple image-text pairs in a sequence (unlike Flamingo with M3W data).
2. **LLM knowledge errors:** Inaccurate knowledge, incorrect reasoning paths, outdated information about new visual content.
3. **Inherited LLM risks:** Offensive language, social bias, privacy leakage from frozen LLM.

---

## 14. Data Flow Summary

```
Stage 1 (Representation Learning):
  Image -> [Frozen ViT] -> image features (257x1024 or larger)
                                |
                          cross-attention
                                |
  Learned Queries (32x768) -> [Q-Former] <- Text tokens
                                |
                    Z (32x768) output
                                |
               ITC loss + ITM loss + ITG loss

Stage 2 (Generative Learning):
  Image -> [Frozen ViT] -> [Q-Former from Stage 1] -> Z (32x768)
                                                          |
                                                    [FC Layer]
                                                          |
                                                Z_proj (32 x LLM_dim)
                                                          |
                                              prepended as soft visual prompts
                                                          |
                                          [Frozen LLM] -> generated text
                                                          |
                                                    LM loss / Prefix LM loss
```
