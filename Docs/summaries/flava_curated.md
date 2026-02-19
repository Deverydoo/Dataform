# FLAVA: A Foundational Language And Vision Alignment Model -- Technical Extraction

**Source:** Singh, Hu, Goswami, Couairon, Galuba, Rohrbach, Kiela (Facebook AI Research / FAIR)
**Paper ID:** arXiv:2112.04482v3, March 2022

---

## 1. ARCHITECTURE

### 1.1 Overall Structure

Three-component transformer architecture:

```
Input Image --> [Image Encoder] --> {h_I} --> [Linear Projection] --\
                                                                      --> [Multimodal Encoder] --> {h_M}
Input Text  --> [Text Encoder]  --> {h_T} --> [Linear Projection] --/
```

All three encoders use the ViT (Vision Transformer) architecture with pre-norm layer normalization (LayerNorm applied BEFORE multi-head attention, not after).

### 1.2 Image Encoder

- **Architecture:** ViT-B/16
- **Input:** Image resized to fixed size, split into non-overlapping patches
- **Patch size:** 16 x 16
- **Input image size (pretraining):** 224 x 224
- **Input image size (VQAv2 fine-tuning):** 480 x 480
- **Hidden size:** 768
- **Number of attention heads:** 12
- **Intermediate (FFN) size:** 3072
- **Number of layers:** 12
- **Dropout:** 0
- **Process:** Patches are linearly embedded, combined with positional embeddings and a classification token [CLS_I]. Output is a set of hidden state vectors {h_I} plus h_{CLS,I}.

### 1.3 Text Encoder

- **Architecture:** ViT-B/16 (identical architecture to image encoder, different parameters)
- **Hidden size:** 768
- **Number of attention heads:** 12
- **Intermediate (FFN) size:** 3072
- **Number of layers:** 12
- **Dropout:** 0
- **Vocabulary size:** 30,522 (BERT vocabulary)
- **Tokenizer:** BERT tokenizer (WordPiece)
- **Process:** Text tokenized and embedded into word vectors, processed through transformer. Output is {h_T} plus h_{CLS,T}.

Key difference from CLIP: FLAVA uses hidden size 768 (vs. CLIP's 512) and ViT architecture (vs. CLIP's GPT-style transformer) for the text encoder.

### 1.4 Multimodal Encoder

- **Architecture:** ViT-based transformer
- **Hidden size:** 768
- **Number of attention heads:** 12
- **Intermediate (FFN) size:** 3072
- **Number of layers:** 6 (half of unimodal encoders)
- **Dropout:** 0
- **Process:**
  1. Two learned linear projections applied to each hidden state in {h_I} and {h_T}
  2. Projected vectors concatenated into a single sequence
  3. Additional [CLS_M] token prepended
  4. Full cross-attention between projected image and text representations
- **Output:** {h_M} plus h_{CLS,M}

### 1.5 Total Model Configuration

| Component | Layers | Hidden | Heads | FFN | Dropout |
|-----------|--------|--------|-------|-----|---------|
| Image Encoder | 12 | 768 | 12 | 3072 | 0 |
| Text Encoder | 12 | 768 | 12 | 3072 | 0 |
| Multimodal Encoder | 6 | 768 | 12 | 3072 | 0 |

### 1.6 Downstream Task Application

- **Vision tasks:** Classifier head on h_{CLS,I} from image encoder
- **Language tasks:** Classifier head on h_{CLS,T} from text encoder
- **Multimodal tasks:** Classifier head on h_{CLS,M} from multimodal encoder
- **Multimodal fine-tuning head:** 2-layer MLP classifier, hidden dimension 1536, applied on h_{CLS,M}

---

## 2. TRAINING OBJECTIVES

### 2.1 Global Contrastive Loss (L_GC)

Applied on: Image-text pairs (multimodal data)
Operates on: h_{CLS,I} and h_{CLS,T} from unimodal encoders (NO masking applied)

```
Procedure:
1. Linearly project h_{CLS,I} and h_{CLS,T} into shared embedding space
   - Projection dimension: 512
2. L2-normalize the projected vectors
3. Compute dot-product similarities between all N image-text pairs in batch (N^2 pairs)
4. Apply softmax loss scaled by learnable temperature parameter
5. Maximize cosine similarity for matched pairs, minimize for unmatched pairs
```

**Critical implementation detail -- Global vs. Local backpropagation:**
- Standard CLIP (open-source) only backpropagates contrastive gradients to embeddings on the local GPU
- FLAVA performs FULL backpropagation across ALL GPU workers through the gathering operation
- This "global contrastive" approach yields +1.65% macro average improvement over "local contrastive" with minor additional compute overhead

### 2.2 Masked Multimodal Modeling (L_MMM) -- Novel Objective

Applied on: Image-text pairs (multimodal data)
Operates on: Output {h_M} from multimodal encoder

```
Procedure:
1. Tokenize input image patches using pretrained dVAE tokenizer (from DALL-E)
   - dVAE codebook size: 8192
   - Maps each image patch to an index in visual codebook
2. Mask rectangular block regions of image patches (following BEiT masking strategy)
3. Mask 15% of text tokens (following BERT masking strategy)
4. Replace masked tokens with [MASK] token
5. Feed masked inputs through image encoder -> text encoder -> multimodal encoder
6. Apply MLP on multimodal encoder outputs {h_M} to predict:
   - Visual codebook index for masked image patches
   - Word vocabulary index for masked text tokens
```

This extends multimodal masked language modeling by jointly masking BOTH modalities simultaneously. L_GC is applied on UNMASKED inputs forwarded separately through the unimodal encoders (separate forward pass from MMM).

### 2.3 Image-Text Matching (L_ITM)

Applied on: Image-text pairs (multimodal data)
Operates on: h_{CLS,M} from multimodal encoder

```
Procedure:
1. Batch contains both matched and unmatched image-text pairs
2. Binary classifier on h_{CLS,M} predicts whether image and text match
```

### 2.4 Masked Image Modeling (L_MIM)

Applied on: Unimodal image data (unpaired images)
Operates on: {h_I} from image encoder

```
Procedure:
1. Tokenize image with pretrained dVAE tokenizer (same as MMM)
2. Mask rectangular block regions following BEiT strategy
3. Predict dVAE token indices of masked patches using classifier on {h_I}
```

### 2.5 Masked Language Modeling (L_MLM)

Applied on: Unimodal text data (unpaired text)
Operates on: {h_T} from text encoder

```
Procedure:
1. Mask 15% of text tokens (BERT-style)
2. Predict original token indices using classifier on {h_T}
```

### 2.6 Loss Application Summary

| Loss | Data Type | Encoder Used | What is Predicted |
|------|-----------|-------------|-------------------|
| L_GC | Image-text pairs | Image + Text encoders (unmasked) | Contrastive matching scores |
| L_MMM | Image-text pairs | Image + Text + Multimodal encoders (masked) | dVAE token indices + word indices |
| L_ITM | Image-text pairs | Image + Text + Multimodal encoders | Binary match/no-match |
| L_MIM | Unpaired images | Image encoder (masked) | dVAE token indices |
| L_MLM | Unpaired text | Text encoder (masked) | Word vocabulary indices |

---

## 3. PRETRAINING PROCEDURE

### 3.1 Encoder Initialization Strategy

**Stage 1 -- Unimodal pretraining:**
- **Image encoder:** Initialize from DINO self-supervised ViT-B/16 pretrained on ImageNet-1K
  - DINO initialization empirically outperforms BEiT initialization
  - Despite switching to MIM objective post-initialization, DINO gives better final performance
- **Text encoder:** Pretrain ViT-based text encoder with MLM loss on CCNews + BookCorpus
  - 125K iterations, batch size 2048, learning rate 5e-4
  - Follows RoBERTa-base hyperparameters
- **Multimodal encoder:** Always initialized randomly

**Stage 2 -- Joint unimodal + multimodal training:**
- Round-robin sampling from three dataset types per iteration
- Each iteration: sample one dataset type, obtain full batch, apply corresponding loss(es)

### 3.2 Dataset Sampling Probabilities

| Dataset | Sampling Probability | Losses Applied |
|---------|---------------------|----------------|
| PMD (multimodal) | 0.70 | L_GC + L_MMM + L_ITM |
| ImageNet-1K (unimodal images) | 0.15 | L_MIM |
| CCNews + BookCorpus (unimodal text) | 0.15 | L_MLM |

### 3.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 8,192 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| AdamW beta_1 | 0.9 |
| AdamW beta_2 | 0.999 |
| Weight decay | 0.1 |
| Warmup iterations | 10,000 |
| Learning schedule | Warmup + cosine decay |
| PMD total iterations | 150,000 |
| Precision | Full FP16 (except LayerNorm) |
| Parallelism | Fully-Sharded Data Parallel (FSDP) |

### 3.4 Checkpoint Selection

- Monitor zero-shot ImageNet classification accuracy every 8K updates
- Select best checkpoint based on ImageNet-1K zero-shot accuracy
- Zero-shot accuracy computed using CLIP's text template methodology

---

## 4. PRETRAINING DATA

### 4.1 Public Multimodal Datasets (PMD) -- 70M pairs total

| Dataset | Image-Text Pairs | Avg. Text Length (words) |
|---------|-----------------|------------------------|
| COCO | 0.9M | 12.4 |
| SBU Captions | 1.0M | 12.1 |
| Localized Narratives | 1.9M | 13.8 |
| Conceptual Captions | 3.1M | 10.3 |
| Visual Genome | 5.4M | 5.1 |
| Wikipedia Image Text (WIT) | 4.8M | 12.8 |
| Conceptual Captions 12M | 11.0M | 17.3 |
| RedCaps | 11.6M | 9.5 |
| YFCC100M (filtered) | 30.3M | 12.7 |
| **Total** | **70M** | **12.1** |

- 68M unique images across 70M pairs
- YFCC100M filtering: discard non-English captions, keep only captions with >2 words
- All datasets are publicly available (unlike CLIP's 400M or ALIGN's 1.8B)

### 4.2 Unimodal Data

- **Images:** ImageNet-1K
- **Text:** CCNews + BookCorpus

---

## 5. FINE-TUNING CONFIGURATIONS

### 5.1 Vision Tasks (22 tasks)

- **Method:** Linear probing (logistic regression with L-BFGS)
- **Features:** Extracted from final layer of image encoder (before multimodal encoder)
- **Iterations:** 1,000
- **Lambda sweep:** 1e-6 to 1e6

### 5.2 NLP Tasks (8 GLUE tasks)

- **Method:** End-to-end fine-tuning of text encoder
- **Head:** Classification head on text encoder (regression head for STS-B)
- **Hyperparameters:** RoBERTa fine-tuning defaults

### 5.3 Multimodal Tasks (VQAv2, SNLI-VE, Hateful Memes)

| Parameter | VQAv2 | SNLI-VE / Hateful Memes |
|-----------|-------|------------------------|
| Learning rate | 1e-4 | 1e-5 |
| Total updates | 44,000 | 24,000 |
| Image size | 480 x 480 | 224 x 224 |
| Batch size | 256 | 256 |
| Weight decay | 1e-2 | 1e-2 |
| Warmup iterations | 2,000 | 2,000 |
| Schedule | Cosine decay | Cosine decay |
| Classifier | 2-layer MLP (hidden=1536) on h_{CLS,M} | 2-layer MLP (hidden=1536) on h_{CLS,M} |

### 5.4 Zero-Shot Retrieval

- Uses cosine similarities from global contrastive loss
- Retrieves text/image with highest matching score to query

---

## 6. RESULTS

### 6.1 Ablation Study -- Average Scores

| Setting | Vision Avg. | NLP Avg. | Multimodal Avg. | Macro Avg. |
|---------|-------------|----------|-----------------|------------|
| MIM only | 57.46 | -- | -- | 19.15 |
| MLM only | -- | 71.55 | -- | 23.85 |
| FLAVA_C (contrastive only) | 64.80 (note: Table 3 = 79.14) | 79.14 (note: Table 3 = 64.80) | 66.25 | 70.06 |
| FLAVA_MM (all multimodal losses) | 74.22 (note: Table 3 = 79.35) | 79.35 (note: Table 3 = 74.22) | 69.11 | 74.23 |
| FLAVA w/o unimodal init | 75.55 (note: Table 3 = 78.29) | 78.29 (note: Table 3 = 75.55) | 67.32 | 73.72 |
| **FLAVA (full)** | **78.19 (note: Table 3 = 79.44)** | **79.44 (note: Table 3 = 78.19)** | **69.92** | **75.85** |

Note: In the paper Table 3, the Vision and NLP columns for FLAVA_C/FLAVA_MM/FLAVA appear transposed relative to Table 4. Using Table 4 as ground truth:

| Setting | Vision Avg. | NLP Avg. | Multimodal Avg. | Macro Avg. |
|---------|-------------|----------|-----------------|------------|
| MIM only | 57.46 | -- | -- | 19.15 |
| MLM only | -- | 71.55 | -- | 23.85 |
| FLAVA_C (contrastive only, PMD) | 79.14 | 64.80 | 66.25 | 70.06 |
| FLAVA_MM (multimodal losses, PMD) | 79.35 | 74.22 | 69.11 | 74.23 |
| FLAVA w/o unimodal init | 78.29 | 75.55 | 67.32 | 73.72 |
| **FLAVA (full)** | **79.44** | **78.19** | **69.92** | **75.85** |

### 6.2 Incremental Ablation Effects

- Adding MMM + ITM to contrastive-only (FLAVA_C -> FLAVA_MM):
  - Multimodal avg: +2.86%
  - NLP avg: +9.42%
  - Vision avg: +0.21%
- Adding unimodal data without init (FLAVA_MM -> FLAVA w/o init):
  - NLP avg: +1.33%
  - Vision avg: -1.06%
  - Multimodal avg: -1.79%
- Adding unimodal pretraining (FLAVA w/o init -> FLAVA full):
  - All tasks improve; macro avg: +2.13%

### 6.3 Key Task Results -- FLAVA (full model)

**NLP Tasks (fine-tuning):**

| Task | FLAVA | BERT_base | CLIP ViT-B/16 (400M) |
|------|-------|-----------|---------------------|
| MNLI | 80.33 | 84.4 | 33.52 |
| CoLA (MCC) | 50.65 | 54.6 | 25.37 |
| MRPC (Acc/F1) | 81.4/86.9 | 81.9/87.6 | 74.9/65.0 |
| QQP (Acc/F1) | 90.4/87.2 | 90.6/87.4 | 76.8/53.9 |
| SST-2 | 90.94 | 92.5 | 88.19 |
| QNLI | 87.31 | 91.0 | 50.54 |
| RTE | 57.76 | 62.5 | 55.23 |
| STS-B (PCC) | 85.67 | 88.1 | 15.98 |

**Vision Tasks (linear probing):**

| Task | FLAVA | CLIP ViT-B/16 (PMD) | CLIP ViT-B/16 (400M) |
|------|-------|--------------------|--------------------|
| ImageNet | 75.54 | 72.95 | 80.20 |
| Food101 | 88.51 | 85.49 | 91.56 |
| CIFAR10 | 92.87 | 91.25 | 94.93 |
| CIFAR100 | 77.68 | 74.40 | 81.10 |
| Cars | 70.87 | 62.84 | 85.92 |
| Aircraft | 47.31 | 40.02 | 51.40 |
| DTD | 77.29 | 73.40 | 78.46 |
| Pets | 84.82 | 79.61 | 91.66 |
| Caltech101 | 95.74 | 93.76 | 95.51 |
| Flowers102 | 96.37 | 94.94 | 97.12 |
| MNIST | 98.42 | 97.38 | 99.01 |
| STL10 | 98.89 | 97.29 | 99.09 |
| EuroSAT | 97.26 | 95.70 | 95.38 |
| GTSRB | 79.46 | 76.34 | 88.61 |
| KITTI | 89.04 | 84.89 | 86.56 |
| PCAM | 85.31 | 83.99 | 83.72 |
| UCF101 | 83.32 | 77.85 | 85.17 |
| CLEVR | 79.66 | 73.64 | 75.89 |
| FER2013 | 61.12 | 57.04 | 68.36 |
| SUN397 | 82.17 | 79.96 | 82.05 |
| SST (image) | 57.11 | 56.84 | 74.68 |
| Country211 | 28.92 | 25.12 | 30.10 |
| **Vision Avg.** | **79.44** | **76.12** | **82.57** |

**Multimodal Tasks:**

| Task | FLAVA | CLIP (PMD) | CLIP (400M) |
|------|-------|-----------|------------|
| VQAv2 (test-dev) | 72.49 | 59.81 | 54.83 |
| SNLI-VE | 78.89 | 73.53 | 74.27 |
| Hateful Memes (AUROC) | 76.09 | 56.59 | 63.93 |
| Flickr30K TR R@1 (zero-shot) | 67.70 | 60.90 | 82.20 |
| Flickr30K TR R@5 (zero-shot) | 94.00 | 88.90 | 96.60 |
| Flickr30K IR R@1 (zero-shot) | 65.22 | 56.48 | 62.08 |
| Flickr30K IR R@5 (zero-shot) | 89.38 | 83.60 | 85.68 |
| COCO TR R@1 (zero-shot) | 42.74 | 37.12 | 52.48 |
| COCO TR R@5 (zero-shot) | 76.76 | 69.48 | 76.68 |
| COCO IR R@1 (zero-shot) | 38.38 | 33.29 | 33.07 |
| COCO IR R@5 (zero-shot) | 67.47 | 62.47 | 58.37 |
| **Multimodal Avg.** | **69.92** | **62.02** | **67.29** |

### 6.4 FLAVA vs. State-of-the-Art Multimodal Models (public data)

| Model | VQAv2 | SNLI-VE | Hateful Memes |
|-------|-------|---------|---------------|
| VisualBERT | 70.8 | 77.3 | 74.1 |
| UNITER_base | 72.7 | 78.3 | -- |
| VL-BERT_base | 71.2 | -- | -- |
| ViLBERT | 70.6 | 75.7 | 74.1 |
| LXMERT | 72.4 | -- | -- |
| UniT | 67.0 | 73.1 | -- |
| **FLAVA** | **72.8** | **79.0** | **76.7** |

### 6.5 FLAVA vs. CLIP -- Direct Comparison

When trained on same PMD data with same ViT-B/16 architecture:
- FLAVA outperforms CLIP on ALL vision, NLP, and multimodal domains
- Architectural optimizations alone (768 hidden, ViT text encoder, BERT tokenizer) contribute:
  - Vision: +0.00%
  - NLP: +16.55%
  - Multimodal: +4.40%
  - Macro: +6.99%

Against CLIP pretrained on 400M (5.7x more data):
- FLAVA significantly better on NLP (+27.69% macro NLP avg)
- FLAVA significantly better on multimodal (+2.63% macro multimodal avg)
- CLIP better on most vision-only tasks (82.57 vs. 79.44 vision avg)

### 6.6 DINO vs. BEiT Initialization

Under full FLAVA pretraining (with unimodal init):

| Metric | BEiT Init | DINO Init |
|--------|-----------|-----------|
| Vision Avg. | 78.65 | 79.44 |
| NLP Avg. | 77.40 | 78.19 |
| Multimodal Avg. | 69.23 | 69.92 |
| Macro Avg. | 78.03 | 78.82 (note: from extended Table C.1) |

DINO initialization is consistently better across all three domains.

---

## 7. KEY DESIGN DECISIONS

1. **Shared ViT architecture across encoders:** Image encoder, text encoder, and multimodal encoder all use ViT-based transformers. Text encoder uses identical architecture (not GPT-style) to image encoder, differing only in parameters.

2. **Pre-norm LayerNorm:** ViT architecture applies LayerNorm before multi-head attention rather than after. This provides more robust learning under large learning rates compared to BERT's post-norm architecture.

3. **dVAE tokenizer for vision:** Both MIM and MMM losses use a pretrained dVAE tokenizer (from DALL-E / same as BEiT) to convert image patches into discrete tokens. Codebook size = 8192.

4. **Separate forward passes for contrastive vs. masked objectives:** Global contrastive loss operates on UNMASKED inputs through unimodal encoders. MMM operates on MASKED inputs through the full pipeline. These are separate forward passes within the same training iteration on multimodal data.

5. **Curriculum via staged initialization:** Training the whole model from scratch with all losses simultaneously is suboptimal (macro avg 73.72 vs. 75.85). Pre-initializing unimodal encoders before joint training provides a natural curriculum.

6. **FP16 everywhere except LayerNorm:** Full FP16 precision with LayerNorm in FP32 for numerical stability.

7. **FSDP parallelism:** Fully-Sharded Data Parallel for memory-efficient multi-GPU training.

---

## 8. EQUATIONS (Implicit)

The paper does not provide explicit mathematical notation for loss functions. The losses are defined procedurally:

**Global Contrastive Loss (L_GC):**
```
z_I = normalize(W_I * h_{CLS,I})    # project + L2-norm image CLS
z_T = normalize(W_T * h_{CLS,T})    # project + L2-norm text CLS
S = z_I @ z_T^T / tau               # N x N similarity matrix scaled by temperature
L_GC = (CrossEntropy(S, labels_row) + CrossEntropy(S^T, labels_col)) / 2
```
Where labels are the diagonal (matched pairs) and tau is learnable temperature.

**Masked Image Modeling (L_MIM):**
```
tokens = dVAE(image)                 # discrete visual tokens per patch
h_I = ImageEncoder(mask(image))      # encode with masked patches
L_MIM = CrossEntropy(MLP(h_I[masked_positions]), tokens[masked_positions])
```

**Masked Language Modeling (L_MLM):**
```
h_T = TextEncoder(mask(text, p=0.15))
L_MLM = CrossEntropy(MLP(h_T[masked_positions]), original_tokens[masked_positions])
```

**Masked Multimodal Modeling (L_MMM):**
```
tokens_I = dVAE(image)
h_M = MultimodalEncoder(ImageEncoder(mask(image)), TextEncoder(mask(text)))
L_MMM = CrossEntropy(MLP(h_M[masked_image_pos]), tokens_I[masked_image_pos])
       + CrossEntropy(MLP(h_M[masked_text_pos]), original_text_tokens[masked_text_pos])
```

**Image-Text Matching (L_ITM):**
```
h_M = MultimodalEncoder(ImageEncoder(image), TextEncoder(text))
L_ITM = BinaryCrossEntropy(Classifier(h_{CLS,M}), match_label)
```

**Joint training iteration (pseudocode):**
```
for each iteration:
    dataset = sample({PMD: 0.70, IN-1K: 0.15, CCNews+BC: 0.15})
    if dataset == PMD:
        loss = L_GC(unmasked) + L_MMM(masked) + L_ITM(unmasked or matched/unmatched)
    elif dataset == IN-1K:
        loss = L_MIM
    elif dataset == CCNews+BC:
        loss = L_MLM
    loss.backward()
    optimizer.step()
```

---

## 9. IMPLEMENTATION

- **Framework:** MMF (Meta) + fairseq
- **Parallelism:** FSDP (Fully-Sharded Data Parallel)
- **Training precision:** FP16 (LayerNorm in FP32)
- **Vision linear probing:** L-BFGS logistic regression (scikit-learn)
- **Code:** https://flava-model.github.io/
