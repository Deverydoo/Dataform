# ALIGN: A Large-scale ImaGe and Noisy-text Embedding
**Source:** ICML 2021 (Jia et al., Google Research)

---

## Core Idea

A dual-encoder model trained on 1.8 billion noisy image-text pairs from raw web alt-text data. No expensive curation, no expert annotation, no complex filtering. Scale compensates for noise. A contrastive loss (normalized softmax) aligns image and text representations in a shared embedding space.

Key difference from CLIP: ALIGN uses raw alt-text data with only minimal frequency-based filtering. CLIP constructs an allowlist of high-frequency visual concepts from English Wikipedia to curate its dataset.

---

## Architecture

### Dual-Encoder Structure

```
Image  -->  [EfficientNet (image encoder)]  -->  global pool  -->  x_i  (L2-normalized)
                                                                        \
                                                                         cosine similarity
                                                                        /
Text   -->  [BERT (text encoder)]  -->  [CLS] token  -->  FC(linear)  -->  y_j  (L2-normalized)
```

### Image Encoder: EfficientNet

- Default variant: EfficientNet-L2
- Global pooling applied (the 1x1 conv layer in the classification head is NOT trained)
- Output: image embedding x_i after L2 normalization
- Training resolution: 289 x 289 pixels
- Input preprocessing: resize to 346 x 346, then random crop (+ random horizontal flip) during training, center crop during evaluation
- Trained from scratch (not pretrained on ImageNet)

### Text Encoder: BERT

- Default variant: BERT-Large
- Vocabulary: 100k wordpiece tokens generated from the training dataset
- Maximum sequence length: 64 tokens
- Output: [CLS] token embedding
- A fully-connected layer with linear activation is added on top to match the image encoder output dimension
- Trained from scratch

### Embedding Dimension

- Scales with EfficientNet backbone size
- EfficientNet-B7: 640 dimensions
- EfficientNet-L2: 1376 dimensions
- For smaller EfficientNet variants (B1, B3, B5), an additional FC layer with linear activation projects the globally-pooled features to match B7's 640-dimensional output
- A similar linear projection layer is applied to all text encoders

---

## Training Objective: Normalized Softmax Contrastive Loss

### Formulation

Given a batch of N image-text pairs, let x_i = L2-normalized image embedding of pair i, y_j = L2-normalized text embedding of pair j.

**Image-to-text loss:**

```
L_i2t = -(1/N) * sum_{i=1}^{N} log [ exp(x_i^T y_i / sigma) / sum_{j=1}^{N} exp(x_i^T y_j / sigma) ]
```

**Text-to-image loss:**

```
L_t2i = -(1/N) * sum_{i=1}^{N} log [ exp(y_i^T x_i / sigma) / sum_{j=1}^{N} exp(y_i^T x_j / sigma) ]
```

**Total loss:**

```
L = L_i2t + L_t2i
```

Where:
- x_i, y_j are L2-normalized embeddings (so x_i^T y_j = cosine similarity)
- sigma is a learned temperature parameter (initialized at 1.0, converges to ~1/64)
- N = batch size (16384 effective)
- Matched pairs (i, i) are positives; all other in-batch pairs (i, j) where i != j are negatives

### Interpretation

- The image-to-text loss treats each text as a "class label" for the image. The text encoder dynamically generates the classifier weights.
- The text-to-image loss is the symmetric counterpart.
- This is equivalent to a symmetric cross-entropy loss over the cosine similarity matrix.

### Temperature Parameter

- Shared between L_i2t and L_t2i
- Initialized at 1.0
- Learned jointly with all other parameters (not manually swept)
- Converges to approximately 1/64
- Quickly decreases to ~1.2x of converged value within first 100k steps, then slowly converges to final value
- Ablation: fixed temperature 1/64 or 1/128 performs slightly better, but learned temperature is competitive and simpler
- Temperature 1/32 degrades performance significantly

### Label Smoothing

- Label smoothing parameter: 0.1 applied in the softmax losses

### In-Batch Negatives

- Embeddings from all TPU cores are concatenated to form the full batch of negatives
- Effective batch size: 16384 (1024 TPUv3 cores x 16 pairs per core)
- Using fewer negatives degrades performance:
  - 50% in-batch negatives: I2T R@1 drops from 51.7 to 50.2 on MSCOCO
  - 25% in-batch negatives: I2T R@1 drops from 51.7 to 48.7 on MSCOCO

---

## Training Data: 1.8B Noisy Image-Text Pairs

### Data Source

Raw English image alt-text pairs from the web, following the Conceptual Captions pipeline methodology but WITHOUT the expensive filtering and post-processing steps.

### Filtering Applied (minimal, frequency-based only)

**Image-based filtering:**
1. Remove pornographic images
2. Keep only images where shorter dimension > 200 pixels
3. Keep only images where aspect ratio < 3
4. Remove images with > 1000 associated alt-texts
5. Remove duplicates/near-duplicates of test images from downstream evaluation datasets (ILSVRC-2012, Flickr30K, MSCOCO)

**Text-based filtering:**
1. Remove alt-texts shared by > 10 images (these are typically irrelevant, e.g., "1920x1080", "alt img")
2. Remove alt-texts containing any rare token (outside 100M most frequent unigrams and bigrams from the raw dataset)
3. Remove alt-texts shorter than 3 unigrams
4. Remove alt-texts longer than 20 unigrams

### Near-Duplicate Detection (for test set decontamination)

1. Train a separate high-quality image embedding model
2. Generate 4K clusters via k-means on all training images
3. For each query/index image, find top-10 nearest clusters
4. Assign each image to C(10,3) = 120 buckets (all 3-cluster combinations from the top 10)
5. For any query-index pair in the same bucket: mark as near-duplicate if embedding cosine similarity > 0.975

### Scale Comparison

| Dataset | Size |
|---|---|
| Conceptual Captions (CC-3M) | ~3M |
| CLIP dataset | 400M |
| ALIGN dataset | 1.8B |

### Data Quality vs. Scale Tradeoff

With the same model (B7 + BERT-base):

| Data | MSCOCO I2T R@1 | MSCOCO T2I R@1 | ImageNet KNN R@1 |
|---|---|---|---|
| CC-3M (clean) | 18.9 | 15.5 | 48.7 |
| ALIGN 3M (noisy) | 8.1 | 6.3 | 41.3 |
| ALIGN 6M (noisy) | 15.8 | 11.9 | 47.9 |
| ALIGN 12M (noisy) | 23.8 | 17.5 | 51.4 |
| ALIGN 10% (~180M) | 52.0 | 39.2 | 68.8 |
| ALIGN full (1.8B) | 55.4 | 41.7 | 69.3 |

Noisy ALIGN data at 3M performs much worse than clean CC-3M at 3M. But noisy ALIGN data surpasses clean CC-3M with only 4x the size (12M). At full scale (1.8B), the advantage is massive.

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | LAMB |
| Weight decay | 1e-5 |
| Learning rate | 1e-3 (peak) |
| LR warmup | Linear, 0 to 1e-3 over 10k steps |
| LR decay | Linear, 1e-3 to 0 over 1.2M steps |
| Total training steps | 1.2M (~12 epochs) |
| Hardware | 1024 Cloud TPUv3 cores |
| Pairs per core | 16 |
| Effective batch size | 16384 |
| Temperature init | 1.0 (learned) |
| Label smoothing | 0.1 |
| Image resolution (train) | 289 x 289 (cropped from 346 x 346) |

---

## Fine-Tuning Details

### Image-Text Retrieval Fine-Tuning (Flickr30K, MSCOCO)

- Same contrastive loss as pretraining
- Reduced global batch size: 2048 (to mitigate false negatives when batch size is comparable to dataset size)
- Initial learning rate: 1e-5 (linear decay)
- Training steps: 3K (Flickr30K), 6K (MSCOCO)
- All other hyperparameters same as pretraining

### ImageNet Classification Fine-Tuning

**Stage 1: Frozen features (train classification head only)**
- Train/eval resolution: 289/360
- Batch size: 1024
- Optimizer: SGD with momentum 0.9
- Initial LR: 0.1, decayed every 30 epochs with ratio 0.2
- Total epochs: 100
- Weight decay: 0

**Stage 2: Full fine-tuning (all layers)**
- Train/eval resolution: 475/600
- Batch size: 1024
- Optimizer: SGD with momentum 0.9
- Initial LR: 0.01 (10x smaller LR on backbone vs. classification head)
- LR decayed every 30 epochs with ratio 0.2
- Total epochs: 100
- Weight decay: 0
- Scale ratio 0.8 between training and evaluation resolution (mitigates random crop resolution discrepancy)

### Fine-Grained Classification Fine-Tuning

- Train/eval resolution: 289/360
- Batch size: 256
- Weight decay: 1e-5
- Stage 1 (head only): LR 1e-2, cosine decay, 20k steps
- Stage 2 (all layers): LR 1e-3, cosine decay, 20k steps, batch norm frozen

---

## Zero-Shot Transfer

### Method

Feed classname text into the text encoder. Classify images by finding the highest cosine similarity between the image embedding and all class text embeddings.

### Prompt Ensembling

Each classname is expanded with multiple prompt templates (same templates as CLIP), e.g., "A photo of a {classname}". The class embedding = average of all template embeddings, followed by L2 normalization. This gives +2.9% improvement on ImageNet top-1 accuracy.

---

## Results

### Zero-Shot Image Classification

| Model | ImageNet | ImageNet-R | ImageNet-A | ImageNet-V2 |
|---|---|---|---|---|
| CLIP | 76.2 | 88.9 | 77.2 | 70.1 |
| ALIGN | 76.4 | 92.2 | 75.8 | 70.1 |

ALIGN slightly outperforms CLIP on ImageNet (+0.2%) and ImageNet-R (+3.3%). CLIP is better on ImageNet-A (+1.4%). Tied on ImageNet-V2.

### ImageNet Classification (Fine-Tuned)

| Model (backbone) | Frozen Features Acc@1 | Fine-Tuned Acc@1 | Acc@5 |
|---|---|---|---|
| WSL (ResNeXt-101 32x48d) | 83.6 | 85.4 | 97.6 |
| CLIP (ViT-L/14) | 85.4 | - | - |
| BiT (ResNet152 x 4) | - | 87.54 | 98.46 |
| NoisyStudent (EfficientNet-L2) | - | 88.4 | 98.7 |
| ViT (ViT-H/14) | - | 88.55 | - |
| Meta-Pseudo-Labels (EfficientNet-L2) | - | 90.2 | 98.8 |
| ALIGN (EfficientNet-L2) | 85.5 | 88.64 | 98.67 |

ALIGN with frozen features: 85.5% (SOTA, slightly above CLIP's 85.4%). ALIGN fine-tuned: 88.64% (above BiT, ViT, NoisyStudent; below Meta-Pseudo-Labels). ALIGN saves 44% FLOPs vs NoisyStudent/Meta-Pseudo-Labels by using 600 eval resolution instead of 800.

### VTAB (19 Tasks)

| Model | All Tasks | Natural | Specialized | Structured |
|---|---|---|---|---|
| BiT-L | 78.72 | - | - | - |
| ALIGN | 79.99 +/- 0.15 | 83.38 | 87.56 | 73.25 |

### Fine-Grained Classification

| Model | Oxford Flowers | Oxford Pets | Stanford Cars | Food101 |
|---|---|---|---|---|
| BiT-L | 99.63 | 96.62 | - | - |
| SAM-baseline | 99.60 | 96.92 | 95.07 | 96.03 |
| SAM-final | 99.65 | 97.10 | 95.96 | 96.18 |
| ALIGN | 99.65 | 96.19 | 96.13 | 95.88 |

### Zero-Shot Image-Text Retrieval

**Flickr30K (1K test set):**

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|
| CLIP | 88.0 | 98.7 | 99.4 | 68.7 | 90.6 | 95.2 |
| ALIGN | 88.6 | 98.7 | 99.7 | 75.7 | 93.8 | 96.8 |

ALIGN beats CLIP by +7.0% on T2I R@1.

**MSCOCO (5K test set):**

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|
| CLIP | 58.4 | 81.5 | 88.1 | 37.8 | 62.4 | 72.2 |
| ALIGN | 58.6 | 83.0 | 89.7 | 45.6 | 69.8 | 78.6 |

ALIGN beats CLIP by +7.8% on T2I R@1.

### Fine-Tuned Image-Text Retrieval

**Flickr30K (1K test set):**

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|
| UNITER | 87.3 | 98.0 | 99.2 | 75.6 | 94.1 | 96.8 |
| VILLA | 87.9 | 97.5 | 98.8 | 76.3 | 94.2 | 96.8 |
| ERNIE-ViL | 88.1 | 98.0 | 99.2 | 76.7 | 93.6 | 96.4 |
| GPO | 88.7 | 98.9 | 99.8 | 76.1 | 94.5 | 97.1 |
| ALIGN | 95.3 | 99.8 | 100.0 | 84.9 | 97.4 | 98.6 |

ALIGN I2T R@10 = 100.0%. ALIGN beats all cross-attention models despite being a simple dual encoder.

**MSCOCO (5K test set):**

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|
| Oscar | 73.5 | 92.2 | 96.0 | 57.5 | 82.8 | 89.8 |
| ALIGN | 77.0 | 93.5 | 96.9 | 59.9 | 83.3 | 89.8 |

### Crisscrossed Captions (CxC) Retrieval

| Model | I2T R@1 | T2I R@1 | T2T R@1 | I2I R@1 |
|---|---|---|---|---|
| DET2T+I2T | 55.9 | 41.7 | 42.4 | 38.5 |
| ALIGN | 78.1 | 61.8 | 45.4 | 49.4 |

ALIGN outperforms previous SOTA by +22.2% on I2T R@1 and +20.1% on T2I R@1. Improvements on intra-modal tasks (T2T, I2I) are less dramatic because the training objective focuses on cross-modal matching.

### CxC Semantic Similarity (Spearman's rho x 100)

| Model | STS | SIS | SITS | Mean Avg |
|---|---|---|---|---|
| DET2T+I2T | 74.2 | 74.5 | 61.9 | 70.2 |
| ALIGN | 72.9 | 77.2 | 67.6 | 72.6 |

ALIGN achieves best mean average (72.6) and best SITS (+5.7%). Slightly worse on STS than previous methods.

### Multilingual (Multi30K, zero-shot, mean Recall)

| Model | en | de | fr | cs |
|---|---|---|---|---|
| M3P (zero-shot) | 57.9 | 36.8 | 27.1 | 20.4 |
| ALIGN_mling (zero-shot) | 90.2 | 84.1 | 84.9 | 63.2 |
| M3P (fine-tuned) | 87.7 | 82.7 | 73.9 | 72.2 |
| UC2 (fine-tuned) | 88.2 | 84.5 | 83.9 | 81.2 |

Zero-shot ALIGN_mling outperforms zero-shot M3P by up to +57.8 mR (on French). Zero-shot ALIGN_mling is comparable to fine-tuned M3P and UC2 on en, de, fr, but lags on Czech.

---

## Ablation: Model Architecture Scaling

### Image Encoder Scaling (with fixed text encoder)

Tested: EfficientNet-B1, B3, B5, B7, L2
- Scaling up image encoder is more important for vision-only tasks
- Even with BERT-Mini text tower, L2 outperforms B7 + BERT-Large on ImageNet KNN

### Text Encoder Scaling (with fixed image encoder)

Tested: BERT-Mini, BERT-Medium, BERT-Base, BERT-Large
- ImageNet KNN saturates from BERT-Base to BERT-Large (with B7 and L2)
- For image-text retrieval, image and text encoder capacities are equally important

### Embedding Dimension Ablation (B5 + BERT-Base baseline)

| Embedding Dim | MSCOCO I2T R@1 | MSCOCO T2I R@1 | ImageNet KNN R@1 |
|---|---|---|---|
| 640 (baseline) | 51.7 | 37.5 | 64.6 |
| 320 | 50.3 | 34.1 | 64.0 |
| 160 | 47.0 | 34.4 | 63.7 |
| 80 | 42.0 | 29.3 | 61.9 |

### Temperature Ablation (B5 + BERT-Base baseline)

| Temperature | MSCOCO I2T R@1 | MSCOCO T2I R@1 | ImageNet KNN R@1 |
|---|---|---|---|
| Learned (~1/64) | 51.7 | 37.5 | 64.6 |
| Fixed 1/128 | 52.2 | 36.5 | 64.8 |
| Fixed 1/64 | 52.2 | 37.3 | 64.8 |
| Fixed 1/32 | 39.6 | 26.9 | 61.2 |

### Dataset Size Ablation (B7 + BERT-Base)

| Data | MSCOCO I2T R@1 | MSCOCO T2I R@1 | ImageNet KNN R@1 |
|---|---|---|---|
| CC-3M | 18.9 | 15.5 | 48.7 |
| ALIGN full (1.8B) | 55.4 | 41.7 | 69.3 |
| ALIGN 10% (~180M) | 52.0 | 39.2 | 68.8 |

On CC-3M, B7+BERT-Base overfits and performs worse than B3+BERT-Mini. Larger models require larger datasets.

---

## Cross-Modal Compositionality

ALIGN embeddings exhibit linear compositionality analogous to word2vec:

```
image_embedding + text_embedding("red") --> retrieves red-colored version of the image subject
image_embedding - text_embedding("cars") --> retrieves the scene without cars
```

Implementation detail: normalize text and image embeddings before combining. Best results with scale factor 2 for text embedding and 1 for image embedding (1:1 also works).

---

## Key Takeaways for Implementation

1. **Scale beats curation.** 1.8B noisy pairs outperform 3M clean pairs by a massive margin. Noisy data at 4x the size of clean data already surpasses it.
2. **Simple architecture suffices.** A dual encoder with contrastive loss beats complex cross-attention models (UNITER, VILLA, Oscar) on retrieval benchmarks.
3. **Temperature matters.** The softmax temperature is crucial since embeddings are L2-normalized. Learning it (initialized at 1.0) is competitive with optimal fixed values and avoids manual tuning.
4. **Large batch = more negatives = better.** 16384 effective batch size via cross-core concatenation. Reducing in-batch negatives directly hurts performance.
5. **Both encoders scale together.** For retrieval, image and text encoder capacities are equally important. For vision-only tasks, the image encoder matters more.
6. **LAMB optimizer.** Outperforms SGD with momentum and ADAM for jointly training CNN + BERT architectures.
