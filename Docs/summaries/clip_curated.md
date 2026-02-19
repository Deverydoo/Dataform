# CLIP: Contrastive Language-Image Pre-training -- Raw Technical Extraction

**Source**: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021, OpenAI)

---

## 1. CORE CONCEPT

Train an image encoder and a text encoder jointly on 400 million (image, text) pairs to predict which text goes with which image via a contrastive objective. At inference, use the text encoder to synthesize zero-shot classifiers from natural language descriptions of target classes.

---

## 2. CONTRASTIVE PRE-TRAINING OBJECTIVE

### 2.1 InfoNCE Loss (N-pair Loss)

Given a batch of N (image, text) pairs, CLIP predicts which of the N x N possible pairings actually occurred.

**Objective**: Maximize cosine similarity of N correct (image, text) pairs while minimizing cosine similarity of N^2 - N incorrect pairings. Optimize a **symmetric cross-entropy loss** over similarity scores.

### 2.2 Pseudocode (Exact from Paper)

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T)  #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

### 2.3 Mathematical Formulation

For a batch of N pairs {(I_i, T_i)}:

**Image-to-text direction** (for image i):

    L_i2t(i) = -log( exp(sim(I_i, T_i) * exp(t)) / sum_j( exp(sim(I_i, T_j) * exp(t)) ) )

**Text-to-image direction** (for text i):

    L_t2i(i) = -log( exp(sim(T_i, I_i) * exp(t)) / sum_j( exp(sim(T_i, I_j) * exp(t)) ) )

**Total loss**:

    L = (1/2N) * sum_i( L_i2t(i) + L_t2i(i) )

Where:
- `sim(A, B) = (A / ||A||) . (B / ||B||)` (cosine similarity)
- `t` = learned log-parameterized temperature scalar
- `exp(t)` scales logits before softmax

### 2.4 Temperature Parameter

- Learnable log-parameterized multiplicative scalar
- Initialized to equivalent of 0.07 (from Wu et al., 2018)
- Clipped to prevent scaling logits by more than 100 (prevents training instability)
- Controls the range/sharpness of the softmax distribution

---

## 3. ARCHITECTURE

### 3.1 Image Encoder -- Option A: Modified ResNet

Base: ResNet-50 (He et al., 2016) with modifications:
- **ResNet-D improvements** (He et al., 2019): modified stem and downsampling
- **Antialiased rect-2 blur pooling** (Zhang, 2019)
- **Attention pooling** replaces global average pooling:
  - Single layer of transformer-style multi-head QKV attention
  - Query is conditioned on the global average-pooled representation
- **Scaling**: EfficientNet-style compound scaling (Tan & Le, 2019)
  - Allocate compute equally to width, depth, and resolution
  - RN50x4 ~ 4x compute of RN50
  - RN50x16 ~ 16x compute of RN50
  - RN50x64 ~ 64x compute of RN50

**ResNet Model Configurations**:

| Model | LR | Embed Dim | Input Res | ResNet Blocks | Width | Text Layers | Text Width | Text Heads |
|---|---|---|---|---|---|---|---|---|
| RN50 | 5e-4 | 1024 | 224 | (3,4,6,3) | 2048 | 12 | 512 | 8 |
| RN101 | 5e-4 | 512 | 224 | (3,4,23,3) | 2048 | 12 | 512 | 8 |
| RN50x4 | 5e-4 | 640 | 288 | (4,6,10,6) | 2560 | 12 | 640 | 10 |
| RN50x16 | 4e-4 | 768 | 384 | (6,8,18,8) | 3072 | 12 | 768 | 12 |
| RN50x64 | 3.6e-4 | 1024 | 448 | (3,15,36,10) | 4096 | 12 | 1024 | 16 |

### 3.2 Image Encoder -- Option B: Vision Transformer (ViT)

Closely follows Dosovitskiy et al. (2020) with:
- Additional layer normalization on combined patch + position embeddings before transformer
- Slightly different initialization scheme
- ViT-L/14 additionally pre-trained at 336px for 1 extra epoch (FixRes-style boost)

**ViT Model Configurations**:

| Model | LR | Embed Dim | Input Res | ViT Layers | ViT Width | ViT Heads | Text Layers | Text Width | Text Heads |
|---|---|---|---|---|---|---|---|---|---|
| ViT-B/32 | 5e-4 | 512 | 224 | 12 | 768 | 12 | 12 | 512 | 8 |
| ViT-B/16 | 5e-4 | 512 | 224 | 12 | 768 | 12 | 12 | 512 | 8 |
| ViT-L/14 | 4e-4 | 768 | 224 | 24 | 1024 | 16 | 12 | 768 | 12 |
| ViT-L/14-336px | 2e-5 | 768 | 336 | 24 | 1024 | 16 | 12 | 768 | 12 |

### 3.3 Text Encoder: Transformer

- Architecture from GPT-2 (Radford et al., 2019)
- Base size: 63M parameters, 12 layers, 512 width, 8 attention heads
- **Tokenization**: Lower-cased byte pair encoding (BPE), vocabulary size = 49,152 (actual 49,408 with special tokens)
- **Max sequence length**: 76 tokens
- **Special tokens**: [SOS] and [EOS] bracket the text sequence
- **Feature extraction**: Activations at [EOS] token from highest transformer layer
- **Post-processing**: Layer normalization then linear projection into joint embedding space
- **Attention**: Masked (causal) self-attention to preserve ability to initialize with pre-trained LM
- **Text encoder scaling**: Only width scales proportional to ResNet width; depth stays fixed at 12 layers

### 3.4 Joint Embedding Space

- **Linear projection only** (no non-linear projection head)
  - W_i: [d_i, d_e] projects image features to embedding
  - W_t: [d_t, d_e] projects text features to embedding
- Both projections followed by L2 normalization
- Interaction between modalities: single dot product in learned joint embedding space

### 3.5 Key Design Simplifications vs. ConVIRT (Zhang et al., 2020)

1. No non-linear projection between representation and contrastive embedding space
2. No text transformation function (no random sentence sampling -- most pairs are single sentences)
3. Simplified image augmentation: random square crop from resized images only
4. Trained from scratch (no ImageNet or pre-trained LM initialization)

---

## 4. TRAINING DETAILS

### 4.1 Dataset: WIT (WebImageText)

- **400 million (image, text) pairs** from internet
- Query construction: 500,000 queries based on:
  - All words occurring >= 100 times in English Wikipedia
  - Augmented with high-PMI bi-grams
  - Names of all Wikipedia articles above certain search volume
  - All WordNet synsets not already in query list
- Class-balanced: up to 20,000 (image, text) pairs per query
- Total word count similar to WebText dataset (used for GPT-2)

### 4.2 Optimization

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam with decoupled weight decay |
| Batch size | 32,768 |
| Training epochs | 32 |
| Weight decay | 0.2 |
| LR schedule | Cosine decay |
| Warm-up iterations | 2,000 |
| Adam beta1 | 0.9 |
| Adam beta2 | 0.999 (ResNet), 0.98 (ViT) |
| Adam epsilon | 1e-8 (ResNet), 1e-6 (ViT) |
| Temperature init | 0.07 |
| Max temperature | 100.0 (clip logit scaling) |
| Precision | Mixed-precision (FP16) |

### 4.3 Memory Optimizations

- Gradient checkpointing
- Half-precision Adam statistics
- Half-precision stochastically rounded text encoder weights
- Embedding similarity computation sharded across GPUs (each GPU computes only its local batch pairwise similarities)

### 4.4 Compute

| Model | Hardware | Training Time |
|---|---|---|
| RN50x64 | 592 V100 GPUs | 18 days |
| ViT-L/14 | 256 V100 GPUs | 12 days |
| ViT-L/14-336px | (above + 1 extra epoch at 336px) | |

- Total images seen: 12.8 billion over 32 epochs (400M x 32)
- At 1 image/second rate: would take 405 years

### 4.5 Data Augmentation

- Random square crop from resized images ONLY
- No other augmentations used during training

---

## 5. ZERO-SHOT CLASSIFICATION PROCEDURE

### 5.1 Steps

1. Encode all class names through the text encoder to get text embeddings {T_k}
2. Encode the test image through the image encoder to get image embedding I
3. Compute cosine similarity between I and each T_k, scaled by temperature
4. Apply softmax to get probability distribution over classes
5. Predict class with highest probability

### 5.2 Mathematical Interpretation

The zero-shot classifier is a **multinomial logistic regression** with:
- L2-normalized inputs (image features)
- L2-normalized weights (text embeddings)
- No bias term
- Temperature scaling

The text encoder acts as a **hypernetwork** generating classifier weights from natural language.

Each pre-training step optimizes a randomly created proxy dataset with:
- 1 example per class
- 32,768 total classes (batch size)
- Classes defined via natural language

### 5.3 Caching

Zero-shot classifier weights are cached after text encoder computes them once; reused for all predictions on a dataset.

---

## 6. PROMPT ENGINEERING AND ENSEMBLING

### 6.1 Problem: Polysemy and Distribution Gap

- Class names alone are ambiguous (e.g., "crane" = bird or machine)
- Pre-training text is usually full sentences, not single words

### 6.2 Default Prompt Template

    "A photo of a {label}."

Improves ImageNet accuracy by +1.3% over bare class names.

### 6.3 Task-Specific Prompts (Examples)

| Task Domain | Prompt Template |
|---|---|
| Fine-grained pets | "A photo of a {label}, a type of pet." |
| Food classification | "A photo of {label}, a type of food." |
| Aircraft classification | "A photo of a {label}, a type of aircraft." |
| OCR / text recognition | Quotes around text: "a photo of the number: \"{label}\"" |
| Satellite imagery | "a satellite photo of a {label}." |
| Action recognition | "a photo of a person {label}." |
| Texture classification | "a photo of a {label} texture." |
| Geo-localization | "a photo i took in {label}." |

### 6.4 Prompt Ensembling

- Use multiple different context prompts per class
- Examples: "A photo of a big {label}", "A photo of a small {label}", etc.
- **Ensemble in embedding space** (not probability space):
  - Average text embeddings across prompts for each class
  - Cache single set of averaged embeddings
  - Same compute cost as single classifier at inference
- ImageNet: 80 different context prompts ensembled
- Ensembling improves ImageNet by +3.5% over single default prompt
- **Total gain from prompt engineering + ensembling: ~5% on ImageNet**

---

## 7. EFFICIENCY COMPARISON: TRAINING OBJECTIVES

Pre-training objective efficiency for zero-shot ImageNet (all using ResNet-50 image encoder):

| Method | Relative Efficiency |
|---|---|
| Transformer language model (63M params, caption prediction) | 1x (baseline) |
| Bag-of-words prediction | 3x faster than transformer LM |
| Bag-of-words contrastive (CLIP approach) | 4x faster than BoW prediction |
| **CLIP contrastive vs. transformer LM** | **~12x more efficient** |

---

## 8. ZERO-SHOT PERFORMANCE RESULTS

### 8.1 CLIP vs. Visual N-Grams (Prior Zero-Shot SOTA)

| Dataset | Visual N-Grams | CLIP |
|---|---|---|
| aYahoo | 72.4 | **98.4** |
| ImageNet | 11.5 | **76.2** |
| SUN397 | 23.0 | **58.5** |

### 8.2 Zero-Shot CLIP vs. Supervised ResNet-50 Linear Probe (27 Datasets)

Zero-shot CLIP wins on 16 of 27 datasets.

**Datasets where zero-shot CLIP wins by large margin**:
- StanfordCars: +28.9%
- Country211: +22.5%
- Food101: +23.2%
- Kinetics700: +14.5%
- SST2: +12.4%
- SUN397: +7.8%
- UCF101: +7.7%

**Datasets where zero-shot CLIP loses significantly**:
- EuroSAT: -37.1%
- KITTI Distance: -34.0%
- PatchCamelyon: -19.5%
- GTSRB: -18.4%
- CLEVRCounts: -18.2%
- DTD: -16.6%
- Flowers102: -12.5%

### 8.3 Zero-Shot CLIP vs. Few-Shot Linear Probes

- Zero-shot CLIP matches **4-shot** linear classifier on same CLIP feature space
- Zero-shot CLIP roughly matches the best **16-shot** classifier (BiT-M ResNet-152x2 on ImageNet-21K)
- Effective data efficiency: median 5.4 labeled examples per class, mean 20.8

### 8.4 ImageNet Zero-Shot Top-1/Top-5

- **Top-1**: 76.2% (matches original ResNet-50)
- **Top-5**: 95.0% (matches Inception-V4)

### 8.5 Full Zero-Shot Results Table (All CLIP Models, 27 Datasets)

| Model | Food101 | CIFAR10 | CIFAR100 | ImageNet | STL10 | ... |
|---|---|---|---|---|---|---|
| RN50 | 81.1 | 75.6 | 41.6 | 59.6 | 94.3 | |
| RN101 | 83.9 | 81.0 | 49.0 | 62.2 | 96.7 | |
| RN50x4 | 86.8 | 79.2 | 48.9 | 65.8 | 96.4 | |
| RN50x16 | 90.5 | 82.2 | 54.2 | 70.5 | 97.8 | |
| RN50x64 | 91.8 | 86.8 | 61.3 | 73.6 | 98.3 | |
| ViT-B/32 | 84.4 | 91.3 | 65.1 | 63.2 | 97.2 | |
| ViT-B/16 | 89.2 | 91.6 | 68.7 | 68.6 | 98.2 | |
| ViT-L/14 | 92.9 | 96.2 | 77.9 | 75.3 | 99.3 | |
| ViT-L/14-336px | 93.8 | 95.7 | 77.5 | 76.2 | 99.4 | |

### 8.6 Scaling Law

Zero-shot error follows a **log-log linear trend** across 44x range of compute (5 ResNet models). Performance is a smoothly predictable function of compute.

Estimated ~1000x increase in compute needed for zero-shot CLIP to reach overall SOTA.

---

## 9. LINEAR PROBE RESULTS

### 9.1 Key Findings

- **ViT-L/14-336px** achieves SOTA on **21 of 27 datasets** in linear probe evaluation
- CLIP ViTs are ~3x more compute-efficient than CLIP ResNets
- Best CLIP model outperforms Noisy Student EfficientNet-L2 on **21 of 27 datasets**

### 9.2 Linear Probe: CLIP vs. Best Prior Models (Selected)

| Dataset | CLIP ViT-L/14-336px | NS EfficientNet-L2-800 | Delta |
|---|---|---|---|
| Food101 | 95.9 | 92.0 | +3.9 |
| Stanford Cars | 91.5 | 75.5 | +16.0 |
| GTSRB | 92.4 | 77.7 | +14.7 |
| SUN397 | 82.2 | 75.7 | +6.5 |
| Country211 | 46.4 | 23.7 | +22.7 |
| SST2 | 80.5 | 56.9 | +23.6 |
| UCF101 | 92.0 | 88.9 | +3.1 |
| Kinetics700 | 73.0 | 66.7 | +6.3 |
| ImageNet | 85.4 | 88.4 | -3.0 |
| CIFAR10 | 97.9 | 98.7 | -0.8 |
| CIFAR100 | 87.4 | 89.0 | -1.6 |

### 9.3 Linear Probe Evaluation Protocol

- Features from penultimate layer (before classification head)
- For CLIP-ViT: features before linear projection (I_f in pseudocode)
- Logistic regression via scikit-learn L-BFGS, max 1000 iterations
- L2 regularization: sweep over [10^-6, 10^6] with 96 log-spaced steps
- Parametric binary search for hyperparameter optimization

---

## 10. ROBUSTNESS TO DISTRIBUTION SHIFT

### 10.1 ImageNet Robustness Benchmarks

| Dataset | NS EfficientNet-L2 | Zero-Shot CLIP | Delta |
|---|---|---|---|
| ImageNet | 88.3 | 76.2 | -12.1 |
| ImageNetV2 | 80.2 | 70.1 | -10.1 |
| ImageNet-A | 84.9 | 77.2 | -7.7 |
| ImageNet-R | 74.7 | 88.9 | **+14.2** |
| ObjectNet | 68.5 | 72.3 | **+3.8** |
| ImageNet Sketch | 47.6 | 60.2 | **+12.6** |
| ImageNet-Vid (PM0) | 88.0 | 95.3 | **+7.3** |
| Youtube-BB (PM0) | 67.7 | 95.2 | **+27.5** |

### 10.2 Key Robustness Findings

- Zero-shot CLIP reduces the "robustness gap" (between ImageNet accuracy and distribution shift accuracy) by up to **75%**
- A ResNet-101 makes **5x more mistakes** on distribution shifts vs. ImageNet val
- Zero-shot CLIP improves **effective robustness** by a large amount on all 7 distribution shift datasets
- Zero-shot CLIP is SOTA on 5 of 7 natural distribution shift datasets

### 10.3 Adaptation Erodes Robustness

- Adapting CLIP to ImageNet (linear probe): +9.2% on ImageNet, ties 2018 SOTA (85.4%)
- But: average distribution shift accuracy **slightly decreases**
- ImageNetV2: +5.8% (closely follows ImageNet creation process)
- ImageNet-R: -4.7%
- ObjectNet: -3.8%
- ImageNet Sketch: -2.8%

### 10.4 Few-Shot Robustness Continuum

- Effective robustness is highest at zero-shot
- Decreases as number of labeled examples increases
- 16-shot matches zero-shot on ImageNet but is less robust
- Fully supervised linear probe: robustness advantage mostly gone

---

## 11. RETRIEVAL PERFORMANCE

### 11.1 Flickr30k Zero-Shot

| Metric | CLIP | Best Fine-tuned (ERNIE-ViL) |
|---|---|---|
| Text R@1 | **88.0** | 88.7 |
| Text R@5 | **98.7** | 98.0 |
| Text R@10 | **99.4** | 99.2 |
| Image R@1 | 68.7 | 76.7 |
| Image R@5 | 90.6 | 93.6 |
| Image R@10 | 95.2 | 96.4 |

### 11.2 MSCOCO Zero-Shot (5k test set)

| Metric | CLIP |
|---|---|
| Text R@1 | 58.4 |
| Text R@5 | 81.5 |
| Text R@10 | 88.1 |
| Image R@1 | 37.8 |
| Image R@5 | 62.4 |
| Image R@10 | 72.2 |

---

## 12. ACTION RECOGNITION

| Setting | UCF101 | Kinetics-700 | RareAct mWAP | RareAct mWSAP |
|---|---|---|---|---|
| Zero-shot CLIP | 80.3 | 69.6 | 40.7 | 44.8 |
| Linear probe CLIP | 92.0 | 73.0 | - | - |
| Prior ZS SOTA (HT100M S3D) | - | - | 30.5 | 34.8 |
| Supervised I3D baseline | - | 70.2 | - | - |
| Fine-tune SOTA | 98.7 | 84.8 | - | - |

---

## 13. GEOLOCALIZATION (IM2GPS)

| Model | 1km | 25km | 200km | 750km | 2500km |
|---|---|---|---|---|---|
| ISNs (SOTA) | **16.9** | **43.0** | **51.9** | **66.7** | **80.2** |
| CPlaNet | 16.5 | 37.1 | 46.4 | 62.0 | 78.5 |
| CLIP | 13.9 | 32.9 | 43.0 | 62.0 | 79.3 |
| PlaNet | 8.4 | 24.5 | 37.6 | 53.6 | 71.3 |

---

## 14. DATA OVERLAP ANALYSIS

- 35 datasets studied; 9 have zero detected overlap
- Median overlap: 2.2%; mean overlap: 3.2%
- Maximum overall accuracy increase from overlap: **0.6% on Birdsnap** (12.1% overlap)
- Largest overlap: Country211 at 21.5% (from YFCC100M subset) but only +0.2% accuracy change
- Only 2 datasets show statistically significant accuracy improvement after Bonferroni correction
- Conclusion: data contamination has minimal impact on reported results

---

## 15. DATASET ABLATION (YFCC100M vs. WIT)

ResNet-50 trained on YFCC100M (15M filtered images) vs. same-sized WIT subset:

| Metric | YFCC | WIT | Delta |
|---|---|---|---|
| Dataset Average (Linear) | 65.5 | 66.6 | -1.1 |
| Dataset Average (Zero-shot) | 29.6 | 30.0 | -0.4 |
| ImageNet (Linear) | 62.0 | 60.8 | +1.2 |
| ImageNet (Zero-shot) | 31.3 | 27.6 | +3.7 |

Similar average performance but large per-dataset variation (up to +/-18.9%). WIT's main advantage: 27x larger than filtered YFCC100M.

---

## 16. HUMAN COMPARISON (Oxford-IIIT Pets)

| Setting | Accuracy | Majority Vote |
|---|---|---|
| Zero-shot Human | 53.7 | 57.0 |
| Zero-shot CLIP | **93.5** | **93.5** |
| One-shot Human | 75.7 | 80.3 |
| Two-shot Human | 75.7 | 85.0 |

- Humans improve from 54% to 76% with one example (almost entirely on uncertain images)
- Suggests humans "know what they don't know" and update from single examples
- Large gap between human and machine few-shot efficiency

---

## 17. KNOWN LIMITATIONS

1. **Compute**: ~1000x more compute needed to reach SOTA on all tasks
2. **Weak tasks**: Satellite imagery, counting, traffic signs, distance estimation, lymph node detection, MNIST
3. **Out-of-distribution**: Fails on data truly absent from pre-training (MNIST handwriting = 88%, below raw pixel logistic regression)
4. **Abstract reasoning**: Struggles with counting objects, systematic/abstract tasks
5. **Fine-grained classification**: Inconsistent (great on cars/food, poor on flowers/aircraft)
6. **No generative output**: Limited to choosing among provided class options
7. **Data efficiency**: Compensates with scale (400M pairs) rather than improving sample efficiency
8. **Social biases**: Learns biases from unfiltered internet data; disparate misclassification rates across demographics
9. **OCR variability**: Strong on rendered text, weak on handwritten/natural scene text

---

## 18. ARCHITECTURAL DECISIONS SUMMARY

| Decision | Choice | Rationale |
|---|---|---|
| Training objective | Contrastive (InfoNCE) | 12x more efficient than caption prediction |
| Image-text interaction | Single dot product | Minimal coupling enables caching |
| Projection head | Linear only | Non-linear showed no efficiency benefit |
| Data augmentation | Random crop only | Sufficient at scale |
| Text attention | Causal (masked) | Preserves LM initialization option |
| Feature extraction (text) | [EOS] token activation | Standard for autoregressive models |
| Feature extraction (image, ResNet) | Attention pooling | Better than global average pooling |
| Feature extraction (image, ViT) | [CLS] token equivalent | Standard ViT approach |
| Temperature | Learned, log-parameterized | Avoids manual tuning |
| Precision | Mixed FP16 | Memory and speed |
| Batch size | 32,768 | Large batch critical for contrastive learning |
