# Machine Learning: Algorithms, Models and Applications -- RAW TECHNICAL EXTRACTION

Source: IntechOpen, Artificial Intelligence Volume 7, Edited by Jaydip Sen, 2021 (154 pages, 6 chapters)

---

## 1. ALGORITHMS COVERED

### 1.1 Stock Price Prediction (Chapter 2)
- **CNN-based regression** (4 variants)
- **LSTM-based regression** (6 variants)
- **Walk-forward validation** with multi-step prediction
- **ARIMA** (Autoregressive Integrated Moving Average)
- **GARCH** (Generalized Autoregressive Conditional Heteroscedasticity)
- **STAR** (Smoothing Transition Autoregressive)
- **VAR** (Vector Autoregression)
- **OLS regression**, **MARS** (Multivariate Adaptive Regression Spline), penalty-based regression, polynomial regression
- **Quantile Regression (QR)**
- **Linear and Quadratic Discriminant Analysis**
- **Backpropagation Neural Network** combined with ARIMA
- **Self-Organizing Fuzzy Neural Network** (sentiment-driven stock prediction, 86% accuracy)
- **Multi-Layer Perceptron** for financial data mining

### 1.2 Human Pose Estimation (Chapter 3)
- **Greedy Part Association Vectors (GPAVs)** -- bottom-up parsing
- **Non-Maximum Suppression** for part candidate detection
- **Bipartite Graph Matching** for joint association
- **Confidence Map / Heatmap generation** via CNN
- **DeepCut / DeeperCut** (joint subset partition and labeling)
- **Convolutional Pose Machines**
- **RMPE** (Regional Multi-person Pose Estimation)
- **Part Affinity Fields** (Cao et al.)

### 1.3 Ensemble Methods for Medical Imaging (Chapter 4)
- **Bagging** (Bootstrap Aggregation)
- **Boosting** (sequential weak learner combination)
- **Stacking** (meta-learner over base classifiers)
- **Random Forest**
- **Logistic Regression**
- **Decision Trees** (Shannon Entropy, Information Gain)
- **Principal Component Analysis (PCA)**
- **Convolutional Neural Networks (CNN)** for image classification
- **Artificial Neural Networks (ANN)**
- **Support Vector Machines (SVM)**
- Ensemble of **ResNet34, DenseNet169, DenseNet201** (shoulder X-ray classification, best kappa = 0.6942)
- Ensemble of **Xception, ResNet, Inception-V3** (ankle fracture detection, 81% accuracy)
- **ResNet101 + Inception-V3** ensemble (ear disease, 93.67% accuracy, 5-fold CV)
- COVID-19 X-ray ensemble: 99.01% accuracy, AUC = 0.9972

### 1.4 Precision Medicine / Continual Learning (Chapter 5)
- **Progressive Learning Framework** (Fayek et al.)
  - Curriculum stage, Progression stage, Pruning stage
- **Convolutional Neural Networks (CNN)** for image classification
- **DenseNet-169** architecture
- **Densely Connected Learning Networks (DCLN)**
- **3D DenseNet** for MRI volumetric data
- **Logistic Regression** and **Linear Regression** for baseline models
- **SoftMax** for categorical prediction output

### 1.5 Predictive Analytics Algorithms (Chapter 6)
- **Linear Regression**
- **Multiple Regression**
- **Multivariate Regression**
- **Decision Tree** (supervised classification)
- **Time Series** (frequency-domain: spectral/wavelet; time-domain: autocorrelation/cross-correlation)
- **Market Segmentation algorithms**
- **Brute Force algorithms**
- **Simple Recursive algorithms**
- **Backtracking algorithms**
- **Randomized algorithms**
- **Dynamic Programming algorithms**

### 1.6 Finance-Specific Methods (Chapter 1)
- **Monte Carlo Simulation** for risk modeling
- **Particle Filtering** for non-linear/non-Gaussian system modeling
- **Reproducing Kernel Hilbert Spaces (RKHS)** for statistical learning
- **Sparsity-aware learning** for model regularization
- **Convex optimization** for parameter learning
- **Reinforcement Learning** for algorithmic trading
- **SHAP** (Shapley Additive Explanations) for model interpretability
- **LIME** (Local Interpretable Model-Agnostic Explanations)
- **Robo-advisors** for portfolio optimization

---

## 2. STOCK PREDICTION METHODS

### 2.1 Data
- **Stock**: Century Textiles (NSE India, diversified sector)
- **Period**: 31 Dec 2012 to 9 Jan 2015
- **Granularity**: 5-minute intervals
- **Training set**: 19,500 records (Year 1: 2013)
- **Test set**: 20,500 records (Year 2: 2014-2015)
- **Features per record**: open, high, low, close, volume (no adjusted attributes)
- **Mean open value (test set)**: 475.70

### 2.2 Prediction Approach
- **Walk-forward validation**: Train on historical data; predict next 5 days (Mon-Fri); after the week completes, extend training set with actual values and re-predict.
- **Univariate models**: predict future `open` from past `open` values only.
- **Multivariate models**: predict future `open` from all 5 attributes (open, high, low, close, volume).
- **Input windows**: 5 days (1 week) or 10 days (2 weeks).

### 2.3 Ten Model Architectures

| # | Model Name | Type | Input Shape | Input Nature | Epochs | Batch Size | #Params |
|---|-----------|------|------------|-------------|--------|-----------|---------|
| 1 | CNN_UNIV_5 | CNN | (5,1) | Univariate, 1 week | 20 | 4 | 289 |
| 2 | CNN_UNIV_10 | CNN | (10,1) | Univariate, 2 weeks | 70 | 16 | 769 |
| 3 | CNN_MULTV_10 | CNN (multi-channel) | (10,5) | Multivariate, 2 weeks | 70 | 16 | 7,373 |
| 4 | CNN_MULTH_10 | CNN (multi-headed) | 5x(10,1) | Multivariate, 2 weeks | 70 | 16 | 132,965 |
| 5 | LSTM_UNIV_5 | LSTM | (5,1) | Univariate, 1 week | 20 | 16 | 182,235 |
| 6 | LSTM_UNIV_10 | LSTM | (10,1) | Univariate, 2 weeks | 20 | 16 | 182,235 |
| 7 | LSTM_UNIV_ED_10 | LSTM Encoder-Decoder | (10,1) | Univariate, 2 weeks | 70 | 16 | 502,601 |
| 8 | LSTM_MULTV_ED_10 | LSTM Encoder-Decoder | (10,5) | Multivariate, 2 weeks | 20 | 16 | 505,801 |
| 9 | LSTM_UNIV_CNN_10 | CNN encoder + LSTM decoder | (10,1) | Univariate, 2 weeks | 20 | 16 | 347,209 |
| 10 | LSTM_UNIV_CONV_10 | ConvLSTM | (10,1) | Univariate, 2 weeks | 20 | 16 | 384,777 |

### 2.4 CNN_UNIV_5 Architecture Detail
```
Input(5,1) -> Conv1D(filters=16, kernel=3) -> MaxPool1D(pool=2) -> Flatten -> Dense(10) -> Dense(5, sigmoid)
```
- Activation: ReLU (all layers except output), Sigmoid (output)
- Optimizer: Adam

### 2.5 CNN_MULTV_10 Architecture Detail
```
Input(10,5) -> Conv1D(32,3) -> Conv1D(32,3) -> MaxPool1D(2) -> Conv1D(16,3) -> MaxPool1D(2) -> Flatten -> Dense(100) -> Dense(5)
```
- Feature dimension formula: f = (k - n) + 1

### 2.6 CNN_MULTH_10 Architecture Detail
```
5 parallel sub-CNNs (one per variable):
  Each: Input(10,1) -> Conv1D(32,3) -> Conv1D(32,3) -> MaxPool1D(2) -> Flatten(96)
Concatenate(5 x 96 = 480) -> Dense(200) -> Dense(100) -> Dense(5)
```

### 2.7 LSTM_UNIV_5 Architecture Detail
```
Input(5,1) -> LSTM(200) -> Dense(100) -> Dense(5) -> Dense(5)
```

### 2.8 LSTM_UNIV_ED_10 Architecture Detail (Encoder-Decoder)
```
Input(10,1) -> LSTM_Encoder(200) -> RepeatVector(5) -> LSTM_Decoder(200) -> TimeDistributed(Dense(100)) -> TimeDistributed(Dense(1))
```

### 2.9 LSTM_UNIV_CNN_10 Architecture Detail
```
Input(10,1) -> Conv1D(64,3) -> Conv1D(64,3) -> MaxPool1D(2) -> Flatten(192) -> RepeatVector(5) -> LSTM(200) -> TimeDistributed(Dense(100)) -> TimeDistributed(Dense(1))
```

### 2.10 LSTM_UNIV_CONV_10 Architecture Detail (ConvLSTM)
```
Input(10,1) -> ConvLSTM2D(64, kernel=3) -> Flatten(192) -> RepeatVector(5) -> LSTM(200) -> TimeDistributed(Dense(100)) -> TimeDistributed(Dense(1))
```
- Uses Keras ConvLSTM2d class tweaked for 1D univariate data

### 2.11 Hardware/Software
- Intel i7, 2.60-2.56 GHz, 16GB RAM
- Python 3.7.4, TensorFlow 2.3.0, Keras 2.4.5

---

## 3. MEDICAL IMAGING / POSE ESTIMATION

### 3.1 Pose Estimation Architecture (Chapter 3)
- **Input**: Fixed-size image
- **Output**: 2D anatomical keypoints per person
- **Architecture**: Two-branch CNN running in parallel
  - Branch 1: Predicts heatmaps H = {H1, H2, ..., Hj} for j body joints
  - Branch 2: Predicts 2D vector fields P = {L1, L2, ..., Lk} for k limb associations
- Outputs are combined via greedy parsing algorithm and fed through multiple convolutional layers

**Preprocessing**: Standard image preprocessing before feeding into CNN

**Evaluation Dataset**: COCO dataset subset
- Training: 3,000 images
- Cross-validation: 1,100 images
- Testing: 568 images (Table 1), 1,000 images (Table 2)

**Evaluation Metric**: OKS (Object Keypoint Similarity), mean Average Precision (mAP), minimum OKS threshold = 0.5, keypoints within 2.77 standard deviations

### 3.2 Medical Image Types (Chapter 4)
- Ultrasound, X-ray, Mammography, Fluoroscopy
- CT (Computed Tomography), CT angiography
- MRI (Magnetic Resonance Imaging), MRA (Magnetic Resonance Angiography)
- Nuclear Medicine, PET (Positron Emission Tomography)

### 3.3 Ensemble CNN Architectures for Medical Imaging
- **ResNet101 + Inception-V3** stacking: ear disease detection, 93.67% accuracy (5-fold CV)
- **ResNet34 + DenseNet169 + DenseNet201** ensemble: shoulder X-ray classification, Cohen's kappa = 0.6942
- **Xception + ResNet + Inception-V3**: ankle fracture, 81% accuracy (596 cases)
- **COVID-19 detection ensemble**: 99.01% accuracy, AUC = 0.9972 (iterative pruned deep learning)
- **TB detection ensemble**: ROC AUC = 0.99 (Shenzhen), 0.97 (Montgomery), hand-crafted + deep CNN features

### 3.4 DCM Precision Medicine System (Chapter 5)
- **Data**: Demographic (numerical + categorical), diagnostic, 3D MRI sequences (T2w sagittal)
- **MRI Preprocessing**: Resampled to 1mm^3 voxel size, signal normalized to [0,1]
- **Categorical data**: One-hot encoded
- **Numerical data**: Scaled to [0,1] for known min/max ranges
- **Target**: mJOA (modified Japanese Orthopedic Association) DCM severity scale (4 classes)
- **Data storage**: MongoDB non-relational document database, BIDS format for MRI
- **Missing data**: Dropped entries with malformed/missing data; imputation for new patient entries
- **Batch effects**: Handled via iterative training integration

---

## 4. ENSEMBLE METHODS

### 4.1 Bagging (Bootstrap Aggregation)
- **Method**: Random sampling with replacement to create multiple datasets from original training data
- **Purpose**: Reduce variance
- **Process**: Train independent models on bootstrap samples; aggregate predictions by averaging (regression) or majority vote (classification)
- Each element can appear multiple times in new dataset

### 4.2 Boosting
- **Method**: Sequential training; each subsequent model corrects errors of previous models
- **Purpose**: Reduce bias
- **Process**: Weak learners trained iteratively; misclassified samples receive higher weight; final prediction is weighted average of all learners
- Transforms weak learners into strong learners
- Examples: AdaBoost, Gradient Boosting

### 4.3 Stacking
- **Method**: Train multiple heterogeneous base classifiers on same data; use a meta-learner to combine their predictions
- **Purpose**: Improve overall prediction quality
- **Process**: Lower-level models trained in parallel on full dataset; their outputs become features for a higher-level meta-model
- Different from boosting: base learners train in parallel, not sequentially

### 4.4 Random Forest
- **Method**: Ensemble of decision trees fitted on bootstrap samples
- **Purpose**: Reduce variance and decorrelate trees
- **Process**: Random subset of features sampled for each tree split; outputs aggregated (average for regression, vote for classification)
- Effective for handling missing data
- Bias toward features with more distinct values (algorithmic bias consideration)

### 4.5 Ensemble Framework for Medical Image Classification
```
Dataset (Medical Images)
  -> Sample 1 -> Learner 1 (e.g., ResNet) -> Prediction 1
  -> Sample 2 -> Learner 2 (e.g., DenseNet) -> Prediction 2
  -> Sample N -> Learner N (e.g., Inception) -> Prediction N
  -> Ensemble Aggregation -> Final Prediction
```
**Pipeline**:
1. Pre-process images
2. Split into training and validation sets
3. Activate input function
4. Train models (compute loss, adjust weights via gradient descent)
5. Evaluate on validation set (adjust learning rate, select best model version)
6. Test set for final comparison
7. Report results

---

## 5. EVALUATION METRICS

### 5.1 RMSE (Root Mean Square Error)
- Primary metric for stock prediction models
- Computed per day (Day1=Monday through Day5=Friday) and aggregated
- Also computed as ratio: **RMSE / Mean(target variable)**

### 5.2 Execution Time
- Time in seconds for one complete round of model execution

### 5.3 BAPC (Best Accuracy Per Cycle)
- Maximum validation accuracy observed in each training cycle
- Mean BAPC averaged across all cycles

### 5.4 mAP (mean Average Precision)
- Used for pose estimation evaluation
- Calculated over different OKS (Object Keypoint Similarity) thresholds
- Per-body-part breakdown: Head, Shoulder, Elbow, Hip, Knee, Ankle, Wrist

### 5.5 OKS (Object Keypoint Similarity)
- COCO evaluation metric for keypoint detection
- Minimum threshold: 0.5

### 5.6 Cohen's Kappa
- Used for medical image classification (shoulder X-ray ensemble: best = 0.6942)

### 5.7 AUC (Area Under Curve)
- ROC AUC for classification (COVID-19 detection: 0.9972; TB detection: 0.99 Shenzhen, 0.97 Montgomery)

### 5.8 Accuracy
- Classification accuracy percentage (e.g., 81%, 93.67%, 99.01%)

### 5.9 Confusion Matrix / Classification Metrics
- Referenced for ensemble classifiers: accuracy, precision, prediction outcomes

### 5.10 mJOA Scale
- Modified Japanese Orthopedic Association scale: 4 severity classes for DCM diagnostic prediction

---

## 6. FEATURE ENGINEERING

### 6.1 Stock Price Features
- **Raw features**: open, high, low, close, volume (5 attributes per record)
- No adjusted attributes (adjusted close, adjusted volume) used
- Univariate models use only `open`; multivariate models use all 5

### 6.2 Principal Component Analysis (PCA)
PCA equations for dimensionality reduction:

**Standardization**:
```
Z = (x - mu) / sigma
```

**Principal Components**:
```
PC1 = w1,1(Feature_A) + w2,1(Feature_B) + ... + wn,1(Feature_N)
PC2 = w1,2(Feature_A) + w2,2(Feature_B) + ... + wn,2(Feature_N)
PCi = wi1*X1 + wi2*X2 + ... + wip*Xp
```
Where: var(PCi) = lambda_i, and w_i1^2 + w_i2^2 + ... + w_ip^2 = 1

**Covariance Matrix**:
```
C = | w11  w12  w13 |
    | w21  w22  w23 |
    | wn1  wn2  wn3 |
```
Eigenvalues of C = variances of principal components.

### 6.3 Medical Image Feature Extraction
- Image pre-processing: suppress distortions, enhance features
- Object detection: localization, segmentation
- Feature extraction via deep learning or statistical methods
- Classification using extracted feature patterns

### 6.4 Financial Feature Engineering Challenges
- Large number of features in ML models
- Unstructured data (text, images, speech) generates enormous feature sets after preprocessing
- AutoML frameworks automatically generate derived features (risk of overfitting)
- Feature strategy differs by application criticality

### 6.5 DCM Clinical Features
- Demographic: age, sex (categorical one-hot encoded; numerical scaled [0,1])
- Diagnostic: mJOA scale and other clinical tests
- MRI: T2w sagittal sequences, resampled to 1mm^3 voxels, signal normalized [0,1]

---

## 7. MODEL ARCHITECTURES

### 7.1 Progressive Learning ConvNet (CIFAR-10 Testing)

| Block | Type | Size | Other |
|-------|------|------|-------|
| 1 | 2DConvolution | 32, 3x3 | Stride=1 |
| | 2DBatchNorm | | |
| | ReLU | | |
| | [Concatenation] | | |
| 2 | 2DConvolution | 32, 3x3 | Stride=1 |
| | 2DBatchNorm | | |
| | ReLU | | |
| | 2DMaxPooling | 2x2 | Stride=2 |
| | Dropout | r=0.25 | |
| | [Concatenation] | | |
| 3 | 2DConvolution | 64, 3x3 | Stride=1 |
| | 2DBatchNorm | | |
| | ReLU | | |
| | [Concatenation] | | |
| 4 | 2DConvolution | 64, 3x3 | Stride=1 |
| | 2DBatchNorm | | |
| | ReLU | | |
| | 2DMaxPooling | 2x2 | Stride=2 |
| | Dropout | r=0.25 | |
| | [Concatenation] | | |
| 5 | Flatten | | |
| | Linear | 512 | |
| | 1DBatchNorm | 512 | |
| | ReLU | | |
| | Dropout | r=0.5 | |
| 6 | [Concatenation] | | |
| | Linear | 20 | |
| | Softmax | | |

### 7.2 DenseNet-169 Architecture (CIFAR-10 Testing)

| Block | Type | Size | Other |
|-------|------|------|-------|
| 1 | 2DConvolution | 64, 7x7 | Stride=2, Padding=3 |
| | 2DBatchNorm | | |
| | ReLU | | |
| | 2DMaxPooling | 3x3 | Stride=2, Padding=1 |
| | [Concatenation] | | |
| 2 | Dense Block | Layers=6 | Bottleneck=4, r=0.2 |
| | Transition Block | | |
| | [Concatenation] | | |
| 3 | Dense Block | Layers=12 | Bottleneck=4, r=0.2 |
| | Transition Block | | |
| | [Concatenation] | | |
| 4 | Dense Block | Layers=32 | Bottleneck=4, r=0.2 |
| | Transition Block | | |
| | [Concatenation] | | |
| 5 | Dense Block | Layers=32 | Bottleneck=4, r=0.2 |
| | Transition Block | | |
| | [Concatenation] | | |
| 6 | 2DBatchNorm | 1664 | |
| | [Concatenation] | | |
| 7 | ReLU | | |
| | 2DAdaptiveAveragePool | 1x1 | |
| | Flatten | | |
| | Linear | 1664 | |

- Growth rate: 32; Initial features: 64
- 4 dense blocks with 6, 12, 32, 32 convolution layers respectively
- Progressive new blocks: half growth rate (16), half initial features (32)

### 7.3 Pose Estimation Multistage Architecture
- **Backbone**: Feed-forward CNN (e.g., VGG-based)
- **Branch 1**: Heatmap prediction (j heatmaps for j body joints)
- **Branch 2**: Greedy Part Association Vector fields (k vector fields for k limb pairs)
- **Fusion**: Greedy parsing algorithm combines outputs
- **Stages**: Multiple refinement stages with two parallel branches feeding forward

### 7.4 DCM Precision Medicine Architecture
- **Parallel branches**: One per data form/type
  - MRI branch: 3D DenseNet (for 3D spatial MRI sequences)
  - Demographic/diagnostic branch: DCLN blocks
- **Merging blocks**: Combine branch outputs
- **Output layer**: Linear -> SoftMax (4-class categorical prediction)
- **Progressive additions**: New blocks at half parameter count, concatenated to prior blocks

---

## 8. TRAINING PROCEDURES

### 8.1 Stock Prediction Models
- **Activation**: ReLU (all hidden layers), Sigmoid (output layer)
- **Optimizer**: Adam (gradient descent)
- **Hyperparameter tuning**: Grid search for node counts, layer sizes, kernel sizes
- **Epochs**: 20-70 depending on model
- **Batch sizes**: 4-16 depending on model
- **10 rounds per model** for robust evaluation; average performance reported

### 8.2 Progressive Learning ConvNet
- **Optimizer**: ADAM, learning rate = 0.001, beta1 = 0.99, beta2 = 0.999, weight decay lambda = 0.001
- **Post-pruning optimizer**: Same ADAM with 1/10 learning rate
- **Training**: 90 epochs per cycle
- **Pruning**: 10 epochs per pruning cycle; repeat until mean accuracy of prior epochs > new epochs; restore to prior state
- **Data augmentation**: Applied to each image to discourage memorization
- **Time limit**: 8 hours wall time
- **Hardware**: Tesla V100-PCIE-16GB GPU, 16GB RAM, 2x Intel Xeon Gold 6148 @ 2.40GHz
- **Framework**: PyTorch 1.8.1

### 8.3 DenseNet-169 Training
- **Optimizer**: SGD, initial learning rate = 0.1, weight decay = 0.0001, momentum = 0.9
- **Learning rate schedule**: Reduce by factor of 10 at 50% and 75% of total epochs per cycle
- **Batch size**: 64
- **Epochs per cycle**: 300
- **Dropout**: r = 0.2 after each dense block

### 8.4 Progressive Learning Pruning Strategy
- **Method**: Drop parameters with lowest absolute value weights
- **Global pruning**: Consider all parameters at once (alternative: layer-by-layer greedy)
- **DCM system**: 10% lowest absolute weight pruning, applied globally, iteratively until accuracy loss observed over 10 post-prune correction epochs
- **Frozen vs Free priors**: Frozen priors prevent catastrophic forgetting but can cause stagnation; free priors allow full model update

### 8.5 Regularization Techniques Referenced
- **Dropout**: r = 0.2 (DenseNet), r = 0.25 (Conv blocks), r = 0.5 (dense blocks)
- **Batch Normalization**: Applied after convolution layers (2D and 1D)
- **Weight decay**: 0.001 (ConvNet), 0.0001 (DenseNet)
- **Model pruning**: Iterative parameter removal to prevent overfitting
- **Sparsity-aware learning**: Alternative regularization for iterative estimation tasks

---

## 9. ALL TABLES AND RESULTS

### Table 1: CNN_UNIV_5 Parameters
| Layer | k | d | f | pprev | pcurr | #params |
|-------|---|---|---|-------|-------|---------|
| Conv1D | 3 | 1 | 16 | - | - | 64 |
| Dense | - | - | - | 16 | 10 | 170 |
| Dense_1 | - | - | - | 10 | 5 | 55 |
| **Total** | | | | | | **289** |

### Table 2: CNN_UNIV_10 Parameters
| Layer | k | d | f | pprev | pcurr | #params |
|-------|---|---|---|-------|-------|---------|
| Conv1D | 3 | 1 | 16 | - | - | 64 |
| Dense | - | - | - | 64 | 10 | 650 |
| Dense_1 | - | - | - | 10 | 5 | 55 |
| **Total** | | | | | | **769** |

### Table 3: CNN_MULTV_10 Parameters
| Layer | k | d | f | pprev | pcurr | #params |
|-------|---|---|---|-------|-------|---------|
| Conv1D_4 | 3*5 | 1 | 32 | - | - | 512 |
| Conv1D_5 | 3 | 32 | 32 | - | - | 3,104 |
| Conv1D_6 | 3 | 32 | 16 | - | - | 1,552 |
| Dense_3 | - | - | - | 16 | 100 | 1,700 |
| Dense_4 | - | - | - | 100 | 5 | 505 |
| **Total** | | | | | | **7,373** |

### Table 4: CNN_MULTH_10 Parameters
| Layer | k | d | f | pprev | pcurr | #params |
|-------|---|---|---|-------|-------|---------|
| Conv1D (x5 first layer) | 3 | 1 | 32 | - | - | 640 |
| Conv1D (x5 second layer) | 3 | 32 | 32 | - | - | 15,520 |
| Dense_1 | - | - | - | 480 | 200 | 96,200 |
| Dense_2 | - | - | - | 200 | 100 | 20,100 |
| Dense_3 | - | - | - | 100 | 5 | 505 |
| **Total** | | | | | | **132,965** |

### Table 5: LSTM_UNIV_5 Parameters
| Layer | x | y | pprev | pcurr | #params |
|-------|---|---|-------|-------|---------|
| LSTM | 200 | 1 | - | - | 161,600 |
| Dense_4 | - | - | 200 | 100 | 20,100 |
| Dense_5 | - | - | 100 | 5 | 505 |
| Dense_6 | - | - | 5 | 5 | 30 |
| **Total** | | | | | **182,235** |

### Table 6: LSTM_UNIV_10 Parameters
(Identical to LSTM_UNIV_5: **182,235** total parameters)

### Table 7: LSTM_UNIV_ED_10 Parameters
| Layer | x | y | pprev | pcurr | #params |
|-------|---|---|-------|-------|---------|
| LSTM_3 (encoder) | 200 | 1 | - | - | 161,600 |
| LSTM_4 (decoder) | 200 | 200 | - | - | 320,800 |
| Dense (time_dist_3) | - | - | 200 | 100 | 20,100 |
| Dense (time_dist_4) | - | - | 100 | 1 | 101 |
| **Total** | | | | | **502,601** |

### Table 8: LSTM_MULTV_ED_10 Parameters
| Layer | x | y | pprev | pcurr | #params |
|-------|---|---|-------|-------|---------|
| LSTM_1 (encoder) | 200 | 5 | - | - | 164,800 |
| LSTM_2 (decoder) | 200 | 200 | - | - | 320,800 |
| Dense (time_dist_1) | - | - | 200 | 100 | 20,100 |
| Dense (time_dist_2) | - | - | 100 | 1 | 101 |
| **Total** | | | | | **505,801** |

### Table 9: LSTM_UNIV_CNN_10 Parameters
| Layer | k/x | d/y | f | pprev | pcurr | #params |
|-------|-----|-----|---|-------|-------|---------|
| Conv1D_4 | 3 | 1 | 64 | - | - | 256 |
| Conv1D_5 | 3 | 64 | 64 | - | - | 12,352 |
| LSTM | 200 | 192 | - | - | - | 314,400 |
| Dense (time_dist_4) | - | - | - | 200 | 100 | 20,100 |
| Dense (time_dist_5) | - | - | - | 100 | 1 | 101 |
| **Total** | | | | | | **347,209** |

### Table 10: LSTM_UNIV_CONV_10 Parameters
| Layer | k/x | d/y | f | pprev | pcurr | #params |
|-------|-----|-----|---|-------|-------|---------|
| ConvLSTM2D | 3 | 1 | 64 | - | - | 50,176 |
| LSTM | 200 | 192 | - | - | - | 314,400 |
| Dense (time_dist) | - | - | - | 200 | 100 | 20,100 |
| Dense (time_dist_1) | - | - | - | 100 | 1 | 101 |
| **Total** | | | | | | **384,777** |

### Table 11: CNN_UNIV_5 Performance (10 rounds)
| Round | Agg RMSE | Day1 | Day2 | Day3 | Day4 | Day5 | Time(s) |
|-------|----------|------|------|------|------|------|---------|
| 1 | 4.058 | 4.00 | 3.40 | 3.90 | 4.40 | 4.50 | 173.95 |
| 2 | 3.782 | 3.10 | 3.30 | 3.80 | 4.10 | 4.40 | 176.92 |
| 3 | 3.378 | 2.80 | 3.00 | 3.40 | 3.60 | 3.90 | 172.21 |
| 4 | 3.296 | 2.60 | 3.00 | 3.30 | 3.60 | 3.90 | 173.11 |
| 5 | 3.227 | 2.60 | 3.00 | 3.30 | 3.50 | 3.70 | 174.72 |
| 6 | 3.253 | 2.60 | 3.00 | 3.30 | 3.50 | 3.70 | 183.77 |
| 7 | 3.801 | 3.60 | 3.60 | 3.80 | 3.80 | 4.10 | 172.29 |
| 8 | 3.225 | 2.60 | 2.90 | 3.30 | 3.50 | 3.70 | 171.92 |
| 9 | 3.306 | 2.80 | 3.00 | 3.30 | 3.50 | 3.70 | 174.92 |
| 10 | 3.344 | 2.70 | 3.10 | 3.40 | 3.60 | 3.80 | 174.01 |
| **Mean** | **3.467** | **2.94** | **3.13** | **3.48** | **3.71** | **3.94** | **174.78** |
| **RMSE/Mean** | **0.007288** | 0.0062 | 0.0066 | 0.0073 | 0.0078 | 0.0083 | |

### Table 12: CNN_UNIV_10 Performance (Mean)
| Metric | Value |
|--------|-------|
| Mean Agg RMSE | 3.3142 |
| RMSE/Mean | 0.006967 |
| Mean Time | 185.01s |
| Day1/Day2/Day3/Day4/Day5 RMSE/Mean | 0.0056/0.0067/0.0070/0.0075/0.0080 |

### Table 13: CNN_MULTV_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 4.4799 | 0.009420 | 202.78s |

### Table 14: CNN_MULTH_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 3.864 | 0.008100 | 215.07s |

### Table 15: LSTM_UNIV_5 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 3.6418 | 0.007770 | 371.62s |

### Table 16: LSTM_UNIV_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 3.5126 | 0.007380 | 554.47s |

### Table 17: LSTM_UNIV_ED_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 3.971 | 0.008350 | 307.27s |

### Table 18: LSTM_MULTV_ED_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 4.897 | 0.010294 | 634.34s |

### Table 19: LSTM_UNIV_CNN_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 3.766 | 0.007916 | 222.48s |

### Table 20: LSTM_UNIV_CONV_10 Performance (Mean)
| Mean Agg RMSE | RMSE/Mean | Mean Time |
|---------------|-----------|-----------|
| 3.563 | 0.007490 | 265.97s |

### Table 21: COMPARATIVE ANALYSIS -- ALL 10 MODELS

| # | Model | #Params | RMSE/Mean | Accuracy Rank | Exec Time(s) | Speed Rank |
|---|-------|---------|-----------|---------------|---------------|------------|
| 1 | CNN_UNIV_5 | 289 | 0.007288 | 2 | 174.78 | 1 |
| 2 | CNN_UNIV_10 | 769 | **0.006967** | **1** | 180.01 | 2 |
| 3 | CNN_MULTV_10 | 7,373 | 0.009420 | 9 | 202.78 | 3 |
| 4 | CNN_MULTH_10 | 132,965 | 0.008100 | 7 | 215.07 | 4 |
| 5 | LSTM_UNIV_5 | 182,235 | 0.007770 | 5 | 371.62 | 8 |
| 6 | LSTM_UNIV_10 | 182,235 | 0.007380 | 3 | 554.47 | 9 |
| 7 | LSTM_UNIV_ED_10 | 502,601 | 0.008350 | 8 | 307.27 | 7 |
| 8 | LSTM_MULTV_ED_10 | 505,801 | 0.010294 | 10 | 634.34 | 10 |
| 9 | LSTM_UNIV_CNN_10 | 347,209 | 0.007916 | 6 | 222.48 | 5 |
| 10 | LSTM_UNIV_CONV_10 | 384,777 | 0.007490 | 4 | 265.97 | 6 |

**Key findings**:
- Best accuracy: CNN_UNIV_10 (RMSE/Mean = 0.006967)
- Fastest execution: CNN_UNIV_5 (174.78s)
- Lowest RMSE ratio overall: 0.006967
- CNN models are uniformly faster than LSTM counterparts
- Univariate models outperform multivariate counterparts on accuracy
- Multivariate models ranked 9th and 10th on accuracy

### Pose Estimation Results -- Table 1 (568 test images)
| Method | Head | Shoulder | Elbow | Hip | Knee | Ankle | Wrist | mAP |
|--------|------|----------|-------|-----|------|-------|-------|-----|
| DeepCut | 73.4 | 71.8 | 57.9 | 56.7 | 44.0 | 32.0 | 39.9 | 54.1 |
| Iqbal et al | 70.0 | 65.2 | 56.4 | 52.7 | 47.9 | 44.5 | 46.1 | 54.7 |
| DeeperCut | 87.9 | 84.0 | 71.9 | 68.8 | 63.8 | 58.1 | 63.9 | 71.2 |
| **Proposed** | **90.7** | **90.9** | **79.8** | **76.1** | **70.2** | **66.3** | **70.5** | **77.7** |

### Pose Estimation Results -- Table 2 (1000 test images)
| Method | Head | Shoulder | Elbow | Hip | Knee | Ankle | Wrist | mAP |
|--------|------|----------|-------|-----|------|-------|-------|-----|
| DeepCut | 78.4 | 72.5 | 60.2 | 57.2 | 52.0 | 45.0 | 51.0 | 54.1 |
| Iqbal et al | 58.4 | 53.9 | 44.5 | 42.2 | 36.7 | 31.1 | 35.0 | 54.7 |
| DeeperCut | 87.9 | 84.0 | 71.9 | 68.8 | 63.8 | 58.1 | 63.9 | 71.2 |
| **Proposed** | **90.1** | **87.9** | **75.8** | **73.1** | **65.2** | **60.3** | **66.5** | **73.7** |

**Improvement**: +6.5% mAP on 568 images; +2.5% mAP on 1000 images vs. DeeperCut

### Progressive Learning Results (CIFAR-10)

| Setup | Data | Max Accuracy | Mean BAPC |
|-------|------|-------------|-----------|
| Static ConvNet | Full | 81.13% | 80.71% |
| Progressive ConvNet (frozen) | Full | 84.80% | 77.44% |
| Progressive ConvNet (free) | Full | **90.66%** | **84.83%** |
| Static ConvNet | Batched | 73.70% | 71.75% |
| Progressive ConvNet (frozen) | Batched | 75.90% | 66.00% |
| Progressive ConvNet (free) | Batched | **82.40%** | **79.02%** |
| Static DenseNet | Full | 90.00% | - |
| Progressive DenseNet (frozen) | Full | 90.66% | - |
| Progressive DenseNet (free) | Full | 90.66% | - |
| Static DenseNet | Batched | 69.90% | 65.87% |
| Progressive DenseNet (frozen) | Batched | 67.10% | 64.28% |
| Progressive DenseNet (free) | Batched | **71.70%** | **68.28%** |

---

## 10. ALL EQUATIONS

### Equation (1): CNN Trainable Parameters in 1D Convolutional Layer
```
n1 = (k * d + 1) * f
```
Where:
- k = kernel size
- d = feature space dimension of previous layer
- f = feature space dimension of current layer
- 1 = bias per feature element

### Equation (2): CNN Dense Layer Parameters
```
n2 = (pcurr * pprev) + 1 * pcurr
```
Simplified: `n2 = pcurr * (pprev + 1)`
Where:
- pcurr = node count in current layer
- pprev = node count in previous layer

### Equation (3): LSTM Gate Parameters
```
n1 = (x + y) * x + x
```
Total LSTM layer parameters = `4 * n1`
Where:
- x = number of LSTM units
- y = input dimension (number of features)
- 4 gates: forget, input, input modulation, output

### Equation (4): LSTM Dense Layer Parameters
```
n2 = pprev * pcurr + pcurr
```
Where:
- pprev = nodes in previous layer
- pcurr = nodes in current layer

### Equation (5): ConvLSTM2D Parameters
```
params = 4 * x * [k * (1 + x) + 1]
```
Where:
- x = number of ConvLSTM units (64)
- k = kernel size (3)

### Equation (6): CNN Feature Space Output Dimension
```
f = (k - n) + 1
```
Where:
- k = input dimension
- n = kernel/filter size

### Equation (7): Pose Estimation -- Confidence Map
```
C*_jk(m) = exp(-Delta^2 / sigma^2)
```
Where:
- sigma = spread from mean
- Delta = |x_jk - m| (absolute difference between empirical joint position and location)
- x_jk = empirical position of body joint j for person k

### Equation (8): Pose Estimation -- Aggregated Confidence Map
```
C*_j(m) = max_k(C*_jk(m))
```
Final map takes maximum of all individual person confidence maps.

### Equation (9): Greedy Part Association Vector
```
G*_j,k = c_hat  if c is on limb j and person k; 0 otherwise
```
Where c_hat is a unit vector along direction of limb:
```
c_hat = (x2 - x1) / sqrt(x2^2 - x1^2)
```

### Equation (10): Averaged GPAV
```
G*_j = sum_k(G*_j,k) / n_j(c)
```
Where n_j(c) = total number of vectors at point c among all people.

### Equation (11): Association Score (Line Integral)
```
E = integral_0^1 G_j(c(m)) . d_hat dm
```
Where:
- G_j(c(m)) = greedy part association vector at interpolated point
- d_hat = unit vector along direction between two detected part candidates t1 and t2

### Equation (12): PCA Standardization
```
Z = (x - mu) / sigma
```

### Equations (13-15): PCA Principal Components
```
PC1 = w1,1(Feature_A) + w2,1(Feature_B) + ... + wn,1(Feature_N)
PC2 = w1,2(Feature_A) + w2,2(Feature_B) + ... + wn,2(Feature_N)
PC3 = w1,3(Feature_A) + w2,3(Feature_B) + ... + wn,3(Feature_N)
```

### Equation (16): PCA General Form
```
PCi = wi1*X1 + wi2*X2 + ... + wip*Xp
```
Where var(PCi) = lambda_i

### Equation (17): PCA Constraint
```
w_i1^2 + w_i2^2 + ... + w_ip^2 = 1
```

### Equation (18): Logistic Regression
```
h_theta(x) = omega*x + b,   where theta = (w, b)
```
For binary: `0 <= h_theta(x) <= 1`

### Equation (19): Sigmoid Function
```
g(z) = 1 / (1 + e^(-z))
```

### Equation (20): Logistic Regression Prediction
```
h_theta(x) = 1 / (1 + e^(-(omega*x + b)))
```

### Equation (21): Prediction Output
```
y_predict = g(z) = 1 / (1 + e^(-z))   [Sigmoid Function, optimized]
```

### Equation (22): Shannon Entropy
```
Entropy(P) = -sum_{i=1}^{n} Pi * log(Pi)
```
Where P = (p1, p2, ..., pn) is a probability distribution.

### Equation (23): Information Gain
```
Gain(P, T) = Entropy(P) - sum_{j=1}^{n} Pj * Entropy(Pj)
```
Where Pj is the set of all values of attribute T.

### Equation (24): Neural Network Weighted Sum
```
a = sum_{j=1}^{n} w_theta * x_j + b
```
Where a = output from multiple inputs; fed into transfer function f to produce y.

### Equation (25): Covariance Matrix
```
C = | w11  w12  w13 |
    | w21  w22  w23 |
    | wn1  wn2  wn3 |
```
Eigenvalues of C = variances of principal components.

---

## APPENDIX: KEY OBSERVATIONS AND FINDINGS

### Stock Prediction
1. CNN models are uniformly faster than LSTM counterparts
2. Univariate models outperform multivariate models on accuracy (counterintuitive)
3. Best model: CNN_UNIV_10 (RMSE/Mean = 0.006967, Rank 1 accuracy, Rank 2 speed)
4. Fastest model: CNN_UNIV_5 (174.78s per round, only 289 parameters)
5. Worst model: LSTM_MULTV_ED_10 (RMSE/Mean = 0.010294, slowest at 634.34s)
6. RMSE increases monotonically from Monday to Friday within each model
7. Encoder-decoder LSTMs faster than vanilla LSTMs despite more parameters

### Pose Estimation
1. Bottom-up approach decouples runtime from number of people in image
2. Proposed method achieves mAP = 77.7 (568 images), 73.7 (1000 images)
3. Inference time 3 orders of magnitude less than competitors
4. mAP improvement: +6.5% over DeeperCut (568 images), +2.5% (1000 images)

### Progressive Learning for Precision Medicine
1. Progressive learning with free (updatable) priors outperforms all other setups
2. ConvNet: 90.66% (full data, progressive free) vs 81.13% (full data, static)
3. ConvNet batched: 82.40% (progressive free) nearly matches 81.13% (static full)
4. Pruning is critical for preventing overfitting in over-parameterized models
5. DenseNet more affected by batching than ConvNet (69.9% batched vs 90% full)
6. Curriculum stage least significant; can be skipped with small clinical batches

### Ensemble Methods for Medical Imaging
1. Ensemble methods consistently outperform single classifiers
2. Best results: COVID-19 detection at 99.01% accuracy, AUC = 0.9972
3. Voting approaches effective for combining views and models
4. Stacking with Random Forest + SVM + neural networks frequently employed
