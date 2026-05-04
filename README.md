# News Recommendation System — MIND Dataset

**Course:** Machine Learning Capstone  
**Dataset:** MIND-small (MIcrosoft News Dataset)  
**Model:** NRMS — Neural News Recommendation with Multi-Head Self-Attention

---

## Introduction

This project implements an end-to-end neural news recommendation system using the MIND-small benchmark dataset. The goal is to predict which news articles a user is likely to click based on their reading history and the content of candidate articles.

News recommendation is uniquely challenging because:
- User interests shift rapidly (news has a very short lifespan)
- Cold-start problems are pervasive — many users have little or no click history
- The candidate pool changes continuously as new articles are published

The model follows the **NRMS architecture** (Wu et al., EMNLP 2019), which uses multi-head self-attention both to encode individual news articles from their titles and to aggregate a user's clicked history into a single user representation.

---

## Methodology

### Dataset
- **Source:** MIND-small — 50,000 users, ~65,000 articles, ~230,000 impressions
- **Files used:** `behaviors.tsv` (user interaction logs) and `news.tsv` (article metadata)
- **Split:** Temporal train/dev split as provided by MIND (no random shuffling across time boundary)
- **Scale:** 236,344 training samples and 73,152 dev samples after preprocessing

### Data Preprocessing
1. **Tokenization:** NLTK word tokenization on news titles (lowercased); vocabulary built from training set with minimum frequency = 2
2. **Word Embeddings:** GloVe 300-dimensional pre-trained vectors (glove.6B.300d); words not in GloVe are randomly initialized. Final embedding matrix: 20,774 × 300
3. **Title Encoding:** Each title is mapped to a fixed-length sequence of word indices (max 30 tokens; padded or truncated)
4. **Behavior Parsing:** Each impression is split into positive (clicked) and negative (not-clicked) articles; 1 positive + 4 negative samples per training example
5. **History:** Each user's clicked history is truncated to the most recent 50 articles and padded to a fixed length with a binary mask

### Model Architecture

```
NEWS ENCODER                        USER ENCODER
────────────                        ────────────
Title Words (30 tokens)             Clicked History [h1, h2, ..., hH]
      │                                      │
Word Embedding (GloVe, 300d)        News Encoder (shared weights)
      │                                      │
Linear Projection → 256d            Multi-Head Self-Attention (16 heads)
      │                                      │
Multi-Head Self-Attention (16h)     Additive Attention (learned query)
      │                                      │
Additive Attention (learned query)  User Vector (256d)
      │                                      │
News Vector (256d)                           │
      └──────────── DOT PRODUCT ─────────────┘
                         │
                   Click Score
```

**Key components:**
- **Additive Attention:** A learned query collapses a variable-length sequence into a single vector, weighting tokens by their importance. This allows the model to focus on the most informative words in a title or the most relevant articles in a user's history, rather than treating all inputs equally.
- **Shared News Encoder:** The same encoder is used for both history articles and candidate articles, reducing parameters and ensuring consistency. This means the model learns a single unified representation space for all news, which is critical for the dot-product scoring to be meaningful.
- **Training objective:** Cross-entropy loss over (1 positive + 4 negative) candidates per impression. The model is trained to rank the clicked article above all four negatives simultaneously, which is a stronger signal than binary classification.

### Hyperparameters

| Parameter | Value |
|---|---|
| Max title length | 30 tokens |
| Max history length | 50 articles |
| Negative samples (K) | 4 |
| Attention heads | 16 |
| Head dimension | 16 (total dim = 256) |
| Dropout | 0.2 |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Epochs | 5 |
| LR scheduler | StepLR (×0.5 every 2 epochs) |

---

## Results

### Training Progress — Baseline

The baseline model trained for 5 epochs with a learning rate of 1e-4 and dropout of 0.2. Loss decreased steadily across all epochs, and AUC improved consistently on the dev set, indicating the model was learning without overfitting.

| Epoch | Train Loss | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| 1 | 1.9249 | 0.5759 | 0.2969 | 0.2734 | 0.3414 |
| 2 | 1.5547 | 0.5920 | 0.3041 | 0.2821 | 0.3500 |
| 3 | 1.5146 | 0.6040 | 0.3108 | 0.2905 | 0.3581 |
| 4 | 1.4992 | 0.6076 | 0.3154 | 0.2957 | 0.3615 |
| 5 | 1.4877 | 0.6105 | 0.3135 | 0.2952 | 0.3617 |

The large loss drop between epoch 1 and epoch 2 (1.9249 → 1.5547) reflects the model initially adjusting the randomly-initialized embeddings and projection weights. By epoch 3 the loss plateaued, suggesting the StepLR scheduler was doing its job of preventing overshooting.

### Hyperparameter Experiments

Three configurations were compared, each trained for 5 epochs on the same data. The best result per run (highest dev AUC) is reported.

| Experiment | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| Baseline (lr=1e-5, drop=0.2) | 0.6105 | 0.3135 | 0.2952 | 0.3617 |
| LR=5e-4 | 0.5935 | 0.3113 | 0.2955 | 0.3562 |
| Dropout=0.1 | **0.6144** | **0.3143** | **0.2974** | **0.3630** |

**Key takeaways:**

- **Higher learning rate (LR=5e-4) hurt performance.** AUC peaked at epoch 2 (0.5935) and then degraded, a classic sign of the optimizer overshooting the loss minimum. The StepLR halving compounded this — by epoch 3 the effective LR was already too large to have converged well, and halving it locked in a suboptimal solution.

- **Lower dropout (0.1) was the best configuration.** Reducing dropout from 0.2 to 0.1 gave a consistent improvement across all four metrics. This suggests the baseline was slightly over-regularizing — with only 300-dimensional GloVe embeddings and titles of at most 30 tokens, the model has limited capacity to begin with, so aggressive dropout was unnecessarily constraining it.

- **Diminishing returns are visible in all runs.** The gap between epoch 4 and epoch 5 is small in every experiment, suggesting 5 epochs is a reasonable stopping point for this dataset size and batch size combination. Additional epochs would likely yield minimal gain without learning rate tuning.

### Final Evaluation — Best Checkpoint (Dropout=0.1, Epoch 5)

| Metric | Random Baseline | Our NRMS (best) | Improvement |
|---|---|---|---|
| AUC | 0.500 | **0.6144** | +22.9% |
| MRR | 0.200 | **0.3143** | +57.2% |
| nDCG@5 | 0.200 | **0.2974** | +48.7% |
| nDCG@10 | 0.300 | **0.3630** | +21.0% |

The model substantially outperforms a random baseline across all metrics. MRR and nDCG@5 show the largest relative gains, indicating the model is particularly good at placing the correct article in the top positions of the ranked list — the positions that matter most in a real recommendation interface.

---

## Error Analysis

### Per-Impression AUC Distribution

Evaluating AUC on each individual impression (rather than aggregating) reveals significant heterogeneity. While mean AUC is approximately 0.614, many impressions score near 0.5 (random), while others score close to 1.0. This wide distribution is typical of recommendation systems — the model is highly effective for some users and essentially random for others.

### AUC by History Length

| History Length | Mean AUC |
|---|---|
| 0–5 clicks | ~0.53 |
| 6–15 clicks | ~0.58 |
| 16–30 clicks | ~0.62 |
| 31–50 clicks | ~0.67 |

Performance scales clearly with history length. Users with 31–50 clicks give the model enough signal to build a meaningful user representation, while cold-start users (0–5 clicks) are barely above random. This is the single largest failure mode of the system.

### Key Findings

1. **Cold-start users suffer most.** Users with fewer than 5 history clicks show significantly lower AUC (~0.53) compared to users with 31–50 history articles (~0.67). With so few clicks, the additive attention over history has almost nothing to aggregate, and the resulting user vector is dominated by padding noise rather than genuine interest signals.

2. **AUC distribution is wide.** While the mean AUC is ~0.614, many individual impressions score near 0.5 (random), suggesting the model is highly confident on some users but struggles with others. This heterogeneity is typical of recommendation systems.

3. **Category imbalance affects quality.** The model was trained on far more news/entertainment examples than niche categories (e.g., health, finance). Performance on underrepresented categories is likely lower, though difficult to measure directly from aggregate metrics.

4. **Temporal drift.** The train/dev temporal boundary may cause some degradation, as user interests and news topics shift across weeks. Articles in the dev set are newer than those in training, meaning some vocabulary and topics may not have been seen during training.

### Improvement Directions

- **Category embeddings:** Concatenating a learned category vector to the news representation could help the model specialize per topic, and would give cold-start users a stronger prior based on the categories of their few clicked articles
- **Abstract encoder:** Many articles have meaningful abstracts; encoding both title and abstract and combining them could improve news representations significantly, since titles alone are often too short to capture article content
- **Cold-start mitigation:** For users with few clicks, falling back to a popularity-based ranker or using demographic signals could improve cold-start performance. Alternatively, the model could be augmented with a user-agnostic content similarity component that activates when history is sparse
- Unfortunately the .gitignore led to the models and .pt files not transferring over from the computer lab into my github, likely due to file size, so I was unable to put those into my final repository.

---

## Conclusions

This project demonstrates a working end-to-end neural news recommendation pipeline. The NRMS model, even with only title-level text and GloVe embeddings, achieves meaningful improvements over a random baseline across all four MIND benchmark metrics — most notably a +57% improvement in MRR, meaning the model places clicked articles much higher in the ranked list than chance. 

Hyperparameter experiments showed that the model is sensitive to learning rate (too high causes divergence after initial progress) but benefits from slightly reduced regularization (dropout=0.1 outperformed dropout=0.2), suggesting the baseline architecture has sufficient inductive bias from the attention mechanism without needing heavy dropout.

The most critical bottleneck is cold-start performance for sparse users — a fundamental challenge in news recommendation that motivates future extensions such as knowledge graph integration and contextualized language model embeddings.

---

## Repository Structure

```
mind-recommender/
├── data/
│   ├── MINDsmall_train/      # Raw train split
│   ├── MINDsmall_dev/        # Raw dev split
│   ├── glove/                # GloVe 300d embeddings
│   └── processed/            # Preprocessed tensors (.pkl, .npy)
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis (Phase 2)
│   ├── 02_preprocessing.ipynb # Tokenization, encoding, sampling (Phase 3)
│   └── 03_training.ipynb      # Model, training, evaluation (Phases 4–6)
├── models/                    # Saved checkpoints
├── results/                   # Saved plots and metric outputs
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install torch pandas numpy scikit-learn nltk matplotlib seaborn wordcloud

# 2. Download MIND-small (run Phase 1 code from the guidelines, or manually)
#    Place in data/MINDsmall_train/ and data/MINDsmall_dev/

# 3. Download GloVe 300d
#    https://nlp.stanford.edu/data/glove.6B.zip
#    Extract glove.6B.300d.txt to data/glove/

# 4. Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_training.ipynb
```

---

## References

1. Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. ACL 2020.
2. Wu, C. et al. (2019). NRMS: Neural News Recommendation with Multi-Head Self-Attention. EMNLP 2019.
3. Pennington, J. et al. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014.
