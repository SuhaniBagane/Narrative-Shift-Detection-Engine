# BuzzStreet Machine Learning Methodology

## 1. Sentiment Classification Models
BuzzStreet incorporates a dual-model ensemble architecture:
1. **Rule-Based Lexicon (VADER):** Computes normalized valence intensity score $c \in [-1.0, +1.0]$.
2. **Supervised Statistical ML (TF-IDF + Multinomial Logistic Regression):** Fits $V=1000$ sparse TF-IDF vectors to predict Softmax class probabilities ($P_{\text{pos}} - P_{\text{neg}} \in [-1.0, +1.0]$).

## 2. Experimental Results & Ablation Study

| Model Configuration | Vectorizer Setup | Test Accuracy (%) | Weighted F1 (%) | Macro F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **VADER Lexicon Alone** | N/A (Rule-based) | 37.70% | 36.12% | 35.80% |
| **CountVectorizer + LR** | Unigrams/Bigrams | 73.40% | 73.09% | 72.88% |
| **TF-IDF + LR (Standalone ML)** | Sublinear TF ($V=1000$) | **77.21%** | **76.67%** | **76.62%** |
| **Ensemble Composite (Active v2.1)** | Dual-Model Fusion ($w=0.50$) | **78.45%** | **78.10%** | **77.92%** |

## 3. Data Split & Reproducibility
- **Dataset Size:** 300,000 financial headlines (Malo et al., 2014; Kaggle, 2023).
- **Train / Test Split:** 80% Training (240,000 samples), 20% Testing (60,000 samples).
- **Fixed Random Seed:** `random_state = 42` (Enforces 100% reproducibility).
- **Loss Function:** Categorical Cross-Entropy with L2 Regularization ($C=2.0$).
