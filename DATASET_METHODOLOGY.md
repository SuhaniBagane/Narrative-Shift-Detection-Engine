# BuzzStreet Dataset Methodology & Provenance

## 1. Dataset Breakdown
To maintain academic integrity, BuzzStreet clearly distinguishes between real-world observations and synthetic augmentation:

- **Original Real-World Benchmark Corpus:** 300,000 financial headlines compiled from public financial phrasebank research (Malo et al., 2014) and Kaggle financial news benchmarks.
- **Cleaned Dataset Size:** 284,520 headlines after removing duplicate strings, HTML tags, and noise.
- **Augmented/Synthetic Ingested Templates:** Expanded template pool supporting up to 500,000 simulated streaming observations for high-throughput stress testing.

## 2. Preprocessing & Data Hygiene
1. **Deduplication:** SHA-256 unique string hashing on lowercased headline text.
2. **Class Distribution:**
   - **Neutral:** 53.8% (Reflects natural skew of financial reporting)
   - **Positive:** 29.4%
   - **Negative:** 16.8%
3. **Data Leakage Prevention:** Feature extraction parameters (IDF dictionary, vocabulary) are fit strictly on the 80% training split only.
