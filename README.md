# Depression Detection from Clinical NLP — DAIC-WOZ

Binary depression detection on the DAIC-WOZ dataset using Classical ML baselines (TF-IDF + Logistic Regression / LinearSVC / Random Forest) and fine-tuned multilingual transformer models (mBERT, XLM-RoBERTa).

---

## Project Structure

```
depression-detection-clinical-nlp/
├── data/
│   ├── labels/          # Official DAIC-WOZ split CSVs
│   ├── transcripts/     # Participant interview transcripts
│   └── processed/       # Cleaned label outputs
├── scripts/
│   ├── classical/       # TF-IDF + Classical ML pipeline
│   ├── sentence_transformers/  # Transformer fine-tuning & hyperparameter search
│   ├── data/            # Preprocessing and label-cleaning utilities
│   └── plot_learning_curves.py
├── results/
│   ├── classical/       # Results and figures for Classical ML experiments
│   └── sentence_transformers/  # Results and figures for Transformer experiments
└── models/              # Saved model checkpoints (not tracked by git, ~14 GB)
```

---

## Experimental Setup

### Hardware

All experiments — Classical ML grid search, transformer fine-tuning, and hyperparameter search — were conducted on a single local machine with the following hardware:

| Component | Specification |
|-----------|--------------|
| **Machine** | Apple MacBook Air |
| **Chip** | Apple M3 (Apple Silicon, unified memory architecture) |
| **CPU** | 8-core Apple M3 (4 performance cores + 4 efficiency cores) |
| **GPU** | 10-core Apple M3 integrated GPU |
| **RAM** | 16 GB unified memory (shared between CPU and GPU) |

> **Note on GPU acceleration:** Apple Silicon uses a unified memory architecture, meaning the CPU and GPU share the same 16 GB memory pool. Transformer fine-tuning (mBERT, XLM-RoBERTa) was accelerated via PyTorch's **MPS (Metal Performance Shaders)** backend, which provides GPU-level acceleration on Apple M-series chips without a discrete GPU. Classical ML experiments (scikit-learn) ran on the CPU.

No external GPU, cloud compute, or HPC cluster was used. All results are reproducible on consumer-grade hardware.

---

## Experiments

### Classical ML Baselines (Experiments 1–3)

Three classifiers combined with TF-IDF feature extraction, tuned via 5-fold stratified `GridSearchCV` (scoring: macro-F1):

| Model | Tuned Hyperparameters |
|-------|-----------------------|
| Logistic Regression | TF-IDF vocab size, n-gram range, regularisation C |
| LinearSVC | TF-IDF vocab size, n-gram range, regularisation C |
| Random Forest | TF-IDF vocab size, n-gram range, n_estimators, max_depth |

All classifiers use `class_weight='balanced'` to address the ~1:2.3 class imbalance (57 depressed vs. 132 non-depressed participants).

### Transformer Fine-Tuning (Experiments 4–5)

Two multilingual pre-trained transformer models fine-tuned for binary sequence classification:

| Model | Experiment |
|-------|-----------|
| `bert-base-multilingual-cased` (mBERT) | Experiment 4 |
| `xlm-roberta-base` (XLM-RoBERTa) | Experiment 5 |

Fine-tuning configuration:
- Optimiser: AdamW with linear warmup
- Loss: `CrossEntropyLoss` with inverse-frequency class weights
- Max sequence length: 512 tokens
- Batch size: 8
- Early stopping on validation macro-F1 (patience = 5)

Hyperparameters searched: learning rate ∈ {5e-6, 1e-5, 2e-5}, depressed class weight ∈ {auto, 4.0}, warmup ratio = 0.0.

**Best configurations (from hyperparameter search):**

| Model | LR | Depressed Weight | Test Macro-F1 | F1 (Depressed) |
|-------|----|-----------------|---------------|----------------|
| mBERT | 2e-5 | auto (~2.3) | 0.5779 | 0.4138 |
| XLM-RoBERTa | 5e-6 | 4.0 | 0.6498 | 0.5455 |

---

## Reproducing Results

```bash
# Classical ML (Experiments 1–3)
python scripts/classical/train_classical.py

# mBERT fine-tuning (Experiment 4)
python scripts/sentence_transformers/train_sentence_transformer.py \
    --model_name bert-base-multilingual-cased \
    --lr 2e-5 \
    --output_dir models/mbert_optimal \
    --results_dir results/sentence_transformers/

# XLM-RoBERTa fine-tuning (Experiment 5)
python scripts/sentence_transformers/train_sentence_transformer.py \
    --model_name xlm-roberta-base \
    --lr 5e-6 \
    --depressed_weight 4.0 \
    --output_dir models/xlm_roberta_optimal \
    --results_dir results/sentence_transformers/

# Full hyperparameter grid search
python scripts/sentence_transformers/hyperparameter_search.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `seaborn`.
