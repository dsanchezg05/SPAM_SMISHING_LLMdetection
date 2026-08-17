# Spam & Smishing Detection with ModernBERT + MLP

A three-class text classifier that labels an incoming SMS/text message as **NORMAL (ham)**, **SPAM**, or **SMISHING**. The model combines contextual embeddings from a fine-tuned [ModernBERT](https://huggingface.co/docs/transformers/en/model_doc/modernbert) backbone with three lightweight rule-based features (presence of a URL, an email address, and a phone number), fed into a small MLP classifier head. A Streamlit app (`scam_scanner.py`) provides an interactive demo.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Reproducing Preprocessing](#reproducing-preprocessing)
  - [Inference in Python](#inference-in-python)
- [Citation](#citation)
- [License](#license)

## Overview

Smishing (SMS phishing) and spam messages share overlapping surface patterns (urgency, promotional language, links), but differ in intent — smishing actively attempts to steal credentials, money, or personal data, while spam is unsolicited but not necessarily malicious. This project trains a classifier to distinguish between the three classes using:

1. **Semantic signal**: a CLS-token embedding produced by a fine-tuned ModernBERT encoder.
2. **Structural signal**: three binary rule-based features — whether the message contains a **URL**, an **email address**, or a **phone number** — extracted with regular expressions.

These are concatenated and passed through a compact multi-layer perceptron (MLP) that outputs a probability distribution over the three classes.


## Pipeline

```
Raw text
   │
   ├─► Regex rules  ──► URL / EMAIL / PHONE flags (0/1)
   │
   └─► ModernBERT tokenizer ──► ModernBERT encoder ──► CLS embedding (768-d)
                                                             │
                                                    StandardScaler (cls_scaler.pkl)
                                                             │
                     concat(scaled CLS embedding, URL, EMAIL, PHONE)
                                                             │
                                                    MLP classifier head
                                                             │
                                          logits → sigmoid → argmax
                                                             │
                                         NORMAL / SPAM / SMISHING
```

- **Backbone**: `ModernBertModel`, 22 hidden layers, hidden size 768, mean-pooled classifier head disabled in favor of a custom MLP on the CLS token (`classifier_pooling` in `config.json` is provided for reference).
- **Classifier head**: ~78K parameters — a 768→96 projection, concatenated with 3 rule-based features, then a 99→40→3 MLP.
- **Labels**: `{0: "ham"/"NORMAL", 1: "spam", 2: "smishing"}`.


## Model Weights

The fine-tuned ModernBERT backbone is hosted on the Hugging Face Hub:
👉 https://huggingface.co/dsanchezg05/modernbert-finetuned-backbone

It is loaded automatically.

## Dataset

- **File**: `Dataset_10191.csv` — 10,191 labeled messages with columns `LABEL, TEXT, URL, EMAIL, PHONE`.
- **Source**: Munoz, Miriam; Islam, Muhammad (2025), *"A Balanced Dataset for Spam and Smishing Detection using Large Language Models (LLMs)"*, Mendeley Data, V1. See [Citation](#citation).
- **Classes**: `ham` (normal), `spam`, `smishing` — approximately balanced across classes.
- Derived artifacts:
  - `preprocessed_df.parquet` — cleaned text plus rule-based features.
  - `embeddings_df.parquet` — preprocessed data plus the 768-d ModernBERT CLS embedding per message.

## Results

Metrics logged during training (`metrics.tsv`, 40 epochs) and on the held-out test split (`test_metrics.tsv`):

| Split | Accuracy | F1-score (weighted) |
|---|---|---|
| Best validation epoch (epoch 29/32/35) | 99.23% | 99.22% |
| **Test set** | **97.59%** | **97.59%** |

Test set confusion matrix (rows = true label, columns = predicted; order `ham, spam, smishing`):

```
[[97  1  0]
 [ 1 95  2]
 [ 0  3 92]]
```

The model rarely confuses `ham` with the other two classes; most residual errors occur between `spam` and `smishing`, which is expected given their overlapping promotional/urgency language.

## Installation

```bash
conda create -n scam_detector python=3.10 -y
conda activate scam_detector

# PyTorch with CUDA support (adjust cu-version to your driver, see `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install transformers accelerate scikit-learn joblib pandas numpy streamlit emoji pyarrow
```

## Usage

### Running the Streamlit App

`scam_scanner.py` currently loads the backbone, scaler, and MLP checkpoint from local absolute paths (`/home/dsanchezg/SPAM_SMISHING_LLMdetection/...`). Update these paths to match your environment before running:

```bash
streamlit run scam_scanner.py
```

The app lets you paste a message, then displays:
- The predicted class (**NORMAL** / **SPAM** / **SMISHING**) with a color-coded badge.
- Per-class confidence bars.
- Detected structural signals (URL / Email / Phone present).


## Citation

If you use the dataset, please cite:

> Munoz, Miriam; Islam, Muhammad (2025), "A Balanced Dataset for Spam and Smishing Detection using Large Language Models (LLMs)", Mendeley Data, V1, doi: [10.17632/vmg875v4xs.1](https://doi.org/10.17632/vmg875v4xs.1)

## License

MIT-license