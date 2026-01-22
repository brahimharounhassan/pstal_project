# Super-Sense Tagging for French via Fine-tuning
![Python 3.12.3](https://img.shields.io/badge/Python-3.12.3-yellow?style=plastic)

A comprehensive deep learning framework for super-sense semantic tagging of French text using transformer-based models with optional LoRA/DoRA fine-tuning.

## Overview

This project implements a super-sense tagging system for French text, identifying semantic categories for nouns and verbs according to the supersense taxonomy. The system supports:

- **Baseline models**: Training classifiers on frozen transformer embeddings
- **Fine-tuned models**: Full parameter fine-tuning or efficient LoRA/DoRA adaptation
- **Multiple architectures**: Support for CamemBERT, BERT, XLM-RoBERTa, DeBERTa, and more
- **Hyperparameter optimization**: Automated tuning with Optuna

### Supersense Categories

The system classifies words into 25 semantic categories including:

- '*', 'Act', 'Animal', 'Artifact', 'Attribute', 'Body', 'Cognition', 'Communication', 'Event', 'Feeling', 'Food', 'Group', 'Institution', 'Object', 'Part', 'Person', 'Phenomenon', 'Plant', 'Possession', 'Quantity', 'Relation', 'State', 'Substance', 'Time', 'Tops'


## Project Structure

```
pstal_project/
├── configs/
│   ├── config.py          # Global configuration
│   └── config.yml         # YAML configuration (optional)
├── data/
│   └── sequoia/           # Training and evaluation data
│       ├── *.train        # Training set
│       ├── *.dev          # Development set
│       └── *.test         # Test set
├── lib/
│   ├── conllulib.py       # CoNLL-U utilities
│   └── evaluate.py        # Evaluation metrics
├── models/
│   ├── *.pt               # Trained model checkpoints
│   ├── peft_adapter*/     # LoRA adapter weights
│   └── checkpoints/       # Best model checkpoints
├── src/
│   ├── train_ssense.py    # Baseline training
│   ├── train_finetuned.py # Training with fine-tuned embeddings
│   ├── fine_tuning.py     # Transformer fine-tuning
│   ├── hp_tuning.py       # Hyperparameter optimization
│   ├── predict_ssense.py  # Prediction script
│   ├── predict_finetuned.py # Prediction with PEFT adapters
│   ├── model_ssense.py    # Classifier architecture
│   └── utils.py           # Data preparation utilities
├── predictions/           # Model predictions
├── outputs/               # Training metrics and visualizations
├── logs/                  # Training logs
└── requirements.txt       # Python dependencies
```

##  Installation

### Prerequisites

- Python >= 3.10+

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Baseline Training

Train a classifier on **frozen** transformer embeddings:

```bash
python src/train_ssense.py \
  --train <train_file> \
  --dev <dev_file> \
  --output <output_model.pt> \
  --model-name <huggingface_model> \
  --n-epochs <num_epochs> \
  --batch-size <batch_size> \
  --dropout <dropout_rate> \
  --lr <learning_rate> \
  --device <cuda or cpu>
```

### Training with Fine-Tuned Models

Train a classifier on fine-tuned embeddings:

```bash
python src/train_finetuned.py \
  --train <train_file> \
  --dev <dev_file> \
  --output models/ssense_finetuned.pt \
  --finetuned-model models/camembert_lora/ \
  --n-epochs 30 \
  --batch-size 32 \
  --dropout 0.5 \
  --lr 0.001
```

### Prediction

#### With Baseline or Fine-Tuned Models

```bash
python src/predict_ssense.py \
  --model <model_path.pt> \
  --input <input_file.conllu> \
  --output <output_file.conllu> \
  --device cuda \
  --normalize  # Optional: normalize embeddings
```

#### With PEFT Adapters

```bash
python src/predict_finetuned.py \
  --peft-adapter models/camembert_base/ \
  --input <input_file.conllu> \
  --output <output_file.conllu>
```

#### Evaluation

```bash
python lib/evaluate.py \      
 --pred <predictions/predicted_file_test.conllu>  \
 --gold <test_file> \
 --tagcolumn frsemcor:noun \
 --train <train_file> \
--upos-filter NOUN PROPN NUM
```


## Models

### Supported Architectures

- **CamemBERT** (`almanach/camembert-base`, `almanach/camembert-large`)
- **BERT** (`dbmdz/bert-base-french-europeana-cased`, `bert-base-multilingual-cased`)
- **XLM-RoBERTa** (`FacebookAI/xlm-roberta-large`)
- **DeBERTa** (`microsoft/deberta-v3-base`)
- **DistilBERT** (`distilbert-base-multilingual-cased`)


## Author
- [Brahim Haroun Hassan]
- [Saal Racim]
- [Anlaoudine B. Moindzé]