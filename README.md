# Super-Sense Tagging for French

A comprehensive deep learning framework for super-sense semantic tagging of French text using transformer-based models with optional LoRA/DoRA fine-tuning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Baseline Training](#baseline-training)
  - [Fine-Tuning with LoRA](#fine-tuning-with-lora)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Training with Fine-Tuned Models](#training-with-fine-tuned-models)
  - [Prediction](#prediction)
- [Configuration](#configuration)
- [Models](#models)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🎯 Overview

This project implements a super-sense tagging system for French text, identifying semantic categories for nouns and verbs according to the supersense taxonomy. The system supports:

- **Baseline models**: Training classifiers on frozen transformer embeddings
- **Fine-tuned models**: Full parameter fine-tuning or efficient LoRA/DoRA adaptation
- **Multiple architectures**: Support for CamemBERT, BERT, XLM-RoBERTa, DeBERTa, and more
- **Hyperparameter optimization**: Automated tuning with Optuna
- **Production-ready**: Efficient inference with automatic memory management

### Supersense Categories

The system classifies words into 25 semantic categories including:
- **Nouns**: Person, Location, Time, Object, Food, etc.
- **Verbs**: Motion, Communication, Cognition, Creation, etc.

## ✨ Features

- 🚀 **Multiple Training Strategies**: Baseline (frozen) vs Fine-tuned embeddings
- 🔧 **LoRA/DoRA Support**: Parameter-efficient fine-tuning with PEFT
- 📊 **Hyperparameter Optimization**: Automated search with Optuna
- 💾 **Memory Management**: Automatic CPU fallback for large models
- 📈 **Comprehensive Logging**: Training metrics and visualization
- 🎯 **Production Ready**: Efficient batch prediction with progress tracking
- 🔒 **Security**: Safetensors support for secure model loading

## 📁 Project Structure

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

## 🔧 Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- Conda or virtualenv

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd pstal_project
```

2. **Create environment**
```bash
conda create -n pstal python=3.12
conda activate pstal
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 🚀 Quick Start

### 1. Train a Baseline Model

```bash
python src/train_ssense.py \
  --train data/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
  --dev data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --output models/baseline_camembert.pt \
  --model-name almanach/camembert-base \
  --n-epochs 50 \
  --batch-size 64 \
  --dropout 0.4 \
  --lr 0.0002
```

### 2. Make Predictions

```bash
python src/predict_ssense.py \
  --model models/baseline_camembert.pt \
  --input data/sequoia/sequoia-ud.parseme.frsemcor.simple.test \
  --output predictions/baseline_test.conllu
```

### 3. Evaluate

```bash
python lib/evaluate.py \
  data/sequoia/sequoia-ud.parseme.frsemcor.simple.test \
  predictions/baseline_test.conllu
```

## 📖 Usage

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
  --device cuda
```

**Arguments:**
- `--train`: Path to training data (CoNLL-U format)
- `--dev`: Path to development data
- `--output`: Output model path
- `--model-name`: HuggingFace model (e.g., `almanach/camembert-base`)
- `--n-epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--dropout`: Dropout rate (default: 0.3)
- `--lr`: Learning rate (default: 3e-4)
- `--device`: Device (`cuda` or `cpu`)

### Fine-Tuning with LoRA

Fine-tune a transformer model with LoRA/DoRA:

```bash
python src/fine_tuning.py \
  --train <train_file> \
  --dev <dev_file> \
  --output-dir models/camembert_lora \
  --model-name almanach/camembert-base \
  --n-epochs 10 \
  --batch-size 8 \
  --lr 4e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --use-dora
```

**LoRA Parameters:**
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha scaling (default: 16)
- `--lora-dropout`: LoRA dropout (default: 0.1)
- `--use-dora`: Use DoRA instead of LoRA
- `--use-rslora`: Use rank-stabilized LoRA

### Hyperparameter Optimization

Optimize hyperparameters with Optuna:

```bash
python src/hp_tuning.py \
  --train <train_file> \
  --dev <dev_file> \
  --model-name almanach/camembert-base \
  --output-dir outputs/ \
  --n-trials 50 \
  --n-epochs 10 \
  --use-dora
```

This will:
1. Run 50 Optuna trials
2. Search for optimal hyperparameters (lr, rank, alpha, dropout, etc.)
3. Save best hyperparameters to `outputs/best_hyperparameters_*.json`
4. Generate visualization plots

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
  --peft-adapter models/camembert_lora/ \
  --input <input_file.conllu> \
  --output <output_file.conllu>
```

## ⚙️ Configuration

Edit `configs/config.py` to customize:

```python
# Data paths
DATA_PATH = Path("data/sequoia")
TRAIN_FILE = "sequoia-ud.parseme.frsemcor.simple.train"
DEV_FILE = "sequoia-ud.parseme.frsemcor.simple.dev"
TEST_FILE = "sequoia-ud.parseme.frsemcor.simple.test"

# Model settings
SEED = 42
TARGET_UPOS = ['NOUN', 'PROPN', 'VERB']  # POS tags to classify
SUPERSENSE_COLUMN = 10  # CoNLL-U column for supersenses

# Paths
LOG_PATH = Path("logs")
MODEL_PATH = Path("models")
OUTPUT_PATH = Path("outputs")
PREDICTION_PATH = Path("predictions")
```

## 🤖 Models

### Supported Architectures

- **CamemBERT** (`almanach/camembert-base`, `almanach/camembert-large`)
- **BERT** (`dbmdz/bert-base-french-europeana-cased`, `bert-base-multilingual-cased`)
- **XLM-RoBERTa** (`FacebookAI/xlm-roberta-large`)
- **DeBERTa** (`microsoft/deberta-v3-base`)
- **DistilBERT** (`distilbert-base-multilingual-cased`)

### Model Types

1. **Baseline Models** (`.pt` files)
   - Frozen transformer embeddings
   - Trained MLP classifier
   - Fast inference
   - Smaller memory footprint

2. **Fine-Tuned Models** (`.pt` files with `is_finetuned=True`)
   - Full model fine-tuned or LoRA-adapted
   - Includes embedding model state
   - Trained MLP classifier
   - Better performance, larger size

3. **PEFT Adapters** (directories with `adapter_*.safetensors`)
   - LoRA/DoRA adapter weights
   - Require base model from HuggingFace
   - Most efficient storage
   - Direct prediction support

## 📊 Results

Training produces:
- **Model checkpoints**: Best model based on validation loss
- **Training metrics**: CSV files with loss/accuracy per epoch
- **Predictions**: CoNLL-U format with supersense annotations
- **Visualizations**: Loss curves, Optuna plots

Example metrics:
```
Epoch 50/50 | Train Loss: 0.1234 | Dev Loss: 0.2345 | Dev Acc: 0.8567
```

## 🔍 Troubleshooting

### Out of Memory Errors

The system automatically detects OOM errors and falls back to CPU. For large models:

```bash
# Force CPU usage
python src/predict_ssense.py --model <model> --input <input> --output <output> --device cpu

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Tokenizer Errors (DeBERTa)

DeBERTa tokenizers are loaded from local directories automatically. Ensure fine-tuned model directories contain tokenizer files:
```
models/deberta_v3_base/
├── tokenizer_config.json
├── tokenizer.json
├── spm.model
└── ...
```

### Import Errors

```bash
# Verify environment
conda activate pstal
pip install -r requirements.txt --upgrade

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Format

Ensure CoNLL-U files follow the correct format:
- Column 10: Supersense labels
- Use `*` for non-supersense tokens
- Sentences separated by blank lines

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{pstal2026supersense,
  title={Super-Sense Tagging for French with Transformer Models},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/pstal_project}}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This project was developed as part of coursework in Natural Language Processing and Deep Learning for NLP tasks.
