"""
utilities for super-sense classification with LoRA fine-tuning.
"""

from configs.config import *
from conllu import parse_incr
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from torch import cat, tensor 
import torch
import numpy as np
from tqdm import tqdm
import logging
from matplotlib import pyplot as plt
from pathlib import Path 
from datetime import datetime
from lib.conllulib import CoNLLUReader


class TuningDataPreparation:
    """
    data preparation for LoRA fine-tuning on super-sense classification.
    """
    
    def __init__(self, in_file: str, full_file: str = DATA_FULL, target_upos: list = None):
        self.in_file = in_file
        self.full_file = full_file
        self.target_upos = target_upos or TARGET_UPOS
        
        # Get all unique sense values from full dataset
        self.sense_values = self._get_all_sense_values()
        
        # Create label mappings (including special tokens)
        self.label2id, self.id2label = self._get_sense_dict()
        
        # Load corpus
        self.word_sent, self.upos_sent, self.sense_sent = self._load_corpus()
        
        self.dataset = None

    def _load_corpus(self) -> tuple[dict, dict, dict]:
        """Load corpus and extract words, UPOS, and supersenses."""
        word_sent = {}
        upos_sent = {}
        sense_sent = {}
        
        with open(self.in_file, mode='r', encoding='UTF-8') as conllu_file:
            sents = parse_incr(conllu_file)
            for i, sent in enumerate(sents):
                word_sent[i] = []
                upos_sent[i] = []
                sense_sent[i] = []
                for token in sent:
                    word_sent[i].append(token["form"])
                    upos_sent[i].append(token["upos"])
                    sense_sent[i].append(token["frsemcor:noun"])
        
        return word_sent, upos_sent, sense_sent

    def _get_all_sense_values(self) -> list:
        """Get all unique sense values from full corpus."""
        with open(self.full_file, mode='r', encoding='UTF-8') as conllu_file:
            sents = parse_incr(conllu_file)
            sense_values = set([
                token["frsemcor:noun"] 
                for sent in sents 
                for token in sent
            ])
        return sorted(sense_values)

    def _get_sense_dict(self) -> tuple[dict, dict]:
        """
        Create label mappings.
        """
        label2id = {label: i for i, label in enumerate(self.sense_values)}
        id2label = {i: label for label, i in label2id.items()}
        return label2id, id2label
    
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset.
        """
        from collections import Counter
        
        # Collect only meaningful labels (target UPOS with actual supersense)
        target_labels = []
        for sent_id in self.sense_sent:
            for i, label in enumerate(self.sense_sent[sent_id]):
                upos = self.upos_sent[sent_id][i]
                # Only include labels for target UPOS and skip special markers
                if upos in self.target_upos and label not in ['_', '*']:
                    target_labels.append(label)
        
        if not target_labels:
            # Fallback: uniform weights
            return torch.ones(len(self.label2id))
        
        # Count occurrences
        label_counts = Counter(target_labels)
        total_samples = len(target_labels)
        
        # Compute inverse frequency weights
        weights = torch.zeros(len(self.label2id))
        for label, count in label_counts.items():
            if label in self.label2id:
                label_id = self.label2id[label]
                weights[label_id] = total_samples / (len(label_counts) * count)
        
        # For labels not in target set, assign weight 0 (will be masked by ignore_index)
        for label in ['_', '*']:
            if label in self.label2id:
                weights[self.label2id[label]] = 0.0
        
        # Normalize weights (excluding zero weights)
        non_zero_mask = weights > 0
        if non_zero_mask.sum() > 0:
            weights[non_zero_mask] = weights[non_zero_mask] / weights[non_zero_mask].sum() * non_zero_mask.sum()
        
        return weights

    def create_dataloader(self, tokenizer: AutoTokenizer, batch_size: int, shuffle_mode: bool = True) -> DataLoader:
        """Create DataLoader with properly prepared dataset."""
        self.dataset = self._prepare_data(tokenizer)
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle_mode)

    def _prepare_data(self, tokenizer: AutoTokenizer) -> TensorDataset:
        """
        Prepare data for training.
        """
        # Find max length in dataset
        max_len = 0
        for sent_id, words in self.word_sent.items():
            encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
            seq_len = len(encoding["input_ids"])
            max_len = max(max_len, seq_len)

        input_ids = []
        attention_masks = []
        all_labels = []

        for i, words in self.word_sent.items():
            supersense_labels = self.sense_sent[i]
            upos_labels = self.upos_sent[i]

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_attention_mask=True,
                return_tensors="pt"
            )

            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

            # Align labels with subword tokens
            word_ids = encoding.word_ids()
            labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    # Special token (CLS, SEP, PAD)
                    labels.append(-100)
                elif word_id != previous_word_id:
                    # First subtoken of a word
                    upos = upos_labels[word_id]
                    supersense = supersense_labels[word_id]
                    
                    # Only assign label if target UPOS and not special marker
                    if upos in self.target_upos and supersense not in ['_', '*']:
                        labels.append(self.label2id[supersense])
                    else:
                        labels.append(-100)
                else:
                    # Continuation subtoken
                    labels.append(-100)
                
                previous_word_id = word_id

            all_labels.append(labels)

        # Convert to tensors
        input_ids = cat(input_ids, dim=0)
        attention_masks = cat(attention_masks, dim=0)
        labels = tensor(all_labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset


class SuperSenseDataPreparation:
    """
    feature-based approach for super-sense classification.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, device: str, normalize_embeddings: bool = False):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.normalize_embeddings = normalize_embeddings

    def _build_label_vocab(self, labels: list) -> dict:
        """Build label vocabulary from list of labels."""
        label_vocab = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        if '*' not in label_vocab:
            label_vocab['*'] = len(label_vocab)
        return label_vocab

    def extract_contextual_embeddings(self, sentences: list, target_upos: list = None) -> tuple[list, list]:
        """
        Extract contextual embeddings for words with target UPOS.
        """
        if target_upos is None:
            target_upos = TARGET_UPOS
            
        embeddings_list = []
        labels_list = []

        with torch.no_grad():
            for sent in tqdm(sentences, desc="Extracting embeddings", ncols=80):
                words = [tok["form"] for tok in sent]
                upos_tags = [tok["upos"] for tok in sent]
                supersense_tags = [tok.get(SUPERSENSE_COLUMN, "*") for tok in sent]

                tok = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True
                )

                word_ids = tok.word_ids()
                tok = {k: v.to(self.device) for k, v in tok.items()}

                emb_sent = self.model(**tok)
                hidden = emb_sent.last_hidden_state[0]  # [T, H]

                for word_idx in range(len(words)):
                    if upos_tags[word_idx] not in target_upos:
                        continue

                    sub_idx = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
                    if not sub_idx:
                        continue

                    # Average subword embeddings
                    word_emb = hidden[sub_idx].mean(dim=0)
                    
                    # Optional normalization (use with caution)
                    if self.normalize_embeddings:
                        word_emb = torch.nn.functional.normalize(word_emb, dim=-1)

                    embeddings_list.append(word_emb.cpu())
                    labels_list.append(supersense_tags[word_idx])

        return embeddings_list, labels_list

    def prepare_data(self, in_file: str):
        """Prepare data from CoNLL-U file."""
        logger = setup_logger("Data preparation", log_dir=LOG_PATH)
        logger.info(f"Loading corpus from {in_file}")

        sentences = list(CoNLLUReader(open(in_file, 'r', encoding='utf-8')).readConllu())
        logger.info(f"Loaded {len(sentences)} sentences")

        embeddings_list, labels_list = self.extract_contextual_embeddings(sentences)
        logger.info(f"Extracted {len(embeddings_list)} embeddings")

        label_vocab = self._build_label_vocab(labels_list)
        labels = [label_vocab.get(lbl, label_vocab['*']) for lbl in labels_list]

        embeddings = torch.stack(embeddings_list)
        labels = torch.tensor(labels, dtype=torch.long)

        return embeddings, labels, label_vocab


def plot_train_val_losses(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    title: str = "Train and Val Losses",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    show_min: bool = True
):
    """Plot training and validation losses."""
    train_losses = np.array(train_losses, dtype=float)
    val_losses = np.array(val_losses, dtype=float)

    epochs_train = np.arange(1, len(train_losses) + 1)
    epochs_val = np.arange(1, len(val_losses) + 1)

    plt.figure()
    plt.plot(epochs_train, train_losses, label="Train Loss")
    plt.plot(epochs_val, val_losses, label="Validation Loss")

    if show_min and len(val_losses) > 0:
        min_epoch = np.argmin(val_losses) + 1
        min_val = np.min(val_losses)

        plt.scatter(min_epoch, min_val)
        plt.annotate(
            f"Min Val Loss = {min_val:.4f}\nEpoch {min_epoch}",
            (min_epoch, min_val),
            textcoords="offset points",
            xytext=(5, 5)
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = OUTPUT_PATH / "train_val_losses.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()


def setup_logger(name: str, log_dir: str = LOG_PATH, level=logging.INFO):
    """Setup logger with file and console handlers."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def setup_training_logger(log_dir: str = LOG_PATH):
    """Setup training logger with CSV metrics file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = setup_logger("training", log_dir)
    metrics_file = log_dir / f"training_metrics_{timestamp}.csv"
    
    with open(metrics_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_f1_macro,val_f1_weighted,val_accuracy,learning_rate,timestamp\n")

    return logger, metrics_file
