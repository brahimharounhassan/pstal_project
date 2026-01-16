from xml.parsers.expat import model
from configs.config import *
from conllu import parse_incr
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from torch import cat, tensor 
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path 
from datetime import datetime
from lib.conllulib import CoNLLUReader
import logging


class TuningDataPreparation:
    def __init__(self, in_file: str, full_file: str = DATA_FULL):
        self.in_file = in_file
        self.full_file = full_file

        self.sense_values = self._get_all_sense_values()

        self.label2id, self.id2label = self._get_sense_dict()

        self.word_sent, self.upos_sent, self.sense_sent = self._load_corpus()

        self.dataset = None

        # self.vocab = {}

    def _load_corpus(self) -> tuple[dict, dict, dict]:
        word_sent = {}
        upos_sent = {}
        sense_sent = {}
        with open(self.in_file, mode='r', encoding='UTF-8' ) as conllu_file :
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

    def _get_all_sense_values(self) -> set:
        with open(self.full_file, mode='r', encoding='UTF-8' ) as conllu_file :
            sents = parse_incr(conllu_file)
            sense_values = set([token["frsemcor:noun"] for sent in sents for token in sent ])# if token["frsemcor:noun"] not in ["_", "*"]])
        return sorted(sense_values)

    def _get_sense_dict(self) -> tuple[dict, dict]:

        label2id = {label: i for i, label in enumerate(self.sense_values)}
        id2label = {i: label for label, i in label2id.items()}
        return label2id, id2label

    def create_dataloader(self, tokenizer: AutoTokenizer, batch_size: int, shuffle_mode: bool = True) -> DataLoader:

        self.dataset = self._prepare_data(tokenizer)

        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle_mode)

    def _prepare_data(self, tokenizer: AutoTokenizer) -> TensorDataset:

        max_len = 0
        # For every sentence...
        for sent_id, words in self.word_sent.items():
            # Tokenize the text and add <s> and </s> tokens.
            encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=True)

            seq_len = len(encoding["input_ids"])
            max_len = max(max_len, seq_len)

        input_ids = []
        attention_masks = []
        all_labels = []

        for i, words in self.word_sent.items():
            supersense_labels = self.sense_sent[i]  # supersense by word

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

            word_ids = encoding.word_ids()
            labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None: # if special token assign -100
                    labels.append(-100)
                elif word_id != previous_word_id: # if it's a new token assign a supersense label
                    labels.append(self.label2id[supersense_labels[word_id]])
                else:
                    labels.append(-100) # if it's a token of the previous word assign -100
                previous_word_id = word_id

            all_labels.append(labels)

        # convert the lists into a torch tensors
        input_ids = cat(input_ids, dim=0)
        attention_masks = cat(attention_masks, dim=0)
        labels = tensor(all_labels)

        # group everything in a tensor dataset
        dataset = TensorDataset(input_ids, attention_masks, labels)

        return dataset

class SuperSenseDataPreparation:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, device: str):
        self.in_file = None
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def _build_label_vocab(self, labels: list) -> dict:
        """
        Builds the super-sense label vocabulary.
        """
        
        label_vocab = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        
        if '*' not in label_vocab:
            label_vocab['*'] = len(label_vocab)
        
        return label_vocab

    def extract_contextual_embeddings(self, sentences: list) -> tuple[list, list]:
        """
        Extracts contextual embeddings for target words (nouns) in all sentences.
        """
        embeddings_list = []
        labels_list = []
        # word_info_list = []
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        with torch.no_grad():
            for sent_idx, sent in enumerate(tqdm(sentences, desc="Extracting embeddings", ncols=80)):
                # Extract words, POS and super-senses from the sentence
                words = [tok["form"] for tok in sent]
                upos_tags = [tok["upos"] for tok in sent]
                supersense_tags = [tok.get(SUPERSENSE_COLUMN, "*") for tok in sent]
                
                # Tokenize the sentence with word-subtoken alignment
                tok_sent = self.tokenizer(words, is_split_into_words=True, return_tensors='pt')
                
                word_ids = tok_sent.word_ids()
                
                # Move tensors to GPU if available
                tok_sent_device = {k: v.to(self.device) for k, v in tok_sent.items()}
                
                # Pass the sentence through the transformer to get embeddings
                embedding_sents = self.model(**tok_sent_device)
                last_hidden_state = embedding_sents['last_hidden_state'][0]  # [num_subtokens, embedding_dim]
                
                # For each word we averaging the embeddings of its sub-tokens
                for word_idx in range(len(words)):
                    # Process only words with a target UPOS
                    if upos_tags[word_idx] not in TARGET_UPOS:
                        continue
                    
                    # We search the indices of sub-tokens corresponding to this word
                    subtoken_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
                    
                    if not subtoken_indices:
                        continue
                    
                    # averaging the embeddings of sub-tokens to get a word embedding
                    word_embeddings = last_hidden_state[subtoken_indices]
                    avg_embedding = word_embeddings.mean(dim=0)
                    
                    embeddings_list.append(avg_embedding.cpu())
                    labels_list.append(supersense_tags[word_idx])
                    # word_info_list.append((sent_idx, word_idx))
        
        return embeddings_list, labels_list


    def prepare_data(self, in_file: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Prepares training/dev data from a CoNLL-U corpus.
        """
        logger = logger = setup_logger("Data preparation", log_dir=LOG_PATH)

        logger.info(f"Loading corpus from {in_file}")
        
        sentences = list(CoNLLUReader(open(in_file, 'r', encoding='utf-8')).readConllu())
        logger.info(f"Loaded {len(sentences)} sentences")
        
        embeddings_list, labels_list = self.extract_contextual_embeddings(sentences)
        logger.info(f"Extracted {len(embeddings_list)} embeddings for target words")
        
        # Build or use existing vocabulary
        label_vocab = self._build_label_vocab(labels_list)
        
        label_indices = [label_vocab.get(label, label_vocab.get('*', 0)) for label in labels_list]
        
        embeddings = torch.stack(embeddings_list)
        labels = torch.LongTensor(label_indices)
        
        return embeddings, labels, label_vocab
    



def plot_train_val_losses(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    title: str="Train and Val Losses",
    xlabel: str="Epoch",
    ylabel: str="Loss",
    show_min: bool=True
):

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
    fname = OUTPUT_PATH /"train_val_losses.png"  
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()


def setup_logger(name: str, log_dir: str = LOG_PATH, level=logging.INFO):
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

    # Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def setup_training_logger(log_dir: str = LOG_PATH):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = setup_logger("training", log_dir)
    metrics_file = log_dir / f"training_metrics_{timestamp}.csv"
    with open(metrics_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,learning_rate,timestamp")

    return logger, metrics_file
