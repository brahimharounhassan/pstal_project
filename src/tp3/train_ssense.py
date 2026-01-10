#!/usr/bin/env python3
"""
Training script for the super-sense classifier with contextual embeddings.

This script uses contextual embeddings from pre-trained transformer models
(CamemBERT, multilingual BERT, DistilBERT) to predict super-senses of nouns.
It extracts embeddings for each target word, uses them to train an
MLP classifier, and saves the trained model.

Usage:
    python train_ssense.py --train corpus_train.conllu --dev corpus_dev.conllu \\
                          --output models/ssense_model.pt \\
                          --model-name almanach/camembert-base \\
                          --epochs 10 --batch-size 64
"""

import sys
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

from model_ssense import SuperSenseClassifier
from lib.conllulib import CoNLLUReader, Util
from utils.logger import get_script_logger

logger = get_script_logger()

# Constants
TARGET_UPOS = {"NOUN", "PROPN", "NUM"}  # POS of words to analyze
SUPERSENSE_COLUMN = "frsemcor:noun"     # Name of the column containing super-senses


def extract_contextual_embeddings(sentences, tokenizer, model, device):
    """
    Extracts contextual embeddings for target words (nouns) in all sentences.
    
    For each sentence:
    1. Tokenizes the words with the transformer tokenizer
    2. Passes the tokenized sentence through the transformer model
    3. For each noun (NOUN/PROPN/NUM), averages the embeddings of its sub-tokens
    4. Associates the averaged embedding with the super-sense label
    
    Args:
        sentences (list): List of sentences (TokenList) from the corpus
        tokenizer: Tokenizer of the transformer model (e.g., CamembertTokenizer)
        model: Pre-trained transformer model (e.g., CamembertModel)
        device (str): Computing device ('cpu' or 'cuda')
    
    Returns:
        tuple: (embeddings_list, labels_list, word_info_list) where:
            - embeddings_list: list of embedding tensors [embedding_dim]
            - labels_list: list of super-sense labels (str)
            - word_info_list: list of tuples (sent_idx, word_idx) for reference
    
    Example:
        >>> from transformers import AutoTokenizer, AutoModel
        >>> tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
        >>> model = AutoModel.from_pretrained("almanach/camembert-base")
        >>> sentences = [...]  # Corpus sentences
        >>> embeddings, labels, infos = extract_contextual_embeddings(sentences, tokenizer, model, 'cpu')
        >>> len(embeddings)  # Number of nouns in the corpus
        1523
        >>> embeddings[0].shape
        torch.Size([768])  # CamemBERT-base embedding dimension
    """
    embeddings_list = []
    labels_list = []
    word_info_list = []
    
    # Put the model in evaluation mode (no transformer training)
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():  # No gradient computation (saves memory)
        for sent_idx, sent in enumerate(tqdm(sentences, desc="Extracting embeddings", ncols=80)):
            # Extract words, POS and super-senses from the sentence
            words = [tok["form"] for tok in sent]
            upos_tags = [tok["upos"] for tok in sent]
            supersense_tags = [tok.get(SUPERSENSE_COLUMN, "*") for tok in sent]
            
            # Tokenize the sentence with word-subtoken alignment
            # is_split_into_words=True indicates that the input is already segmented into words
            tok_sent = tokenizer(words, is_split_into_words=True, return_tensors='pt')
            
            # Get the word <-> sub-tokens alignment BEFORE moving to device
            # word_ids() returns a list where each element indicates which word the sub-token belongs to
            # E.g.: [None, 0, 0, 1, 2, 2, 2, 3, None] for 4 words
            word_ids = tok_sent.word_ids()
            
            # Move tensors to device (GPU if available)
            tok_sent_device = {k: v.to(device) for k, v in tok_sent.items()}
            
            # Pass the sentence through the transformer to get embeddings
            outputs = model(**tok_sent_device)
            last_hidden_state = outputs['last_hidden_state'][0]  # [num_subtokens, embedding_dim]
            
            # For each word, average the embeddings of its sub-tokens
            for word_idx in range(len(words)):
                # Process only words with a target POS (nouns)
                if upos_tags[word_idx] not in TARGET_UPOS:
                    continue
                
                # Find the indices of sub-tokens corresponding to this word
                # E.g.: if word_ids = [None, 0, 0, 1, ...], then for word_idx=0, subtoken_indices = [1, 2]
                subtoken_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
                
                if not subtoken_indices:
                    continue
                
                # Average the embeddings of sub-tokens to get a word embedding
                word_embeddings = last_hidden_state[subtoken_indices]
                avg_embedding = word_embeddings.mean(dim=0)
                
                # Save the embedding, label and position information
                embeddings_list.append(avg_embedding.cpu())
                labels_list.append(supersense_tags[word_idx])
                word_info_list.append((sent_idx, word_idx))
    
    return embeddings_list, labels_list, word_info_list


def build_label_vocab(labels):
    """
    Builds the super-sense label vocabulary.
    
    Args:
        labels (list): List of super-sense labels (str)
    
    Returns:
        dict: Vocabulary {label: index}
    
    Example:
        >>> labels = ['Person', 'Location', 'Person', '*', 'Time']
        >>> vocab = build_label_vocab(labels)
        >>> print(vocab)
        {'*': 0, 'Location': 1, 'Person': 2, 'Time': 3}
    """
    label_vocab = {}
    for label in sorted(set(labels)):
        if label not in label_vocab:
            label_vocab[label] = len(label_vocab)
    
    # Ensures that '*' is in the vocabulary
    if '*' not in label_vocab:
        label_vocab['*'] = len(label_vocab)
    
    return label_vocab


def prepare_data(corpus_path, tokenizer, model, device, label_vocab=None):
    """
    Prepares training/dev data from a CoNLL-U corpus.
    
    Reads the corpus, extracts contextual embeddings for each noun,
    and builds/uses the label vocabulary.
    
    Args:
        corpus_path (str): Path to the CoNLL-U corpus file
        tokenizer: Tokenizer of the transformer model
        model: Pre-trained transformer model
        device (str): Computing device ('cpu' or 'cuda')
        label_vocab (dict, optional): Existing vocabulary. If None, builds a new one.
    
    Returns:
        tuple: (embeddings, labels, label_vocab) where:
            - embeddings: tensor of embeddings [num_words, embedding_dim]
            - labels: tensor of label indices [num_words]
            - label_vocab: vocabulary {label: index}
    
    Example:
        >>> train_emb, train_labels, label_vocab = prepare_data(
        ...     'train.conllu', tokenizer, model, 'cuda'
        ... )
        >>> dev_emb, dev_labels, _ = prepare_data(
        ...     'dev.conllu', tokenizer, model, 'cuda', label_vocab
        ... )
    """
    logger.info(f"Loading corpus from {corpus_path}")
    
    # Read sentences from the CoNLL-U corpus
    reader = CoNLLUReader(open(corpus_path, 'r', encoding='utf-8'))
    sentences = list(reader.readConllu())
    logger.info(f"Loaded {len(sentences)} sentences")
    
    # Extract embeddings and labels
    embeddings_list, labels_list, _ = extract_contextual_embeddings(
        sentences, tokenizer, model, device
    )
    logger.info(f"Extracted {len(embeddings_list)} embeddings for target words")
    
    # Build or use existing vocabulary
    if label_vocab is None:
        label_vocab = build_label_vocab(labels_list)
        logger.info(f"Built label vocabulary with {len(label_vocab)} labels")
    
    # Convert labels to indices
    label_indices = [label_vocab.get(label, label_vocab.get('*', 0)) for label in labels_list]
    
    # Stack embeddings into a single tensor
    embeddings = torch.stack(embeddings_list)
    labels = torch.LongTensor(label_indices)
    
    return embeddings, labels, label_vocab


def fit(model, train_loader, loss_fn, optimizer, device):
    """
    Trains the classifier on one epoch of the training corpus.
    
    Args:
        model (SuperSenseClassifier): The model to train
        train_loader (DataLoader): Iterator over training batches
        loss_fn (nn.CrossEntropyLoss): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Computing device
    
    Returns:
        float: Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    
    for embeddings, labels in tqdm(train_loader, desc="Training batches", ncols=80, leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, dev_loader, loss_fn, device):
    """
    Evaluates performance on the development corpus.
    
    Args:
        model (SuperSenseClassifier): The model to evaluate
        dev_loader (DataLoader): Iterator over dev batches
        loss_fn (nn.CrossEntropyLoss): Loss function
        device (str): Computing device
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for embeddings, labels in dev_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            logits = model(embeddings)
            loss = loss_fn(logits, labels)
            
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dev_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train super-sense classifier')
    parser.add_argument('--train', required=True, help='Path to training corpus')
    parser.add_argument('--dev', required=True, help='Path to dev corpus')
    parser.add_argument('--output', default='models/ssense_model.pt', help='Output model file')
    parser.add_argument('--model-name', default='distilbert/distilbert-base-multilingual-cased',
                        help='Transformer model name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Loading transformer model: {args.model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    transformer_model = AutoModel.from_pretrained(args.model_name)
    
    # Get embedding dimension from model config
    embedding_dim = transformer_model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Prepare training data
    logger.info("Preparing training data")
    train_embeddings, train_labels, label_vocab = prepare_data(
        args.train, tokenizer, transformer_model, args.device
    )
    
    # Prepare dev data
    logger.info("Preparing dev data")
    dev_embeddings, dev_labels, _ = prepare_data(
        args.dev, tokenizer, transformer_model, args.device, label_vocab=label_vocab
    )
    
    logger.info(f"Training samples: {len(train_labels)}")
    logger.info(f"Dev samples: {len(dev_labels)}")
    logger.info(f"Number of labels: {len(label_vocab)}")
    
    # Create DataLoaders
    train_loader = Util.dataloader(
        [train_embeddings], [train_labels],
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = Util.dataloader(
        [dev_embeddings], [dev_labels],
        batch_size=args.batch_size, shuffle=False
    )
    
    # Initialize classifier
    logger.info("Initializing classifier")
    classifier = SuperSenseClassifier(
        embedding_dim=embedding_dim,
        num_labels=len(label_vocab),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    classifier = classifier.to(args.device)
    
    logger.info(f"Classifier parameters: {Util.count_params(classifier):,}")
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    
    # Training loop
    logger.info("Starting training")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss = fit(classifier, train_loader, loss_fn, optimizer, args.device)
        dev_loss, dev_acc = evaluate(classifier, dev_loader, loss_fn, args.device)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Dev Loss: {dev_loss:.4f} | "
                   f"Dev Acc: {dev_acc:.4f}")
        
        if dev_acc > best_acc:
            best_acc = dev_acc
            logger.info(f"New best accuracy: {best_acc:.4f}")
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Reverse label vocab for prediction
    label_vocab_dict = dict(label_vocab)
    
    checkpoint = {
        'model_state': classifier.state_dict(),
        'label_vocab': label_vocab_dict,
        'embedding_dim': embedding_dim,
        'num_labels': len(label_vocab),
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'model_name': args.model_name
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")
    logger.info(f"Best dev accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
