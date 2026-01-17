#!/usr/bin/env python3
"""
Prediction script for baseline super-sense classifier.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from model_ssense import SuperSenseClassifier
from src.model_lightgbm import LightGBMClassifier
from src.model_tabnet import TabNetClassifier
from configs.config import *
from src.utils import setup_logger, write_conllu_predictions
from lib.conllulib import CoNLLUReader

logger = setup_logger('PREDICT_SSENSE')


def predict_sentence(sentence, model, classifier, tokenizer, label_vocab_rev, target_upos, device='cpu', normalize=False):
    """
    Predict super-sense labels for a single sentence using baseline approach.
    """
    words = [token['form'] for token in sentence]
    upos_tags = [token['upos'] for token in sentence]
    
    # Tokenize
    encodings = tokenizer(
        words,
        is_split_into_words=True,
        padding=False,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # Extract embeddings
    with torch.no_grad():
        outputs = model(**encodings)
        embeddings = outputs.last_hidden_state.squeeze(0)
    
    # Get word-level embeddings
    word_ids = encodings.word_ids()
    predictions = []
    
    for word_idx in range(len(words)):
        if upos_tags[word_idx] not in target_upos:
            predictions.append('*')
            continue
            
        # Get token indices for this word
        token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
        
        if not token_indices:
            predictions.append('*')
            continue
        
        # Average embeddings for subword tokens
        word_embedding = embeddings[token_indices].mean(dim=0)
        
        if normalize:
            word_embedding = word_embedding / (word_embedding.norm() + 1e-8)
        
        # Predict with classifier
        with torch.no_grad():
            logits = classifier(word_embedding.unsqueeze(0))
            predicted_label_id = torch.argmax(logits, dim=1).item()
        
        predictions.append(label_vocab_rev[predicted_label_id])
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict super-sense labels with baseline model')
    parser.add_argument('--model', required=True, help='Path to trained baseline model (.pt)')
    parser.add_argument('--input', required=True, help='Input CoNLL-U file')
    parser.add_argument('--output', required=True, help='Output CoNLL-U file with predictions')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('--normalize', action='store_true', 
                        help='Normalize embeddings before classification')
    
    args = parser.parse_args()
    
    logger.info(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    
    model_name = checkpoint['model_name']
    label_vocab = checkpoint['label_vocab']
    embedding_dim = checkpoint['embedding_dim']
    num_labels = checkpoint['num_labels']
    dropout = checkpoint['dropout']
    is_finetuned = checkpoint.get('is_finetuned', False)
    
    logger.info(f"Base model: {model_name}")
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    if is_finetuned:
        logger.info("Using FINE-TUNED embeddings")
        
        model = AutoModel.from_pretrained(model_name).to(args.device)
        model.load_state_dict(checkpoint['embedding_model_state'], strict=False)
        model.eval()
        logger.info("Fine-tuned encoder loaded locally.")
    else:
        logger.info("Using BASELINE frozen embeddings")
        
        # Load baseline frozen model
        model = AutoModel.from_pretrained(model_name).to(args.device)
        model.eval()
        logger.info("Baseline model loaded from HuggingFace")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine classifier type
    classifier_type = checkpoint.get('classifier_type', 'mlp')
    logger.info(f"Classifier type: {classifier_type}")
    
    # Load classifier based on type
    if classifier_type == 'lightgbm':
        classifier = LightGBMClassifier(
            num_labels=num_labels
        ).to(args.device)
        logger.info("Using LightGBM classifier")
    elif classifier_type == 'tabnet':
        classifier = TabNetClassifier(
            num_labels=num_labels
        ).to(args.device)
        logger.info("Using TabNet classifier")
    else:
        classifier = SuperSenseClassifier(
            embedding_dim=embedding_dim,
            num_labels=num_labels,
            dropout=dropout
        ).to(args.device)
        logger.info("Using MLP classifier")
    
    classifier.load_state_dict(checkpoint['model_state'])
    classifier.eval()
    
    logger.info("Classifier loaded successfully")
    
    # Create reverse label vocabulary
    label_vocab_rev = {v: k for k, v in label_vocab.items()}
    
    # Read input corpus
    logger.info(f"Reading input from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = CoNLLUReader(f)
        sentences = list(reader.readConllu())
    
    logger.info(f"Processing {len(sentences)} sentences...")
    
    # Generate predictions for all sentences
    predictions_list = []
    for sent in tqdm(sentences, desc="Predicting"):
        predictions = predict_sentence(
            sent,
            model,
            classifier,
            tokenizer,
            label_vocab_rev,
            TARGET_UPOS,
            args.device,
            normalize=args.normalize
        )
        predictions_list.append(predictions)
    
    # Write predictions to file
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = CoNLLUReader(f)
        num_sentences, num_words, num_predicted = write_conllu_predictions(
            sentences, 
            predictions_list, 
            reader, 
            args.output,
            supersense_column=SUPERSENSE_COLUMN
        )
    
    logger.info(f"Processed {num_sentences} sentences ({num_words} words)")
    logger.info(f"Predicted supersenses for {num_predicted} tokens")
    logger.info(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()