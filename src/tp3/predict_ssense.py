#!/usr/bin/env python3
"""
Super-sense prediction script from a trained model.

This script loads a pre-trained super-sense classifier and uses it
to predict super-sense labels on a new corpus in CoNLL-U format.
It uses the same transformer model as the one used during training
to extract contextual embeddings.

Usage:
    python predict_ssense.py --model models/ssense_model.pt \\
                            --input data/test.conllu \\
                            --output predictions/test_pred.conllu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from model_ssense import SuperSenseClassifier
from utils.logger import get_script_logger

logger = get_script_logger()

# Constants (same as for training)
TARGET_UPOS = {"NOUN", "PROPN", "NUM"}
SUPERSENSE_COLUMN = "frsemcor:noun"


def load_checkpoint(filepath, device='cpu'):
    """
    Loads a super-sense classifier from a saved checkpoint.
    
    Args:
        filepath (str): Path to the .pt file containing the checkpoint
        device (str, optional): Device where to load the model. Default 'cpu'.
    
    Returns:
        tuple: (classifier, label_vocab_rev, model_name) where:
            - classifier: the loaded SuperSenseClassifier model
            - label_vocab_rev: reversed vocabulary {index: label}
            - model_name: name of the transformer model used
    
    Example:
        >>> classifier, label_vocab_rev, model_name = load_checkpoint('model.pt', 'cuda')
        >>> print(f"Loaded {model_name} with {len(label_vocab_rev)} labels")
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Rebuilds the classifier with the saved hyperparameters
    classifier = SuperSenseClassifier(
        embedding_dim=checkpoint['embedding_dim'],
        num_labels=checkpoint['num_labels'],
        hidden_dim=checkpoint['hidden_dim'],
        dropout=checkpoint['dropout']
    )
    
    # Loads the classifier weights
    classifier.load_state_dict(checkpoint['model_state'])
    classifier = classifier.to(device)
    classifier.eval()
    
    # Reverse the vocabulary for prediction (index -> label)
    label_vocab_rev = {v: k for k, v in checkpoint['label_vocab'].items()}
    
    return classifier, label_vocab_rev, checkpoint['model_name']


def predict_sentence(sentence, classifier, tokenizer, transformer_model, 
                     label_vocab_rev, device='cpu'):
    """
    Predicts super-senses for all words in a sentence.
    
    For each word in the sentence:
    1. If the word is not a noun (NOUN/PROPN/NUM), predicts '*'
    2. Otherwise, extracts its contextual embedding via the transformer
    3. Passes the embedding through the classifier to get the prediction
    
    Args:
        sentence (TokenList): Sentence in CoNLL-U format
        classifier (SuperSenseClassifier): Super-sense classifier
        tokenizer: Tokenizer of the transformer model
        transformer_model: Pre-trained transformer model
        label_vocab_rev (dict): Reversed vocabulary {index: label}
        device (str, optional): Computing device. Default 'cpu'.
    
    Returns:
        list: List of predicted super-sense labels (one per word)
    
    Example:
        >>> sentence = [...]  # "Quelle surprise ! Arturo arrive..."
        >>> predictions = predict_sentence(sentence, classifier, tokenizer, 
        ...                                transformer_model, label_vocab_rev)
        >>> print(predictions)
        ['*', 'Feeling', '*', 'Person', '*', ...]
    """
    classifier.eval()
    transformer_model.eval()
    transformer_model = transformer_model.to(device)
    
    with torch.no_grad():
        # Extract words and POS from the sentence
        words = [tok["form"] for tok in sentence]
        upos_tags = [tok["upos"] for tok in sentence]
        
        # Tokenize the sentence with word-subtoken alignment
        tok_sent = tokenizer(words, is_split_into_words=True, return_tensors='pt')
        
        # Get the alignment before moving to device
        word_ids = tok_sent.word_ids()
        
        # Move to device
        tok_sent_device = {k: v.to(device) for k, v in tok_sent.items()}
        
        # Get embeddings from the transformer
        outputs = transformer_model(**tok_sent_device)
        last_hidden_state = outputs['last_hidden_state'][0]
        
        # Predict for each word
        predictions = []
        for word_idx in range(len(words)):
            # Check if the word has a target POS
            if upos_tags[word_idx] not in TARGET_UPOS:
                predictions.append('*')
                continue
            
            # Find sub-token indices for this word
            subtoken_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if not subtoken_indices:
                predictions.append('*')
                continue
            
            # Average the embeddings of sub-tokens
            word_embeddings = last_hidden_state[subtoken_indices]
            avg_embedding = word_embeddings.mean(dim=0).unsqueeze(0)  # [1, embedding_dim]
            avg_embedding = avg_embedding.to(device)
            
            # Predict the label
            logits = classifier(avg_embedding)
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_label = label_vocab_rev.get(pred_idx, '*')
            
            predictions.append(pred_label)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict super-senses')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to input corpus (CoNLL-U)')
    parser.add_argument('--output', default='predictions/ssense_pred.conllu', 
                        help='Path to output file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    logger.info(f"Loading model from {args.model}")
    classifier, label_vocab_rev, model_name = load_checkpoint(args.model, device=args.device)
    logger.info(f"Model loaded on device: {args.device}")
    logger.info(f"Transformer model: {model_name}")
    
    # Load tokenizer and transformer model
    logger.info(f"Loading tokenizer and transformer model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_model = AutoModel.from_pretrained(model_name)
    
    logger.info(f"Processing {args.input}")
    
    # Read and process sentences
    from lib.conllulib import CoNLLUReader
    reader = CoNLLUReader(open(args.input, 'r', encoding='utf-8'))
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as outfile:
        num_sentences = 0
        num_words = 0
        
        for sent in tqdm(reader.readConllu(), desc="Predicting", ncols=80):
            # Predict super-senses for this sentence
            predictions = predict_sentence(
                sent, classifier, tokenizer, transformer_model, 
                label_vocab_rev, args.device
            )
            
            # Write sentence with predictions
            # Write metadata
            if sent.metadata:
                for key, value in sent.metadata.items():
                    outfile.write(f"# {key} = {value}\n")
            
            # Write tokens with predicted super-sense
            for tok_idx, token in enumerate(sent):
                # Convert feats dict to string if needed
                feats_str = '_'
                if token['feats']:
                    if isinstance(token['feats'], dict):
                        feats_str = '|'.join([f"{k}={v}" for k, v in sorted(token['feats'].items())])
                    else:
                        feats_str = str(token['feats'])
                
                # Convert misc dict to string if needed
                misc_str = '_'
                if token['misc']:
                    if isinstance(token['misc'], dict):
                        misc_str = '|'.join([f"{k}={v}" if v else k for k, v in sorted(token['misc'].items())])
                    else:
                        misc_str = str(token['misc'])
                
                fields = [
                    str(token['id']),
                    token['form'],
                    token['lemma'] if token['lemma'] else '_',
                    token['upos'] if token['upos'] else '_',
                    token['xpos'] if token['xpos'] else '_',
                    feats_str,
                    str(token['head']) if token['head'] else '_',
                    token['deprel'] if token['deprel'] else '_',
                    token['deps'] if token['deps'] else '_',
                    misc_str,
                ]
                
                # Add extra columns if they exist in the input
                # Check if there are extra columns in the header
                extra_cols = []
                for col_name in reader.header[10:]:  # Columns after standard 10
                    if col_name == SUPERSENSE_COLUMN:
                        extra_cols.append(predictions[tok_idx])
                    else:
                        extra_cols.append(token.get(col_name, '_'))
                
                # If no extra columns in header but we need to add the supersense column
                if SUPERSENSE_COLUMN not in reader.header:
                    extra_cols.append(predictions[tok_idx])
                
                # Write all fields
                all_fields = fields + extra_cols
                outfile.write('\t'.join(all_fields) + '\n')
            
            outfile.write('\n')
            num_sentences += 1
            num_words += len(sent)
        
        logger.info(f"Processed {num_sentences} sentences ({num_words} words)")
        logger.info(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
