#!/usr/bin/env python3
"""
Prediction script for super-sense classification with feature-based approach.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from model_ssense import SuperSenseClassifier
from configs.config import *
from src.utils import setup_logger
from lib.conllulib import CoNLLUReader

logger = setup_logger('PREDICT_SUPERSENSE')


def load_finetuned_model(model_path: str, device: str):
    """
    Load fine-tuned LoRA model for embedding extraction.
    """
    logger.info(f"Loading fine-tuned LoRA model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_name = checkpoint['model_name']
    hyperparameters = checkpoint['hyperparameters']
    num_labels = checkpoint['num_labels']
    
    logger.info(f"Base model: {model_name}")
    logger.info(f"LoRA params: r={hyperparameters['r']}, alpha={hyperparameters.get('lora_alpha', hyperparameters['r']*2)}")
    
    # Load base model
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    
    # Add LoRA
    lora_config = LoraConfig(
        r=hyperparameters['r'],
        lora_alpha=hyperparameters.get('lora_alpha', hyperparameters['r'] * 2),
        target_modules=["query", "value", "key"],
        lora_dropout=hyperparameters['lora_dropout'],
        bias="none",
        task_type="TOKEN_CLS",
        use_dora=hyperparameters.get('use_dora', False)
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    
    # Load trained weights
    lora_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    lora_model = lora_model.to(device)
    lora_model.eval()
    
    # Extract the base transformer for embedding extraction
    finetuned_model = lora_model.base_model.model.roberta
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()
    
    logger.info("Fine-tuned model loaded for embedding extraction")
    
    return finetuned_model, model_name


def load_classifier(checkpoint_path, device='cpu'):
    """
    Load the trained MLP classifier (SuperSenseClassifier).
    """
    logger.info(f"Loading classifier from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    embedding_dim = checkpoint['embedding_dim']
    num_labels = checkpoint['num_labels']
    dropout = checkpoint['dropout']
    label_vocab = checkpoint['label_vocab']
    finetuned_model_path = checkpoint['finetuned_model_path']
    
    # Reverse label vocab for prediction
    label_vocab_rev = {v: k for k, v in label_vocab.items()}
    
    # Initialize classifier
    classifier = SuperSenseClassifier(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        dropout=dropout
    )
    
    # Load trained weights
    classifier.load_state_dict(checkpoint['model_state'])
    classifier = classifier.to(device)
    classifier.eval()
    
    logger.info(f"Classifier loaded: {num_labels} labels, {embedding_dim}D embeddings")
    
    return classifier, label_vocab_rev, finetuned_model_path


def predict_sentence(sentence, finetuned_model, classifier, tokenizer, 
                     label_vocab_rev, target_upos, device='cpu', 
                     normalize_embeddings=False):
    """
    Predict super-senses for all words in a sentence using feature-based approach.
    """
    finetuned_model.eval()
    classifier.eval()
    
    with torch.no_grad():
        # Extract words and UPOS
        words = [tok["form"] for tok in sentence]
        upos_tags = [tok["upos"] for tok in sentence]
        
        # Tokenize
        tok_sent = tokenizer(words, is_split_into_words=True, return_tensors='pt')
        word_ids = tok_sent.word_ids()
        
        # Move to device
        tok_sent_device = {k: v.to(device) for k, v in tok_sent.items()}
        
        # Extract embeddings from fine-tuned model
        emb_sent = finetuned_model(**tok_sent_device)
        hidden = emb_sent.last_hidden_state[0]  # [T, embedding_dim]
        
        # Predict for each word
        predictions = []
        for word_idx in range(len(words)):
            # Check if target UPOS
            if upos_tags[word_idx] not in target_upos:
                predictions.append('*')
                continue
            
            # Find subtoken indices for this word
            subtoken_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if not subtoken_indices:
                predictions.append('*')
                continue
            
            # Average subword embeddings (comme dans le TP)
            word_emb = hidden[subtoken_indices].mean(dim=0)
            
            # Optional normalization
            if normalize_embeddings:
                word_emb = torch.nn.functional.normalize(word_emb, dim=-1)
            
            # Predict with MLP classifier
            word_emb = word_emb.unsqueeze(0)  # [1, embedding_dim]
            logits = classifier(word_emb)  # [1, num_labels]
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_label = label_vocab_rev.get(pred_idx, '*')
            
            predictions.append(pred_label)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Predict super-senses with feature-based approach (fine-tuned embeddings + MLP)'
    )
    parser.add_argument('--classifier', required=True, 
                        help='Path to trained MLP classifier (.pt from train_finetuned.py)')
    parser.add_argument('--input', required=True, help='Path to input corpus (CoNLL-U)')
    parser.add_argument('--output', default='predictions/ssense_finetuned.conllu', 
                        help='Path to output file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize embeddings before classification')
    
    args = parser.parse_args()
    
    logger.info("Super-sense Prediction")
    logger.info(f"Classifier: {args.classifier}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Load MLP classifier
    classifier, label_vocab_rev, finetuned_model_path = load_classifier(
        args.classifier, 
        device=args.device
    )
    
    # Load fine-tuned model for embedding extraction
    finetuned_model, model_name = load_finetuned_model(
        finetuned_model_path,
        device=args.device
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer loaded: {model_name}")
    
    # Process input file
    logger.info(f"Processing {args.input}")
    reader = CoNLLUReader(open(args.input, 'r', encoding='utf-8'))
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as outfile:
        num_sentences = 0
        num_words = 0
        num_predicted = 0
        
        for sent in tqdm(reader.readConllu(), desc="Predicting", ncols=80):
            # Predict super-senses
            predictions = predict_sentence(
                sent, finetuned_model, classifier, tokenizer, 
                label_vocab_rev, TARGET_UPOS, args.device,
                normalize_embeddings=args.normalize
            )
            
            # Write metadata
            if sent.metadata:
                for key, value in sent.metadata.items():
                    outfile.write(f"# {key} = {value}\n")
            
            # Write tokens with predictions
            for tok_idx, token in enumerate(sent):
                # Convert feats dict to string
                feats_str = '_'
                if token['feats']:
                    if isinstance(token['feats'], dict):
                        feats_str = '|'.join([f"{k}={v}" for k, v in sorted(token['feats'].items())])
                    else:
                        feats_str = str(token['feats'])
                
                # Convert misc dict to string
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
                
                # Add supersense column
                extra_cols = []
                for col_name in reader.header[10:]:
                    if col_name == SUPERSENSE_COLUMN:
                        extra_cols.append(predictions[tok_idx])
                        if predictions[tok_idx] not in ['*', '_']:
                            num_predicted += 1
                    else:
                        extra_cols.append(token.get(col_name, '_'))
                
                if SUPERSENSE_COLUMN not in reader.header:
                    extra_cols.append(predictions[tok_idx])
                    if predictions[tok_idx] not in ['*', '_']:
                        num_predicted += 1
                
                all_fields = fields + extra_cols
                outfile.write('\t'.join(all_fields) + '\n')
            
            outfile.write('\n')
            num_sentences += 1
            num_words += len(sent)
        
        logger.info(f"Processed {num_sentences} sentences ({num_words} words)")
        logger.info(f"Predicted supersenses for {num_predicted} tokens")
        logger.info(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
