#!/usr/bin/env python3
"""
Super-sense prediction script from a trained model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from model_ssense import SuperSenseClassifier
from configs.config import *

from src.utils import setup_logger

logger = setup_logger('EVAL SUPERSENSE')


def load_checkpoint(filepath, device='cpu'):
    """
    Loads a super-sense classifier from a saved checkpoint.
    
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Rebuilds the classifier with the saved hyperparameters
    model = SuperSenseClassifier(
        embedding_dim=checkpoint['embedding_dim'],
        num_labels=checkpoint['num_labels'],
        dropout=checkpoint['dropout']
    )
    
    # Loads the model weights
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    # Reverse the vocabulary for prediction (index -> label)
    label_vocab_rev = {v: k for k, v in checkpoint['label_vocab'].items()}
    
    # Get finetuned model path if available
    finetuned_model_path = checkpoint.get('finetuned_model_path', None)
    
    return model, label_vocab_rev, checkpoint['model_name'], finetuned_model_path


def predict_sentence(sentence, model, tokenizer, transformer_model, 
                     label_vocab_rev, device='cpu'):
    """
    Predicts super-senses for all words in a sentence.
    """
    model.eval()
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
            logits = model(avg_embedding)
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
    parser.add_argument('--finetuned-model', default=None,
                        help='Path to fine-tuned LoRA model (.pth)')
    
    args = parser.parse_args()
    
    logger.info(f"Loading model from {args.model}")
    classifier, label_vocab_rev, model_name, finetuned_from_checkpoint = load_checkpoint(args.model, device=args.device)
    logger.info(f"Model loaded on device: {args.device}")
    logger.info(f"Transformer model: {model_name}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine which fine-tuned model to use (CLI arg overrides checkpoint value)
    finetuned_model_path = args.finetuned_model if args.finetuned_model else finetuned_from_checkpoint
    
    # If using a fine-tuned model, load it with LoRA weights merged
    if finetuned_model_path:
        logger.info(f"Loading fine-tuned LoRA model from: {finetuned_model_path}")
        from train_finetuned import load_finetuned_model
        transformer_model, _ = load_finetuned_model(finetuned_model_path, args.device)
        logger.info("Fine-tuned model loaded with LoRA weights merged")
    else:
        logger.info("Using base transformer model (no fine-tuning)")
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
