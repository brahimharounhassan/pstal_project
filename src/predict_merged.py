"""
END-TO-END prediction using MERGED LoRA model.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Use shared utility
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lib.conllulib import CoNLLUReader
from src.utils import write_conllu_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PREDICT_MERGED")


def load_merged_model(model_path: str, device: str = 'cpu'):
    """
    Load merged LoRA model (standard AutoModelForTokenClassification).
    
    Args:
        model_path: Path to merged model checkpoint
        device: Device to load model on
        
    Returns:
        model, model_name, id2label
    """
    logger.info(f"Loading merged model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_name = checkpoint['model_name']
    num_labels = checkpoint['num_labels']
    id2label = checkpoint['id2label']
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Labels: {num_labels}")
    
    # Load model structure
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Load trained weights (classifier + encoder with merged LoRA)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("Merged model loaded successfully (no LoRA wrapper)")
    logger.info("Ready for END-TO-END prediction")
    
    return model, model_name, id2label


def predict_sentence_merged(sentence, model, tokenizer, id2label, target_upos, device='cpu'):
    """
    Predict super-senses using merged LoRA model.
    """
    model.eval()
    
    with torch.no_grad():
        words = [tok["form"] for tok in sentence]
        upos_tags = [tok["upos"] for tok in sentence]
        
        # Tokenize
        encodings = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding=False,
            truncation=True
        ).to(device)
        
        word_ids = encodings.word_ids()
        
        # Forward pass
        outputs = model(**encodings)
        logits = outputs.logits[0]  # [seq_len, num_labels]
        
        # Get predictions
        token_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Align with words
        predictions = []
        for word_idx in range(len(words)):
            if upos_tags[word_idx] not in target_upos:
                predictions.append('*')
                continue
            
            # Find tokens for this word
            token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if not token_indices:
                predictions.append('*')
                continue
            
            # Use first subtoken prediction
            pred_label_id = token_predictions[token_indices[0]]
            pred_label = id2label.get(pred_label_id, '*')
            
            predictions.append(pred_label)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='END-TO-END Super-sense prediction with merged LoRA model.')
    parser.add_argument('--model', required=True, help='Path to merged model (.pt)')
    parser.add_argument('--input', required=True, help='Input CoNLL-U file')
    parser.add_argument('--output', required=True, help='Output CoNLL-U file')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info("END-TO-END PREDICTION WITH MERGED MODEL")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    model, model_name, id2label = load_merged_model(args.model, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer loaded: {model_name}")
    
    # Get target UPOS
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    target_upos = set(checkpoint.get('target_upos', ['NOUN', 'PROPN', 'NUM']))
    logger.info(f"Target UPOS: {target_upos}")
    
    # Read input
    logger.info(f"Reading input from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = CoNLLUReader(f)
        sentences = list(reader.readConllu())
    logger.info(f"Processing {len(sentences)} sentences...")
    
    # Predict
    all_prediction_lists = []
    
    for sentence in tqdm(sentences, desc="Predicting (merged model)"):
        predictions = predict_sentence_merged(
            sentence, model, tokenizer, id2label, target_upos, args.device
        )
        all_prediction_lists.append(predictions)
    
    # Write output
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = CoNLLUReader(f)
        write_conllu_predictions(
            sentences=sentences,
            predictions_list=all_prediction_lists,
            reader=reader,
            output_path=args.output,
            supersense_column='frsemcor:noun'
        )
    
    # Stats
    total_words = sum(len(sent) for sent in sentences)
    target_tokens = sum(len([p for p in preds if p != '*']) for preds in all_prediction_lists)
    
    logger.info(f"Processed {len(sentences)} sentences ({total_words} words)")
    logger.info(f"Predicted supersenses for {target_tokens} tokens")
    logger.info(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
