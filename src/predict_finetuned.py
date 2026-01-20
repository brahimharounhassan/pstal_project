#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from peft import PeftModel
from tqdm import tqdm
import json

from configs.config import *
from src.utils import setup_logger, write_conllu_predictions
from lib.conllulib import CoNLLUReader

logger = setup_logger('PREDICT_FINETUNED')


def load_peft_adapter(peft_adapter_path: str, device: str):
    adapter_path = Path(peft_adapter_path)
    
    if not (adapter_path / "adapter_config.json").exists():
        peft_dirs = list(adapter_path.glob("peft_adapter*"))
        if not peft_dirs:
            raise FileNotFoundError(f"No peft_adapter* directory found in {peft_adapter_path}")
        peft_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        adapter_path = peft_dirs[0]
        logger.info(f"Using the most recent adapter: {adapter_path.name}")
    
    # Load metadata
    metadata_file = adapter_path / "training_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"training_metadata.json file not found in {adapter_path}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata.get('model_name')
    num_labels = metadata.get('num_labels', 25)  # 25 labels incluant '*' (extrait du corpus complet)
    label2id = metadata.get('label2id', {})
    id2label = metadata.get('id2label', {})
    
    # Convert id2label keys to int if necessary
    if id2label and isinstance(list(id2label.keys())[0], str):
        id2label = {int(k): v for k, v in id2label.items()}
    
    logger.info(f"Base model: {model_name}")
    logger.info(f"Number of labels: {num_labels}")
    
    # Load base model with classification head
    base_config = AutoConfig.from_pretrained(model_name)
    base_config.num_labels = num_labels
    
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=base_config
    )
    
    # Load LoRA adapters
    logger.info("Loading LoRA adapters...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    peft_model = peft_model.to(device)
    peft_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return peft_model, tokenizer, id2label


def predict_sentence(sentence, model, tokenizer, id2label, target_upos, device='cpu'):
    words = [token['form'] for token in sentence]
    upos_tags = [token['upos'] for token in sentence]
    
    # Tokenization
    tok_sent = tokenizer(
        words,
        is_split_into_words=True,
        padding=False,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**tok_sent)
        logits = outputs.logits.squeeze(0)  # [seq_len, num_labels]
    
    word_ids = tok_sent.word_ids()
    predictions = []
    
    for word_idx in range(len(words)):
        # If not in target_upos, predict '*'
        if upos_tags[word_idx] not in target_upos:
            predictions.append('*')
            continue
        
        # Retrieve token indices for this word
        token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
        
        if not token_indices:
            predictions.append('*')
            continue
        
        # Average the logits of subwords
        word_logits = logits[token_indices].mean(dim=0)
        
        # Predict the class with highest logit
        predicted_label_id = torch.argmax(word_logits).item()
        predicted_label = id2label.get(predicted_label_id, '*')
        
        predictions.append(predicted_label)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Prediction with fine-tuned model from peft_adapter'
    )
    parser.add_argument('--peft-adapter', required=True, 
                        help='Path to peft_adapter (e.g., models/peft_adapter_*)')
    parser.add_argument('--input', required=True, 
                        help='Input CoNLL-U file')
    parser.add_argument('--output', required=True, 
                        help='Output CoNLL-U file with predictions')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    logger.info(f"Adapter: {args.peft_adapter}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Load fine-tuned model from peft_adapter
    model, tokenizer, id2label = load_peft_adapter(args.peft_adapter, args.device)
    
    # Load input corpus
    logger.info(f"\nLoading corpus: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = CoNLLUReader(f)
        sentences = list(reader.readConllu())
    
    logger.info(f"{len(sentences)} sentences loaded")
    
    # Make predictions
    predictions_list = []
    
    for sentence in tqdm(sentences, desc="Prediction", ncols=80, colour="green"):
        predictions = predict_sentence(
            sentence, 
            model, 
            tokenizer,
            id2label,
            TARGET_UPOS,
            device=args.device
        )
        predictions_list.append(predictions)
    
    # Write predictions
    logger.info(f"\nWriting predictions: {args.output}")
    
    # Reload reader for writing
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = CoNLLUReader(f)
        sentences = list(reader.readConllu())
    
    num_sentences, num_words, num_predicted = write_conllu_predictions(
        sentences,
        predictions_list,
        reader,
        args.output,
        supersense_column=SUPERSENSE_COLUMN
    )
    
    logger.info(f"{num_sentences} sentences written")
    logger.info(f"{num_words} tokens in total")
    

if __name__ == '__main__':
    main()
