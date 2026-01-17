#!/usr/bin/env python3
"""
Training script for the super-sense classifier with contextual embeddings.
"""

import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import SuperSenseDataPreparation, setup_logger

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model, PeftModel
import json

from model_ssense import SuperSenseClassifier
from lib.conllulib import Util
from configs.config import *
from src.train_ssense import train_epochs

logger = setup_logger('TRAIN SSENSE FINE-TUNED')

def load_finetuned_model(finetuned_model_path: str, device: str):
    """
    Loads a fine-tuned model from the specified path.
    """

    finetuned_path = Path(finetuned_model_path)
    
    adapter_config_file = finetuned_path / "adapter_config.json"
    metadata_file = finetuned_path / "training_metadata.json"
    
    if not adapter_config_file.exists() and not metadata_file.exists():
        # Find all peft_adapter_* directories
        peft_dirs = list(finetuned_path.glob("peft_adapter*"))
        
        # get the most recent adapter directory
        peft_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        finetuned_path = peft_dirs[0]
        logger.info(f"Loading fine-tuned model from: {finetuned_model_path}/{finetuned_path.name}")
    
    # METADATA
    metadata_file = finetuned_path / "training_metadata.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata.get('model_name')
    
    # Load base model config
    num_labels = metadata.get('num_labels', 25)
    logger.info(f"Base model: {model_name}")
    
    base_config = AutoConfig.from_pretrained(model_name)
    base_config.num_labels = num_labels
    
    # Load as AutoModelForTokenClassification since it was fine-tuned with task_type="TOKEN_CLS"
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=base_config
    )
    
    # Load LoRA adapters
    finetuned_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()
    
    logger.info(f"LoRA model loaded successfully")
    logger.info(f"Config: hidden_size={finetuned_model.config.hidden_size}")
    
    # Merge LoRA adapters into base model to get standard weights
    # This combines the LoRA weights with the base weights, removing PEFT wrappers
    logger.info("Merging LoRA adapters into base model...")
    merged_model = finetuned_model.merge_and_unload()
    
    # Extract ONLY the encoder (RoBERTa/CamemBERT) - NOT the classification head
    # The classification head was trained for 25-class token classification,
    # but we want to train our own MLP classifier on top of embeddings
    if hasattr(merged_model, 'roberta'):
        embedding_model = merged_model.roberta
    elif hasattr(merged_model, 'bert'):
        embedding_model = merged_model.bert
    elif hasattr(merged_model, 'distilbert'):
        embedding_model = merged_model.distilbert
    else:
        # Fallback: try to get the base model
        embedding_model = merged_model.base_model
    
    embedding_model.eval()
    
    logger.info("LoRA adapters merged successfully")
    
    return embedding_model, model_name
    
    

def main():
    parser = argparse.ArgumentParser(description='Train super-sense classifier with fine-tuned model')
    parser.add_argument('--train', required=True, help='Path to training corpus')
    parser.add_argument('--dev', required=True, help='Path to dev corpus')
    parser.add_argument('--output', default='models/ssense_finetuned.pt', help='Output model file')
    parser.add_argument('--finetuned-model', default='models/', 
                        help='Path to fine-tuned model directory (peft_adapter_*)')
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info("Chargement du modèle fine-tuné...")
    
    # Load fine-tuned model (returns the embedding model without classification head)
    embedding_model, model_name = load_finetuned_model(args.finetuned_model, args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get embedding dimension from model config
    embedding_dim = embedding_model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")

    data_prep = SuperSenseDataPreparation(
        tokenizer=tokenizer, 
        model=embedding_model,
        device=args.device
        )
    
    # Prepare training data with predefined vocab
    logger.info("Preparing training data")
    train_embeddings, train_labels, label_vocab = data_prep.prepare_data(args.train)
    
    # Prepare dev data with same vocab
    logger.info("Preparing dev data")
    dev_embeddings, dev_labels, _ = data_prep.prepare_data(args.dev)
    
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
        dropout=args.dropout
    ).to(args.device)
    
    logger.info(f"Classifier parameters: {Util.count_params(classifier):,}")

    model = train_epochs(
        classifier, 
        train_loader, 
        dev_loader, 
        epochs=args.n_epochs, 
        lr=args.lr, 
        device=args.device
        )
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Reverse label vocab for prediction
    label_vocab_dict = dict(label_vocab)
    
    checkpoint = {
        'model_state': model.state_dict(),  # MLP weights
        'embedding_model_state': embedding_model.state_dict(),  # Fine-tuned encoder weights
        'label_vocab': label_vocab_dict,
        'embedding_dim': embedding_dim,
        'num_labels': len(label_vocab),
        'dropout': args.dropout,
        'model_name': model_name,
        'is_finetuned': True  # Flag to indicate this is a fine-tuned model
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
