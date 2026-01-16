#!/usr/bin/env python3
"""
Training script for super-sense classifier with fine-tuned embeddings.
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
from transformers import AutoTokenizer, AutoConfig, AutoModel

from model_ssense import SuperSenseClassifier
from lib.conllulib import Util
from configs.config import *
from src.train_ssense import train_epochs

logger = setup_logger('TRAIN_SSENSE_FINETUNED')


def load_finetuned_model(model_path: str, device: str):
    """
    Load fine-tuned LoRA model for embedding extraction.
    """
    from transformers import AutoModelForTokenClassification
    from peft import LoraConfig, get_peft_model
    
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_name = checkpoint['model_name']
    hyperparameters = checkpoint['hyperparameters']
    num_labels = checkpoint['num_labels']
    
    logger.info(f"Base model: {model_name}")
    logger.info(f"Number of labels: {num_labels}")
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
    
    logger.info("LoRA model loaded successfully")
    
    # Extract the base transformer (RoBERTa) with proper structure
    # We need to wrap it to have the right interface
    finetuned_model = lora_model.base_model.model.roberta
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()
    
    logger.info(" Extracted transformer model for embedding generation")
    
    return finetuned_model, model_name


def main():
    parser = argparse.ArgumentParser(
        description='Train super-sense classifier with fine-tuned embeddings',
    )
    parser.add_argument('--train', required=True, help='Path to training corpus')
    parser.add_argument('--dev', required=True, help='Path to dev corpus')
    parser.add_argument('--output', default='models/ssense_finetuned.pth', help='Output model file')
    parser.add_argument('--finetuned-model', required=True, help='Fine-tuned LoRA model file path (.pt)')
    parser.add_argument('--n-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    parser.add_argument('--no-normalize', action='store_true', default=True,
                        help='Disable embedding normalization')
    
    args = parser.parse_args()
    
    logger.info("Feature-based Training")
    logger.info(f"Device: {args.device}")
    
    finetuned_model, model_name = load_finetuned_model(
        args.finetuned_model, 
        args.device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    embedding_dim = finetuned_model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")

    data_prep = SuperSenseDataPreparation(
        tokenizer=tokenizer, 
        model=finetuned_model,
        device=args.device,
        normalize_embeddings=(not args.no_normalize)
    )
    
    # Prepare training data
    logger.info("Preparing training data (extracting embeddings)...")
    train_embeddings, train_labels, label_vocab = data_prep.prepare_data(args.train)
    
    # Prepare dev data
    logger.info("Preparing dev data (extracting embeddings)...")
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

    # Train
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
    
    label_vocab_dict = dict(label_vocab)
    
    checkpoint = {
        'model_state': model.state_dict(),
        'label_vocab': label_vocab_dict,
        'embedding_dim': embedding_dim,
        'num_labels': len(label_vocab),
        'dropout': args.dropout,
        'model_name': model_name,
        'finetuned_model_path': args.finetuned_model,
        'approach': 'feature-based'
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Fine-tuned model used: {args.finetuned_model}")


if __name__ == '__main__':
    main()
