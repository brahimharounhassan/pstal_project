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
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model

from model_ssense import SuperSenseClassifier
from lib.conllulib import Util
from configs.config import *
from src.train_ssense import fit, evaluate, train_epochs

logger = setup_logger('TRAIN SSENSE FINE-TUNED')

def load_finetuned_model(model_name: str, device: str):
    """
    Charge le dernier modèle fine-tuné depuis le répertoire models/.
    """

    logger.info(f"Chargement du modèle fine-tuné: {model_name}")
    
    checkpoint = torch.load(model_name, map_location=device)
    
    model_name = checkpoint['model_name']
    hyperparameters = checkpoint['hyperparameters']
    num_labels = checkpoint['num_labels']

    
    logger.info(f"Finetuned model: {model_name}")
    logger.info(f"Labels num: {num_labels}")
    logger.info(f"Hyperparams LoRA: r={hyperparameters['r']}, alpha={hyperparameters['lora_alpha']}")
    
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    
    lora_config = LoraConfig(
        r=hyperparameters['r'],
        lora_alpha=hyperparameters['lora_alpha'],
        target_modules=["query", "value", "key"],
        lora_dropout=hyperparameters['lora_dropout'],
        bias="none",
        task_type="TOKEN_CLS",
        use_dora=hyperparameters.get('use_dora', False),
        use_rslora=hyperparameters.get('use_rslora', False)
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    
    lora_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    lora_model.eval()
    
    lora_model = lora_model.merge_and_unload()
    
    logger.info("LoRa weights merged into the base model")
    
    # Extract the basic transformer (RoBERTa) from the merged model
    finetuned_model = lora_model.roberta
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()
    
    return finetuned_model, model_name

# final_model_epochs_50_2026-01-15_13-56-21

def main():
    parser = argparse.ArgumentParser(description='Train super-sense classifier with fine-tuned model')
    parser.add_argument('--train', required=True, help='Path to training corpus')
    parser.add_argument('--dev', required=True, help='Path to dev corpus')
    parser.add_argument('--output', default='models/ssense_finetuned.pth', help='Output model file')
    parser.add_argument('--finetuned-model', default='models/final_model_epochs_50.pth', help='Fine-tuned model file path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info("Chargement du modèle fine-tuné...")
    
    # Load fine-tuned model
    finetuned_model, model_name = load_finetuned_model(args.finetuned_model, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get embedding dimension from model config
    embedding_dim = finetuned_model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")

    data_prep = SuperSenseDataPreparation(
        tokenizer=tokenizer, 
        model=finetuned_model,
        device=args.device
        )
    
    # Prepare training data
    logger.info("Preparing training data")
    train_embeddings, train_labels, label_vocab = data_prep.prepare_data(args.train)
    
    # Prepare dev data
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
        epochs=args.epochs, 
        lr=args.lr, 
        device=args.device
        )
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Reverse label vocab for prediction
    label_vocab_dict = dict(label_vocab)
    
    checkpoint = {
        'model_state': model.state_dict(),
        'label_vocab': label_vocab_dict,
        'embedding_dim': embedding_dim,
        'num_labels': len(label_vocab),
        'dropout': args.dropout,
        'model_name': model_name,
        'finetuned_model_path': args.finetuned_model  # CRUCIAL: save which finetuned model was used
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")
    logger.info(f"Fine-tuned model path saved: {args.finetuned_model}")


if __name__ == '__main__':
    main()
