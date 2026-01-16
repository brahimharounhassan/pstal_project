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
from peft import LoraConfig, get_peft_model, PeftModel

from model_ssense import SuperSenseClassifier
from lib.conllulib import Util
from configs.config import *
from src.train_ssense import train_epochs

logger = setup_logger('TRAIN SSENSE FINE-TUNED')

def load_finetuned_model(finetuned_model_path: str, device: str):
    """
    Charge le modèle fine-tuné depuis un répertoire (format Hugging Face).
    Si c'est un ancien fichier .pt, charge avec l'ancienne méthode pour compatibilité.
    """

    logger.info(f"Chargement du modèle fine-tuné: {finetuned_model_path}")
    
    # Check if it's a directory (new format) or a .pt file (old format)
    finetuned_path = Path(finetuned_model_path)
    
    if finetuned_path.is_dir():
        # Load metadata
        import json
        metadata_file = finetuned_path / "training_metadata.json"
        is_lora_format = False
        model_name = None
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model trained on: {metadata.get('timestamp', 'unknown')}")
            logger.info(f"Number of labels: {metadata.get('num_labels')}")
            is_lora_format = metadata.get('lora_format', False)
            model_name = metadata.get('model_name')
        
        # Check if it's a LoRA model (PeftModel format)
        adapter_config_file = finetuned_path / "adapter_config.json"
        if adapter_config_file.exists() or is_lora_format:
            logger.info("Loading LoRA model using PeftModel.from_pretrained()")
            
            # Load base model first
            if not model_name:
                # Try to read from adapter config
                with open(adapter_config_file, 'r') as f:
                    adapter_config = json.load(f)
                model_name = adapter_config.get('base_model_name_or_path', 'camembert/camembert-base')
            
            logger.info(f"Base model: {model_name}")
            
            # Load base model config with correct num_labels from metadata
            num_labels = metadata.get('num_labels', 25)
            logger.info(f"Loading base model with num_labels={num_labels}")
            
            base_config = AutoConfig.from_pretrained(model_name)
            base_config.num_labels = num_labels
            base_model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                config=base_config
            )
            
            # Load LoRA adapters
            lora_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
            lora_model = lora_model.to(device)
            lora_model.eval()
            
            logger.info(f"LoRA model loaded successfully")
            logger.info(f"Config: hidden_size={lora_model.config.hidden_size}")
            
            # Extract the RoBERTa/CamemBERT model (with embeddings + encoder)
            # NOT just the encoder - we need the full model for input processing
            finetuned_model = lora_model.base_model.roberta
            finetuned_model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
            
            return finetuned_model, model_name
        
        else:
            # Merged model format (old but still valid)
            logger.info("Loading merged model using from_pretrained() (Hugging Face format)")
            
            merged_model = AutoModelForTokenClassification.from_pretrained(finetuned_model_path)
            tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
            
            # Get model name from config
            model_name = merged_model.config._name_or_path
            
            logger.info(f"Model loaded: {model_name}")
            logger.info(f"Config: hidden_size={merged_model.config.hidden_size}")
            
            # Extract the encoder from the merged model
            finetuned_encoder = merged_model.roberta.encoder
            finetuned_encoder = finetuned_encoder.to(device)
            finetuned_encoder.eval()
            
            return finetuned_encoder, model_name
    
    else:
        # Old format: Load from .pt checkpoint (backward compatibility)
        logger.info("Loading model from .pt checkpoint (legacy format)")
        
        checkpoint = torch.load(finetuned_model_path, map_location=device)
        
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

        # To Freeze backbone parameters
        for name, param in lora_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        
        lora_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        lora_model.eval()
        
        logger.info("LoRa weights loaded (legacy format)")
        
        # Extract the basic transformer (RoBERTa) from the model
        finetuned_model = lora_model.roberta.encoder
        finetuned_model = finetuned_model.to(device)
        finetuned_model.eval()
        
        return finetuned_model, model_name


def main():
    parser = argparse.ArgumentParser(description='Train super-sense classifier with fine-tuned model')
    parser.add_argument('--train', required=True, help='Path to training corpus')
    parser.add_argument('--dev', required=True, help='Path to dev corpus')
    parser.add_argument('--output', default='models/ssense_finetuned.pth', help='Output model file')
    parser.add_argument('--finetuned-model', default='models/final_model_epochs_50.pth', help='Fine-tuned model file path')
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
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
    
    # Build label vocabulary from both train and dev to avoid label mismatch
    logger.info("Building global label vocabulary from train and dev")
    from lib.conllulib import CoNLLUReader
    
    all_labels = set()
    for file_path in [args.train, args.dev]:
        sentences = list(CoNLLUReader(open(file_path, 'r', encoding='utf-8')).readConllu())
        for sent in sentences:
            for token in sent:
                if hasattr(token, 'supersenses') and token.supersenses:
                    all_labels.add(token.supersenses[0])
    
    # Create label vocab with all labels
    label_vocab = {label: idx for idx, label in enumerate(sorted(all_labels))}
    if '*' not in label_vocab:
        label_vocab['*'] = len(label_vocab)
    
    logger.info(f"Global label vocabulary: {len(label_vocab)} labels")
    
    # Prepare training data with predefined vocab
    logger.info("Preparing training data")
    train_embeddings, train_labels, _ = data_prep.prepare_data(args.train, label_vocab=label_vocab)
    
    # Prepare dev data with same vocab
    logger.info("Preparing dev data")
    dev_embeddings, dev_labels, _ = data_prep.prepare_data(args.dev, label_vocab=label_vocab)
    
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
        'model_state': model.state_dict(),
        'label_vocab': label_vocab_dict,
        'embedding_dim': embedding_dim,
        'num_labels': len(label_vocab),
        'dropout': args.dropout,
        'model_name': model_name,
        'finetuned_model_path': args.finetuned_model 
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")
    logger.info(f"Fine-tuned model path saved: {args.finetuned_model}")


if __name__ == '__main__':
    main()
