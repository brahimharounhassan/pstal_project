import sys
from pathlib import Path
from tqdm import tqdm
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from model_ssense import SuperSenseClassifier
from lib.conllulib import Util
from src.utils import SuperSenseDataPreparation, setup_logger
from configs.config import *

logger = setup_logger('TRAIN SUPERSENSE BASE')

def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}min {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}min {secs:02d}s"
    else:
        return f"{secs}s"

def fit(model: SuperSenseClassifier, train_loader: DataLoader, loss_fn, optimizer, device):
    """
    Trains the classifier on one epoch of the training corpus.
    """
    model.train()
    total_loss = 0.0
    
    for embeddings, labels in tqdm(train_loader, desc="Training batches", ncols=80, leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train_epochs(
    model: SuperSenseClassifier,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str
) -> SuperSenseClassifier:

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_acc = 0.0
    patience = 10
    no_improve = 0

    training_start_time = time.time()
    logger.info(f"Classifier training started ({epochs} epochs)")

    for epoch in range(epochs):
        train_loss = fit(model, train_loader, loss_fn, optimizer, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, loss_fn, device)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.4f}"
        )

        if dev_acc > best_acc:
            best_acc = dev_acc
            no_improve = 0
            logger.info(f"New best accuracy: {best_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered")
                break

    elapsed = time.time() - training_start_time
    logger.info(f"Training finished in {format_time(elapsed)}")
    logger.info(f"Best dev accuracy: {best_acc:.4f}")

    return model


def evaluate(model: SuperSenseClassifier, dev_loader: DataLoader, loss_fn: nn.CrossEntropyLoss, device: str) -> tuple[float, float]:
    """
    Evaluates performance on the development corpus.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for embeddings, labels in dev_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            logits = model(embeddings)
            loss = loss_fn(logits, labels)
            
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dev_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train super-sense classifier')
    parser.add_argument('--train', required=True, help='Path to training corpus')
    parser.add_argument('--dev', required=True, help='Path to dev corpus')
    parser.add_argument('--output', default='models/ssense_model.pt', help='Output model file')
    parser.add_argument('--model-name', default='almanach/camembert-base',
                        help='Transformer model name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    # parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Loading transformer model: {args.model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    transformer_model = AutoModel.from_pretrained(args.model_name)
    
    # Get embedding dimension from model config
    embedding_dim = transformer_model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    data_prep = SuperSenseDataPreparation(
        tokenizer=tokenizer, 
        model=transformer_model,
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
        # hidden_dim=embedding_dim, # args.hidden_dim,
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
        # 'hidden_dim': embedding_dim,
        'dropout': args.dropout,
        'model_name': args.model_name
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
