import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_ssense import SuperSenseClassifier
from lib.conllulib import Util
from src.utils import SuperSenseDataPreparation, setup_logger
from configs.config import *

logger = setup_logger("TRAIN SUPERSENSE")


def fit(
    model: SuperSenseClassifier,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for embeddings, labels in tqdm(
        train_loader, desc="Training batches", ncols=80, leave=False
    ):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model: SuperSenseClassifier,
    dev_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for embeddings, labels in dev_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)
        loss = loss_fn(logits, labels)

        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(dev_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_epochs(
    model: SuperSenseClassifier,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device
) -> SuperSenseClassifier:

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    best_dev_loss = float("inf")
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        train_loss = fit(model, train_loader, loss_fn, optimizer, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, loss_fn, device)

        scheduler.step(dev_loss)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.4f}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
            logger.info(f"New best dev loss: {best_dev_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered")
                break


    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Best model restored")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train super-sense classifier")
    parser.add_argument("--train", required=True, help="Path to training corpus")
    parser.add_argument("--dev", required=True, help="Path to dev corpus")
    parser.add_argument("--output", default="models/ssense_model.pt")
    parser.add_argument("--model-name", default="almanach/camembert-base")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    Util.init_seed(SEED)

    logger.info(f"Using device: {device}")
    logger.info(f"Loading transformer: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    transformer_model = AutoModel.from_pretrained(args.model_name)

    transformer_model.eval()
    for p in transformer_model.parameters():
        p.requires_grad = False

    embedding_dim = transformer_model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")

    data_prep = SuperSenseDataPreparation(
        tokenizer=tokenizer,
        model=transformer_model,
        device=device
    )

    logger.info("Preparing training data")
    train_embeddings, train_labels, label_vocab = data_prep.prepare_data(args.train)

    logger.info("Preparing dev data")
    dev_embeddings, dev_labels, _ = data_prep.prepare_data(args.dev)

    train_loader = Util.dataloader(
        [train_embeddings],
        [train_labels],
        batch_size=args.batch_size,
        shuffle=True
    )

    dev_loader = Util.dataloader(
        [dev_embeddings],
        [dev_labels],
        batch_size=args.batch_size,
        shuffle=False
    )

    logger.info(f"Training samples: {len(train_labels)}")
    logger.info(f"Dev samples: {len(dev_labels)}")
    logger.info(f"Number of labels: {len(label_vocab)}")

    classifier = SuperSenseClassifier(
        embedding_dim=embedding_dim,
        num_labels=len(label_vocab),
        dropout=args.dropout
    ).to(device)

    logger.info(f"Classifier parameters: {Util.count_params(classifier):,}")

    model = train_epochs(
        classifier,
        train_loader,
        dev_loader,
        epochs=args.n_epochs,
        lr=args.lr,
        device=device
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "label_vocab": dict(label_vocab),
        "embedding_dim": embedding_dim,
        "num_labels": len(label_vocab),
        "dropout": args.dropout,
        "model_name": args.model_name
    }

    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
