import sys
from pathlib import Path
import time
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np

from src.utils import TuningDataPreparation, setup_training_logger
from configs.config import *
import glob
import json

tqdm.pandas()

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


logger, metrics_file = setup_training_logger(log_dir=LOG_PATH)

def train_final_model(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_labels: int,
    best_hyperparameters: dict,
    model_name: str,
    epochs: int=30,
    device: str="cuda",
    accumulation_steps: int=1,
    metrics_file: str=None
    ):

    # Load model config
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # Base model
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    # LoRA config
    lora_config = LoraConfig(
        r=best_hyperparameters['r'],
        lora_alpha=best_hyperparameters['lora_alpha'],
        target_modules=["query", "value", "key"],
        lora_dropout=best_hyperparameters['lora_dropout'],
        bias="none",
        task_type="TOKEN_CLS",
        use_dora=best_hyperparameters['use_dora']
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.gradient_checkpointing_enable()  # to save memory
    lora_model.to(device)

    # Optimizer
    optimizer = AdamW(
        lora_model.parameters(),
        lr=best_hyperparameters['lr'],
        weight_decay=best_hyperparameters.get('weight_decay', 1e-4),
        eps=1e-8
        )

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * best_hyperparameters.get('warmup_ratio', 0.1))

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
        # verbose=True
        )

    # Mixed precision
    scaler = GradScaler()

    # Early stopping params
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    best_model_fname = None
    
    training_start_time = time.time()
    logger.info(f"Finetuning start - {epochs} epochs max")

    for epoch in range(epochs):
      lora_model.train()
      total_loss = 0.0
      optimizer.zero_grad()

      for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch +1}")):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        with autocast(device_type=device):
          outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

      avg_train_loss = total_loss / len(train_loader)
      train_losses.append(avg_train_loss)

      learning_rates.append(optimizer.param_groups[0]['lr'])

      # Validation
      lora_model.eval()
      val_loss_total = 0.0
      with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validation"):
          input_ids, attention_mask, labels = [x.to(device) for x in batch]
          with autocast(device_type=device):
            outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          val_loss_total += outputs.loss.item()

      avg_val_loss = val_loss_total / len(dev_loader)
      val_losses.append(avg_val_loss)

      logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

      # Update scheduler
      scheduler.step(avg_val_loss)

      # Early stopping check
      if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        
        # Delete old checkpoint to save space
        if best_model_fname is not None and best_model_fname.exists():
            best_model_fname.unlink()
            logger.info(f"Deleted old checkpoint to save space")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        best_model_fname = CHECKPOINT_PATH / f"best_model_epoch_{epoch+1}_{timestamp}.pt"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': lora_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'hyperparameters': best_hyperparameters
        }, best_model_fname)
        logger.info(f"Saved new best checkpoint: {best_model_fname.name}")

      else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
          training_elapsed = time.time() - training_start_time
          logger.info(f"Early stopping at epoch {epoch+1}")
          logger.info(f"Finetuning end - at time: {format_time(training_elapsed)}")
          break

    # Save metrics
    training_elapsed = time.time() - training_start_time
    if epochs_no_improve < patience:
        logger.info(f"Finetuning end - at time: {format_time(training_elapsed)}")
    
    np.save(OUTPUT_PATH / "train_losses.npy", np.array(train_losses))
    np.save(OUTPUT_PATH / "val_losses.npy", np.array(val_losses))
    np.save(OUTPUT_PATH / "learning_rates.npy", np.array(learning_rates))

    # Load best model
    checkpoint = torch.load(best_model_fname)
    lora_model.load_state_dict(checkpoint['model_state_dict'])

    # Load best weights
    return lora_model

if __name__ == "__main__":

    # Set up device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_data_prep = TuningDataPreparation(
    in_file=DATA_TRAIN,
    full_file=DATA_FULL
    )

    dev_data_prep = TuningDataPreparation(
        in_file=DATA_DEV,
        full_file=DATA_FULL
        )

    print("All supersense:", train_data_prep.sense_values)
    print(train_data_prep.id2label)

    try:
        logger.info("Final model training.")

        hp_files = glob.glob(os.path.join(OUTPUT_PATH, "best_hyperparameters_*.json"))
        if not hp_files:
            logger.error("No hyperparameters file found!")
            raise FileNotFoundError("No hyperparameter files found. Run hp_tuning.py first.")

        latest_hp_file = max(hp_files, key=os.path.getctime)
        logger.info(f"Loading hyperparameters from: {latest_hp_file}")

        with open(latest_hp_file, "r") as f:
            best_hyperparameters = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        batch_size = best_hyperparameters['bs']

        ModelConfig.batch_size = batch_size

        logger.info(f"Training configuration:")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Epochs: {N_EPOCHS}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Number of labels: {len(train_data_prep.label2id)}")

        train_loader = train_data_prep.create_dataloader(
            tokenizer,
            batch_size=batch_size,
            shuffle_mode=True
        )
        dev_loader = dev_data_prep.create_dataloader(
            tokenizer,
            batch_size=batch_size,
            shuffle_mode=False
        )

        model = train_final_model(
            train_loader=train_loader,
            dev_loader=dev_loader,
            num_labels=len(train_data_prep.label2id),
            best_hyperparameters=best_hyperparameters,
            model_name=MODEL_NAME,
            epochs=N_EPOCHS,
            device=DEVICE,
            accumulation_steps=1,
            metrics_file=metrics_file
        )

        # Saving model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_model_fname = os.path.join(
            MODEL_PATH,
            f"final_model_{timestamp}.pt"
        )

        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': best_hyperparameters,
            'num_labels': len(train_data_prep.label2id),
            'model_name': MODEL_NAME,
            'label2id': train_data_prep.label2id,
            'id2label': {v: k for k, v in train_data_prep.label2id.items()}
        }, final_model_fname)

        logger.info(f"model saved to: {final_model_fname}")

    except Exception as e:
        logger.error(f"training failed: {e}", exc_info=True)
        raise