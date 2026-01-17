"""
Fine-tuning script with LoRA for super-sense classification.
"""

from pathlib import Path
import sys
import time
import os
import glob
import argparse

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import json


from src.utils import TuningDataPreparation, setup_training_logger
from configs.config import *

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
    id2label: dict,
    label2id: dict,
    tokenizer,
    epochs: int = 30,
    device: str = "cuda",
    accumulation_steps: int = 1,
    metrics_file: str = None
):
    """
    Train final model with best hyperparameters.
    """

    # Load model config
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # Base model - use AutoModelForTokenClassification for proper token classification
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    # LoRA config
    lora_config = LoraConfig(
        r=best_hyperparameters['r'],
        lora_alpha=best_hyperparameters.get('lora_alpha', best_hyperparameters['r'] * 2),
        target_modules=["query", "value", "key"],
        lora_dropout=best_hyperparameters['lora_dropout'],
        bias="none",
        task_type="TOKEN_CLS",
        use_dora= True, #best_hyperparameters.get('use_dora', False),
        use_rslora=False
    )

    lora_model = get_peft_model(base_model, lora_config)
    
    # Freeze backbone params, keep LoRA and classifier trainable
    for name, param in lora_model.named_parameters():
        if "lora_" not in name and "modules_to_save" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())

    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    lora_model.to(device)

    # Optimizer
    optimizer = AdamW(
        lora_model.parameters(),
        lr=best_hyperparameters['lr'],
        weight_decay=best_hyperparameters.get('weight_decay', 1e-4),
        eps=1e-8
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs // accumulation_steps
    warmup_steps = int(total_steps * best_hyperparameters.get('warmup_ratio', 0.1))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision
    scaler = GradScaler()

    # Early stopping params
    patience = PATIENCE
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_f1_scores_macro = []
    val_f1_scores_weighted = []
    val_accuracies = []
    learning_rates = []
    best_model_fname = None
    best_epoch = 0
    
    training_start_time = time.time()
    training_start_timestamp = datetime.now().isoformat()
    logger.info(f"Fine-tuning started - {epochs} epochs max")


    logger.info(f"Training start device: {device}")
    
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        lora_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", colour="blue", leave=False, ncols=80)):
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
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Validation
        lora_model.eval()
        val_loss_total = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validation", colour="yellow", leave=False, ncols=80):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                
                with autocast(device_type=device):
                    outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                val_loss_total += outputs.loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Flatten and filter out ignored indices (-100)
                active_labels = labels.view(-1)
                active_predictions = predictions.view(-1)
                mask = active_labels != -100
                
                all_predictions.extend(active_predictions[mask].cpu().numpy())
                all_labels.extend(active_labels[mask].cpu().numpy())

        avg_val_loss = val_loss_total / len(dev_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        val_f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        val_f1_scores_macro.append(val_f1_macro)
        val_f1_scores_weighted.append(val_f1_weighted)
        val_accuracies.append(val_accuracy)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1 Macro: {val_f1_macro:.4f} | "
            f"Val F1 Weighted: {val_f1_weighted:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Log metrics to CSV
        if metrics_file:
            with open(metrics_file, 'a') as f:
                f.write(
                    f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},"
                    f"{val_f1_macro:.6f},{val_f1_weighted:.6f},{val_accuracy:.6f},"
                    f"{optimizer.param_groups[0]['lr']:.8e},{datetime.now().isoformat()}\n"
                )

        # Early stopping check : based on accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_no_improve = 0
            

            # Save checkpoint in PeftModel adapter format
            if best_model_fname is None:
                best_model_fname = CHECKPOINT_PATH / "best_model_checkpoint"
            
            logger.info(f"Saving checkpoint: epoch {epoch+1}, Accuracy={val_accuracy:.4f}, F1 macro={val_f1_macro:.4f}")
            
            # Save LoRA adapters
            lora_model.save_pretrained(best_model_fname)
            tokenizer.save_pretrained(best_model_fname)
            
            # Save training metadata
            checkpoint_metadata = {
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_f1_macro': val_f1_macro,
                'val_f1_weighted': val_f1_weighted,
                'val_accuracy': val_accuracy,
                'hyperparameters': best_hyperparameters,
                'num_labels': num_labels,
                'model_name': model_name,
                'label2id': label2id,
                'id2label': id2label,
                'timestamp': datetime.now().isoformat(),
                'device_start': device,
                'device_current': str(next(lora_model.parameters()).device),
                'training_start_time': training_start_timestamp,
                'checkpoint_saved_time': datetime.now().isoformat(),
                'elapsed_time_seconds': time.time() - training_start_time,
                'elapsed_time_formatted': format_time(time.time() - training_start_time)
            }
            
            with open(best_model_fname / "checkpoint_metadata.json", 'w') as f:
                json.dump(checkpoint_metadata, f, indent=2)
            
            logger.info(f"Saved best checkpoint to: {best_model_fname}")
            
            # Print classification report for best model
            try:
                target_names = [id2label[i] for i in sorted(set(all_labels))]
                report = classification_report(
                    all_labels, 
                    all_predictions, 
                    target_names=target_names,
                    zero_division=0,
                    digits=4
                )
                logger.info(f"\nClassification Report (Best Epoch {epoch+1}):\n{report}")
            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{patience} epochs")
            
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Save metrics
    training_elapsed = time.time() - training_start_time
    training_end_timestamp = datetime.now().isoformat()
    
    # Detect actual device at end
    device_end = str(next(lora_model.parameters()).device)
    
    if epochs_no_improve < patience:
        logger.info(f"Training completed all {epochs} epochs")
    logger.info(f"Training time: {format_time(training_elapsed)}")
    logger.info(f"Training end device: {device_end}")
    
    if device != device_end:
        logger.warning(f"Device changed during training: {device} --> {device_end}")
    
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Accuracy: {best_val_accuracy:.4f}")

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    np.save(OUTPUT_PATH / "train_losses.npy", np.array(train_losses))
    np.save(OUTPUT_PATH / "val_losses.npy", np.array(val_losses))
    np.save(OUTPUT_PATH / "val_f1_scores_macro.npy", np.array(val_f1_scores_macro))
    np.save(OUTPUT_PATH / "val_f1_scores_weighted.npy", np.array(val_f1_scores_weighted))
    np.save(OUTPUT_PATH / "val_accuracies.npy", np.array(val_accuracies))
    np.save(OUTPUT_PATH / "learning_rates.npy", np.array(learning_rates))

    # Load best model
    if best_model_fname and best_model_fname.exists():
        from peft import PeftModel
        logger.info(f"Loading best checkpoint from: {best_model_fname}")
        
        # Load checkpoint metadata
        with open(best_model_fname / "checkpoint_metadata.json", 'r') as f:
            checkpoint_metadata = json.load(f)
        
        # load base model
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
        
        # Load LoRA adapters
        lora_model = PeftModel.from_pretrained(base_model, best_model_fname)
        lora_model.to(device)
        
        logger.info(f"Loaded best model from epoch {checkpoint_metadata['epoch']}")
    else:
        logger.warning("No best model checkpoint found, using final model state")

    # Store training info for metadata
    lora_model.training_info = {
        'start_time': training_start_timestamp,
        'end_time': training_end_timestamp,
        'elapsed_seconds': training_elapsed,
        'elapsed_formatted': format_time(training_elapsed),
        'device_start': device,
        'device_end': device_end,
        'device_changed': (device != device_end),
        'best_epoch': best_epoch,
        'best_accuracy': best_val_accuracy
    }

    return lora_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train hyper-parameter tuning with Optuna for LoRA fine-tuning')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--n-epochs', type=int, default=N_EPOCH_TUNER, help='Number of epochs')

    args = parser.parse_args()


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_data_prep = TuningDataPreparation(
        in_file=DATA_TRAIN,
        full_file=DATA_FULL,
        target_upos=TARGET_UPOS
    )

    dev_data_prep = TuningDataPreparation(
        in_file=DATA_DEV,
        full_file=DATA_FULL,
        target_upos=TARGET_UPOS
    )

    logger.info(f"Number of unique supersenses: {len(train_data_prep.sense_values)}")
    logger.info(f"Number of labels: {len(train_data_prep.label2id)}")
    logger.info(f"Label to ID mapping sample: {list(train_data_prep.label2id.items())[:5]}")

    try:
        logger.info("model training")

        # Find latest hyperparameters file 
        hp_files = glob.glob(str(Path(OUTPUT_PATH) / "best_hyperparameters*.json"))
        if not hp_files:
            hp_files = glob.glob(str(Path(OUTPUT_PATH) / "best_hyperparameters_*.json"))
        
        if not hp_files:
            raise FileNotFoundError(
                "No hyperparameters file found."
            )

        latest_hp_file = max(hp_files, key=os.path.getctime)
        logger.info(f"Loading hyperparameters from: {latest_hp_file}")

        import json
        with open(latest_hp_file, "r") as f:
            best_hyperparameters = json.load(f)

        
        model_name = args.model_name
        n_epochs = args.n_epochs

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = best_hyperparameters['bs']

        ModelConfig.batch_size = batch_size

        logger.info(f"Training configuration:")
        logger.info(f"Model: {model_name}")
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Number of labels: {len(train_data_prep.label2id)}")
        logger.info(f"Target UPOS: {TARGET_UPOS}")
        logger.info(f"Hyperparameters: {best_hyperparameters}")

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
            model_name=model_name,
            id2label=train_data_prep.id2label,
            label2id=train_data_prep.label2id,
            tokenizer=tokenizer,
            epochs=n_epochs,
            device=DEVICE,
            accumulation_steps=ACCUMULATION_STEPS,
            metrics_file=metrics_file
        )

        Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
        
        # Save LoRA model with PeftModel format (adapters only)
        logger.info("Saving LoRA model with PeftModel format...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_model_dir = Path(MODEL_PATH) / f"peft_adapter_{timestamp}"
        
        # Save LoRA adapters and base model config
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        logger.info(f"LoRA model type: {type(model)}")
        
        # Save additional metadata in a separate JSON file
        metadata = {
            'hyperparameters': best_hyperparameters,
            'num_labels': len(train_data_prep.label2id),
            'model_name': model_name,
            'label2id': train_data_prep.label2id,
            'id2label': train_data_prep.id2label,
            'target_upos': list(TARGET_UPOS),
            'merged': False,
            'lora_format': True,
            'timestamp': timestamp,
            'training_device_start': model.training_info['device_start'],
            'training_device_end': model.training_info['device_end'],
            'training_device_changed': model.training_info['device_changed'],
            'training_start_time': model.training_info['start_time'],
            'training_end_time': model.training_info['end_time'],
            'training_duration_seconds': model.training_info['elapsed_seconds'],
            'training_duration_formatted': model.training_info['elapsed_formatted'],
            'best_epoch': model.training_info['best_epoch'],
            'best_accuracy': model.training_info['best_accuracy']
        }
        
        with open(final_model_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LoRA model saved to: {final_model_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
