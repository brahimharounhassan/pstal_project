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

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np

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
    epochs: int = 30,
    device: str = "cuda",
    accumulation_steps: int = 1,
    metrics_file: str = None,
    class_weights: torch.Tensor = None
):
    """
    Train final model with best hyperparameters.
    """

    # Load model config
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # Base model
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    # LoRA config
    lora_config = LoraConfig(
        r=best_hyperparameters['r'],
        lora_alpha=best_hyperparameters.get('lora_alpha', best_hyperparameters['r'] * 2),
        target_modules=["query", "value", "key"],
        lora_dropout=best_hyperparameters['lora_dropout'],
        bias="none",
        task_type="TOKEN_CLS",
        use_dora=best_hyperparameters.get('use_dora', False)
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
    
    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights for imbalanced data")

    # Early stopping params
    patience = PATIENCE
    best_val_f1_macro = 0.0
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
    logger.info(f"Fine-tuning started - {epochs} epochs max")
    
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        lora_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"), color="green", leave=False, ncols=80):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            with autocast(device_type=device):
                outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # Apply class weights
                if class_weights is not None:
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                else:
                    loss = outputs.loss
                
                loss = loss / accumulation_steps

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
            for batch in tqdm(dev_loader, desc="Validation", color="yellow", leave=False, ncols=80):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                
                with autocast(device_type=device):
                    outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                    # Compute loss with class weights
                    if class_weights is not None:
                        logits = outputs.logits
                        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                    else:
                        loss = outputs.loss
                
                val_loss_total += loss.item()
                
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
        
        # Calculate metrics - ADDED macro F1
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

        # Early stopping check : based on macro F1
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            best_epoch = epoch + 1
            epochs_no_improve = 0
            

            # Overwrite same checkpoint fil
            if best_model_fname is None:
                best_model_fname = CHECKPOINT_PATH / "best_model_checkpoint.pt"
            
            logger.info(f"Saving checkpoint: epoch {epoch+1}, F1 macro={val_f1_macro:.4f}")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': lora_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_f1_macro': val_f1_macro,
                'val_f1_weighted': val_f1_weighted,
                'val_accuracy': val_accuracy,
                'hyperparameters': best_hyperparameters,
                'all_predictions': all_predictions,
                'all_labels': all_labels
            }, best_model_fname)
            
            logger.info(f"Saved best checkpoint: {best_model_fname.name}")
            
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
    if epochs_no_improve < patience:
        logger.info(f"Training completed all {epochs} epochs")
    logger.info(f"Training time: {format_time(training_elapsed)}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best F1 Macro: {best_val_f1_macro:.4f}")

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    np.save(OUTPUT_PATH / "train_losses.npy", np.array(train_losses))
    np.save(OUTPUT_PATH / "val_losses.npy", np.array(val_losses))
    np.save(OUTPUT_PATH / "val_f1_scores_macro.npy", np.array(val_f1_scores_macro))
    np.save(OUTPUT_PATH / "val_f1_scores_weighted.npy", np.array(val_f1_scores_weighted))
    np.save(OUTPUT_PATH / "val_accuracies.npy", np.array(val_accuracies))
    np.save(OUTPUT_PATH / "learning_rates.npy", np.array(learning_rates))

    # Load best model
    if best_model_fname and best_model_fname.exists():
        checkpoint = torch.load(best_model_fname, map_location=device, weights_only=False)
        lora_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    else:
        logger.warning("No best model checkpoint found, using final model state")

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
            # Fallback to original files
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
        
        # Compute class weights
        class_weights = train_data_prep.compute_class_weights()
        logger.info(f"Class weights: Min={class_weights.min():.3f}, Max={class_weights.max():.3f}")
        logger.info(f"Non-zero weights: {(class_weights > 0).sum()}/{len(class_weights)}")

        model = train_final_model(
            train_loader=train_loader,
            dev_loader=dev_loader,
            num_labels=len(train_data_prep.label2id),
            best_hyperparameters=best_hyperparameters,
            model_name=model_name,
            id2label=train_data_prep.id2label,
            epochs=n_epochs,
            device=DEVICE,
            accumulation_steps=ACCUMULATION_STEPS,
            metrics_file=metrics_file,
            class_weights=class_weights
        )

        Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
        
        # Save LoRA model with PeftModel format (adapters only)
        logger.info("Saving LoRA model with PeftModel format...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_model_dir = Path(MODEL_PATH) / f"final_model_lora_{timestamp}"
        
        # Save LoRA adapters and base model config
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        logger.info(f"LoRA model type: {type(model)}")
        
        # Save additional metadata in a separate JSON file
        import json
        metadata = {
            'hyperparameters': best_hyperparameters,
            'num_labels': len(train_data_prep.label2id),
            'model_name': model_name,
            'label2id': train_data_prep.label2id,
            'id2label': train_data_prep.id2label,
            'target_upos': TARGET_UPOS,
            'merged': False,
            'lora_format': True,
            'timestamp': timestamp
        }
        
        with open(final_model_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LoRA model saved to: {final_model_dir}")
        logger.info(f"This model can be loaded with PeftModel.from_pretrained()")
        logger.info(f"Adapters size is much smaller than full model")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
