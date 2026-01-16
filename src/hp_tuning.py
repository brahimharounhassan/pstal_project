"""
Hyperparameter tuning with Optuna for LoRA fine-tuning.
"""

from pathlib import Path
import sys
import time
import gc
import math

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from src.utils import TuningDataPreparation, setup_logger
from configs.config import *

from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torch.nn import utils
import torch
import optuna
from optuna import visualization as vis
from datetime import datetime


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


logger = setup_logger("hyperparameter_search", log_dir=LOG_PATH)


def train_eval_model(
    trial,
    tokenizer: AutoTokenizer,
    train_data_prep: TuningDataPreparation,
    dev_data_prep: TuningDataPreparation,
    num_labels: int,
    model_name: str,
    class_weights: torch.Tensor,
    epochs: int = 5,
    device: str = "cuda",
    accumulation_steps: int = None
) -> float:
    """
    Train and evaluate model with given hyperparameters.
    """
    
    if accumulation_steps is None:
        accumulation_steps = ACCUMULATION_STEPS

    # Hyperparameters - OPTIMIZED RANGES for CamemBERT + LoRA
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    r = trial.suggest_int("r", 8, 32)  # Extended range for complex tasks
    alpha = r * 2  # Standard ratio: alpha = 2*r
    dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
    bs = trial.suggest_categorical('bs', [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)

    try:
        # Create dataloaders
        train_loader = train_data_prep.create_dataloader(tokenizer, batch_size=bs, shuffle_mode=True)
        dev_loader = dev_data_prep.create_dataloader(tokenizer, batch_size=bs, shuffle_mode=False)

        # Load base model
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

        # LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query", "value", "key"],  # For CamemBERT (RoBERTa-based)
            lora_dropout=dropout,
            bias="none",
            task_type="TOKEN_CLS",
            use_dora=False  # Disabled for stability
        )

        lora_model = get_peft_model(base_model, lora_config)
        
        # Freeze backbone parameters
        for name, param in lora_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        logger.info(f"Trial {trial.number} | Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

        lora_model.to(device)

        # Optimizer
        optimizer = AdamW(
            lora_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8
        )

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs // accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Mixed precision
        scaler = GradScaler()
        
        # Move class weights to device
        class_weights_device = class_weights.to(device)

        best_val_f1_macro = 0.0

        for epoch in range(epochs):
            # Training
            lora_model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", leave=False)):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

                with autocast(device_type=device):
                    outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                    # Apply class weights
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_device, ignore_index=-100)
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps

            avg_train_loss = train_loss / len(train_loader)

            # Evaluation
            lora_model.eval()
            val_loss = 0.0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validation", leave=False):
                    input_ids, attention_mask, labels = [x.to(device) for x in batch]
                    
                    with autocast(device_type=device):
                        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        
                        # Compute loss with class weights
                        logits = outputs.logits
                        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_device, ignore_index=-100)
                        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                    
                    val_loss += loss.item()
                    
                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)
                    active_labels = labels.view(-1)
                    active_predictions = predictions.view(-1)
                    mask = active_labels != -100
                    
                    all_predictions.extend(active_predictions[mask].cpu().numpy())
                    all_labels.extend(active_labels[mask].cpu().numpy())

            avg_val_loss = val_loss / len(dev_loader)
            
            # Calculate metrics - ADDED: macro F1 for better imbalanced class handling
            val_accuracy = accuracy_score(all_labels, all_predictions)
            val_f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            val_f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            
            best_val_f1_macro = max(best_val_f1_macro, val_f1_macro)

            logger.info(
                f"Trial {trial.number} Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            # Report for pruning (negative because Optuna minimizes)
            trial.report(-val_f1_macro, epoch+1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return negative macro F1 (Optuna minimizes, we want to maximize)
        return -best_val_f1_macro
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise
    finally:
        # Memory cleanup
        if 'lora_model' in locals():
            del lora_model
        if 'base_model' in locals():
            del base_model
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler' in locals():
            del scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ModelConfig.device = device

    # Load data with preparation class
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
    
    # Compute class weights ONCE (not in every trial)
    class_weights = train_data_prep.compute_class_weights()
    logger.info(f"Class weights computed. Min: {class_weights.min():.3f}, Max: {class_weights.max():.3f}")
    logger.info(f"Non-zero weights: {(class_weights > 0).sum()}/{len(class_weights)}")

    try:
        logger.info("Hyperparameter tuning with Optuna")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Device: {device}")
        logger.info(f"Number of labels: {len(train_data_prep.label2id)}")
        logger.info(f"Number of trials: {N_TRIALS_TUNER}")
        logger.info(f"Epochs per trial: {N_EPOCH_TUNER}")

        def objective(trial):
            return train_eval_model(
                trial=trial,
                tokenizer=tokenizer,
                train_data_prep=train_data_prep,
                dev_data_prep=dev_data_prep,
                num_labels=len(train_data_prep.label2id),
                model_name=MODEL_NAME,
                class_weights=class_weights,
                epochs=N_EPOCH_TUNER,
                device=device,
                accumulation_steps=ACCUMULATION_STEPS
            )

        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna_study.db",
            study_name="lora_supersense",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=2
            )
        )

        logger.info(f"Starting Optuna optimization - {N_TRIALS_TUNER} trials")
        optuna_start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=N_TRIALS_TUNER,
            show_progress_bar=True,
            n_jobs=1,
            callbacks=[
                lambda study, trial: logger.info(
                    f"Trial {trial.number} finished | "
                    f"F1 Macro: {-trial.value:.4f} | "
                    f"Params: {trial.params}"
                )
            ]
        )

        optuna_elapsed = time.time() - optuna_start_time
        logger.info(f"Optuna optimization completed in {format_time(optuna_elapsed)}")
        
        best_hyperparameters = study.best_params
        best_hyperparameters['lora_alpha'] = best_hyperparameters['r'] * 2  # Add computed alpha
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = OUTPUT_PATH / f"best_hyperparameters_{timestamp}.json"

        import json
        with open(filepath, "w") as f:
            json.dump(best_hyperparameters, f, indent=2)

        logger.info(f"Best hyperparameters saved to: {filepath}")
        logger.info(f"Best F1 Macro: {-study.best_value:.4f}")
        logger.info(f"Best trial: #{study.best_trial.number}")
        logger.info("Best hyperparameters:")
        for key, value in best_hyperparameters.items():
            logger.info(f"{key}: {value}")

        # Statistics
        completed_trials = [t for t in study.trials if t.value is not None and math.isfinite(t.value)]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        logger.info(f"Completed trials: {len(completed_trials)}")
        logger.info(f"Pruned trials: {len(pruned_trials)}")
        logger.info(f"Failed trials: {len(failed_trials)}")

        # Generate visualizations
        if len(completed_trials) >= 2:
            try:
                # Optimization history
                fig = vis.plot_optimization_history(study)
                fig.write_html(OUTPUT_PATH / f"plot_optimization_history{timestamp}.html")
                
                # Parameter importances
                fig = vis.plot_param_importances(study)
                fig.write_html(OUTPUT_PATH / f"plot_param_importance{timestamp}.html")
                
                # Parallel coordinates
                fig = vis.plot_parallel_coordinate(study)
                fig.write_html(OUTPUT_PATH / f"plot_parallel_coordinates{timestamp}.html")
                
                logger.info("Visualizations saved to outputs/")
            except Exception as e:
                logger.warning(f"Could not generate visualizations: {e}")

        # Save study summary
        study_summary = {
            "best_f1_macro": -study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_completed": len(completed_trials),
            "n_pruned": len(pruned_trials),
            "n_failed": len(failed_trials),
            "best_trial_number": study.best_trial.number,
            "timestamp": timestamp,
            "model_name": MODEL_NAME,
            "target_upos": TARGET_UPOS
        }

        with open(OUTPUT_PATH / f"study_summary{timestamp}.json", "w") as f:
            json.dump(study_summary, f, indent=2)

        logger.info("Hyperparameter tuning completed successfully")

    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}", exc_info=True)
        raise
