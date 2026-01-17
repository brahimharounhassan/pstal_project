"""
Hyperparameter tuning with Optuna for LoRA fine-tuning.
"""

import argparse
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
    epochs: int = 5,
    device: str = "cuda",
    accumulation_steps: int = None
) -> float:
    """
    Train and evaluate model with given hyperparameters.
    """
    
    if accumulation_steps is None:
        accumulation_steps = ACCUMULATION_STEPS

    # Hyperparameters - CamemBERT + LoRA
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    r = trial.suggest_int("r", 8, 32)
    alpha = trial.suggest_int("lora_alpha", 8, 64) 
    dora = trial.suggest_categorical("use_dora", [True, False])
    dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
    bs = trial.suggest_categorical('bs', [8, 16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

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
            target_modules=["query", "value", "key"],
            lora_dropout=dropout,
            bias="none",
            task_type="TOKEN_CLS",
            use_dora=dora
        )

        lora_model = get_peft_model(base_model, lora_config)
        
        # Freeze backbone parameters, keep LoRA adapters and classifier trainable
        for name, param in lora_model.named_parameters():
            if "lora_" not in name and "modules_to_save" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        logger.info(f"Trial {trial.number+1 } | Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

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

        best_val_accuracy = 0.0

        for epoch in range(epochs):
            # Training
            lora_model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Trial {trial.number+1} Epoch {epoch+1}", leave=False)):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

                with autocast(device_type=device):
                    outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / accumulation_steps

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

            # Clear GPU cache after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                    
                    val_loss += outputs.loss.item()
                    
                    # Get predictions
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    active_labels = labels.view(-1)
                    active_predictions = predictions.view(-1)
                    mask = active_labels != -100
                    
                    all_predictions.extend(active_predictions[mask].cpu().numpy())
                    all_labels.extend(active_labels[mask].cpu().numpy())

            avg_val_loss = val_loss / len(dev_loader)
            
            # Calculate metrics
            val_accuracy = accuracy_score(all_labels, all_predictions)
            val_f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            val_f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            
            best_val_accuracy = max(best_val_accuracy, val_accuracy)

            # Clear predictions and labels to free memory
            del all_predictions, all_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"Trial {trial.number+1} Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            # Report for pruning (negative because Optuna minimizes)
            trial.report(-val_accuracy, epoch+1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return negative accuracy (Optuna minimizes, we want to maximize)
        return -best_val_accuracy
        
    except Exception as e:
        logger.error(f"Trial {trial.number+1} failed: {e}")
        raise
    finally:
        # Aggressive memory cleanup
        if 'train_loader' in locals():
            del train_loader
        if 'dev_loader' in locals():
            del dev_loader
        if 'lora_model' in locals():
            lora_model.cpu()
            del lora_model
        if 'base_model' in locals():
            del base_model
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler' in locals():
            del scheduler
        if 'scaler' in locals():
            del scaler
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train hyper-parameter tuning with Optuna for LoRA fine-tuning')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS_TUNER, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--n-epochs', type=int, default=N_EPOCH_TUNER, help='Number of epochs for each trial')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')
    
    args = parser.parse_args()
    

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
    
    try:
        logger.info("Hyperparameter tuning with Optuna")

        model_name = args.model_name

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        n_epoch = args.n_epochs
        n_trials = args.n_trials

        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Number of labels: {len(train_data_prep.label2id)}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Epochs per trial: {n_epoch}")

        def objective(trial):
            return train_eval_model(
                trial=trial,
                tokenizer=tokenizer,
                train_data_prep=train_data_prep,
                dev_data_prep=dev_data_prep,
                num_labels=len(train_data_prep.label2id),
                model_name=model_name,
                epochs=n_epoch,
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
        logger.info(f"Optimization device: {device}")
        optuna_start_time = time.time()
        optuna_start_timestamp = datetime.now().isoformat()
        
        # Callback to clean memory between trials
        def trial_callback(study, trial):
            logger.info(
                f"Trial {trial.number+1} finished | "
                f"F1 Macro: {-trial.value:.4f} | "
                f"Params: {trial.params}"
            )
            # Force GPU cleanup between trials
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1,
            callbacks=[trial_callback]
        )

        optuna_elapsed = time.time() - optuna_start_time
        optuna_end_timestamp = datetime.now().isoformat()
        
        device_end = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Optuna optimization completed in {format_time(optuna_elapsed)}")
        logger.info(f"Optimization end device: {device_end}")
        
        if device != device_end:
            logger.warning(f"Device changed during optimization: {device} --> {device_end}")
        
        best_hyperparameters = study.best_params
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        filepath = OUTPUT_PATH / f"best_hyperparameters_{timestamp}.json"

        # Add metadata to hyperparameters
        hyperparameters_with_metadata = {
            **best_hyperparameters,
            '_metadata': {
                'device_start': device,
                'device_end': device_end,
                'device_changed': (device != device_end),
                'optimization_start_time': optuna_start_timestamp,
                'optimization_end_time': optuna_end_timestamp,
                'optimization_duration_seconds': optuna_elapsed,
                'optimization_duration_formatted': format_time(optuna_elapsed),
                'best_f1_macro': -study.best_value,
                'best_trial_number': study.best_trial.number + 1,
                'timestamp': timestamp
            }
        }

        import json
        with open(filepath, "w") as f:
            json.dump(hyperparameters_with_metadata, f, indent=2)

        logger.info(f"Best hyperparameters saved to: {filepath}")
        logger.info(f"Best F1 Macro: {-study.best_value:.4f}")
        logger.info(f"Best trial: #{study.best_trial.number+1}")
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
            "best_trial_number": study.best_trial.number+1,
            "timestamp": timestamp,
            "model_name": model_name,
            "target_upos": list(TARGET_UPOS),
            "device_start": device,
            "device_end": device_end,
            "device_changed": (device != device_end),
            "optimization_start_time": optuna_start_timestamp,
            "optimization_end_time": optuna_end_timestamp,
            "optimization_duration_seconds": optuna_elapsed,
            "optimization_duration_formatted": format_time(optuna_elapsed)
        }
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH / f"study_summary{timestamp}.json", "w") as f:
            json.dump(study_summary, f, indent=2)

        logger.info("Hyperparameter tuning completed successfully")

    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}", exc_info=True)
        raise
