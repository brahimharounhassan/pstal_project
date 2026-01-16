import sys
from pathlib import Path
import time

# Add workspace root to Python path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from transformers import (AutoConfig, AutoModelForTokenClassification, AutoTokenizer)
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import utils
import torch
import optuna
from src.utils import TuningDataPreparation, setup_logger
from configs.config import *
from configs.config import ModelConfig
from datetime import datetime
import json 
from optuna import visualization as vis
import math

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
    train_data_prep: DataLoader,
    dev_data_prep: DataLoader,
    num_labels: int,
    model_name: str,
    epochs: int=5,
    device: str="cuda",
    accumulation_steps: int=1
    ) -> float:

    # hyperparameters to search - IMPROVED RANGES for better performance
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    r = trial.suggest_int("r", 16, 64)  # Increased from 4-16 to 16-64
    alpha = trial.suggest_int("lora_alpha", 32, 128)  # Increased from 8-32 to 32-128
    dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    dora = trial.suggest_categorical("use_dora", [True, False])
    bs = trial.suggest_categorical('bs', [8, 16, 32])  # Smaller batches for larger LoRA rank

    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    # if the rank is more than 8 don't use dora for memory safety usage
    # if dora and r > 8:
    #   raise optuna.TrialPruned()
    try:
      train_loader = train_data_prep.create_dataloader(tokenizer, batch_size=bs, shuffle_mode=True)
      dev_loader = dev_data_prep.create_dataloader(tokenizer, batch_size=bs, shuffle_mode=False)

      # Load base model
      config = AutoConfig.from_pretrained(model_name)
      config.num_labels = num_labels
      base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

      # Add LoRA - including "key" for better adaptation
      lora_config = LoraConfig(
          r=r,
          lora_alpha=alpha,
          target_modules=["query", "value", "key"],  # Added "key"
          lora_dropout=dropout,
          bias="none",
          task_type="TOKEN_CLS",
          use_dora=dora
      )

      model = get_peft_model(base_model, lora_config)
      model.gradient_checkpointing_enable()
      model.to(device)


      # Optimizer with weight decay
      optimizer = AdamW(
          model.parameters(),
          lr=lr,
          weight_decay=weight_decay,
          eps=1e-8
          )

      # Learning rate scheduler
      total_steps = len(train_loader) * epochs
      warmup_steps = int(total_steps * warmup_ratio)
      scheduler = CosineAnnealingLR(
          optimizer,
          T_max=total_steps - warmup_steps
          )

      # Mixed precision
      scaler = GradScaler()

      best_val_loss = float("inf")

      for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
          input_ids, attention_mask, labels = [x.to(device) for x in batch]

          with autocast(device_type=device):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps  # Scale loss

          # optimizer.zero_grad(set_to_none=True)

          scaler.scale(loss).backward()

          if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step >= warmup_steps:
              scheduler.step()

          train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)


        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
          for batch in tqdm(dev_loader, desc="Validation"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            with autocast(device_type=device):
              outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(dev_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        trial.report(avg_val_loss, epoch+1)
        if trial.should_prune():
          raise optuna.TrialPruned()

      return best_val_loss
    except Exception as e:
      logger.error(f"Trial failed: {e}")
      raise
    finally:
      # memory cleanup
      del model, optimizer
      if 'scheduler' in locals():
        del scheduler
      torch.cuda.empty_cache()
      import gc; gc.collect()



logger = setup_logger("hyperparameter_search", log_dir=LOG_PATH)

if __name__ == "__main__":

    ModelConfig.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    train_data_prep = TuningDataPreparation(
    in_file=DATA_TRAIN,
    full_file=DATA_FULL
    )

    dev_data_prep = TuningDataPreparation(
        in_file=DATA_DEV,
        full_file=DATA_FULL
        )

    try:
        logger.info("hyper params tuning.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Number of trials: 20")
        logger.info(f"Epochs per trial: 10")

        def objective(trial):
            logger.info(f"--- Trial {trial.number+1} ---")
            result = train_eval_model(
                trial=trial,
                tokenizer=tokenizer,
                train_data_prep=train_data_prep,
                dev_data_prep=dev_data_prep,
                model_name=MODEL_NAME,
                epochs=10,
                num_labels=len(train_data_prep.label2id)
            )
            logger.info(f"Trial {trial.number} completed with loss: {result:.4f}")
            return result

        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna_study.db",
            study_name="lora_ner_optimization",
            load_if_exists=True, 
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3
            )
        )

        logger.info(f"Obtuna research start - {N_TRIALS_TUNER} trials")
        optuna_start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=N_TRIALS_TUNER,
            show_progress_bar=True,
            n_jobs=1,
            callbacks=[
                lambda study, trial: logger.info(
                    f"Trial {trial.number} finished with value: {trial.value} "
                    f"and parameters: {trial.params}"
                )
            ]
        )

        optuna_elapsed = time.time() - optuna_start_time
        logger.info(f"Otuna research ended - time : {format_time(optuna_elapsed)}")
        
        best_hyperparameters = study.best_params
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = OUTPUT_PATH / f"best_hyperparameters_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(best_hyperparameters, f, indent=2)

        logger.info(f"Best hyperparameters saved to: {filepath}")
        logger.info(f"Best validation loss: {study.best_value:.4f}")
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info(f"Best hyperparameters:")
        for key, value in best_hyperparameters.items():
            logger.info(f"  {key}: {value}")

        # Statistics
        completed_trials = [t for t in study.trials if t.value is not None and math.isfinite(t.value)]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        # Optuna vis
        if len(completed_trials) < 1:
            logger.warning("No valid trials for visualizations.")
        elif len(completed_trials) == 1:
            logger.warning("Only 1 valid trial. Limited visualizations available.")
            try:
                fname = OUTPUT_PATH / f"plot_optimization_history_{timestamp}.html"
                vis.plot_optimization_history(study).write_html(
                    fname,
                    include_plotlyjs="cdn"
                )
                logger.info("Optimization history plot saved")
            except Exception as e:
                logger.warning(f"Could not generate optimization history: {e}")
        else:
            plots_generated = 0

            # Optim. history
            try:
                fname = OUTPUT_PATH / f"plot_optimization_history_{timestamp}.html"
                vis.plot_optimization_history(study).write_html(
                    fname,
                    include_plotlyjs="cdn"
                )
                logger.info("Optimization history plot saved")
                plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate optimization history: {e}")
            # Params importances
            try:
                fname = OUTPUT_PATH / f"plot_param_importance_{timestamp}.html"
                vis.plot_param_importances(study).write_html(
                    fname,
                    include_plotlyjs="cdn"
                )
                logger.info("Parameter importance plot saved")
                plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate parameter importances: {e}")

            # Parallel coordinate
            try:
                fname = OUTPUT_PATH / f"plot_parallel_coordinates_{timestamp}.html"
                vis.plot_parallel_coordinate(study).write_html(
                    fname,
                    include_plotlyjs="cdn"
                )
                logger.info("Parallel coordinate plot saved")
                plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate parallel coordinate: {e}")
            # Slice plot
            try:
                fname = OUTPUT_PATH / f"plot_slice_{timestamp}.html"
                vis.plot_slice(study).write_html(
                    fname,
                    include_plotlyjs="cdn"
                )
                logger.info("Slice plot saved")
                plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate slice plot: {e}")
            # Contour plots
            try:
                params_to_plot = []
                if "lr" in best_hyperparameters and "r" in best_hyperparameters:
                    params_to_plot.append(["lr", "r"])
                if "lora_alpha" in best_hyperparameters and "lora_dropout" in best_hyperparameters:
                    params_to_plot.append(["lora_alpha", "lora_dropout"])
                if "lr" in best_hyperparameters and "weight_decay" in best_hyperparameters:
                    params_to_plot.append(["lr", "weight_decay"])

                for params in params_to_plot:
                    param_str = "_".join(params)
                    fname = OUTPUT_PATH / f"plot_contour_{param_str}_{timestamp}.html"
                    vis.plot_contour(study, params=params).write_html(
                       fname,
                       include_plotlyjs="cdn"
                    )
                    logger.info(f"Contour plot saved for {params}")
                    plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate contour plots: {e}")

            try:
                fname = OUTPUT_PATH / f"plot_edf_{timestamp}.html"
                vis.plot_edf(study).write_html(
                   fname,
                   include_plotlyjs="cdn"
                )
                logger.info("EDF plot saved")
                plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate EDF plot: {e}")

            # Intermediate vals
            try:
                fname = OUTPUT_PATH / f"plot_intermediate_values_{timestamp}.html"
                vis.plot_intermediate_values(study).write_html(
                    fname,
                    include_plotlyjs="cdn"
                )
                logger.info("Intermediate values plot saved")
                plots_generated += 1
            except Exception as e:
                logger.warning(f"Could not generate intermediate values plot: {e}")

            logger.info(f"Total plots generated: {plots_generated}")

        study_summary = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_completed": len(completed_trials),
            "n_pruned": len(pruned_trials),
            "n_failed": len(failed_trials),
            "best_trial_number": study.best_trial.number,
            "timestamp": timestamp
        }

        with open(OUTPUT_PATH / f"study_summary_{timestamp}.json", "w") as f:
            json.dump(study_summary, f, indent=2)

    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}", exc_info=True)
        raise

