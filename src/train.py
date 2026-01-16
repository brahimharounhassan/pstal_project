import argparse

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch

from configs.config import MODEL_NAME, N_EPOCH_TUNER

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train hyper-parameter tuning with Optuna for LoRA fine-tuning')
    parser.add_argument('--baseline-model', type=str, default=MODEL_NAME, help='Baseline model name')
    parser.add_argument('--finetuned-model', type=str, help='Finetuned model name')
    parser.add_argument('--n-epochs', type=int, default=N_EPOCH_TUNER, help='Number of epochs')

    args = parser.parse_args()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = args.baseline_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModel.from_pretrained(model_name)
    base_model.to(device)

    # Load PEFT 
    model = PeftModel.from_pretrained(base_model, args.finetuned_model)
    model.eval()



