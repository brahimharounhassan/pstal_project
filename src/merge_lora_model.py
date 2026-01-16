"""
Merge LoRA weights into base model and save for easy loading.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoConfig
from peft import LoraConfig, get_peft_model
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MERGE_LORA")


def merge_and_save_model(checkpoint_path: str, output_path: str):
    """
    Load LoRA model, merge weights, and save as standard model.
    """
    logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model_name = checkpoint['model_name']
    num_labels = checkpoint['num_labels']
    hyperparameters = checkpoint['hyperparameters']
    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']
    target_upos = checkpoint['target_upos']
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Labels: {num_labels}")
    logger.info(f"LoRA r={hyperparameters['r']}, alpha={hyperparameters.get('lora_alpha', hyperparameters['r']*2)}")
    
    # Create base model with classification head
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    
    # Add LoRA
    lora_config = LoraConfig(
        r=hyperparameters['r'],
        lora_alpha=hyperparameters.get('lora_alpha', hyperparameters['r'] * 2),
        target_modules=["query", "value", "key"],
        lora_dropout=hyperparameters['lora_dropout'],
        bias="none",
        task_type="TOKEN_CLS",
        use_dora=hyperparameters.get('use_dora', False)
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    
    # Load trained weights
    logger.info("Loading LoRA weights...")
    missing_keys, unexpected_keys = lora_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
        for key in missing_keys[:5]:
            logger.warning(f" - {key}")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys[:5]:
            logger.warning(f" - {key}")
    
    # Merge LoRA weights into base model
    logger.info("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()
    
    # Verify it's a standard model now
    logger.info(f"Merged model type: {type(merged_model)}")
    logger.info(f"Has base_model attr: {hasattr(merged_model, 'base_model')}")
    
    # Save merged model
    logger.info(f"Saving merged model to: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': merged_model.state_dict(),
        'model_name': model_name,
        'num_labels': num_labels,
        'label2id': label2id,
        'id2label': id2label,
        'target_upos': target_upos,
        'hyperparameters': hyperparameters,
        'merged': True
    }, output_path)
    
    logger.info("Merged model saved successfully")
    logger.info(f"This model can be loaded with standard AutoModelForTokenClassification")
    
    return merged_model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge LoRA weights into base model'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to LoRA checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save merged model (.pt file)'
    )
    
    args = parser.parse_args()
    
    merge_and_save_model(args.checkpoint, args.output)
