from peft import LoraConfig, get_peft_model
import os 
from pytorch_lightning.utilities import rank_zero_info

def apply_lora_peft(model, lora_config_dict):
    
    if lora_config_dict.adaptation_scope == "full":
        target_modules = list(lora_config_dict.target_modules) + list(lora_config_dict.ff_modules)
    
    elif lora_config_dict.adaptation_scope == "cross":
        target_modules = ".*attn2.*(" + "|".join(lora_config_dict.target_modules) + ")"
    
    elif lora_config_dict.adaptation_scope == "attn":
        target_modules = ".*attn.*(" + "|".join(lora_config_dict.target_modules) + ")"

    else:
        raise ValueError(f"Unknown adaptation_scope: {lora_config_dict.adaptation_scope}")
        
    peft_config = LoraConfig(
        r=lora_config_dict.rank,
        lora_alpha=lora_config_dict.alpha,
        target_modules=target_modules,
        lora_dropout=lora_config_dict.dropout,
        bias="none"
    )

    unet = model.model.diffusion_model
    if not hasattr(unet, "config"):
        class MockConfig:
            def to_dict(self): return {}
        unet.config = MockConfig()

    model.model.diffusion_model = get_peft_model(unet, peft_config)
    
    print("--- PEFT Status ---")
    model.model.diffusion_model.print_trainable_parameters()

    return model

def export_lora_weights(model, logdir):

    lora_model = model.model.diffusion_model
    
    save_path = os.path.join(logdir, "lora_adapter")
    os.makedirs(save_path, exist_ok=True)

    lora_model.save_pretrained(save_path)
    
    rank_zero_info(f">>> LoRA adapter weights exported to: {save_path}")