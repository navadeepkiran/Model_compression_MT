import os
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

print("=== WMT26 Gemma 3 12B Hierarchical Slicing (Step 3: Layers CS-DE) ===")

model_id = "google/gemma-3-12b-it"
fisher_dir = "pruning"
output_dir = "/kaggle/working/outputs/gemma3-12b-40L-csde"
os.makedirs(output_dir, exist_ok=True)

LAYERS_TO_DROP = 8  

print("[*] Loading Layer Fisher Scores...")
with open(os.path.join(fisher_dir, "layer_fisher_csde.json"), "r") as f:
    layer_scores = np.array(json.load(f))
    
num_layers = len(layer_scores)
# Using the exact same non-contiguous lowest-score logic from the Chinese run
kept_layer_indices = sorted(np.argsort(layer_scores)[LAYERS_TO_DROP:])
print(f"[*] Keeping {len(kept_layer_indices)}/{num_layers} layers: {kept_layer_indices}")

print(f"[*] Loading original model to CPU RAM in BFloat16 (~24GB)...")
hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

# Load to CPU to manipulate the tensors safely without VRAM limits
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cpu", 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    token=hf_token
)

text_config = model.config.text_config if hasattr(model.config, "text_config") else model.config

# Bulletproof Layer Finder
target_layers = None
layer_parent_module = None
layer_attr_name = None

for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and len(module) == num_layers:
        target_layers = module
        if '.' in name:
            parent_name, layer_attr_name = name.rsplit('.', 1)
            layer_parent_module = model.get_submodule(parent_name)
        else:
            layer_parent_module = model
            layer_attr_name = name
        break

new_layers = nn.ModuleList()

print(f"[*] Dropping the {LAYERS_TO_DROP} absolute weakest layers...")
for old_idx in kept_layer_indices:
    new_layers.append(target_layers[old_idx])

# Reassign the pruned ModuleList back into the model
setattr(layer_parent_module, layer_attr_name, new_layers)

# Update Config
text_config.num_hidden_layers = len(kept_layer_indices)
if hasattr(text_config, "layer_types"):
    text_config.layer_types = [text_config.layer_types[i] for i in kept_layer_indices]

print(f"[*] Slicing complete! New architecture has {text_config.num_hidden_layers} Layers.")
print(f"[*] Saving 40-layer model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

repo_id = "navadeepkiran/gemma3-12b-40L-csde"
print(f"[*] Pushing 40-layer model to Hugging Face ({repo_id})...")
model.push_to_hub(repo_id, token=hf_token, private=True)
tokenizer.push_to_hub(repo_id, token=hf_token, private=True)

print("[*] Done! The 8 lowest layers are gone and model is safe on Hugging Face. You can now use this model for Step 4!")
