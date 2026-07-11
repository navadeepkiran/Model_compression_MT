import os
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

print("=== WMT26 Gemma 3 12B Hierarchical Slicing (Step 5: Neurons CS-DE) ===")

model_id = "nani-nav/gemma-3-12b-40L-csde"
fisher_dir = "pruning"
repo_id = "nani-nav/gemma-3-12b-final-csde"

hf_token = os.environ.get("HF_TOKEN")

# We are keeping 70% (dropping 30%)
FFN_KEEP_RATIO = 0.7  

print("[*] Loading Neuron Fisher Scores...")
with open(os.path.join(fisher_dir, "neuron_fisher_csde.json"), "r") as f:
    neuron_scores = np.array(json.load(f))
    
print(f"[*] Loading 40-layer model to CPU RAM in BFloat16 (~20GB)...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cpu", 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    token=hf_token
)

text_config = model.config.text_config if hasattr(model.config, "text_config") else model.config
old_intermediate_size = text_config.intermediate_size
new_intermediate_size = int(old_intermediate_size * FFN_KEEP_RATIO)

print(f"[*] Target FFN size: {new_intermediate_size}/{old_intermediate_size}")

# Find the layers
num_layers = text_config.num_hidden_layers
target_layers = None

for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and len(module) == num_layers:
        target_layers = module
        break

if target_layers is None:
    raise ValueError("Could not find the transformer layers inside the model!")

print("[*] Slicing FFN Neurons for all layers...", flush=True)

import gc

for layer_idx in range(num_layers):
    print(f"  -> Slicing Layer {layer_idx}/{num_layers}", flush=True)
    layer = target_layers[layer_idx]
    
    # Get scores for this specific layer
    layer_neuron_scores = neuron_scores[layer_idx]
    
    # Find the indices of the highest scoring neurons to KEEP
    kept_neurons = np.argsort(layer_neuron_scores)[-new_intermediate_size:]
    kept_neurons = sorted(kept_neurons) 
    
    mlp = layer.mlp
    
    # Slice the Linear layers
    new_gate = nn.Linear(text_config.hidden_size, new_intermediate_size, bias=False, dtype=torch.bfloat16)
    new_gate.weight.data = mlp.gate_proj.weight.data[kept_neurons, :].clone()
    
    new_up = nn.Linear(text_config.hidden_size, new_intermediate_size, bias=False, dtype=torch.bfloat16)
    new_up.weight.data = mlp.up_proj.weight.data[kept_neurons, :].clone()
    
    new_down = nn.Linear(new_intermediate_size, text_config.hidden_size, bias=False, dtype=torch.bfloat16)
    new_down.weight.data = mlp.down_proj.weight.data[:, kept_neurons].clone()
    
    # AGGRESSIVE RAM SAVING: Delete the old massive tensors from RAM immediately!
    del mlp.gate_proj
    del mlp.up_proj
    del mlp.down_proj
    gc.collect()
    
    # Replace old massive FFN with shrunken FFN
    layer.mlp.gate_proj = new_gate
    layer.mlp.up_proj = new_up
    layer.mlp.down_proj = new_down

# Update Config
text_config.intermediate_size = new_intermediate_size

print(f"[*] Slicing complete! New architecture:")
print(f"    Layers: {text_config.num_hidden_layers}")
print(f"    FFN Size: {text_config.intermediate_size}")

print(f"[*] Pushing final pruned model directly to Hugging Face ({repo_id}) to save Disk Space...")
model.push_to_hub(repo_id, token=hf_token, private=True)
tokenizer.push_to_hub(repo_id, token=hf_token, private=True)

print("[*] Model successfully pruned and pushed! It is now permanently smaller and ready for QLoRA fine-tuning!")
