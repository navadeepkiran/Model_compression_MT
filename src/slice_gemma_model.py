import os
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def slice_model():
    print("=== WMT26 Gemma 3 12B Architectural Slicing ===")
    
    model_id = "google/gemma-3-12b-it"
    fisher_dir = "outputs/fisher_scores"
    output_dir = "outputs/gemma3-12b-sliced"
    
    # Pruning Ratios
    LAYERS_TO_DROP = 8  # Drop 8 out of 40 layers
    FFN_KEEP_RATIO = 0.7  # Keep 70% of FFN neurons
    
    print("[*] Loading Fisher Scores...")
    with open(os.path.join(fisher_dir, "layer_fisher.json"), "r") as f:
        layer_scores = np.array(json.load(f))
    with open(os.path.join(fisher_dir, "neuron_fisher.json"), "r") as f:
        neuron_scores = np.array(json.load(f))
        
    # 1. Determine Layers to Keep
    num_layers = len(layer_scores)
    # Sort layers by score descending
    kept_layer_indices = sorted(np.argsort(layer_scores)[LAYERS_TO_DROP:])
    print(f"[*] Keeping {len(kept_layer_indices)}/{num_layers} layers.")
    
    # 2. Load Model on CPU in BF16 (requires ~24GB RAM, fits in Kaggle's 30GB)
    print(f"[*] Loading original model to CPU in BFloat16...")
    hf_token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cpu", 
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    config = model.config
    old_intermediate_size = config.intermediate_size
    new_intermediate_size = int(old_intermediate_size * FFN_KEEP_RATIO)
    
    print(f"[*] Target FFN size: {new_intermediate_size}/{old_intermediate_size}")
    
    # 3. Perform Slicing
    import torch.nn as nn
    
    new_layers = nn.ModuleList()
    
    for old_idx in kept_layer_indices:
        layer = model.model.layers[old_idx]
        
        # Determine FFN neurons to keep for this specific layer
        layer_neuron_scores = neuron_scores[old_idx]
        kept_neurons = np.argsort(layer_neuron_scores)[-new_intermediate_size:]
        kept_neurons = sorted(kept_neurons) # keep order to preserve spatial correlation if any
        
        # Slice MLP matrices
        mlp = layer.mlp
        
        # gate_proj: [intermediate_size, hidden_size]
        new_gate = nn.Linear(config.hidden_size, new_intermediate_size, bias=False)
        new_gate.weight.data = mlp.gate_proj.weight.data[kept_neurons, :].clone()
        
        # up_proj: [intermediate_size, hidden_size]
        new_up = nn.Linear(config.hidden_size, new_intermediate_size, bias=False)
        new_up.weight.data = mlp.up_proj.weight.data[kept_neurons, :].clone()
        
        # down_proj: [hidden_size, intermediate_size]
        new_down = nn.Linear(new_intermediate_size, config.hidden_size, bias=False)
        new_down.weight.data = mlp.down_proj.weight.data[:, kept_neurons].clone()
        
        # Assign back
        layer.mlp.gate_proj = new_gate
        layer.mlp.up_proj = new_up
        layer.mlp.down_proj = new_down
        
        new_layers.append(layer)
        
    # Replace layers in model
    model.model.layers = new_layers
    
    # Update Config
    model.config.num_hidden_layers = len(kept_layer_indices)
    model.config.intermediate_size = new_intermediate_size
    
    print(f"[*] Slicing complete! New architecture:")
    print(f"    Layers: {model.config.num_hidden_layers}")
    print(f"    FFN Size: {model.config.intermediate_size}")
    
    # 4. Save Model
    print(f"[*] Saving shrunken model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[*] Model saved successfully. Ready for INT4 QLoRA!")

if __name__ == "__main__":
    slice_model()
