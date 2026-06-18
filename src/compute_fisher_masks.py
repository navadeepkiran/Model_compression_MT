import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm

def compute_fisher():
    print("=== WMT26 Gemma 3 12B Fisher Mask Computation ===")
    
    # 1. Configuration
    model_id = "google/gemma-3-12b-it"
    num_calibration_samples = 1000 # Adjust based on Kaggle limits
    max_length = 256
    output_dir = "outputs/fisher_scores"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Model in INT4
    print(f"[*] Loading {model_id} in INT4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    # Custom device map to save VRAM (keep heavy embeddings on CPU if needed)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    
    # 3. Setup Trainable Masks
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size
    
    print(f"[*] Initializing Masks for {num_layers} layers, {num_heads} heads, and {intermediate_size} FFN neurons...")
    
    layer_masks = nn.Parameter(torch.ones(num_layers, device=model.device))
    head_masks = nn.Parameter(torch.ones(num_layers, num_heads, device=model.device))
    neuron_masks = nn.Parameter(torch.ones(num_layers, intermediate_size, device=model.device))
    
    # 4. Inject Masks into the Model
    # A. Layer Masks
    for i in range(num_layers):
        def make_layer_hook(idx):
            def hook(module, args, output):
                # output is typically a tuple (hidden_states, cache, ...)
                hidden_states = output[0]
                masked_hidden = hidden_states * layer_masks[idx]
                return (masked_hidden,) + output[1:]
            return hook
        model.model.layers[i].register_forward_hook(make_layer_hook(i))
        
        # B. Head Masks (Hooking input to o_proj)
        def make_head_hook(idx):
            def hook(module, args):
                x = args[0] # Shape: (batch, seq, hidden_size)
                batch, seq, hidden = x.shape
                head_dim = hidden // num_heads
                x_reshaped = x.view(batch, seq, num_heads, head_dim)
                masked_x = x_reshaped * head_masks[idx].view(1, 1, -1, 1)
                return (masked_x.view(batch, seq, hidden),)
            return hook
        model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(make_head_hook(i))
        
        # C. FFN Neuron Masks (Patching MLP forward)
        mlp = model.model.layers[i].mlp
        # Save original forward
        mlp._orig_forward = mlp.forward
        def make_mlp_forward(mlp_module, idx):
            def new_forward(x):
                # Standard Llama/Gemma MLP: down_proj(act_fn(gate_proj(x)) * up_proj(x))
                gate = mlp_module.gate_proj(x)
                up = mlp_module.up_proj(x)
                act = mlp_module.act_fn(gate) * up
                # act shape: (batch, seq, intermediate_size)
                masked_act = act * neuron_masks[idx].view(1, 1, -1)
                return mlp_module.down_proj(masked_act)
            return new_forward
        mlp.forward = make_mlp_forward(mlp, i)
        
    print("[*] Masks successfully injected into the computational graph.")
    
    # 5. Load Calibration Data
    print("[*] Loading English-Chinese calibration data...")
    # Using a small slice of Opus for calibration
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split=f"train[:{num_calibration_samples}]")
    
    # 6. Compute Fisher Information
    print("[*] Computing Fisher Information...")
    layer_fisher = torch.zeros_like(layer_masks)
    head_fisher = torch.zeros_like(head_masks)
    neuron_fisher = torch.zeros_like(neuron_masks)
    
    model.eval()
    
    batch_size = 1 # Keep small to fit in Kaggle
    optimizer = torch.optim.SGD([layer_masks, head_masks, neuron_masks], lr=0.0) # Dummy optimizer
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_items = dataset[i:i+batch_size]["translation"]
        
        for item in batch_items:
            src = item["en"]
            tgt = item["zh"]
            
            prompt = f"Translate English to Chinese.\nEnglish: {src}\nChinese: {tgt}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass (computes gradients for our masks only)
            model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher (gradient squared)
            with torch.no_grad():
                if layer_masks.grad is not None:
                    layer_fisher += layer_masks.grad ** 2
                if head_masks.grad is not None:
                    head_fisher += head_masks.grad ** 2
                if neuron_masks.grad is not None:
                    neuron_fisher += neuron_masks.grad ** 2
                    
            # Clear gradients
            layer_masks.grad = None
            head_masks.grad = None
            neuron_masks.grad = None
            
    # Normalize Fisher
    layer_fisher /= len(dataset)
    head_fisher /= len(dataset)
    neuron_fisher /= len(dataset)
    
    # 7. Save Scores
    print("[*] Saving Fisher scores...")
    layer_scores = layer_fisher.cpu().numpy().tolist()
    head_scores = head_fisher.cpu().numpy().tolist()
    neuron_scores = neuron_fisher.cpu().numpy().tolist()
    
    with open(os.path.join(output_dir, "layer_fisher.json"), "w") as f:
        json.dump(layer_scores, f)
    with open(os.path.join(output_dir, "head_fisher.json"), "w") as f:
        json.dump(head_scores, f)
    with open(os.path.join(output_dir, "neuron_fisher.json"), "w") as f:
        json.dump(neuron_scores, f)
        
    print(f"[*] Done! Scores saved to {output_dir}")

if __name__ == "__main__":
    compute_fisher()
