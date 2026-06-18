import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm

print("=== WMT26 Gemma 3 12B Fisher Mask Computation ===")

# ==========================================
# CHANGE THIS TO COMPUTE DIFFERENT SCORES
MODE = "layers"  # Options: "layers", "heads", "neurons"
# ==========================================

model_id = "google/gemma-3-12b-it"
num_calibration_samples = 500 
max_length = 256
output_dir = "/kaggle/working/outputs/fisher_scores"
os.makedirs(output_dir, exist_ok=True)

print(f"[*] Loading {model_id} in INT4... (Cached)")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token=hf_token)

model.requires_grad_(False)

text_config = model.config.text_config if hasattr(model.config, "text_config") else model.config
num_layers = text_config.num_hidden_layers
num_heads = text_config.num_attention_heads
intermediate_size = text_config.intermediate_size

target_layers = None
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and len(module) == num_layers:
        target_layers = module
        break

print(f"[*] Initializing ONLY {MODE} masks to save VRAM...")

if MODE == "layers":
    masks = nn.Parameter(torch.ones(num_layers, device=model.device))
elif MODE == "heads":
    masks = nn.Parameter(torch.ones(num_layers, num_heads, device=model.device))
elif MODE == "neurons":
    masks = nn.Parameter(torch.ones(num_layers, intermediate_size, device=model.device))

# Inject ONLY the requested Mask
for i in range(num_layers):
    if MODE == "layers":
        def make_layer_hook(idx):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    mask = masks[idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
                    return (hidden_states * mask,) + output[1:]
                else:
                    hidden_states = output
                    mask = masks[idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
                    return hidden_states * mask
            return hook
        target_layers[i].register_forward_hook(make_layer_hook(i))
        
    elif MODE == "heads":
        def make_head_hook(idx):
            def hook(module, args):
                x = args[0]
                batch, seq, hidden = x.shape
                head_dim = hidden // num_heads
                mask = masks[idx].to(device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
                masked_x = x.view(batch, seq, num_heads, head_dim) * mask
                return (masked_x.view(batch, seq, hidden),)
            return hook
        target_layers[i].self_attn.o_proj.register_forward_pre_hook(make_head_hook(i))
        
    elif MODE == "neurons":
        mlp = target_layers[i].mlp
        mlp._orig_forward = mlp.forward
        def make_mlp_forward(mlp_module, idx):
            def new_forward(x):
                gate = mlp_module.gate_proj(x)
                up = mlp_module.up_proj(x)
                act = mlp_module.act_fn(gate) * up
                mask = masks[idx].to(device=act.device, dtype=act.dtype).view(1, 1, -1)
                return mlp_module.down_proj(act * mask)
            return new_forward
        mlp.forward = make_mlp_forward(mlp, i)
    
print(f"[*] Loading calibration data...")
dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split=f"train[:{num_calibration_samples}]")

fisher_scores = torch.zeros_like(masks)

model.gradient_checkpointing_enable()
model.train() 

optimizer = torch.optim.SGD([masks], lr=0.0) 

print(f"[*] Computing Fisher Information for {MODE}...")
for i in tqdm(range(0, len(dataset))):
    item = dataset[i]["translation"]
    prompt = f"Translate English to Chinese.\nEnglish: {item['en']}\nChinese: {item['zh']}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    
    # Gemma 3 is a multimodal model and requires token_type_ids during training
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        if masks.grad is not None: 
            fisher_scores += masks.grad ** 2
            
    masks.grad = None
    torch.cuda.empty_cache()
    
fisher_scores /= len(dataset)

with open(os.path.join(output_dir, f"{MODE[:-1]}_fisher.json"), "w") as f:
    json.dump(fisher_scores.cpu().numpy().tolist(), f)
    
print(f"[*] Done! Scores for {MODE} saved successfully.")
