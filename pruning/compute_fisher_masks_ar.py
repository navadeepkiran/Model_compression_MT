import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm

print("=== WMT26 Gemma 3 12B Fisher Mask Computation (Egyptian Arabic) ===")

# ==========================================
MODE = "layers"  # Step 1: Compute layer scores first
# ==========================================

model_id = "google/gemma-3-12b-it"
num_calibration_samples = 500 
max_length = 256
output_dir = "/kaggle/working/outputs/fisher_scores_ar"
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

target_layers = None
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and len(module) == num_layers:
        target_layers = module
        break

print(f"[*] Initializing ONLY {MODE} masks to save VRAM...")
masks = nn.Parameter(torch.ones(num_layers, device=model.device))

# Inject ONLY the requested Mask
for i in range(num_layers):
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
        
print(f"[*] Loading FLORES-200 Egyptian Arabic calibration data...")
# Load Egyptian Arabic (arz_Arab) and English (eng_Latn) from FLORES
dataset = load_dataset("facebook/flores", "eng_Latn-arz_Arab", split=f"dev[:{num_calibration_samples}]")

fisher_scores = torch.zeros_like(masks)

model.gradient_checkpointing_enable()
model.train() 

optimizer = torch.optim.SGD([masks], lr=0.0) 

print(f"[*] Computing Fisher Information for {MODE}...")
for i in tqdm(range(0, len(dataset))):
    item = dataset[i]
    prompt = f"Translate English to Egyptian Arabic.\nEnglish: {item['sentence_eng_Latn']}\nArabic: {item['sentence_arz_Arab']}"
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
    
print(f"[*] Done! Scores for {MODE} saved successfully. Check {output_dir}/layer_fisher.json")
