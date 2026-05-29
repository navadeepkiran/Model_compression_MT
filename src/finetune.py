import os

# Set BOTH env var names to cover all PyTorch versions (name changed in 2.x)
# Must be before 'import torch' so the CUDA caching allocator reads it at init time.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # PyTorch < 2.x
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"       # PyTorch >= 2.x

# Force cuBLAS to use a deterministic/alternative workspace to bypass T4 Float16 NOT_SUPPORTED bugs
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F

# Disable reduced precision reduction which triggers cuBLAS unsupported kernels on T4
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
# Disable TF32 since T4 doesn't support it anyway, just to be safe
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Redirect HF Cache to /kaggle/tmp or /tmp on Linux environments (Kaggle/Colab) to prevent home directory disk full errors
if os.name != "nt":
    if os.path.exists("/kaggle"):
        os.environ["HF_HOME"] = "/kaggle/tmp/huggingface_cache"
        os.environ["HF_DATASETS_CACHE"] = "/kaggle/tmp/huggingface_cache/datasets"
        os.environ["HF_HUB_CACHE"] = "/kaggle/tmp/huggingface_cache/hub"
    else:
        os.environ["HF_HOME"] = "/tmp/huggingface_cache"
        os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface_cache/datasets"
        os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache/hub"

# Automatically load Kaggle Secrets for HuggingFace token if available
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass

import gc
import json
import torch
import argparse
import random
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# --- FLORES-200 VALIDATION DATA LOADER ---
def load_flores_validation(base_dir="flores200_dataset", num_samples=100):
    val_data = []
    possible_paths = [
        base_dir,
        os.path.join("..", base_dir),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", base_dir),
        os.path.join("c:/Users/navad/Documents/WMT", base_dir)
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "dev", "eng_Latn.dev")):
            actual_path = p
            break
            
    if actual_path is None:
        print("[!] Warning: flores200_dataset directory not found. Validation callback will be skipped.")
        return None
        
    print(f"[*] Found validation dataset at: {actual_path}")
    
    # Load Czech-German
    cs_path = os.path.join(actual_path, "dev", "ces_Latn.dev")
    de_path = os.path.join(actual_path, "dev", "deu_Latn.dev")
    if os.path.exists(cs_path) and os.path.exists(de_path):
        with open(cs_path, "r", encoding="utf-8") as f_src, open(de_path, "r", encoding="utf-8") as f_ref:
            src_lines = [line.strip() for line in f_src][:num_samples]
            ref_lines = [line.strip() for line in f_ref][:num_samples]
            for src, ref in zip(src_lines, ref_lines):
                val_data.append({
                    "src": src,
                    "ref": ref,
                    "src_lang": "Czech",
                    "tgt_lang": "German"
                })
                
    # Load English-Chinese
    en_path = os.path.join(actual_path, "dev", "eng_Latn.dev")
    zh_path = os.path.join(actual_path, "dev", "zho_Hans.dev")
    if os.path.exists(en_path) and os.path.exists(zh_path):
        with open(en_path, "r", encoding="utf-8") as f_src, open(zh_path, "r", encoding="utf-8") as f_ref:
            src_lines = [line.strip() for line in f_src][:num_samples]
            ref_lines = [line.strip() for line in f_ref][:num_samples]
            for src, ref in zip(src_lines, ref_lines):
                val_data.append({
                    "src": src,
                    "ref": ref,
                    "src_lang": "English",
                    "tgt_lang": "Chinese (Simplified)"
                })
                
    # Load English-Arabic
    ar_path = os.path.join(actual_path, "dev", "arz_Arab.dev")
    if os.path.exists(en_path) and os.path.exists(ar_path):
        with open(en_path, "r", encoding="utf-8") as f_src, open(ar_path, "r", encoding="utf-8") as f_ref:
            src_lines = [line.strip() for line in f_src][:num_samples]
            ref_lines = [line.strip() for line in f_ref][:num_samples]
            for src, ref in zip(src_lines, ref_lines):
                val_data.append({
                    "src": src,
                    "ref": ref,
                    "src_lang": "English",
                    "tgt_lang": "Arabic"
                })
                
    return val_data

# --- CUSTOM EVALUATION CALLBACK FOR CHECKPOINTING ---
class CometEvaluationCallback(TrainerCallback):
    def __init__(self, val_dataset, tokenizer, output_dir, comet_model_name="Unbabel/wmt22-comet-da"):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_comet_score = -9999.0
        self.best_model_dir = os.path.join(output_dir, "best_model")
        self.comet_model = None
        
        if val_dataset:
            try:
                from comet import download_model, load_from_checkpoint
                print("[*] Pre-loading COMET model for callback...")
                model_path = download_model(comet_model_name)
                self.comet_model = load_from_checkpoint(model_path)
                # Note: With 4-bit QLoRA, we have enough VRAM to keep COMET on GPU
            except Exception as e:
                print(f"[!] Error loading COMET model: {e}. Comet scoring will be disabled.")
                
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.val_dataset is None or self.comet_model is None:
            return
            
        print(f"\n[*] Epoch {state.epoch:.1f} ended. Starting COMET evaluation...")
        
        # Set to eval mode and free GPU cache so generation has max headroom
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        
        predictions = []
        data_to_grade = []
        
        for item in tqdm(self.val_dataset, desc="Generating validation translations"):
            src_text = item["src"]
            ref_text = item["ref"]
            src_lang = item["src_lang"]
            tgt_lang = item["tgt_lang"]
            
            # Format using prompt template
            messages = [
                {"role": "user", "content": f"Translate the following text from {src_lang} to {tgt_lang}. Output ONLY the raw translation, without any introductory text, explanation, markdown formatting, or surrounding conversation. The output must contain only the translated text.\n\nText to translate:\n{src_text}"}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
            num_gen = outputs.shape[1] - inputs.input_ids.shape[1]
            pred = self.tokenizer.decode(outputs[0][-num_gen:], skip_special_tokens=True).strip().replace('\n', ' ')
            
            predictions.append(pred)
            data_to_grade.append({
                "src": src_text,
                "mt": pred,
                "ref": ref_text
            })
            
        # Run COMET evaluation on GPU
        try:
            import pytorch_lightning as pl
            if int(pl.__version__.split(".")[0]) >= 2:
                comet_results = self.comet_model.predict(data_to_grade, batch_size=8, devices=1, accelerator="gpu")
            else:
                comet_results = self.comet_model.predict(data_to_grade, batch_size=8, gpus=1)
            
            current_score = comet_results.system_score
            
            print(f"\n========================================")
            print(f"Epoch {state.epoch:.1f} COMET Score: {current_score:.4f} (Previous Best: {self.best_comet_score:.4f})")
            print(f"========================================")
            
            if current_score > self.best_comet_score:
                print(f"[+] New best model found! Saving weights to {self.best_model_dir}...")
                self.best_comet_score = current_score
                model.save_pretrained(self.best_model_dir)
                self.tokenizer.save_pretrained(self.best_model_dir)
                
                # Save the score metadata
                with open(os.path.join(self.best_model_dir, "best_comet_score.json"), "w") as f:
                    json.dump({"epoch": state.epoch, "comet_score": current_score}, f)
        except Exception as e:
            print(f"[!] Error during COMET evaluation: {e}")
            
        # Set back to train mode
        model.train()

def prepare_model_for_kbit_training_custom(model):
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Gemma 3 WMT Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it", help="Model HF ID")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-12b-wmt-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--subset_size", type=int, default=50000, help="Dataset size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length (lower = less VRAM; 256 perfectly fits 8-bit on 15GB T4)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"STARTING FINE-TUNING PIPELINE")
    print(f"Model ID: {args.model_id}")
    print(f"Target Size: {args.subset_size} sentences")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Helper to extract text pair dynamically from various translation dataset schemas (WMT, OPUS, Tatoeba, etc.)
    def extract_text_pair(item, src_code, tgt_code):
        if "translation" in item:
            trans = item["translation"]
            src_key, tgt_key = None, None
            for k in trans.keys():
                if k.lower().startswith(src_code[:2].lower()) or src_code.lower().startswith(k[:2].lower()):
                    src_key = k
                if k.lower().startswith(tgt_code[:2].lower()) or tgt_code.lower().startswith(k[:2].lower()):
                    tgt_key = k
            if src_key and tgt_key:
                return trans[src_key], trans[tgt_key]
                
        src_key, tgt_key = None, None
        for k in item.keys():
            if k.lower().startswith(src_code[:2].lower()) or src_code.lower().startswith(k[:2].lower()):
                src_key = k
            if k.lower().startswith(tgt_code[:2].lower()) or tgt_code.lower().startswith(k[:2].lower()):
                tgt_key = k
        if src_key and tgt_key:
            return item[src_key], item[tgt_key]
            
        return None
        
    # 1. Load Datasets
    print("[*] Loading Czech-German dataset...")
    ds_cs_de = None
    # Czech-German is WMT News-commentary / Europarl
    for dataset_name, config in [("Helsinki-NLP/europarl", "cs-de"), ("Helsinki-NLP/news_commentary", "cs-de"), ("Helsinki-NLP/tatoeba_mt", "ces-deu")]:
        try:
            print(f" - Attempting: {dataset_name} ({config})...")
            ds_cs_de = load_dataset(dataset_name, config, split="train[:333000]")
            print(f"   ✅ Successfully loaded {dataset_name}")
            break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    if ds_cs_de is None:
        raise RuntimeError("[!] Fatal: Could not load Czech-German dataset from any fallback source.")
        
    print("[*] Loading English-Chinese dataset...")
    ds_zh_en = None
    for dataset_name, config in [("wmt19", "zh-en"), ("Helsinki-NLP/opus-100", "en-zh"), ("Helsinki-NLP/tatoeba_mt", "eng-zho")]:
        try:
            print(f" - Attempting: {dataset_name} ({config})...")
            ds_zh_en = load_dataset(dataset_name, config, split="train[:333000]")
            print(f"   ✅ Successfully loaded {dataset_name}")
            break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    if ds_zh_en is None:
        raise RuntimeError("[!] Fatal: Could not load English-Chinese dataset from any fallback source.")
        
    print("[*] Loading English-Arabic dataset...")
    ds_ar_en = None
    for dataset_name, config in [("Helsinki-NLP/opus-100", "ar-en"), ("Helsinki-NLP/opus-100", "en-ar"), ("Helsinki-NLP/tatoeba_mt", "ara-eng")]:
        try:
            print(f" - Attempting: {dataset_name} ({config})...")
            ds_ar_en = load_dataset(dataset_name, config, split="train[:333000]")
            print(f"   ✅ Successfully loaded {dataset_name}")
            break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    if ds_ar_en is None:
        raise RuntimeError("[!] Fatal: Could not load English-Arabic dataset from any fallback source.")
        
    # 2. Extract and Process Pairs
    combined_data = []
    
    print("[*] Extracting translation pairs...")
    
    # Extract cs-de
    print(f" - Processing Czech-German ({len(ds_cs_de)} pairs)...")
    for item in ds_cs_de:
        pair = extract_text_pair(item, "cs", "de")
        if pair:
            combined_data.append({
                "src": pair[0],
                "tgt": pair[1],
                "src_lang": "Czech",
                "tgt_lang": "German"
            })
        
    # Extract zh-en
    print(f" - Processing English-Chinese ({len(ds_zh_en)} pairs)...")
    for item in ds_zh_en:
        pair = extract_text_pair(item, "en", "zh")
        if pair:
            combined_data.append({
                "src": pair[0],
                "tgt": pair[1],
                "src_lang": "English",
                "tgt_lang": "Chinese (Simplified)"
            })
        
    # Extract ar-en
    print(f" - Processing English-Arabic ({len(ds_ar_en)} pairs)...")
    for item in ds_ar_en:
        pair = extract_text_pair(item, "en", "ar")
        if pair:
            combined_data.append({
                "src": pair[0],
                "tgt": pair[1],
                "src_lang": "English",
                "tgt_lang": "Arabic"
            })
        
    # 3. Filter by character length and sort
    print("[*] Filtering sentences based on length (20 < characters < 1500)...")
    filtered_data = [
        item for item in combined_data
        if item["src"] and item["tgt"] and 20 < len(item["src"]) < 1500
    ]
    print(f"[*] Total pairs after filtering: {len(filtered_data)}")
    
    print("[*] Sorting dataset by character length (descending)...")
    filtered_data.sort(key=lambda x: len(x["src"]), reverse=True)
    
    subset_size = min(args.subset_size, len(filtered_data))
    print(f"[*] Selecting top {subset_size} longest sentences...")
    final_subset = filtered_data[:subset_size]
    
    # Interleave languages by shuffling
    print("[*] Shuffling dataset to interleave languages...")
    random.seed(42)
    random.shuffle(final_subset)
    
    # 4. Load Tokenizer & Model
    print("[*] Loading Tokenizer...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, token=hf_token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Convert final subset list to HF Dataset
    train_dataset = Dataset.from_list(final_subset)
    
    # Helper to apply template
    # Gemma-3-IT has a ~650-token BUILT-IN system prompt injected automatically by apply_chat_template.
    # Without overriding it, max_length=256 only captures the system prompt — no translation content.
    # Fix: pass a short custom system message to override the default.
    # Budget: ~10 (system) + ~20 (roles/specials) + ~20 (instruction) + 80 (src) + 70 (tgt) = ~200 ✓
    SRC_MAX_TOKENS = 80
    TGT_MAX_TOKENS = 70

    def format_prompts(example):
        # Truncate at the token level, then decode back to text
        src_ids = tokenizer.encode(example['src'], add_special_tokens=False)[:SRC_MAX_TOKENS]
        tgt_ids = tokenizer.encode(example['tgt'], add_special_tokens=False)[:TGT_MAX_TOKENS]
        src = tokenizer.decode(src_ids, skip_special_tokens=True)
        tgt = tokenizer.decode(tgt_ids, skip_special_tokens=True)

        messages = [
            # Short system message overrides Gemma-3-IT's ~650-token built-in system prompt
            {"role": "system", "content": "You are a machine translation assistant. Output only the translation."},
            {"role": "user",   "content": f"Translate from {example['src_lang']} to {example['tgt_lang']}:\n{src}"},
            {"role": "model",  "content": tgt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
        
    print("[*] Formatting prompts using tokenizer chat template...")
    train_dataset = train_dataset.map(format_prompts, remove_columns=["src", "tgt", "src_lang", "tgt_lang"])

    # ── GUARANTEED TRUNCATION ──────────────────────────────────────────────────
    # Pre-tokenize the entire dataset HERE with truncation=True and max_length capped.
    # This runs BEFORE SFTTrainer sees the data, so it cannot be bypassed by any
    # SFTTrainer version, chat template system-prompt, or collator quirk.
    # The Gemma-3-IT built-in system prompt alone is ~650 tokens; without this step
    # the sequences reaching the model are 888+ tokens → OOM on MLP gate_proj.
    print(f"[*] Pre-tokenizing with hard truncation to {args.max_seq_length} tokens...")
    def pre_tokenize(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()  # causal LM: labels = input_ids
        return enc

    train_dataset = train_dataset.map(
        pre_tokenize,
        remove_columns=["text"],
        batched=False,
        desc="Tokenizing + truncating"
    )
    # ──────────────────────────────────────────────────────────────────────────

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()
    # Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set at module top (before import torch)
    # so the CUDA caching allocator can merge fragmented blocks to satisfy large contiguous requests.
    
    # Config for 4-bit NF4 QLoRA - using pure float32 compute
    # Gemma 3's massive MLPs natively overflow float16 (65504 max value) in the forward pass,
    # instantly causing NaN loss. Since Kaggle T4s cannot handle bfloat16 properly, we must 
    # use pure float32 for compute to mathematically guarantee no overflows.
    print("[*] Configuring 4-bit NF4 quantization settings (Strict Float32 for T4 compatibility)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            
        bnb_4bit_compute_dtype=torch.float32, 
        bnb_4bit_use_double_quant=False,       
    )
    
    # Load model
    print("[*] Loading Gemma-3-12B in 4-bit precision (Float32 base)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Use 'auto' so accelerate distributes it
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",  # Eager attention: no SDPA peak-memory spikes
        token=hf_token
    )
    model.config.torch_dtype = torch.float32
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    
    # Prepare model
    model = prepare_model_for_kbit_training_custom(model)
    
    # Configure LoRA settings targeting all linear blocks
    print("[*] Configuring LoRA settings targeting all linear blocks...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Force trainable parameters (LoRA) and norms to float32 natively
    print("[*] Ensuring LoRA layers and layernorms are natively float32...")
    for name, module in model.named_modules():
        if "lora_" in name.lower():
            module.to(torch.float32)
        elif any(x in name.lower() for x in ["layernorm", "layer_norm", "norm"]):
            module.to(torch.float32)
    # Load FLORES validation set
    print("[*] Loading FLORES-200 validation subsets...")
    val_dataset = load_flores_validation(num_samples=100)
    
    # Setup custom evaluation callback
    comet_callback = CometEvaluationCallback(
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir
    )
    
    # Setup Trainer configs
    print("[*] Configuring Trainer...")
    
    # Core TrainingArguments params
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": False,
        "optim": "paged_adamw_8bit",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 20,
        "learning_rate": args.learning_rate,
        "fp16": False,   # Disable AMP GradScaler to completely bypass the BFloat16 crash!
        "bf16": False,   # Explicitly disable bf16 to prevent BFloat16 crashes on T4
        "group_by_length": True,
        "lr_scheduler_type": "cosine",
        "push_to_hub": False,
        "report_to": "none",
        "num_train_epochs": args.epochs,
        "remove_unused_columns": False,  # Keep all columns including token_type_ids
        "torch_compile": False,          # Explicitly disable compilation as per advice
    }

    # ── SIMPLE COLLATOR ───────────────────────────────────────────────────────
    # Dataset is already tokenized + truncated. This collator only pads to the
    # longest sequence in the batch and injects token_type_ids for Gemma-3.
    # Sequences are guaranteed ≤ max_seq_length; no complex truncation needed.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def gemma3_collator(features):
        input_ids      = [torch.tensor(f["input_ids"],      dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels         = [torch.tensor(f.get("labels", f["input_ids"]), dtype=torch.long) for f in features]

        # Pad strictly to max_seq_length to guarantee perfectly aligned matrix sizes 
        # (e.g. 256 or 512) for cuBLAS Tensor Cores, avoiding any NOT_SUPPORTED errors.
        target_len = args.max_seq_length

        def pad_tensor(t, pad_val):
            pad_size = target_len - len(t)
            return torch.nn.functional.pad(t, (0, pad_size), value=pad_val) if pad_size > 0 else t

        input_ids_padded      = torch.stack([pad_tensor(x, pad_id) for x in input_ids])
        attention_mask_padded = torch.stack([pad_tensor(x, 0) for x in attention_mask])
        labels_padded         = torch.stack([pad_tensor(x, -100) for x in labels])

        return {
            "input_ids":      input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels":         labels_padded,
            "token_type_ids": torch.zeros_like(input_ids_padded),
        }
    # ──────────────────────────────────────────────────────────────────────────

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "callbacks": [comet_callback],
        "data_collator": gemma3_collator,
    }

    # Dataset is pre-tokenized → do NOT pass dataset_text_field or packing.
    # Route tokenizer / processing_class (needed for generation in callbacks)
    import inspect
    sft_trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    print(f"[*] Instantiating SFTConfig with arguments: {list(config_kwargs.keys())}")
    sft_config = SFTConfig(**config_kwargs)

    trainer_kwargs["args"] = sft_config
    print(f"[*] Instantiating SFTTrainer with arguments: {list(trainer_kwargs.keys())}")
    trainer = SFTTrainer(**trainer_kwargs)

    
    # Check for resume checkpoints
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(args.output_dir)
        
    print("\n" + "=" * 40)
    if last_checkpoint is not None:
        print(f"[🚀] Checkpoint found! Resuming training from: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("[🌟] No previous checkpoint found. Starting fresh training...")
        trainer.train()
        
    # Save final model
    trainer.model.save_pretrained(args.output_dir)
    print(f"\n[🎉] Training complete! Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
