import os

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
            except Exception as e:
                print(f"[!] Error loading COMET model: {e}. Comet scoring will be disabled.")
                
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.val_dataset is None or self.comet_model is None:
            return
            
        print(f"\n[*] Epoch {state.epoch:.1f} ended. Starting COMET evaluation...")
        
        # Set to eval mode
        model.eval()
        
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
            
        # Run COMET evaluation
        try:
            gpus = 1 if torch.cuda.is_available() else 0
            comet_results = self.comet_model.predict(data_to_grade, batch_size=8, gpus=gpus)
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

def prepare_model_for_kbit_training_custom(model, use_gradient_checkpointing=True):
    """
    Custom wrapper to prepare the model for QLoRA training without upcasting the massive 
    input/output embeddings (embed_tokens and lm_head) to float32. This saves ~4GB+ of VRAM 
    for Gemma-3 (which has a 256k vocabulary) and avoids CUDA OOM on 16GB GPUs.
    """
    for param in model.parameters():
        param.requires_grad = False
        
    # Cast layernorms/norms to float32 for training stability
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ["layernorm", "layer_norm", "norm"]) and not any(x in name.lower() for x in ["embed_tokens", "lm_head"]):
            module.to(torch.float32)
            
    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    return model

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Gemma 3 WMT Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it", help="Model HF ID")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-12b-wmt-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--subset_size", type=int, default=50000, help="Dataset size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
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
    def format_prompts(example):
        messages = [
            {"role": "user", "content": f"Translate the following text from {example['src_lang']} to {example['tgt_lang']}. Output ONLY the raw translation, without any introductory text, explanation, markdown formatting, or surrounding conversation. The output must contain only the translated text.\n\nText to translate:\n{example['src']}"},
            {"role": "model", "content": example['tgt']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
        
    print("[*] Formatting prompts using tokenizer chat template...")
    train_dataset = train_dataset.map(format_prompts, remove_columns=["src", "tgt", "src_lang", "tgt_lang"])
    
    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()
    
    # Config for INT8 precision
    print("[*] Configuring INT8 quantization settings...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    # Load model
    print("[*] Loading Gemma-3-12B in INT8 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},  # Force single GPU execution to prevent deadlocks and memory dispersion
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Force fp16 computation path for T4 compatibility
        token=hf_token
    )
    model.config.torch_dtype = torch.float16
    
    # Prepare model
    model = prepare_model_for_kbit_training_custom(model)
    
    # Configure LoRA
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
    
    # Force trainable params to float32 to prevent bfloat16 casting errors during gradient updates
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            
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
    import inspect
    
    sft_config_sig = inspect.signature(SFTConfig.__init__).parameters
    sft_trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    
    # Core TrainingArguments params
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 20,
        "learning_rate": args.learning_rate,
        "fp16": False,  # Bypass GradScaler BFloat16 issues on T4
        "group_by_length": True,
        "lr_scheduler_type": "cosine",
        "push_to_hub": False,
        "report_to": "none",
        "num_train_epochs": args.epochs,
        "remove_unused_columns": False  # Crucial: Prevent Trainer from stripping token_type_ids
    }
    
    # Setup custom data collator to inject token_type_ids (required by Gemma-3 during training)
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    
    def gemma3_collator(features):
        # Filter features to keep only tokenized numeric keys and drop any string metadata columns (like 'text')
        allowed_keys = {"input_ids", "attention_mask", "labels"}
        cleaned_features = []
        for f in features:
            cleaned_f = {k: v for k, v in f.items() if k in allowed_keys}
            cleaned_features.append(cleaned_f)
            
        batch = base_collator(cleaned_features)
        if "input_ids" in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch
        
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "callbacks": [comet_callback],
        "data_collator": gemma3_collator
    }
    
    # Route sequence length parameter (max_seq_length vs max_length)
    if "max_seq_length" in sft_config_sig:
        config_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_config_sig:
        config_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_trainer_sig:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_trainer_sig:
        trainer_kwargs["max_length"] = args.max_seq_length
        
    # Route dataset_text_field parameter
    if "dataset_text_field" in sft_config_sig:
        config_kwargs["dataset_text_field"] = "text"
    elif "dataset_text_field" in sft_trainer_sig:
        trainer_kwargs["dataset_text_field"] = "text"
        
    # Route packing parameter
    if "packing" in sft_config_sig:
        config_kwargs["packing"] = False
    elif "packing" in sft_trainer_sig:
        trainer_kwargs["packing"] = False
        
    # Route tokenizer / processing_class
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
