import os
import glob

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

# ── BUGFIX FOR PYTORCH 2.6 ──────────────────────────────────────────────────
try:
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass
# ────────────────────────────────────────────────────────────────────────────

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

def prepare_model_for_kbit_training_custom(model, use_gradient_checkpointing=True):
    for param in model.parameters():
        param.requires_grad = False
    
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
        model.gradient_checkpointing_enable({"use_reentrant": True})
        
    return model

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Gemma 3 WMT Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it", help="Model HF ID")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-12b-wmt-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--subset_size", type=int, default=4000, help="Dataset size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"STARTING FINE-TUNING PIPELINE (POLISHED DATASET)")
    print(f"Model ID: {args.model_id}")
    print(f"Target Size: {args.subset_size} sentences")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # 1. Load Polished Dataset dynamically
    print("[*] Searching for polished Parquet dataset...")
    parquet_files = glob.glob('/kaggle/input/**/*.parquet', recursive=True)
    if not parquet_files:
        # Fallback to local data folder if running locally
        parquet_files = glob.glob('data/*.parquet')
        
    if not parquet_files:
        raise RuntimeError("[!] Fatal: Could not find any .parquet dataset file in /kaggle/input/ or local data/ folder.")
        
    target_parquet = parquet_files[0]
    for p in parquet_files:
        if "wmt" in p.lower() or "stage" in p.lower() or "36k" in p.lower():
            target_parquet = p
            break
            
    print(f"[*] Loading dataset from {target_parquet}...")
    ds_zh_en = load_dataset('parquet', data_files=target_parquet, split='train')
    
    # 2. Extract and Process Pairs
    combined_data = []
    print(f"[*] Processing {len(ds_zh_en)} pairs...")
    for item in ds_zh_en:
        if "source" in item and "target" in item:
            combined_data.append({
                "src": item["source"],
                "tgt": item["target"],
                "src_lang": "English",
                "tgt_lang": "Chinese (Simplified)"
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
    print("[*] Shuffling dataset...")
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
    
    SRC_MAX_TOKENS = 80
    TGT_MAX_TOKENS = 70

    def format_prompts(example):
        src_ids = tokenizer.encode(example['src'], add_special_tokens=False)[:SRC_MAX_TOKENS]
        tgt_ids = tokenizer.encode(example['tgt'], add_special_tokens=False)[:TGT_MAX_TOKENS]
        src = tokenizer.decode(src_ids, skip_special_tokens=True)
        tgt = tokenizer.decode(tgt_ids, skip_special_tokens=True)

        messages = [
            {"role": "system", "content": "You are a machine translation assistant. Output only the translation."},
            {"role": "user",   "content": f"Translate from {example['src_lang']} to {example['tgt_lang']}:\n{src}"},
            {"role": "model",  "content": tgt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
        
    print("[*] Formatting prompts using tokenizer chat template...")
    train_dataset = train_dataset.map(format_prompts, remove_columns=["src", "tgt", "src_lang", "tgt_lang"])

    print(f"[*] Pre-tokenizing with hard truncation to {args.max_seq_length} tokens...")
    def pre_tokenize(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        
        labels = enc["input_ids"].copy()
        
        model_turn_str = "<start_of_turn>model\n"
        idx = example["text"].find(model_turn_str)
        if idx != -1:
            prompt_enc = tokenizer(example["text"][:idx + len(model_turn_str)])
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len
            
        enc["labels"] = labels
        return enc

    train_dataset = train_dataset.map(
        pre_tokenize,
        remove_columns=["text"],
        batched=False,
        desc="Tokenizing + truncating"
    )

    gc.collect()
    torch.cuda.empty_cache()
    
    print("[*] Configuring 4-bit NF4 quantization settings (Pure BFloat16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True  
    )
    
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        use_unsloth = False

    if use_unsloth:
        print("[*] Loading model using UNSLOTH (Highly Optimized 4-bit)...")
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=256,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            token=hf_token,
        )
        print("[*] Configuring LoRA settings via Unsloth...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        model.config.use_cache = False
    else:
        print("[*] Loading model in 4-bit precision (BFloat16 base)...")
        custom_device_map = {
            "model.embed_tokens": "cpu",
            "lm_head": "cpu",
            "": "cuda:0"
        }
    
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map=custom_device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            token=hf_token
        )
        
        model.is_model_parallel = True
        model.is_loaded_in_4bit = True
        model.config.torch_dtype = torch.bfloat16
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        
        model = prepare_model_for_kbit_training_custom(model)
        
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
    
    print("[*] Ensuring LoRA layers and layernorms are natively float32...")
    for name, module in model.named_modules():
        if "lora_" in name.lower():
            module.to(torch.float32)
        elif any(x in name.lower() for x in ["layernorm", "layer_norm", "norm"]):
            module.to(torch.float32)
    
    print("[*] Configuring Trainer...")
    
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": True},
        "optim": "paged_adamw_8bit",
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "logging_steps": 20,
        "learning_rate": args.learning_rate,
        "fp16": False,
        "bf16": False,
        "lr_scheduler_type": "cosine",
        "push_to_hub": False,
        "report_to": "none",
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "remove_unused_columns": False,
        "torch_compile": False,
    }

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def gemma3_collator(features):
        input_ids      = [torch.tensor(f["input_ids"],      dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels         = [torch.tensor(f.get("labels", f["input_ids"]), dtype=torch.long) for f in features]

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

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "data_collator": gemma3_collator,
    }

    import inspect
    sft_trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    print(f"[*] Instantiating SFTConfig with arguments: {list(config_kwargs.keys())}")
    sft_config = SFTConfig(**config_kwargs)

    class BugfixTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            if num_items_in_batch is not None and hasattr(self, "_compute_loss_scaling_factor"):
                loss = self._compute_loss_scaling_factor(loss, num_items_in_batch)
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs["args"] = sft_config
    print(f"[*] Instantiating BugfixTrainer with arguments: {list(trainer_kwargs.keys())}")
    trainer = BugfixTrainer(**trainer_kwargs)
    
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
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[🎉] Training complete! Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
