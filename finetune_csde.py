import os

# Set BOTH env var names to cover all PyTorch versions (name changed in 2.x)
# Must be before 'import torch' so the CUDA caching allocator reads it at init time.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # PyTorch < 2.x
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"       # PyTorch >= 2.x

# FORCE GPU 0 ONLY to completely disable DataParallel and prevent cross-device FX dynamo errors!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Force cuBLAS to use a deterministic/alternative workspace to bypass T4 Float16 NOT_SUPPORTED bugs
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F

# Disable reduced precision reduction which triggers cuBLAS unsupported kernels on T4
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
# Disable TF32 since T4 doesn't support it anyway, just to be safe
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Redirect HF Cache to /kaggle/working on Kaggle to use the 73GB disk instead of the tiny /kaggle/tmp RAM disk!
if os.name != "nt":
    if os.path.exists("/kaggle"):
        os.environ["HF_HOME"] = "/kaggle/working/huggingface_cache"
        os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/huggingface_cache/datasets"
        os.environ["HF_HUB_CACHE"] = "/kaggle/working/huggingface_cache/hub"
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
import argparse
import random
import pandas as pd
from tqdm import tqdm

# ─── BUGFIX FOR PYTORCH 2.6 ──────────────────────────────────────────────────
# PyTorch 2.6 sets weights_only=True by default for torch.load().
# When transformers Trainer tries to load rng_state.pth, it crashes because 
# the numpy RNG state contains a numpy _reconstruct object which is not allowed.
# We completely monkey-patch torch.load to forcefully disable this security check.
try:
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

from datasets import Dataset
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
        if os.path.exists(os.path.join(p, "dev", "ces_Latn.dev")):
            actual_path = p
            break
            
    if actual_path is None:
        print("[!] Warning: flores200_dataset directory not found. Validation callback will be skipped.")
        return None
        
    print(f"[*] Found validation dataset at: {actual_path}")
    # Load Czech-German exclusively
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
                
    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.val_dataset is None or self.comet_model is None:
            return
            
        print(f"\n[*] Training ended. Starting final COMET evaluation...")
        
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
                comet_results = self.comet_model.predict(data_to_grade, batch_size=8, devices=[0], accelerator="gpu")
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



# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Gemma 3 WMT Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, default="nani-nav/gemma-3-12b-final-csde", help="Model HF ID")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-12b-40L-csde-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--subset_size", type=int, default=25000, help="Dataset size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    args = parser.parse_args()
    
    # Automatically save to Google Drive if it's mounted, preventing Colab from deleting checkpoints
    if os.path.exists("/content/drive/MyDrive"):
        args.output_dir = "/content/drive/MyDrive/Gemma3_WMT_Outputs"
        print(f"[*] Google Drive detected! Saving checkpoints securely to {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"STARTING FINE-TUNING PIPELINE (CS-DE)")
    print(f"Model ID: {args.model_id}")
    print(f"Target Size: {args.subset_size} sentences")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
        
    # 1. Load Dataset
    print("[*] Loading Czech-German dataset...")
    df = pd.read_parquet("data/wmt26_ce_de_stage6_filtered_0.75.parquet")
    print(f"[*] Loaded dataset with {len(df)} rows.")
    
    # 2. Extract and Process Pairs
    combined_data = []
    
    print("[*] Extracting translation pairs...")
    for idx, row in df.iterrows():
        combined_data.append({
            "src": str(row["cs"]).strip(),
            "tgt": str(row["de"]).strip(),
            "src_lang": "Czech",
            "tgt_lang": "German"
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
    
    # Helper to apply template
    # Gemma-3-IT has a ~650-token BUILT-IN system prompt injected automatically by apply_chat_template.
    # Without overriding it, max_length=256 only captures the system prompt — no translation content.
    # Fix: pass a short custom system message to override the default.
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

    # ─── GUARANTEED TRUNCATION ───────────────────────────────────────────────────
    # Pre-tokenize the entire dataset HERE with truncation=True and max_length capped.
    # This runs BEFORE SFTTrainer sees the data, so it cannot be bypassed by any
    # SFTTrainer version, chat template system-prompt, or collator quirk.
    # The Gemma-3-IT built-in system prompt alone is ~650 tokens; without this step
    # the sequences reaching the model are 888+ tokens -> OOM on MLP gate_proj.
    print(f"[*] Pre-tokenizing with hard truncation to {args.max_seq_length} tokens...")
    def pre_tokenize(example):
        # We manually tokenize to find the exact boundary of the model's response
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        
        # Causal LM: labels = input_ids
        labels = enc["input_ids"].copy()
        
        # Mask everything before the translation so the model doesn't try to predict the random English inputs
        # Gemma 3 chat template adds <start_of_turn>model\n right before the target translation.
        # Find where this occurs in the tokenized sequence.
        # Gemma special token IDs: <start_of_turn> = 106, model = 2516, \n = 108 (approx, varies by vocab)
        # Instead of guessing tokens, we use string matching to find the boundary.
        model_turn_str = "<start_of_turn>model\n"
        idx = example["text"].find(model_turn_str)
        if idx != -1:
            # Tokenize just the prompt to find its length
            prompt_enc = tokenizer(example["text"][:idx + len(model_turn_str)])
            prompt_len = len(prompt_enc["input_ids"])
            # Mask the prompt tokens
            labels[:prompt_len] = [-100] * prompt_len
            
        enc["labels"] = labels
        return enc

    train_dataset = train_dataset.map(
        pre_tokenize,
        remove_columns=["text"],
        batched=False,
        desc="Tokenizing + truncating"
    )
    # ─────────────────────────────────────────────────────────────────────────────

    print("[*] Configuring 4-bit NF4 quantization settings (Hardware-Accelerated Float16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True
    )
    
    # We save the sharded model to /tmp/ (RAM Disk) to completely bypass Kaggle's 20GB output disk limit!
    # 16.5GB in RAM + 2GB active load chunk perfectly fits inside Kaggle's 30GB CPU RAM!
    local_sharded_dir = "/tmp/sharded_model"
    if not os.path.exists(local_sharded_dir):
        print(f"[*] WARNING: The Hugging Face repo {args.model_id} contains a single massive 16.5GB safetensors file!")
        print(f"[*] Downloading into CPU RAM to safely shard it into 2GB chunks first...")
        
        # Load without device_map so it just sits in CPU RAM safely without quantization overhead
        temp_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=hf_token
        )
        print(f"[*] Model loaded into CPU RAM. Sharding to {local_sharded_dir}...")
        temp_model.save_pretrained(local_sharded_dir, max_shard_size="2GB")
        tokenizer.save_pretrained(local_sharded_dir)
        
        print("[*] Sharding complete! Nuking temp model from RAM...")
        del temp_model
        gc.collect()
        torch.cuda.empty_cache()
        print("[*] RAM safely cleared!")
        
    # Overwrite the model_id so the 4-bit loader reads our local sharded directory!
    args.model_id = local_sharded_dir
    
    print("[*] Loading pruned sharded model in 4-bit precision ENTIRELY on GPU 0...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",  # Eager attention: no SDPA peak-memory spikes
        token=hf_token
    )
    
    # CRITICAL FIX: Prevent Trainer from wrapping the model in DataParallel!
    model.is_model_parallel = True
    model.is_loaded_in_4bit = True
        
    model.config.torch_dtype = torch.float16
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    
    # Prepare model using official PEFT function
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
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
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": True},
        "optim": "adamw_torch", # 8-bit optimizer crashes PyTorch fp16 GradScaler! Standard AdamW is only 300MB extra for LoRA.
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "logging_steps": 20,
        "learning_rate": args.learning_rate,
        "fp16": False,   # Disabled to completely bypass the GradScaler AssertionError bug!
        "bf16": False,
        "lr_scheduler_type": "cosine",
        "push_to_hub": False,
        "report_to": "none",
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "remove_unused_columns": False,  # Keep all columns including token_type_ids
        "torch_compile": False,          # Explicitly disable compilation as per advice
    }

    # ─── SIMPLE COLLATOR ─────────────────────────────────────────────────────────
    # Dataset is already tokenized + truncated. This collator only pads to the
    # longest sequence in the batch and injects token_type_ids for Gemma-3.
    # Sequences are guaranteed <= max_seq_length; no complex truncation needed.
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
    # ─────────────────────────────────────────────────────────────────────────────

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "callbacks": [comet_callback],
        "data_collator": gemma3_collator,
    }

    # Dataset is pre-tokenized -> do NOT pass dataset_text_field or packing.
    # Route tokenizer / processing_class (needed for generation in callbacks)
    import inspect
    sft_trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    print(f"[*] Instantiating SFTConfig with arguments: {list(config_kwargs.keys())}")
    sft_config = SFTConfig(**config_kwargs)

    # ─── BUGFIX TRAINER ──────────────────────────────────────────────────────────
    # Unsloth's new Gemma 3 patch returns a custom tensor object for logits where 
    # `.shape` is accidentally a method instead of a property. This causes the newest
    # version of `trl` to crash when it tries to calculate entropy. 
    # We bypass this completely by overriding compute_loss to never touch the logits!
    class BugfixTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Hugging Face >= 4.46 requires compute_loss to scale the loss by gradient_accumulation_steps 
            if num_items_in_batch is not None and hasattr(self, "_compute_loss_scaling_factor"):
                loss = self._compute_loss_scaling_factor(loss, num_items_in_batch)
                
            # CRITICAL MATHEMATICAL FIX FOR T4 FLOAT16 UNDERFLOW
            # Because we must use fp16=False to bypass PyTorch GradScaler crashes, the tiny 
            # LoRA gradients underflow to exactly 0.0 in float16. 
            # We fix this by manually scaling the loss by 1024.0. The gradients become 1024x larger,
            # surviving the float16 backward pass. The adamw optimizer is inherently scale-invariant,
            # so the 1024x multiplier perfectly cancels out during the parameter update!
            loss = loss * 1024.0
            
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs["args"] = sft_config
    print(f"[*] Instantiating BugfixTrainer with arguments: {list(trainer_kwargs.keys())}")
    trainer = BugfixTrainer(**trainer_kwargs)

    
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
    
    # Nuke the temporary sharded model
    import shutil
    shutil.rmtree("/tmp/sharded_model", ignore_errors=True)
    # Also nuke the huggingface cache so it doesn't get zipped!
    shutil.rmtree("/kaggle/working/huggingface_cache", ignore_errors=True)
    print("[*] Cleaned up temporary sharded files.")

if __name__ == "__main__":
    main()
