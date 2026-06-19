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

# ── BUGFIX FOR PYTORCH 2.6 ──────────────────────────────────────────────────
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
# ────────────────────────────────────────────────────────────────────────────

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
    # Load English-Chinese exclusively
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

def prepare_model_for_kbit_training_custom(model, use_gradient_checkpointing=True):
    for param in model.parameters():
        param.requires_grad = False
    
    if use_gradient_checkpointing:
        # CRITICAL: Inputs MUST require grad when using device_map="auto" + gradient checkpointing.
        # If inputs don't require grad, the autograd graph disconnects across GPU boundaries,
        # causing the infamous AccumulateGrad stream deadlock!
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
        # Initialize checkpointing on the model
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
    
    # Automatically save to Google Drive if it's mounted, preventing Colab from deleting checkpoints
    if os.path.exists("/content/drive/MyDrive"):
        args.output_dir = "/content/drive/MyDrive/Gemma3_WMT_Outputs"
        print(f"[*] Google Drive detected! Saving checkpoints securely to {args.output_dir}")
    
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
    import glob
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
    # ──────────────────────────────────────────────────────────────────────────

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()
    # Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set at module top (before import torch)
    # so the CUDA caching allocator can merge fragmented blocks to satisfy large contiguous requests.
    
    # Config for 4-bit NF4 QLoRA - using pure BFloat16 compute
    # Gemma 3 natively overflows Float16 (NaN loss). Float32 causes CUDA OOM.
    # BFloat16 is the ONLY precision that has the dynamic range to prevent NaN loss
    # while maintaining the memory efficiency to prevent OOM. 
    # T4 supports BFloat16 in software natively. We bypass GradScaler crashes
    # by keeping fp16=False in the SFTTrainer args.
    print("[*] Configuring 4-bit NF4 quantization settings (Pure BFloat16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True  # Required to allow embed_tokens and lm_head on CPU
    )
    
    # Check if unsloth is available to bypass the HuggingFace threaded loader memory leak
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        use_unsloth = False

    if use_unsloth:
        print("[*] Loading Gemma-3-12B using UNSLOTH (Highly Optimized 4-bit)...")
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
        
        # Disable cache for gradient checkpointing
        model.config.use_cache = False
    else:
        # Fallback to buggy HuggingFace loader
        print("[*] Loading Gemma-3-12B in 4-bit precision (BFloat16 base)...")
        # Gemma 3 has a massive 256,000 vocabulary. The embed_tokens and lm_head alone take 4.2 GB of VRAM!
        # By forcing these two specific BFloat16 tensors to the CPU, we instantly free up 4.2 GB of VRAM.
        # This guarantees the 4-bit layers have enough room to materialize without the 14.5 GB OOM spike.
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
            attn_implementation="eager",  # Eager attention: no SDPA peak-memory spikes
            token=hf_token
        )
        
        # CRITICAL FIX: Prevent Trainer from wrapping the model in DataParallel!
        # Because we put the entire model on cuda:0, Trainer thinks the model isn't parallelized.
        # Since it sees 2 GPUs on the Kaggle machine, it forcefully wraps it in nn.DataParallel.
        # DataParallel attempts to copy the 4-bit model to GPU 1, which instantly corrupts the 
        # bitsandbytes quant_state pointers, causing a CUDA illegal memory access.
        # Setting these flags completely disables DataParallel.
        model.is_model_parallel = True
        model.is_loaded_in_4bit = True
        
        model.config.torch_dtype = torch.bfloat16
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
    # ── LOGITS FLOAT32 UPCAST ─────────────────────────────────────────────────
    # Since we disabled fp16 autocast to avoid the BFloat16 GradScaler crash,
    # the lm_head naturally outputs float16/bfloat16 logits. Gemma 3 has a massive 256,000
    # token vocabulary, so the exp(logits) in CrossEntropyLoss WILL overflow,
    # resulting in NaN loss and NaN gradients. We MUST upcast logits to float32!
    print("[*] Registering float32 upcast hook on lm_head to prevent NaN loss...")
    def cast_logits_to_fp32(module, input, output):
        return output.to(torch.float32)
        
    output_layer = model.get_output_embeddings()
    if output_layer is not None:
        output_layer.register_forward_hook(cast_logits_to_fp32)
    elif hasattr(model, "lm_head"):
        model.lm_head.register_forward_hook(cast_logits_to_fp32)
    # ──────────────────────────────────────────────────────────────────────────

    print("[*] Configuring Trainer...")
    
    # Core TrainingArguments params
    # Core TrainingArguments params
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
        "fp16": False,   # Disable AMP GradScaler to completely bypass the BFloat16 crash!
        "bf16": False,   # Explicitly disable bf16 to prevent BFloat16 crashes on T4
        "lr_scheduler_type": "cosine",
        "push_to_hub": False,
        "report_to": "none",
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
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

    # ── BUGFIX TRAINER ────────────────────────────────────────────────────────
    # Unsloth's new Gemma 3 patch returns a custom tensor object for logits where 
    # `.shape` is accidentally a method instead of a property. This causes the newest
    # version of `trl` to crash when it tries to calculate entropy. 
    # We bypass this completely by overriding compute_loss to never touch the logits!
    class BugfixTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Hugging Face >= 4.46 requires compute_loss to scale the loss by gradient_accumulation_steps 
            # if num_items_in_batch is passed. If we skip this, gradients become 8x too large!
            if num_items_in_batch is not None and hasattr(self, "_compute_loss_scaling_factor"):
                loss = self._compute_loss_scaling_factor(loss, num_items_in_batch)
                
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

if __name__ == "__main__":
    main()
