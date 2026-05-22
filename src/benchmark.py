import os
import argparse
import time
import json

# Automatically load Kaggle Secrets for HuggingFace if available
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
except Exception:
    pass

import torch
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

# Language code to human-readable name mapping for WMT26 Model Compression
LANG_MAP = {
    "eng_Latn": "English",
    "ces_Latn": "Czech",
    "deu_Latn": "German",
    "zho_Hans": "Chinese (Simplified)",
    "arz_Arab": "Egyptian Arabic",
}

def get_lang_name(code):
    return LANG_MAP.get(code, code.split("_")[0].capitalize())

def clean_translation(translation, src_lang_name, tgt_lang_name):
    # 1. Clean up any trailing/leading whitespace
    translation = translation.strip()
    
    # 2. Check if the model hallucinated another prompt block of the source language
    if f"{src_lang_name}:" in translation:
        translation = translation.split(f"{src_lang_name}:")[0].strip()
        
    # Split by newlines to inspect lines
    lines = [line.strip() for line in translation.split("\n") if line.strip()]
    if not lines:
        return ""
        
    # Check if lines[0] is an introductory prefix (e.g. "Sure, here's the translation:")
    def is_intro_line(line):
        line_lower = line.lower().strip("*_# :.+-")
        if not line_lower:
            return True
            
        # Common single-word markers
        if line_lower in ["translation", "translated", "übersetzung", "traduction", "traducción", "raw translation"]:
            return True
            
        # Ends with colon and looks like an intro
        words = line_lower.split()
        if len(words) < 15:
            prefix_keywords = {
                "sure", "here", "translation", "translate", "translated", "text", 
                "übersetzung", "deutsch", "arabic", "chinese", "czech", "english", 
                "german", "laute", "in", "into", "to", "is", "the", "below", "following",
                "of", "sentence", "phrase", "this", "here's", "hereis", "here!s", "german:"
            }
            # If it starts with common phrase markers
            starts_with_prefix = (
                line_lower.startswith("sure") or
                line_lower.startswith("here") or
                line_lower.startswith("translation") or
                line_lower.startswith("the translation") or
                line_lower.startswith("this is") or
                line_lower.startswith("die übersetzung") or
                line_lower.startswith("hier ist") or
                line_lower.startswith("übersetzung")
            )
            ends_with_colon = line.strip().endswith(":")
            
            if (starts_with_prefix or ends_with_colon) and all(w in prefix_keywords or w.isdigit() or len(w) <= 2 for w in words if w.isalnum()):
                return True
                
            # Additional check for common German, Arabic, Chinese translation prefix markers
            common_phrases = [
                "here is the translation", 
                "here is the translated", 
                "sure, here's", 
                "sure! here's", 
                "sure, here is", 
                "sure! here is", 
                "translation into", 
                "translation to",
                "übersetzung ins",
                "übersetzung zum",
                "die übersetzung lautet",
                "hier ist die",
                "here's the translation",
                "translation of the",
                "translation of:",
                "translated text:"
            ]
            if any(p in line_lower for p in common_phrases):
                return True
                
        return False

    cleaned_lines = []
    skipped_intro = False
    for i, line in enumerate(lines):
        if i == 0 and is_intro_line(line) and len(lines) > 1:
            skipped_intro = True
            continue
        if i == 1 and skipped_intro and is_intro_line(line) and len(lines) > 2:
            continue
        cleaned_lines.append(line)
        
    if cleaned_lines:
        res = " ".join(cleaned_lines).strip()
    else:
        res = lines[-1].strip()
        
    # Strip quotes or markdown bold/italic markers from the final translation if they wrap the whole text
    # e.g., "**Translation**" or "**[actual translation]**" or '"[translation]"'
    res = res.strip("*_`\"'")
    return res.strip()

def parse_args():
    parser = argparse.ArgumentParser(description="WMT Model Benchmarker")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--precision", type=str, required=True, choices=["fp16", "bf16", "int8", "int4"], help="Model precision format")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="FLORES source language code")
    parser.add_argument("--tgt_lang", type=str, default="hin_Deva", help="FLORES target language code")
    parser.add_argument("--limit", type=int, default=100, help="Number of sentences to run benchmark on")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Forced attention implementation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
    return parser.parse_args()

def load_quantized_model(model_name, precision, attn_implementation=None):
    print(f"[*] Loading model config for: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_seq2seq = config.is_encoder_decoder
    
    print(f"[*] Architecture detection: {'Seq2Seq' if is_seq2seq else 'Decoder-only (CausalLM)'}")
    
    # Configure quantization settings
    bnb_config = None
    torch_dtype = torch.float32
    
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    model_class = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    
    # Force single-GPU loading ("cuda:0") to prevent accelerate from splitting the model
    # across multiple GPUs or offloading layers to CPU.
    # This prevents multi-GPU deadlocks (common with bitsandbytes/transformers on Kaggle dual T4)
    # and avoids slow inference caused by CPU offloading (since unquantized model size estimation
    # might exceed a single GPU's VRAM, but the quantized INT8/INT4 model fits comfortably).
    device_map = "auto"
    if torch.cuda.is_available():
        device_map = "cuda:0"
        
    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True
    }
    if attn_implementation:
        print(f"[*] Forcing attention implementation: {attn_implementation}")
        load_kwargs["attn_implementation"] = attn_implementation
        
    t0 = time.time()
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
        model = model_class.from_pretrained(model_name, **load_kwargs)
    else:
        load_kwargs["torch_dtype"] = torch_dtype
        model = model_class.from_pretrained(model_name, **load_kwargs)
    load_time = time.time() - t0
    
    return model, is_seq2seq, load_time

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_alias = args.model.replace("/", "_")
    output_json_path = os.path.join(args.output_dir, f"{model_alias}_{args.precision}_{args.src_lang}_{args.tgt_lang}_translations.json")
    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    
    print(f"=== Starting Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Precision: {args.precision}")
    print(f"Task: {args.src_lang} -> {args.tgt_lang} (Limit: {args.limit})")
    
    # Load dataset
    print("[*] Loading FLORES-200 dataset...")
    import urllib.request
    import tarfile
    
    dataset_dir = "flores200_dataset"
    tar_path = "flores200_dataset.tar.gz"
    url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    
    loaded_successfully = False
    src_sentences, tgt_sentences = [], []
    
    # 1. Try downloading directly from Meta Research CDN (fast, script-free, bypasses HuggingFace Hub bugs)
    if not os.path.exists(dataset_dir):
        print(f"[*] Downloading FLORES-200 raw dataset from {url}...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, tar_path)
            
            print("[*] Extracting FLORES-200 dataset...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall()
            if os.path.exists(tar_path):
                os.remove(tar_path)
        except Exception as e:
            print(f"[!] Direct download from Meta failed: {e}. Will attempt Hugging Face fallback.")
            
    # 2. Try loading from local extracted files
    if os.path.exists(dataset_dir):
        try:
            for split in ["dev", "devtest"]:
                src_path = os.path.join(dataset_dir, split, f"{args.src_lang}.{split}")
                tgt_path = os.path.join(dataset_dir, split, f"{args.tgt_lang}.{split}")
                
                if os.path.exists(src_path) and os.path.exists(tgt_path):
                    print(f"[*] Reading sentences from local FLORES-200 files (split: {split})...")
                    with open(src_path, "r", encoding="utf-8") as f:
                        src_sentences = [line.strip() for line in f if line.strip()][:args.limit]
                    with open(tgt_path, "r", encoding="utf-8") as f:
                        tgt_sentences = [line.strip() for line in f if line.strip()][:args.limit]
                    loaded_successfully = True
                    break
        except Exception as e:
            print(f"[!] Failed to read local FLORES-200 files: {e}")
            
    # 3. Fallback to HF Datasets library if direct download / extraction failed
    if not loaded_successfully:
        print("[!] Local loading failed. Falling back to Hugging Face datasets library...")
        try:
            try:
                dataset_src = load_dataset("tomasmajercik/flores-parquet", name=args.src_lang, split="dev")
                dataset_tgt = load_dataset("tomasmajercik/flores-parquet", name=args.tgt_lang, split="dev")
            except Exception:
                try:
                    dataset_src = load_dataset("tomasmajercik/flores-parquet", name=args.src_lang, split="devtest")
                    dataset_tgt = load_dataset("tomasmajercik/flores-parquet", name=args.tgt_lang, split="devtest")
                except Exception:
                    dataset_src = load_dataset("tomasmajercik/flores-parquet", name=args.src_lang, split="validation")
                    dataset_tgt = load_dataset("tomasmajercik/flores-parquet", name=args.tgt_lang, split="validation")
                    
            src_sentences = [item["sentence"] for item in dataset_src][:args.limit]
            tgt_sentences = [item["sentence"] for item in dataset_tgt][:args.limit]
            loaded_successfully = True
        except Exception as e:
            print(f"[!] Hugging Face Parquet loading failed: {e}. Trying legacy Muennighoff/flores200...")
            try:
                dataset_src = load_dataset("Muennighoff/flores200", args.src_lang, split="dev", trust_remote_code=True)
                dataset_tgt = load_dataset("Muennighoff/flores200", args.tgt_lang, split="dev", trust_remote_code=True)
            except Exception:
                dataset_src = load_dataset("Muennighoff/flores200", args.src_lang, split="dev")
                dataset_tgt = load_dataset("Muennighoff/flores200", args.tgt_lang, split="dev")
            src_sentences = [item["sentence"] for item in dataset_src][:args.limit]
            tgt_sentences = [item["sentence"] for item in dataset_tgt][:args.limit]
        
    print(f"[*] Loaded {len(src_sentences)} sentences.")
    
    # Load Model and Tokenizer
    model, is_seq2seq, load_time = load_quantized_model(args.model, args.precision, args.attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[*] Model loaded in {load_time:.2f} seconds.")
    
    # Determine custom end/stopping tokens for generation to handle instruction-tuned models correctly
    eos_token_ids = []
    if isinstance(tokenizer.eos_token_id, list):
        eos_token_ids.extend(tokenizer.eos_token_id)
    elif tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
        
    # Standard conversation end/stopping tokens for various instruction models
    stop_words = ["<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|END_OF_TURN_TOKEN|>", "<|endoftext|>", "<|end|>"]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for word in stop_words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id is not None and token_id != unk_id:
            if token_id not in eos_token_ids:
                eos_token_ids.append(token_id)
                
    # Also incorporate model config-specific eos_token_ids if present
    model_eos = getattr(model.config, "eos_token_id", None)
    if model_eos is not None:
        if isinstance(model_eos, list):
            for eid in model_eos:
                if eid not in eos_token_ids:
                    eos_token_ids.append(eid)
        elif isinstance(model_eos, int):
            if model_eos not in eos_token_ids:
                eos_token_ids.append(model_eos)
                
    eos_token_ids = [eid for eid in eos_token_ids if eid is not None]
    if not eos_token_ids:
        eos_token_ids = None
    else:
        print(f"[*] Configured stop/EOS token IDs for generation: {eos_token_ids}")
    
    # VRAM Tracking Setup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    # Warm-up phase (5 sentences) to ensure memory metrics and CUDA graph cache are settled
    print("[*] Performing 5-sentence warm-up...")
    warmup_limit = min(5, len(src_sentences))
    for i in range(warmup_limit):
        src_text = src_sentences[i]
        if is_seq2seq:
            inputs = tokenizer(src_text, return_tensors="pt").to(model.device)
            _ = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
                eos_token_id=eos_token_ids
            )
        else:
            if tokenizer.chat_template is not None:
                messages = [
                    {"role": "user", "content": f"Translate the following text from {get_lang_name(args.src_lang)} to {get_lang_name(args.tgt_lang)}.\n\nText to translate:\n{src_text}"}
                ]
                try:
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                except TypeError:
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)
                    inputs = {"input_ids": input_ids}
            else:
                prompt = f"Translate the following text from {get_lang_name(args.src_lang)} to {get_lang_name(args.tgt_lang)}.\n{get_lang_name(args.src_lang)}: {src_text}\n{get_lang_name(args.tgt_lang)}:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
            _ = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
                eos_token_id=eos_token_ids
            )
            
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    # Run full evaluation
    results = []
    total_tokens = 0
    total_duration = 0.0
    
    src_lang_name = get_lang_name(args.src_lang)
    tgt_lang_name = get_lang_name(args.tgt_lang)
    
    print("[*] Starting translation inference...")
    for idx, (src_text, ref_text) in enumerate(zip(src_sentences, tgt_sentences)):
        # Construct Prompt
        if is_seq2seq:
            prompt = src_text
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:
            # Detect Chat Template
            if tokenizer.chat_template is not None:
                messages = [
                    {"role": "user", "content": f"Translate the following text from {src_lang_name} to {tgt_lang_name}. Output ONLY the raw translation, without any introductory text, explanation, markdown formatting, or surrounding conversation. The output must contain only the translated text.\n\nText to translate:\n{src_text}"}
                ]
                # Ensure special tokens are tokenized correctly as control IDs, not raw text subwords
                try:
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                except TypeError:
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)
                    inputs = {"input_ids": input_ids}
                # Keep prompt as a string for debugging and metadata
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"Translate the following text from {src_lang_name} to {tgt_lang_name}.\n{src_lang_name}: {src_text}\n{tgt_lang_name}:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
                eos_token_id=eos_token_ids
            )
        duration = time.time() - t0
        
        # Decode and postprocess
        if is_seq2seq:
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            num_generated_tokens = len(outputs[0])
            raw_gen = tokenizer.decode(outputs[0], skip_special_tokens=False)
        else:
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            num_generated_tokens = len(generated_ids)
            raw_gen = tokenizer.decode(generated_ids, skip_special_tokens=False)
            
        # Debug print for first sentence to diagnose generation length and stop tokens
        if idx == 0:
            print(f"\n[DEBUG] Prompt: {repr(prompt)}")
            print(f"[DEBUG] Raw Generated Text: {repr(raw_gen)}")
            print(f"[DEBUG] Cleaned Translation: {repr(translation)}")
            print(f"[DEBUG] Generated token count: {num_generated_tokens}")
            print(f"[DEBUG] Tokenizer chat template: {tokenizer.chat_template is not None}\n")
            
        # Clean up causal LM response format if model includes extra context
        if not is_seq2seq:
            translation = clean_translation(translation, src_lang_name, tgt_lang_name)
            
        tokens_per_sec = num_generated_tokens / duration if duration > 0 else 0
        
        results.append({
            "id": idx,
            "source": src_text,
            "reference": ref_text,
            "translation": translation,
            "duration_sec": duration,
            "generated_tokens": num_generated_tokens,
            "tokens_per_sec": tokens_per_sec
        })
        
        total_tokens += num_generated_tokens
        total_duration += duration
        
        print(f"    - Processed {idx + 1}/{len(src_sentences)} sentences (time: {duration:.2f}s, speed: {tokens_per_sec:.1f} tok/s)...")
            
    # Measure VRAM
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        peak_vram = 0.0
        
    avg_tokens_per_sec = total_tokens / total_duration if total_duration > 0 else 0
    
    # Save translations
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[*] Translations saved to {output_json_path}")
    
    # Save CSV Summary Row
    summary_data = {
        "model": [args.model],
        "precision": [args.precision],
        "src_lang": [args.src_lang],
        "tgt_lang": [args.tgt_lang],
        "load_time_sec": [load_time],
        "peak_vram_mb": [peak_vram],
        "avg_tokens_per_sec": [avg_tokens_per_sec],
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    df_new = pd.DataFrame(summary_data)
    
    if os.path.exists(summary_csv_path):
        df_old = pd.read_csv(summary_csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        # Drop duplicates based on model, precision, and language pair, keeping the latest run
        df_combined = df_combined.drop_duplicates(subset=["model", "precision", "src_lang", "tgt_lang"], keep="last")
        df_combined.to_csv(summary_csv_path, index=False)
    else:
        df_new.to_csv(summary_csv_path, index=False)
        
    print(f"[*] Summary appended to {summary_csv_path}")
    print(f"=== Benchmark Complete ===")
    print(f"Load Time: {load_time:.2f}s | Peak VRAM: {peak_vram:.2f}MB | Avg Speed: {avg_tokens_per_sec:.2f} tok/s")

if __name__ == "__main__":
    main()
