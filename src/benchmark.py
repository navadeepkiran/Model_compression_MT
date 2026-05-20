import os
import argparse
import time
import json
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

def parse_args():
    parser = argparse.ArgumentParser(description="WMT Model Benchmarker")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--precision", type=str, required=True, choices=["fp16", "bf16", "int8", "int4"], help="Model precision format")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="FLORES source language code")
    parser.add_argument("--tgt_lang", type=str, default="hin_Deva", help="FLORES target language code")
    parser.add_argument("--limit", type=int, default=100, help="Number of sentences to run benchmark on")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    return parser.parse_args()

def load_quantized_model(model_name, precision):
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
    
    t0 = time.time()
    if bnb_config:
        model = model_class.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = model_class.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
    load_time = time.time() - t0
    
    return model, is_seq2seq, load_time

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_alias = args.model.replace("/", "_")
    output_json_path = os.path.join(args.output_dir, f"{model_alias}_{args.precision}_translations.json")
    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    
    print(f"=== Starting Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Precision: {args.precision}")
    print(f"Task: {args.src_lang} -> {args.tgt_lang} (Limit: {args.limit})")
    
    # Load dataset
    print("[*] Loading FLORES-200 dataset...")
    try:
        # Try loading as aligned pair
        dataset = load_dataset("Muennighoff/flores200", f"{args.src_lang}-{args.tgt_lang}", split="dev")
        src_sentences = dataset[f"sentence_{args.src_lang}"][:args.limit]
        tgt_sentences = dataset[f"sentence_{args.tgt_lang}"][:args.limit]
    except Exception as e:
        print(f"[!] Aligned pair load failed: {e}. Falling back to single-language loads...")
        dataset_src = load_dataset("Muennighoff/flores200", args.src_lang, split="dev")
        dataset_tgt = load_dataset("Muennighoff/flores200", args.tgt_lang, split="dev")
        src_sentences = [item["sentence"] for item in dataset_src][:args.limit]
        tgt_sentences = [item["sentence"] for item in dataset_tgt][:args.limit]
        
    print(f"[*] Loaded {len(src_sentences)} sentences.")
    
    # Load Model and Tokenizer
    model, is_seq2seq, load_time = load_quantized_model(args.model, args.precision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[*] Model loaded in {load_time:.2f} seconds.")
    
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
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        else:
            prompt = f"Translate the following text from {get_lang_name(args.src_lang)} to {get_lang_name(args.tgt_lang)}.\n{get_lang_name(args.src_lang)}: {src_text}\n{get_lang_name(args.tgt_lang)}:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            
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
                    {"role": "user", "content": f"Translate the following text from {src_lang_name} to {tgt_lang_name}:\n\n{src_text}"}
                ]
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
                num_beams=1
            )
        duration = time.time() - t0
        
        # Decode and postprocess
        if is_seq2seq:
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            num_generated_tokens = len(outputs[0])
        else:
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            num_generated_tokens = len(generated_ids)
            
        # Clean up causal LM response format if model includes extra context
        if not is_seq2seq and "\n" in translation:
            # Check if model hallucinated another prompt block
            translation = translation.split(f"{src_lang_name}:")[0].strip()
            translation = translation.split(f"\n\n")[0].strip()
            
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
        
        if (idx + 1) % 10 == 0:
            print(f"    - Processed {idx + 1}/{len(src_sentences)} sentences...")
            
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
        "load_time_sec": [load_time],
        "peak_vram_mb": [peak_vram],
        "avg_tokens_per_sec": [avg_tokens_per_sec],
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    df_new = pd.DataFrame(summary_data)
    
    if os.path.exists(summary_csv_path):
        df_old = pd.read_csv(summary_csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        # Drop duplicates based on model and precision, keeping the latest run
        df_combined = df_combined.drop_duplicates(subset=["model", "precision"], keep="last")
        df_combined.to_csv(summary_csv_path, index=False)
    else:
        df_new.to_csv(summary_csv_path, index=False)
        
    print(f"[*] Summary appended to {summary_csv_path}")
    print(f"=== Benchmark Complete ===")
    print(f"Load Time: {load_time:.2f}s | Peak VRAM: {peak_vram:.2f}MB | Avg Speed: {avg_tokens_per_sec:.2f} tok/s")

if __name__ == "__main__":
    main()
