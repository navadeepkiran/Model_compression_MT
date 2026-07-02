import os
# Force transformers to ignore TensorFlow and JAX to avoid Kaggle's broken protobuf environment
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Suppress warnings that might leak to stdout
import warnings
warnings.filterwarnings("ignore")

# Automatically load Kaggle Secrets for HuggingFace token if testing locally on Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass

LANG_MAP = {
    "eng": "English",
    "zho_Hans": "Chinese (Simplified)",
}

def get_lang_name(code):
    return LANG_MAP.get(code, code)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--use_stdin", action="store_true")
    return parser.parse_args()

def clean_translation(translation, src_lang, tgt_lang):
    # Split into lines and strip whitespace
    lines = [line.strip() for line in translation.split("\n") if line.strip()]
    
    # Filter out markdown formatting and conversational fluff
    valid_lines = []
    for line in lines:
        if line.startswith("```"): continue
        if line.lower().startswith("here is the translation"): continue
        if line.lower().startswith("translation:"): continue
        if line.lower().startswith("the translated text"): continue
        if line.lower() == "text to translate:": continue
        valid_lines.append(line)
        
    if valid_lines:
        return valid_lines[-1].strip("*_`\"'")
    return translation.strip("*_`\"'")

def main():
    args = parse_args()
    
    try:
        src, tgt = args.lang_pair.split("-")
    except ValueError:
        src, tgt = "eng", "zho_Hans"
        
    src_name = get_lang_name(src)
    tgt_name = get_lang_name(tgt)
    
    # Send all print statements to stderr so they don't break WMT's stdout eval
    def log(msg):
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    log(f"[*] Starting Inference for {src_name} -> {tgt_name}")
    
    lines = []
    if args.use_stdin:
        lines = [line.strip() for line in sys.stdin if line.strip()]
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
    if not lines:
        return

    # Load Model (Honor WMT MODEL_DIR requirement)
    model_id = os.environ.get("MODEL_DIR", "nani-nav/gemma-3-12b-final-wmt-4488")
    log(f"[*] Loading model from: {model_id} in INT4 NF4...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    except AttributeError:
        log("[!] Tokenizer failed to load from local directory due to a transformers bug. Falling back to base Hugging Face tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", trust_remote_code=True, token=hf_token)
        
    tokenizer.padding_side = "left"
    
    # Left padding is required for batched generation!
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        token=hf_token
    )
    model.eval()
    
    out_f = None
    if not args.use_stdin and args.output_file:
        out_f = open(args.output_file, "w", encoding="utf-8")
        
    log(f"[*] Processing {len(lines)} sentences using exact training prompt AND correct tokenization...")
        
    for i in range(0, len(lines), args.batch_size):
        batch_lines = lines[i:i+args.batch_size]
        
        batch_inputs = []
        for src in batch_lines:
            messages = [
                {"role": "system", "content": "You are a machine translation assistant. Output only the translation."},
                {"role": "user", "content": f"Translate from {src_name} to {tgt_name}:\n{src}"}
            ]
            
            # We MUST use tokenize=True to ensure Gemma's <start_of_turn> control tokens are mapped 
            # to their special integer IDs. If we use tokenize=False and pass the string to tokenizer(), 
            # it will shred the control tokens into literal character pieces, causing the model to output EOS.
            try:
                encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
            except TypeError:
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            batch_inputs.append(encoded)
            
        inputs = tokenizer.pad(batch_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        prompt_length = inputs.input_ids.shape[1]
        
        for out_tokens in outputs:
            generated_tokens = out_tokens[prompt_length:]
            translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # --- DIAGNOSTIC LOGGING ---
            print(f"\\n[DIAGNOSTIC] Raw Generated Tokens: {generated_tokens.tolist()}", file=sys.stderr)
            print(f"[DIAGNOSTIC] Raw Decoded String: {repr(translation)}", file=sys.stderr)
            # --------------------------
            
            # Clean up the output exactly as required
            translation = clean_translation(translation, src_name, tgt_name)
            translation = translation.replace('\n', ' ').replace('\r', ' ').strip()
            
            if out_f:
                out_f.write(translation + '\n')
                out_f.flush()
            else:
                sys.stdout.write(translation + '\n')
                sys.stdout.flush()
                
    if out_f:
        out_f.close()
    
    log("[*] Inference complete.")

if __name__ == "__main__":
    main()
