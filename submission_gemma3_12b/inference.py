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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        token=hf_token
    )
    model.eval()
    
    # Align pad token ID
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    out_f = None
    if not args.use_stdin and args.output_file:
        out_f = open(args.output_file, "w", encoding="utf-8")
        
    log(f"[*] Processing {len(lines)} sentences with batch size {args.batch_size}...")
        
    for i in range(0, len(lines), args.batch_size):
        batch = lines[i:i + args.batch_size]
        
        batch_inputs = []
        for text in batch:
            messages = [
                {"role": "system", "content": "You are a machine translation assistant. Output only the translation."},
                {"role": "user", "content": f"Translate the following text from {src_name} to {tgt_name}. Output ONLY the raw translation, without any introductory text, explanation, markdown formatting, or surrounding conversation. The output must contain only the translated text.\n\nText to translate:\n{text}"}
            ]
            
            try:
                encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
            except TypeError:
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
                
            batch_inputs.append(encoded)
            
        # Use HuggingFace's native pad function to correctly handle left-padding and attention masks
        inputs = tokenizer.pad(batch_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for tr in decoded:
            clean_tr = clean_translation(tr, src_name, tgt_name)
            # Ensure it's strictly one line
            clean_tr = clean_tr.replace('\n', ' ').replace('\r', '')
            if out_f:
                out_f.write(clean_tr + "\n")
                out_f.flush()
            else:
                sys.stdout.write(clean_tr + "\n")
                sys.stdout.flush()
                
    if out_f:
        out_f.close()
    
    log("[*] Inference complete.")

if __name__ == "__main__":
    main()
