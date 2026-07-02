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
    
    out_f = None
    if not args.use_stdin and args.output_file:
        out_f = open(args.output_file, "w", encoding="utf-8")
        
    log(f"[*] Processing {len(lines)} sentences (Sequential Mode for maximum stability)...")
        
    for i, text in enumerate(lines):
        messages = [
            {"role": "user", "content": f"Translate the following text from {src_name} to {tgt_name}.\n\nText to translate:\n{text}"}
        ]
        
        try:
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_dict=True, 
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(model.device)
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)
            
        gen_inputs = {"input_ids": input_ids}
        
        with torch.no_grad():
            outputs = model.generate(
                **gen_inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Decode only the newly generated tokens
        input_len = input_ids.shape[1]
        generated_tokens = outputs[0, input_len:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        clean_tr = clean_translation(decoded, src_name, tgt_name)
        out_str = clean_tr.replace("\n", " ").replace("\r", " ").strip()
        
        if out_f:
            out_f.write(out_str + "\n")
            out_f.flush()
        else:
            sys.stdout.write(out_str + "\n")
            sys.stdout.flush()
                
    if out_f:
        out_f.close()
    
    log("[*] Inference complete.")

if __name__ == "__main__":
    main()
