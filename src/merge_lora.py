import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("=== WMT26 LoRA Merge Script ===")
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default="nani-nav/gemma-3-12b-final-wmt", help="Path or HF ID of base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-12b-merged", help="Output path for merged model")
    parser.add_argument("--push_to_hub", type=str, default=None, help="If provided, pushes merged model to this HF repo")
    args = parser.parse_args()

    # Automatically load Kaggle Secrets for HuggingFace token if available
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[!] Warning: HF_TOKEN not found. May fail if base model is a private repository.")
    
    print(f"[*] Loading base model: {args.base_model} in BFloat16...")
    # CRITICAL: We must load the base model in 16-bit (BFloat16) to merge. 
    # You cannot mathematically merge FP32/BF16 LoRA weights directly into a 4-bit quantized base model.
    # We load onto the CPU to completely prevent GPU OOM crashes during the massive matrix additions.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=hf_token
    )
    
    print(f"[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    
    print(f"[*] Loading LoRA adapter from: {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    
    print(f"[*] Merging LoRA weights into base model...")
    print(f"    (This will take a few minutes as CPU calculates billions of additions...)")
    merged_model = model.merge_and_unload()
    
    print(f"[*] Saving unified model locally to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir, max_shard_size="2GB", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[+] Merged model saved successfully!")
    
    if args.push_to_hub:
        print(f"[*] Pushing unified model to HuggingFace Hub: {args.push_to_hub}...")
        merged_model.push_to_hub(args.push_to_hub, token=hf_token)
        tokenizer.push_to_hub(args.push_to_hub, token=hf_token)
        print(f"[+] Successfully pushed to Hub! Your model is now a standalone checkpoint.")

if __name__ == "__main__":
    main()
