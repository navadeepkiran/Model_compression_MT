import os
import subprocess
import sys
import argparse
from datetime import datetime

# Check required imports
try:
    import comet
    import bitsandbytes
    import datasets
    import transformers
    import accelerate
except ImportError as e:
    print(f"\n[!] WARNING: Missing required dependency: {e.name}")
    print("Please install the dependencies by running this command in a notebook cell first:")
    print("!pip install -r requirements.txt")
    print("\nStarting the pipeline anyway, but it may fail during execution.\n")


# Automatically load Kaggle Secrets for HuggingFace token if available
# This propagates the token to all spawned benchmark subprocesses
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
except Exception:
    pass


# Define the models and their target precisions (from the max-feasible table)
# You can edit the HF model IDs below if you use local paths or specific model versions.
MODELS = [
    {"name": "Gemma-2B", "id": "google/gemma-2b-it", "precision": "fp16"},
    {"name": "Gemma-7B", "id": "google/gemma-7b-it", "precision": "int8"},
    {"name": "Aya-Expanse-8B", "id": "CohereForAI/aya-expanse-8b", "precision": "int8"},
    {"name": "Llama-3.1-8B", "id": "meta-llama/Llama-3.1-8B-Instruct", "precision": "int8"},
    {"name": "Qwen-2.5-7B", "id": "Qwen/Qwen2.5-7B-Instruct", "precision": "int8"},
    {"name": "Qwen-2.5-14B", "id": "Qwen/Qwen2.5-14B-Instruct", "precision": "int4"},
    {"name": "Mistral-7B", "id": "mistralai/Mistral-7B-Instruct-v0.3", "precision": "int8"},
    {"name": "EuroLLM-9B", "id": "utter-project/EuroLLM-9B-Instruct", "precision": "int8"},
]

# WMT26 Language combinations
LANG_PAIRS = [
    {"src": "ces_Latn", "tgt": "deu_Latn", "desc": "Czech -> German"},
    {"src": "eng_Latn", "tgt": "zho_Hans", "desc": "English -> Chinese (Simplified)"},
    {"src": "eng_Latn", "tgt": "arz_Arab", "desc": "English -> Egyptian Arabic"},
]

def parse_args():
    parser = argparse.ArgumentParser(description="WMT26 Automation Pipeline Runner")
    parser.add_argument("--limit", type=int, default=100, help="Number of sentences to run benchmark on")
    parser.add_argument("--only_model", type=str, default=None, help="Run only this model name (e.g. Gemma-2B)")
    parser.add_argument("--skip_models", type=str, default=None, help="Comma-separated list of model names to skip (e.g. Gemma-2B,Gemma-7B)")
    parser.add_argument("--only_lang", type=str, default=None, help="Run only this lang pair (e.g. ces_Latn-deu_Latn)")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Attention implementation to use")
    return parser.parse_args()

def run_command(cmd, log_file=None):
    """Runs a shell command, streams its output to console in real-time, and logs it."""
    print(f"Executing: {' '.join(cmd)}")
    
    log_f = None
    if log_file:
        log_f = open(log_file, "a", encoding="utf-8")
        log_f.write(f"\n\n--- COMMAND RUN AT {datetime.now()} ---\n")
        log_f.write(f"Command: {' '.join(cmd)}\n\n")
        log_f.flush()
        
    try:
        # Start the subprocess with stdout and stderr piped
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line-buffered
        )
        
        # Read stdout line by line as it becomes available
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_f:
                log_f.write(line)
                log_f.flush()
                
        process.wait()
        returncode = process.returncode
    except Exception as e:
        print(f"[!] Error running command: {e}")
        returncode = -1
    finally:
        if log_f:
            log_f.close()
            
    return returncode == 0


def main():
    args = parse_args()
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline_log = os.path.join(output_dir, "pipeline_run.log")
    error_log = os.path.join(output_dir, "pipeline_errors.log")
    
    # Filter models if specified
    models_to_run = MODELS
    if args.only_model:
        models_to_run = [m for m in MODELS if m["name"].lower() == args.only_model.lower()]
        if not models_to_run:
            print(f"[!] Model '{args.only_model}' not found in configuration list.")
            sys.exit(1)
            
    if args.skip_models:
        skip_list = [s.strip().lower() for s in args.skip_models.split(",")]
        models_to_run = [m for m in models_to_run if m["name"].lower() not in skip_list]
            
    # Filter language pairs if specified
    langs_to_run = LANG_PAIRS
    if args.only_lang:
        langs_to_run = []
        for pair in LANG_PAIRS:
            pair_alias = f"{pair['src']}-{pair['tgt']}"
            if pair_alias.lower() == args.only_lang.lower():
                langs_to_run.append(pair)
        if not langs_to_run:
            print(f"[!] Language pair '{args.only_lang}' not found (format: src-tgt, e.g. eng_Latn-zho_Hans).")
            sys.exit(1)
            
    total_runs = len(models_to_run) * len(langs_to_run)
    current_run = 0
    successful_runs = 0
    failed_runs = []
    
    print("=" * 60)
    print(f"WMT26 Model Compression Evaluation Pipeline")
    print(f"Total Configurations to Run: {total_runs}")
    print(f"Outputs will be saved in: '{output_dir}'")
    print(f"Detailed logs saved to: '{pipeline_log}'")
    print("=" * 60)
    
    for model_cfg in models_to_run:
        model_name = model_cfg["name"]
        model_id = model_cfg["id"]
        precision = model_cfg["precision"]
        
        for lang_pair in langs_to_run:
            src = lang_pair["src"]
            tgt = lang_pair["tgt"]
            lang_desc = lang_pair["desc"]
            
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] PROCESSING: {model_name} ({precision}) on {lang_desc}")
            print("-" * 50)
            
            # Step 1: Run Benchmarking
            benchmark_cmd = [
                sys.executable, "-u", "src/benchmark.py",
                "--model", model_id,
                "--precision", precision,
                "--src_lang", src,
                "--tgt_lang", tgt,
                "--limit", str(args.limit),
                "--output_dir", output_dir
            ]
            if args.attn_implementation:
                benchmark_cmd.extend(["--attn_implementation", args.attn_implementation])
            
            bench_ok = run_command(benchmark_cmd, log_file=pipeline_log)
            
            if not bench_ok:
                print(f"[!] BENCHMARK FAILED for {model_name} ({src}->{tgt}). Check logs: {pipeline_log}")
                with open(error_log, "a", encoding="utf-8") as err_f:
                    err_f.write(f"[{datetime.now()}] Benchmark failed: {model_name} ({precision}) for {src}->{tgt}\n")
                failed_runs.append(f"{model_name} ({precision}) on {src}->{tgt} [Benchmarking stage]")
                continue
                
            # Step 2: Identify output translation file and run evaluation
            model_alias = model_id.replace("/", "_")
            translation_file = os.path.join(output_dir, f"{model_alias}_{precision}_{src}_{tgt}_translations.json")
            
            if not os.path.exists(translation_file):
                print(f"[!] ERROR: Expected translation file not found: {translation_file}")
                failed_runs.append(f"{model_name} ({precision}) on {src}->{tgt} [File missing stage]")
                continue
                
            print(f"[+] Benchmark succeeded. Evaluating quality with COMET...")
            
            evaluate_cmd = [
                sys.executable, "-u", "src/evaluate.py",
                "--translation_file", translation_file,
                "--summary_csv", os.path.join(output_dir, "benchmark_summary.csv")
            ]
            
            eval_ok = run_command(evaluate_cmd, log_file=pipeline_log)
            
            if not eval_ok:
                print(f"[!] EVALUATION FAILED for {model_name} ({src}->{tgt}). Check logs: {pipeline_log}")
                with open(error_log, "a", encoding="utf-8") as err_f:
                    err_f.write(f"[{datetime.now()}] Evaluation failed: {model_name} ({precision}) for {src}->{tgt}\n")
                failed_runs.append(f"{model_name} ({precision}) on {src}->{tgt} [Evaluation stage]")
                continue
                
            print(f"[+] SUCCESS: Completed pipeline for {model_name} ({precision}) on {lang_desc}")
            successful_runs += 1
            
        # Clean up model cache from disk to prevent disk full errors
        try:
            import shutil
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = os.path.join(hf_home, "hub")
            else:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                
            model_cache_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
            if os.path.exists(model_cache_dir):
                print(f"[*] Cleaning up disk cache for {model_name} ({model_id}) to free space...")
                shutil.rmtree(model_cache_dir, ignore_errors=True)
                print(f"[+] Disk space freed!")
        except Exception as e:
            print(f"[!] Failed to clean cache for {model_name}: {e}")
            
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total runs attempted: {total_runs}")
    print(f"Successfully completed: {successful_runs}")
    print(f"Failed configs: {len(failed_runs)}")
    
    if failed_runs:
        print("\nFailed configurations list:")
        for fail in failed_runs:
            print(f" - {fail}")
        print(f"See '{error_log}' for timestamped failures.")
        
    print(f"\nUnified results matrix saved to: '{os.path.join(output_dir, 'benchmark_summary.csv')}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
