"""
eval_checkpoint900.py
=====================
Thin wrapper that:
  1. Downloads the base model to local cache (bulletproof, bypasses XetHub)
  2. Calls the existing src/benchmark.py for each language pair
  3. Calls the existing src/evaluate.py for each translation JSON

On Kaggle:
  - Make sure 900_csde_model dataset is attached (contains checkpoint-900/)
  - Run: python Model_compression_MT/eval_checkpoint900.py
"""

import os
import sys
import subprocess
import shutil

# Must be set before any transformers/peft import to stop TF protobuf crash
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

# ─── Kaggle Secrets ──────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
except Exception:
    hf_token = os.environ.get("HF_TOKEN", "")

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR    = "/kaggle/working/Model_compression_MT"
BENCHMARK   = os.path.join(REPO_DIR, "src", "benchmark.py")
EVALUATE    = os.path.join(REPO_DIR, "src", "evaluate.py")
LORA_PATH   = "/kaggle/input/900_csde_model/checkpoint-900"
OUTPUT_DIR  = "/kaggle/working/eval_checkpoint900"
CACHE_DIR   = "/kaggle/tmp/model_cache"

BASE_MODEL_ID = "nani-nav/gemma-3-12b-final-csde"

LANG_PAIRS = [
    ("eng_Latn", "ces_Latn"),
    ("eng_Latn", "deu_Latn"),
    ("eng_Latn", "zho_Hans"),
    ("eng_Latn", "arz_Arab"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── Step 1: Download model to local cache (bulletproof, bypasses XetHub) ────
safetensors = os.path.join(CACHE_DIR, "model.safetensors")
config_file = os.path.join(CACHE_DIR, "config.json")

if os.path.exists(config_file) and os.path.exists(safetensors) and \
        os.path.getsize(safetensors) > 15 * 1024 ** 3:
    print(f"[*] Model already cached at {CACHE_DIR} — skipping download!")
else:
    print(f"[*] Downloading {BASE_MODEL_ID} to {CACHE_DIR}...")

    # Wipe .lock files (prevent hf_hub_download deadlock)
    for lock_root in [os.path.expanduser("~/.cache/huggingface"), CACHE_DIR]:
        if os.path.exists(lock_root):
            for rt, dirs, fls in os.walk(lock_root, topdown=False):
                if ".locks" in dirs:
                    shutil.rmtree(os.path.join(rt, ".locks"), ignore_errors=True)
                for fn in fls:
                    if fn.endswith(".lock"):
                        try: os.remove(os.path.join(rt, fn))
                        except: pass

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "hf_xet"],
                   capture_output=True, timeout=30)

    from huggingface_hub import hf_hub_download, hf_hub_url
    import requests

    # Small config files
    small_files = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json", "added_tokens.json", "chat_template.jinja"
    ]
    print("[*] Downloading config files...")
    for f_name in small_files:
        if not os.path.exists(os.path.join(CACHE_DIR, f_name)):
            try:
                hf_hub_download(repo_id=BASE_MODEL_ID, filename=f_name,
                                token=hf_token, local_dir=CACHE_DIR)
            except Exception:
                pass

    # Big model file via direct HTTP stream (the ONLY thing that works with XetHub)
    if not os.path.exists(safetensors) or os.path.getsize(safetensors) < 15 * 1024 ** 3:
        print("[*] Streaming 16.5GB model.safetensors via direct HTTP...")
        url = hf_hub_url(BASE_MODEL_ID, "model.safetensors")
        with requests.get(url, headers={"Authorization": f"Bearer {hf_token}"},
                          stream=True, allow_redirects=True, timeout=(60, 300)) as resp:
            resp.raise_for_status()
            written = 0
            with open(safetensors, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024 * 1024):
                    if chunk:
                        f.write(chunk); f.flush()
                        written += len(chunk)
                        if written % (1024 ** 3) < 64 * 1024 * 1024:
                            print(f"    -> {written / (1024**3):.1f} GB downloaded")
        print(f"[*] Download complete: {written / (1024**3):.1f} GB")

print(f"[*] Model ready at {CACHE_DIR}\n")

# ─── Step 2: Run benchmark.py for each language pair ─────────────────────────
# We pass CACHE_DIR as --model so benchmark.py loads from local files (no HF download)
translation_files = []

for src_lang, tgt_lang in LANG_PAIRS:
    print(f"\n{'='*60}")
    print(f"[*] Benchmarking: {src_lang} -> {tgt_lang}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, BENCHMARK,
        "--model",        CACHE_DIR,
        "--precision",    "int4",
        "--lora_path",    LORA_PATH,
        "--src_lang",     src_lang,
        "--tgt_lang",     tgt_lang,
        "--limit",        "100",
        "--max_new_tokens", "128",
        "--output_dir",   OUTPUT_DIR,
    ]

    result = subprocess.run(cmd, check=True)

    # benchmark.py saves to: {output_dir}/{model_alias}_{precision}_{src}_{tgt}_translations.json
    # model_alias replaces "/" with "_"
    model_alias = CACHE_DIR.replace("/", "_")
    json_name = f"{model_alias}_int4_{src_lang}_{tgt_lang}_translations.json"
    json_path = os.path.join(OUTPUT_DIR, json_name)
    if os.path.exists(json_path):
        translation_files.append(json_path)
        print(f"[*] Saved: {json_path}")
    else:
        # Fallback: find any matching JSON just in case alias differs
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(f"{src_lang}_{tgt_lang}_translations.json"):
                translation_files.append(os.path.join(OUTPUT_DIR, f))
                print(f"[*] Found: {f}")
                break

# ─── Step 3: Run evaluate.py (COMET) for each translation JSON ───────────────
print(f"\n{'='*60}")
print("[*] Running COMET evaluation on all language pairs...")
print(f"{'='*60}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "unbabel-comet", "sacrebleu"], check=True)

for json_path in translation_files:
    print(f"\n[*] Scoring: {os.path.basename(json_path)}")
    subprocess.run([
        sys.executable, EVALUATE,
        "--translation_file", json_path,
        "--comet_model",      "Unbabel/wmt22-comet-da",
        "--batch_size",       "8",
        "--summary_csv",      os.path.join(OUTPUT_DIR, "benchmark_summary.csv"),
    ], check=True)

print("\n=== All done! Check", OUTPUT_DIR, "for results ===")
