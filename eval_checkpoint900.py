"""
eval_checkpoint900.py
=====================
Runs FLORES-200 translation benchmark + COMET scoring on the checkpoint-900 
LoRA fine-tuned model, using exactly the same methodology as the other benchmarked models.

On Kaggle: 
  - Mount the 900_csde_model dataset (contains checkpoint-900/)
  - Run: python Model_compression_MT/eval_checkpoint900.py

Language pairs benchmarked (matching the other models):
  - eng_Latn -> ces_Latn  (English -> Czech)
  - eng_Latn -> deu_Latn  (English -> German)
  - eng_Latn -> zho_Hans  (English -> Chinese Simplified)
  - eng_Latn -> arz_Arab  (English -> Egyptian Arabic)
"""

import os
import sys
import json
import time
import subprocess
import tarfile
import urllib.request

# CRITICAL: Must be set before ANY transformers/peft import.
# Kaggle has broken TensorFlow/protobuf that crashes when transformers tries to auto-detect backends.
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

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "nani-nav/gemma-3-12b-final-csde"
LORA_PATH      = "/kaggle/input/900_csde_model/checkpoint-900"   # Kaggle Dataset mount
PRECISION      = "int4"
LIMIT          = 100    # sentences per language pair (same as other benchmarks)
MAX_NEW_TOKENS = 128
OUTPUT_DIR     = "/kaggle/working/eval_checkpoint900"

LANG_PAIRS = [
    ("eng_Latn", "ces_Latn"),
    ("eng_Latn", "deu_Latn"),
    ("eng_Latn", "zho_Hans"),
    ("eng_Latn", "arz_Arab"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Install dependencies ─────────────────────────────────────────────────────
print("[*] Installing evaluation dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "unbabel-comet", "sacrebleu"], check=True)

# ─── FLORES-200 dataset ───────────────────────────────────────────────────────
# Prioritize the copy that came with the git clone of the repo
REPO_FLORES = "/kaggle/working/Model_compression_MT/flores200_dataset"
FLORES_DIR  = "/kaggle/working/flores200_dataset"

if os.path.exists(REPO_FLORES):
    FLORES_DIR = REPO_FLORES
    print(f"[*] Using FLORES-200 from repo: {FLORES_DIR}")
elif not os.path.exists(FLORES_DIR):
    print("[*] Downloading FLORES-200 from Meta Research CDN...")
    url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    tar_path = "/kaggle/working/flores200_dataset.tar.gz"
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, tar_path)
    print("[*] Extracting FLORES-200...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("/kaggle/working")
    os.remove(tar_path)
else:
    print(f"[*] FLORES-200 already exists at {FLORES_DIR}")

# ─── Load model ONCE, evaluate all language pairs ────────────────────────────
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─── Model Download / Cache ───────────────────────────────────────────────────
# Check if already cached from a previous training run in this session
CACHE_DIR = "/kaggle/tmp/model_cache"
safetensors_path = os.path.join(CACHE_DIR, "model.safetensors")
config_path = os.path.join(CACHE_DIR, "config.json")

if os.path.exists(config_path) and os.path.exists(safetensors_path) and \
   os.path.getsize(safetensors_path) > 15 * 1024**3:
    print(f"[*] Model already cached at {CACHE_DIR} — skipping download!")
    model_source = CACHE_DIR
else:
    print(f"[*] Model not cached. Downloading {BASE_MODEL_ID}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Wipe orphaned .lock files to prevent hf_hub_download deadlock
    for lock_root in [os.path.expanduser("~/.cache/huggingface"), CACHE_DIR]:
        if os.path.exists(lock_root):
            for rt, dirs, fls in os.walk(lock_root, topdown=False):
                if ".locks" in dirs:
                    shutil.rmtree(os.path.join(rt, ".locks"), ignore_errors=True)
                for fn in fls:
                    if fn.endswith(".lock"):
                        try: os.remove(os.path.join(rt, fn))
                        except: pass

    # Download small config files via hf_hub_download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "hf_xet"],
                   capture_output=True, timeout=30)

    from huggingface_hub import hf_hub_download, hf_hub_url
    import requests as req_lib

    small_files = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json", "added_tokens.json", "chat_template.jinja"
    ]
    print("[*] Downloading config files...")
    for f_name in small_files:
        fp = os.path.join(CACHE_DIR, f_name)
        if not os.path.exists(fp):
            try:
                hf_hub_download(repo_id=BASE_MODEL_ID, filename=f_name,
                                token=hf_token, local_dir=CACHE_DIR)
            except Exception:
                pass

    # Download the massive safetensors via direct HTTP streaming (bypasses XetHub deadlock)
    if not os.path.exists(safetensors_path) or os.path.getsize(safetensors_path) < 15 * 1024**3:
        print("[*] Streaming 16.5GB model.safetensors via direct HTTP...")
        url = hf_hub_url(BASE_MODEL_ID, 'model.safetensors')
        with req_lib.get(url, headers={'Authorization': f'Bearer {hf_token}'},
                         stream=True, allow_redirects=True, timeout=(60, 300)) as resp:
            resp.raise_for_status()
            written = 0
            with open(safetensors_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        written += len(chunk)
                        if written % (1024**3) < 64 * 1024 * 1024:
                            print(f"    -> {written/(1024**3):.1f} GB downloaded")
        print(f"[*] Download complete: {written/(1024**3):.1f} GB")
    model_source = CACHE_DIR

print(f"\n[*] Loading base model from: {model_source}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_source,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    token=hf_token,
)
model.eval()

print(f"[*] Loading LoRA adapter from: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# EOS / stop tokens
eos_token_ids = []
for t in [tokenizer.eos_token_id, *([tokenizer.eos_token_id] if isinstance(tokenizer.eos_token_id, list) else [])]:
    if t is not None and t not in eos_token_ids:
        eos_token_ids.append(t)
for word in ["<end_of_turn>", "<|im_end|>", "<|eot_id|>"]:
    tid = tokenizer.convert_tokens_to_ids(word)
    if tid and tid != getattr(tokenizer, "unk_token_id", None) and tid not in eos_token_ids:
        eos_token_ids.append(tid)

LANG_MAP = {
    "eng_Latn": "English",
    "ces_Latn": "Czech",
    "deu_Latn": "German",
    "zho_Hans": "Chinese (Simplified)",
    "arz_Arab": "Egyptian Arabic",
}

def get_lang_name(code):
    return LANG_MAP.get(code, code.split("_")[0].capitalize())

def load_flores(src_lang, tgt_lang, limit):
    src_sentences, tgt_sentences = [], []
    for split in ["devtest", "dev"]:
        src_path = os.path.join(FLORES_DIR, split, f"{src_lang}.{split}")
        tgt_path = os.path.join(FLORES_DIR, split, f"{tgt_lang}.{split}")
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            with open(src_path, "r", encoding="utf-8") as f:
                src_sentences = [l.strip() for l in f if l.strip()][:limit]
            with open(tgt_path, "r", encoding="utf-8") as f:
                tgt_sentences = [l.strip() for l in f if l.strip()][:limit]
            print(f"    -> Loaded {len(src_sentences)} sentences from FLORES-200 ({split})")
            return src_sentences, tgt_sentences
    raise FileNotFoundError(f"FLORES-200 files not found for {src_lang}/{tgt_lang}")

all_results = {}

for src_lang, tgt_lang in LANG_PAIRS:
    print(f"\n{'='*60}")
    print(f"Evaluating: {src_lang} -> {tgt_lang}")
    print(f"{'='*60}")
    
    src_sentences, tgt_sentences = load_flores(src_lang, tgt_lang, LIMIT)
    src_name = get_lang_name(src_lang)
    tgt_name = get_lang_name(tgt_lang)
    
    results = []
    torch.cuda.empty_cache()
    
    for idx, (src_text, ref_text) in enumerate(zip(src_sentences, tgt_sentences)):
        messages = [
            {"role": "system", "content": "You are a machine translation assistant. Output only the translation."},
            {"role": "user", "content": f"Translate the following text from {src_name} to {tgt_name}. Output ONLY the raw translation, without any introductory text, explanation, or formatting.\n\nText to translate:\n{src_text}"}
        ]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda:0")
            inputs = {"input_ids": input_ids}
        
        gen_inputs = {"input_ids": inputs["input_ids"]}
        
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **gen_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos_token_ids if eos_token_ids else None,
            )
        duration = time.time() - t0
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        num_tokens = len(generated_ids)
        
        if idx == 0:
            print(f"  [DEBUG] First translation: {repr(translation[:200])}")
        
        results.append({
            "id": idx,
            "source": src_text,
            "reference": ref_text,
            "translation": translation,
            "duration_sec": duration,
            "generated_tokens": num_tokens,
            "tokens_per_sec": num_tokens / duration if duration > 0 else 0,
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(src_sentences)}] processed...")
    
    # Save translations JSON
    output_json = os.path.join(OUTPUT_DIR, f"checkpoint900_{src_lang}_{tgt_lang}_translations.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[*] Translations saved to {output_json}")
    all_results[f"{src_lang}->{tgt_lang}"] = output_json

# ─── COMET Scoring ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Running COMET scoring on all language pairs...")
print(f"{'='*60}")

from comet import download_model, load_from_checkpoint

print("[*] Downloading COMET model (Unbabel/wmt22-comet-da)...")
comet_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_path)

summary = []
for lang_pair, json_path in all_results.items():
    print(f"\n[*] Scoring {lang_pair}...")
    with open(json_path, "r", encoding="utf-8") as f:
        translations = json.load(f)
    
    data = [{"src": t["source"], "mt": t["translation"], "ref": t["reference"]} for t in translations]
    
    gpus = 1 if torch.cuda.is_available() else 0
    predictions = comet_model.predict(data, batch_size=8, gpus=gpus)
    system_score = predictions.system_score
    scores = predictions.scores
    
    # Write individual scores back
    for i, t in enumerate(translations):
        t["comet_score"] = float(scores[i])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    
    summary.append({
        "model": f"checkpoint-900 ({BASE_MODEL_ID})",
        "precision": PRECISION,
        "lang_pair": lang_pair,
        "comet_score": system_score,
        "num_sentences": len(translations),
    })
    print(f"  COMET Score [{lang_pair}]: {system_score:.4f}")

# ─── Final Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL COMET SCORES — checkpoint-900")
print(f"{'='*60}")
for row in summary:
    print(f"  {row['lang_pair']:25s}  COMET: {row['comet_score']:.4f}")

summary_path = os.path.join(OUTPUT_DIR, "checkpoint900_comet_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n[*] Summary saved to {summary_path}")
print("=== Evaluation Complete ===")
