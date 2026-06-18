import os
from datasets import load_dataset

print("Testing Czech-German dataset loading...")
try:
    ds_cs_de = load_dataset("wmt19", "cs-de", split="train", streaming=True)
    item = next(iter(ds_cs_de))
    print("✅ cs-de loaded successfully. Sample:", item)
except Exception as e:
    print("❌ cs-de failed to load:", e)

print("\nTesting English-Chinese dataset loading...")
try:
    ds_zh_en = load_dataset("wmt19", "zh-en", split="train", streaming=True)
    item = next(iter(ds_zh_en))
    print("✅ zh-en loaded successfully. Sample:", item)
except Exception as e:
    print("❌ zh-en failed to load:", e)

print("\nTesting English-Arabic dataset loading...")
try:
    ds_ar_en = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train", streaming=True)
    item = next(iter(ds_ar_en))
    print("✅ ar-en loaded successfully. Sample:", item)
except Exception as e:
    print("❌ ar-en failed to load:", e)
