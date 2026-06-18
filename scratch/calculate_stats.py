import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("outputs/benchmark_summary.csv")

# Filter out rows with no language pairs or empty scores
df = df.dropna(subset=["comet_score", "src_lang"])

# Group by model and precision
groups = df.groupby(["model", "precision"])

results = []
for (model, precision), group in groups:
    # We want to verify we have exactly 3 language pairs
    comet_scores = group["comet_score"].values
    avg_comet = np.mean(comet_scores)
    std_comet = np.std(comet_scores, ddof=1) if len(comet_scores) > 1 else 0.0
    
    avg_speed = group["avg_tokens_per_sec"].mean()
    max_vram = group["peak_vram_mb"].max()
    
    results.append({
        "Model": model,
        "Precision": precision,
        "Avg COMET": avg_comet,
        "Std Dev COMET": std_comet,
        "Avg Speed (tok/s)": avg_speed,
        "Max VRAM (MB)": max_vram
    })

# Convert to DataFrame and sort by Avg COMET descending
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by="Avg COMET", ascending=False)

# Format to markdown table manually to avoid tabulate dependency
headers = ["Model", "Precision", "Avg COMET", "Std Dev COMET", "Avg Speed (tok/s)", "Max VRAM (MB)"]
print("| " + " | ".join(headers) + " |")
print("| " + " | ".join([":---" for _ in headers]) + " |")
for _, r in res_df.iterrows():
    row = [
        r["Model"],
        r["Precision"],
        f"{r['Avg COMET']:.4f}",
        f"{r['Std Dev COMET']:.4f}",
        f"{r['Avg Speed (tok/s)']:.4f}",
        f"{r['Max VRAM (MB)']:.2f}"
    ]
    print("| " + " | ".join(row) + " |")

