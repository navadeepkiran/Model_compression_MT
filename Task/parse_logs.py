import ast
import csv

input_file = r"c:\Users\navad\Documents\WMT\Task\output.txt"
output_file = r"c:\Users\navad\Documents\WMT\Task\training_metrics.csv"

metrics = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("{'loss':"):
            try:
                data = ast.literal_eval(line)
                metrics.append(data)
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")

if metrics:
    # Ensure consistent keys
    keys = ["epoch", "loss", "grad_norm", "learning_rate"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Successfully wrote {len(metrics)} rows to {output_file}")
else:
    print("No metrics found.")
