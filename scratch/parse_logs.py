import ast
import pandas as pd
import re

file_path = r'C:\Users\navad\Documents\WMT\Task\output.txt'
out_path = r'C:\Users\navad\Documents\WMT\outputs\training_logs.csv'

data = []

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if line.startswith("{'loss':"):
            try:
                # Use ast to parse the dictionary safely
                dict_obj = ast.literal_eval(line)
                data.append(dict_obj)
            except Exception as e:
                print(f"Failed to parse line: {line} | Error: {e}")

if data:
    df = pd.DataFrame(data)
    # Add step column based on epoch (total steps = 1875)
    df['step'] = (df['epoch'] * 1875).round().astype(int)
    
    # Reorder columns
    cols = ['step', 'epoch', 'loss', 'grad_norm', 'learning_rate']
    df = df[[c for c in cols if c in df.columns]]
    
    df.to_csv(out_path, index=False)
    print(f"✅ Successfully extracted {len(df)} log records and saved to {out_path}")
else:
    print("❌ No log lines found in output.txt!")
