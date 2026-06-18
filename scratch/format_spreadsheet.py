import ast
import pandas as pd
import re

file_path = r'C:\Users\navad\Documents\WMT\Task\output.txt'
out_path = r'C:\Users\navad\Documents\WMT\outputs\Professional_Training_Metrics.csv'

data = []

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if line.startswith("{'loss':"):
            try:
                dict_obj = ast.literal_eval(line)
                data.append(dict_obj)
            except Exception as e:
                pass

if data:
    df = pd.DataFrame(data)
    
    # Calculate exact step since it logged every 20 steps
    # We ignore the 'epoch' rounding issue by strictly using the row index
    df['Step'] = (df.index + 1) * 20
    
    # Create professional columns
    df['Model Name'] = 'Gemma-3-12B-WMT-LoRA'
    df['Training Loss'] = df['loss'].round(4)
    df['Gradient Norm'] = df['grad_norm'].round(4)
    df['Learning Rate'] = df['learning_rate'].apply(lambda x: f"{x:.2e}")
    df['Epoch'] = df['epoch']
    
    # Add final COMET score to the very last row only
    df['Final COMET Score'] = ""
    df.loc[df.index[-1], 'Final COMET Score'] = "0.6976"
    
    # Reorder and select professional columns
    cols = ['Model Name', 'Step', 'Epoch', 'Training Loss', 'Gradient Norm', 'Learning Rate', 'Final COMET Score']
    df = df[cols]
    
    df.to_csv(out_path, index=False)
    print(f"Successfully generated professional spreadsheet with {len(df)} rows to {out_path}")
else:
    print("No data found!")
