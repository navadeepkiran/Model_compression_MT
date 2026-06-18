import ast
import pandas as pd
import re

file_path = r'C:\Users\navad\Documents\WMT\Task\output.txt'
out_path = r'C:\Users\navad\Documents\WMT\outputs\Professional_Training_Metrics.csv'

data = []
last_step = None

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        
        # Extract the exact step number from the tqdm progress bar (e.g. 1600/1600 or 820/1875)
        match = re.search(r'(\d+)/(?:1875|1600)', line)
        if match:
            last_step = int(match.group(1))
            
        if line.startswith("{'loss':"):
            try:
                dict_obj = ast.literal_eval(line)
                
                step_val = last_step if last_step is not None else int(round(dict_obj['epoch'] * 1875 / 20) * 20)
                
                data.append({
                    'Model Name': 'Gemma-3-12B-WMT-LoRA',
                    'Step': step_val,
                    'Epoch': dict_obj['epoch'],
                    'Training Loss': round(dict_obj['loss'], 4),
                    'Gradient Norm': round(dict_obj['grad_norm'], 4),
                    'Learning Rate': f"{dict_obj['learning_rate']:.2e}"
                })
            except Exception as e:
                pass

if data:
    df = pd.DataFrame(data)
    
    # Because you restarted from step 800 after the crash, the text file has overlapping steps.
    # We drop the old crashed steps and keep the final successful run to make it perfectly clean!
    df = df.drop_duplicates(subset=['Step'], keep='last').sort_values('Step').reset_index(drop=True)
    
    # Add the final COMET score to the final row
    df['Final COMET Score'] = ""
    df.loc[df.index[-1], 'Final COMET Score'] = "0.6976"
    
    df.to_csv(out_path, index=False)
    print(f"Successfully generated pristine spreadsheet with {len(df)} rows to {out_path}")
else:
    print("No data found!")
