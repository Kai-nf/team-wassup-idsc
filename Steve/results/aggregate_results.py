import os
import glob
import json
import pandas as pd
import numpy as np

def aggregate_results():
    result_files = glob.glob("results/*_results.json")
    rows = []
    
    for file in result_files:
        model_name = os.path.basename(file).replace("_results.json", "")
        with open(file, "r") as f:
            data = json.load(f)
            
        metrics = ["f1_macro", "roc_auc"] # Add other metrics
        agg_data = {"Model": model_name}
        
        for m in metrics:
            vals = [data[str(fold)].get(m, 0) for fold in range(5) if str(fold) in data]
            if len(vals) > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                agg_data[m] = f"{mean_val:.3f} +- {std_val:.3f}"
                
        rows.append(agg_data)
        
    df = pd.DataFrame(rows)
    df.to_csv("results/master_ablation_table.csv", index=False)
    
    latex_table = df.to_latex(index=False, escape=False)
    with open("results/master_ablation_table.tex", "w") as f:
        f.write(latex_table)
        
    print("Results aggregated and exported to results/master_ablation_table.csv and .tex")

if __name__ == "__main__":
    aggregate_results()