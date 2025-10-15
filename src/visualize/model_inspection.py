from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

def export_lr_weights(model, cfg, out_dir: Path = None, basename="lr_weights"):

    coef = model.coef_
    intercept = model.intercept_

    trg_names = cfg['features']['target']
    ftr_names = cfg['features']['predictors']

    dict_a1 = {}
    dict_a2 = {}
    dict_a1['intercept'] = intercept[0]
    dict_a2['intercept'] = intercept[1]
    for i,feature in zip(range(5), ftr_names):
        dict_a1[feature] = coef[0, i]
    for i,feature in zip(range(5),ftr_names):
        dict_a2[feature] = coef[1, i]
    export_weights = {
        trg_names[0]: dict_a1,
        trg_names[1]: dict_a2
    } 

    with open(f"{out_dir}/{basename}.json", 'w') as f:
        json.dump(export_weights, f, indent=4)


import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_lr_abs_coeffs(exp_folder, weights_file_name='lr_weights.json'):
    
    artifacts_path = exp_folder / 'artifacts'
    weights_path = artifacts_path / weights_file_name
    figure_path = exp_folder / 'figures' 

    with open(weights_path, 'r') as f:
        weights_data = json.load(f)

    abs_weights = pd.DataFrame(weights_data).T
    abs_weights = abs_weights.abs()
    abs_weights = abs_weights.drop(columns = ['intercept'])

    abs_weights.plot(kind='bar', figsize=(8,5))
    plt.title("Feature importance (|coefficients|)")
    plt.ylabel("Absolute coefficient value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{figure_path}/feature_importance.jpg", dpi = 150)

    
