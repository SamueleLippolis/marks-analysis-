from pathlib import Path
import numpy as np
import pandas as pd
import json

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