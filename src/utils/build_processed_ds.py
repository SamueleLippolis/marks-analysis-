from pathlib import Path
import pandas as pd
from src.utils.io import load_cfg, ensure_dir

# interim to processed  
def run_build_processed_ds(cfg):
    # Set
    interim_dir = Path(cfg['paths']['interim']) 
    df = pd.read_csv(f'{interim_dir}/clean_ds.csv')

    # interim to processed 
    mean_dict = {}
    std_dict = {}
    for col in list(df.columns):
        mean_dict[col] = df[col].mean()
        std_dict[col] = df[col].std()

    for col in list(set(df.columns) - set(['id'])):
        df[col] = (df[col] - mean_dict[col]) / std_dict[col]

    # export 
    out = Path(cfg['paths']['processed']) / 'normalized_df.csv'
    ensure_dir(out.parent)
    df.to_csv(out, index = False)

    return out 
