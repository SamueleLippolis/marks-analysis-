from pathlib import Path
import pandas as pd
from src.utils.io import ensure_dir

# raw to interim
def run_make_ds(cfg):
    # Set and import 
    raw_dir = Path(cfg["paths"]["raw"])
    df = pd.read_csv(raw_dir / "ds.csv")

    # Clean: raw to interim 
    col_to_save = [
        'a1_mean',
        'a2_mean',
        'd1_mean',
        'd2_mean', 
        'd3_mean',
        'd4_mean',
        'd5_mean',
        'id '
    ]
    df = df.drop(columns = [col for col in df.columns if col not in col_to_save])

    raname_dict = {
        'a1_mean': 'aggregated1',
        'a2_mean': 'aggregated2',
        'd1_mean': 'granular1',
        'd2_mean': 'granular2',
        'd3_mean': 'granular3',
        'd4_mean': 'granular4',
        'd5_mean': 'granular5',
        'id ': 'id'
    }
    df = df.rename(columns = raname_dict)
    
    # Export
    out = Path(cfg["paths"]["interim"]) / "clean_ds.csv"
    ensure_dir(out.parent)
    df.to_csv(out, index=False)

    return out