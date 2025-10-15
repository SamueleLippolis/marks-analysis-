from pathlib import Path
import pandas as pd
from src.utils.io import load_cfg
from src.utils.data_utils import get_columns_stats
from src.visualize.eda_plots import plot_columns_stats, plot_df_stats


if __name__ == "__main__":
    cfg = load_cfg()
    df = pd.read_csv(Path(cfg["paths"]["interim"]) / "clean_ds.csv")  

    export_path =  Path(cfg["paths"]["reports"]) / "data_analysis"

    stats = get_columns_stats(df, cfg)
    plot_columns_stats(stats, export_path)
    plot_df_stats(df, cfg, export_path)
