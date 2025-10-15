from pathlib import Path
import json, pandas as pd
import datetime, os
from src.utils.io import load_cfg
from src.utils.data_utils import split_dataset
from src.models.linear_regression import fit_lr, predict_lr, save_lr
from src.models.metrics import regression_metrics
from src.visualize.results_plots import plot_multi_target_points
from src. visualize.model_inspection import export_lr_weights

if __name__ == "__main__":
    cfg = load_cfg()
    df = pd.read_csv(Path(cfg["paths"]["interim"]) / "clean_ds.csv") # for semplcity i use the interim data
    feats = cfg["features"]["predictors"]
    trg = cfg["features"]["target"]
    test_size = float(cfg['split']['test_size'])
    seed = int(cfg['split']['random_state'])
    shuffle_data = bool(cfg['split']['shuffle'])

    X_train, X_test, y_train, y_test = split_dataset(df, feats, trg, test_size, seed, shuffle_data)
    model = fit_lr(X_train, y_train, fit_intercept=True)
    y_pred = predict_lr(model, X_test)
    metrics = regression_metrics(y_test, y_pred)
    print(metrics)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f'lr_seed{seed}_ts{test_size}_shuffle{shuffle_data}'
    out = Path(cfg["paths"]["experiments"]) / f'lr_{timestamp}_{model_id}'
    out.mkdir(parents=True, exist_ok=True)

    save_lr(model, Path(out) / 'artifacts' / f'{model_id}.pkl')
    (out / "metrics.json").write_text(json.dumps({model_id: metrics}, indent=2))
    fig_path = out / "figures" / "preds_visualization.png"  
    plot_multi_target_points(
        y_true=y_test,
        y_pred=y_pred,
        outpath=fig_path
    )
    export_lr_weights(
        model=model,
        cfg = cfg,
        out_dir = out / "artifacts",           # JSON + CSV saved here
        basename ="lr_weights"
    )

