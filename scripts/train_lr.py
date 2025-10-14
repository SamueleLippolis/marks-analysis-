from pathlib import Path
import json, pandas as pd
from src.utils.io import load_cfg
from src.utils.data_utils import split_dataset
from src.models.linear_regression import fit_lr, predict_lr, save_lr
from src.models.metrics import regression_metrics

if __name__ == "__main__":
    cfg = load_cfg()
    df = pd.read_csv(Path(cfg["paths"]["interim"]) / "clean_ds.parquet") # for semplcity i use the interim data
    feats = cfg["features"]["predictors"]
    trg = cfg["features"]["target"]
    test_size = 0.2
    seed = 0

    X_train, X_test, y_train, y_test = split_dataset(df, feats, trg, test_size, seed)
    model = fit_lr(X_train, y_train, fit_intercept=True)
    y_pred = predict_lr(model, X_test)
    metrics = regression_metrics(y_test, y_pred)
    print(metrics)

    model_id = 'lr_model_seed{seed}_ts{test_size}'
    save_lr(model, Path(cfg["paths"]["models"]) / f"{model_id}.pkl")
    Path(cfg["paths"]["reports"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["reports"]) / "metrics.json").write_text(json.dumps({f"{model_id}": metrics}, indent=2))
