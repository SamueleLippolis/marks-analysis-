from pathlib import Path
import json, joblib, numpy as np, pandas as pd, datetime, subprocess
from src.utils.io import load_cfg
from src.models.metrics import regression_metrics
from src.utils.data_utils import split_dataset
from src.models.neural_network import build_nn_model
from src.models.train_nn import train_one_model, predict
from src.visualize.results_plots import plot_multi_target_points, plot_learning_curves
import torch


if __name__ == '__main__':
    cfg = load_cfg()
    df = pd.read_csv(Path(cfg['paths']['interim']) / 'clean_ds.csv')
    features = cfg['features']['predictors']
    targets = cfg['features']['target']
    test_size = float(cfg['split']['test_size'])
    split_seed = int(cfg['split']['random_state'])
    shuffle_data = bool(cfg['split']['shuffle'])
    X_train, X_test, y_train, y_test = split_dataset(df, features, targets, test_size, split_seed, shuffle_data)

    input_dim  = int(cfg['nn']['input_dim'])
    hidden     = tuple(cfg['nn']['hidden_dims'])
    output_dim = int(cfg['nn']['output_dim'])
    epochs     = int(cfg['nn']['epochs'])
    batch_size = int(cfg['nn']['batch_size'])
    lr         = float(cfg['nn']['lr'])
    nn_seed    = int(cfg['nn']['seed'])
    model = build_nn_model(input_dim, hidden, output_dim)
    model, hist = train_one_model(
        model, 
        X_train,y_train,X_test,y_test,
        epochs,lr,batch_size, nn_seed
    )
    y_pred = predict(model, X_test)
    metrics = regression_metrics(y_test, y_pred)
    print(metrics)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f'nn_hidden{hidden}_lr{lr}'
    exp_repo = Path(cfg['paths']['experiments']) / f'nn_{timestamp}_{model_id}'
    exp_repo.mkdir(parents=True, exist_ok=True)
    Path(exp_repo/'artifacts').mkdir()

    torch.save(model.state_dict(), exp_repo / 'artifacts' /'model.pt')
    (exp_repo / 'metrics.json').write_text(json.dumps({model_id: metrics}, indent = 2))
    fig_path = exp_repo / "figures"   
    plot_multi_target_points(
        y_true=y_test,
        y_pred=y_pred,
        outpath=fig_path /  "preds_visualization.png"
    )
    plot_learning_curves(hist, fig_path / 'hist.png')

