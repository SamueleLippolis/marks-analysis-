from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(df: pd.DataFrame, features, target, test_size=0.2, seed=0, shuffle = True):
    X = df[features].values
    y = df[target].values
    return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle = shuffle)

def get_columns_stats(df, cfg):

    aggregated_cols = cfg['features']['target']
    granular_cols = cfg['features']['predictors']
    stats = {}

    for col in aggregated_cols + granular_cols:
        mean = df[col].mean()
        std = df[col].std()
        stats[col] = {'mean': mean, 'std': std}

    return stats 
    
