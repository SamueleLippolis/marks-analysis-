from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(df: pd.DataFrame, features, target, test_size=0.2, seed=0):
    X = df[features].values
    y = df[target].values
    return train_test_split(X, y, test_size=test_size, random_state=seed)
