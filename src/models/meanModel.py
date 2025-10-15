from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def meanModel_predict(X): # The * means "pack all positional arguments into the tuple X"
    """
    Given one or more lists/arrays of numbers, return their means.
    - If one list is given → return a single mean (float)
    - If multiple lists are given → return a list of means
    """
    means = np.array([[np.mean(x),np.mean(x)] for x in X])
    return means[0] if len(means) == 1 else means 