from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

def fit_lr(X_train, y_train, fit_intercept=True):
    m = LinearRegression(fit_intercept=fit_intercept).fit(X_train, y_train)
    return m

def predict_lr(model, X_test):
    return model.predict(X_test)

def save_lr(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path) # it saves your model into a file 
