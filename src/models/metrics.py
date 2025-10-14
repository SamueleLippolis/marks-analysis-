from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

def regression_metrics(y_true, y_pred):
    return {
        "overall": {
            "R2": float(r2_score(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            'MAPE': float(mean_absolute_percentage_error(y_true, y_pred))
        },

        "agg1": {
            "R2": float(r2_score(y_true[:,0], y_pred[:,0])),
            "MAE": float(mean_absolute_error(y_true[:,0], y_pred[:,0])),
            'MAPE': float(mean_absolute_percentage_error(y_true[:,0], y_pred[:,0]))
        },

        "agg2": {
            "R2": float(r2_score(y_true[:,1], y_pred[:,1])),
            "MAE": float(mean_absolute_error(y_true[:,1], y_pred[:,1])),
            'MAPE': float(mean_absolute_percentage_error(y_true[:,1], y_pred[:,1]))
        }
    }
