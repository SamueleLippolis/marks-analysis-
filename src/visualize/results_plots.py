# src/visualize/results_plots.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_multi_target_points(y_true, y_pred, ylim = [0,10], outpath: Path = None):
    """
    Scatter-style point plots of y_true vs y_pred for 1+ targets.
    - y_true, y_pred: array-like [N] or [N, T]
    - target_names: list of length T (optional)
    - ylim: tuple (ymin, ymax) or None
    - outpath: Path to save (PNG). If None, returns without saving.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    plt.figure(figsize=(10, 6))
    plt.suptitle("Predictions visualization", fontsize=16, y=0.95)

    plt.subplot(1,2,1)
    plt.title("Aggregated1")
    plt.plot(yt[:,0], marker='o', linestyle='None', label='Observations')
    plt.plot(yp[:,0], marker='x', linestyle='None', label='Predictions')
    plt.ylim((ylim[0],ylim[1]))
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.title("Aggregated2")
    plt.plot(yt[:,1], marker='o', linestyle='None', label='Observations')
    plt.plot(yp[:,1], marker='x', linestyle='None', label='Predictions')
    plt.ylim((ylim[0],ylim[1]))
    plt.legend()
    plt.grid(True)

    # improve layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)
    plt.close()   # don't show in non-interactive scripts
