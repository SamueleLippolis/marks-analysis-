from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_columns_stats(stats, export_path):
    # --- Convert to DataFrame ---
    stats_df = pd.DataFrame(stats).T  # rows = features, columns = mean, std

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    colors = ["#ca851dff" if 'aggregated' in col else '#87cefa' for col in stats_df.index]
    plt.bar(stats_df.index, stats_df['mean'], yerr=stats_df['std'], capsize=4, color=colors, edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value')
    plt.ylim((5,9.5))
    plt.title('Mean ± 1 Std for Aggregated and Granular Variables')
    plt.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(export_path / 'columns_means', dpi = 150)
    plt.close()

def plot_df_stats(df, cfg, export_path):
    # --- Plot ---
    aggregated_cols = cfg['features']['target']
    granular_cols = cfg['features']['predictors']

    df[aggregated_cols].plot.box()
    plt.title("Aggregated box-plot")
    plt.savefig(export_path / 'aggregated_boxplots', dpi = 150)
    plt.close()

    df[granular_cols].plot.box()
    plt.title("Granular box-plot")
    plt.savefig(export_path / 'granular_boxplots', dpi = 150)
    plt.close()

    # -- Plot --
    bins = [i for i in range(3, 11)]
    all_cols = aggregated_cols + granular_cols


    n_cols = 3
    n_rows = (len(all_cols) + n_cols - 1) // n_cols  # automatic rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        df[col].hist(bins=bins, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_ylim(0, 25)
        axes[i].grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Histograms of Aggregated and Granular Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(export_path / 'columns_hists', dpi = 150)
    plt.close()

    # -- Plot -- 
    sns.heatmap(df[granular_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Granular features correlation')
    plt.savefig(export_path / 'granular_feature_corr', dpi = 150)
    plt.close()

    # corr between granular and aggregated
    sns.heatmap(df[granular_cols + aggregated_cols].corr().loc[granular_cols, aggregated_cols], annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Granular vs Aggregated features correlation')
    plt.savefig(export_path / 'granular_aggregated_corr', dpi = 150)
    plt.close()

    # corr between granular and aggregated
    plt.figure(figsize=(11,9))
    sns.heatmap(df[granular_cols + aggregated_cols].corr(), annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Granular vs Aggregated features correlation')
    plt.savefig(export_path / 'variables_corr', dpi = 150)
    plt.close()