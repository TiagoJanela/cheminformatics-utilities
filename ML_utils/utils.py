#imports
from typing import List
import os
import numpy as np
from scipy.stats import stats

# Plotting
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

# sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def convert_to_class_actives(y: List[float], class_type: str = None, min_pot=None, max_pot=None) -> tuple[
    List[int], List[str]]:
    if class_type is None:
        raise ValueError("class_type must be specified. Must be one of ['5-6-7-8-9-10-11'].")

    if min_pot is None:
        raise ValueError("min_pot must be specified.")
    if max_pot is None:
        raise ValueError("max_pot must be specified.")

    if class_type in ['5-6-7-8-9-10-11']:
        potency_thresholds = {'5-6-7-8-9-10-11': [6, 7, 8, 9, 10]}
        class_tags = {'5-6-7-8-9-10-11': [f'{min_pot}-6', '6-7', '7-8', '8-9', '9-10', f'10-{max_pot}']}
        idx = 1
    elif class_type in ['5-6-7-8-9-10']:
        potency_thresholds = {'5-6-7-8-9-10': [6, 7, 8, 9]}
        class_tags = {'5-6-7-8-9-10': [f'{min_pot}-6', '6-7', '7-8', '8-9', f'9-{max_pot}']}
        idx = 1
    else:
        raise ValueError("Invalid class_type. Must be one of ['5-6-7-8-9-10-11' or '5-6-7-8-9-10']")

    pot_thresh = potency_thresholds[class_type]
    class_tags = class_tags[class_type]

    y_class, y_class_tag = [], []

    for yp in y:
        for i, pt in enumerate(pot_thresh):
            if yp < pt:
                y_class.append(i + idx)
                y_class_tag.append(class_tags[i])
                break
        else:
            y_class.append(len(pot_thresh) + idx)
            y_class_tag.append(class_tags[-1])

    return y_class, y_class_tag


def class_to_potency(y_class, potency_values=None) -> List[int]:
    if potency_values is None:
        raise ValueError("potency_values must be specified.")

    y_potency = []
    for y in y_class:
        for i, value in enumerate(potency_values):
            if y == i+1:
                y_potency.append(value)
            else:
                continue
    return y_potency


def conf_matrix(labels, predictions):
    # Confusion matrix
    cnf_matrix = confusion_matrix(labels, predictions)

    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - ((cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)) + (
            cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)) + (np.diag(cnf_matrix)))
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)

    return TP, TN, FP, FN


def catplot(df, ct, metrics, x, y, hue, kind, hue_order, col, row, palette, sharey, aspect, bbox_to_anchor,
            save_path=None, file_name=None, font_scale=None, xlab=None, ylab=None):
    sns.set(font_scale=font_scale, style='whitegrid')

    df = df.query(f'Class_type in {ct} and Metric in {metrics}')

    t = sns.catplot(data=df, x=x, y=y, hue=hue,
                    kind=kind, hue_order=hue_order, col=col, palette=palette, row=row, sharey=sharey, aspect=aspect)
    if row and col:
        t.set_titles("{row_name} - {col_name}")
    elif row:
        t.set_titles("{row_name}")
    elif col:
        t.set_titles("{col_name}")
    t.set_xlabels(f'{xlab}')
    t.set_ylabels(f'{ylab}')
    sns.move_legend(t, "lower center", bbox_to_anchor=bbox_to_anchor, ncol=len(hue_order), )
    if save_path:
        t.savefig(os.path.join(save_path, file_name), dpi=300)

    return t


def catplot_regression(df, metrics, x, y, hue, kind, hue_order, col,  palette, aspect,
                       bbox_to_anchor,
                       legend_title='',
                       row_order=None, row=None,
                       save_path=None, file_name=None, font_scale=None, xlab=None, ylab=None, **kwargs):
    sns.set(font_scale=font_scale, style='whitegrid')

    t = sns.catplot(data=df.query(f'Metric in {metrics}'), x=x, y=y, hue=hue,
                    kind=kind, hue_order=hue_order, col=col, palette=palette,
                    aspect=aspect, row_order=row_order, row=row, **kwargs)
    if row and col:
        t.set_titles("{row_name} - {col_name}")
    elif col:
        t.set_titles("{col_var}: {col_name}")
    if xlab:
        t.set_xlabels(f'{xlab}')
    if ylab:
        t.set_ylabels(f'{ylab}')
    sns.move_legend(t, "lower center", bbox_to_anchor=bbox_to_anchor, ncol=len(hue_order), title=legend_title,
                    ) #label_pad=0.1
    if save_path:
        t.savefig(os.path.join(save_path, file_name), dpi=300)

    return t


def plot_regression_models_cat(df, metric, kind='box',
                               filename=None, results_path=None,
                               x=None, y=None,
                               col=None,
                               row=None,
                               ymin=None, ymax=None, yticks=None,
                               xticks=None,
                               palette='tab10',
                               x_labels='',
                               y_labels='',
                               hue=None, hue_order=None, title=True,
                               order=None,
                               legend_title="",
                               font_size=30, height=5,
                               font_scale=1.2,
                               aspect=1.2,
                               bbox_to_anchor=(-0.01, -0.15),
                               sharey=True,
                               theme='whitegrid',
                               show=False,
                               fig=None,
                               sub_fig_title=None,
                               **kwargs):
    # database
    performance_df_ = df.loc[df.Metric.isin(metric)]

    # plt parameters
    sns.set_theme('paper', style=theme, font_scale=font_scale)

    g = sns.catplot(data=performance_df_, x=x, y=y,
                    kind=kind,
                    height=height, aspect=aspect,
                    order=order, palette=palette,
                    hue=hue,
                    hue_order=hue_order,
                    col=col,
                    row=row,
                    legend=True, sharey=sharey, **kwargs)

    g.set_ylabels(y_labels, labelpad=15, fontsize=font_size)
    g.set_xlabels(f'{x_labels}', labelpad=15, fontsize=font_size)
    g.set(ylim=(ymin, ymax))
    g.tick_params(labelsize=font_size)

    if title:
        if row and col:
            g.set_titles(r"{row_name} - {col_name}", size=font_size)
        elif isinstance(title, str):
            g.set_titles(title)
        else:
            if fig is not None:
                g.set_titles(r"{row_var}: {row_name} - {col_name}", size=font_size)
            else:
                g.set_titles("{col_var}: {col_name}", size=font_size)
    if yticks:
        g.set(ylim=(ymin, ymax), yticks=yticks)
    if xticks:
        g.set_xticklabels(xticks, fontsize=font_size)

    plt.tight_layout()
    g.despine(right=True, top=True)
    sns.move_legend(g, "lower center", bbox_to_anchor=bbox_to_anchor, ncol=len(hue_order), title=legend_title,
                    labelspacing=1.5)

    if sub_fig_title:
        plt.suptitle(f'{sub_fig_title}', fontsize=45, x=0, y=1, fontweight='bold')

    if results_path:
        plt.savefig(results_path + f'{filename}.png', dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return g


def get_regression_metrics(labels, predictions, *args):
    # calculate metrics
    mae = metrics.mean_absolute_error(labels, predictions)
    mse = metrics.mean_squared_error(labels, predictions)
    rmse = metrics.root_mean_squared_error(labels, predictions)
    r2 = metrics.r2_score(labels, predictions)
    r = stats.pearsonr(labels, predictions)[0]

    results_list = {"MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2,
                    "r": r,
                    "r2": r ** 2,
                    "Dataset size": len(labels),
                    "Target ID": args[0],
                    "Algorithm": args[1],
                    "trial": args[2],
                    "Class_type": args[3],
                    "Potency_class": args[4],
                    "Potency range": args[5],
                    }

    return results_list


def plot_heatmap_stat_analysis(df, x, y, value, pvalue_boundaries=[0, 0.003, 0.01, 1],
                               font_size=16, clrs=['green', 'orange', 'red'],
                               height=10, aspect=1.5, square=False,
                               order=None,
                               results_path=None, filename=None,
                               sub_fig_title=None, **kwargs):
    norm = matplotlib.colors.BoundaryNorm(pvalue_boundaries, ncolors=len(pvalue_boundaries))
    cmap = matplotlib.colors.ListedColormap(clrs)

    sns.set(font_scale=1, style='white')
    pvalue_boundaries = pvalue_boundaries
    font = {'size': font_size}
    matplotlib.rc('font', **font)

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    ax = sns.FacetGrid(df, height=height, aspect=aspect, **kwargs).map_dataframe(draw_heatmap, x, y, value,
                                                                                 norm=norm, cmap=cmap, cbar=False,
                                                                                 annot=True, square=square)

    ax.set_titles("Exp: {col_name} Vs Pred: {row_name}")

    fig = ax.figure
    cbar_ax = fig.add_axes([1.01, 0.36, 0.02, 0.3])
    cbar_ticks = np.linspace(0, 1, (len(pvalue_boundaries) - 1) * 2 + 1)[1:][::2]

    cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap.reversed(), ticks=cbar_ticks)

    cbar.set_ticks(ticks=cbar_ticks,
                   labels=[f'p < {p}' if p < 1 else 'Not significant' for p in pvalue_boundaries[1:]][::-1])

    cbar.outline.set_edgecolor('0.5')
    cbar.outline.set_linewidth(0.1)
    cbar.ax.tick_params(size=0)

    if sub_fig_title:
        plt.suptitle(f'{sub_fig_title}', fontsize=25, x=0, y=1, fontweight='bold')
    if filename:
        fig.savefig(f'{results_path}/{filename}.png', dpi=300, bbox_inches='tight')
    return ax
