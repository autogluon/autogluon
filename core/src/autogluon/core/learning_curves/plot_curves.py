import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Tuple


def plot_curves(learning_curves: Tuple[dict, dict], model: str, metric: str) -> Figure:
    """
    Plots learning curves across all evaluation sets for specified model-metric pairing.

    Parameters:
    -----------
    learning_curves: Tuple[dict, dict]
        Learning curve output from TabularPredictor's learning_curves() method.
    model: str
        The model to plot curves for. Model must be included in TabularPredictor's
        training process, and thus the associated learning curves.
    metric: str
        The metric to plot curves for. Metric must be specified in TabularPredictor's
        training process, and thus the associated learning curves.  

    Returns:
    --------
    The matplotlib Figure object with the plotted learning curve data.
    """
    meta_data, model_data = learning_curves

    metric_index = model_data[model][0].index(metric)
    curve = model_data[model][1][metric_index]

    eval_set_count, iteration_count = np.array(curve).shape
    eval_sets = ["train", "val", "test"][:eval_set_count]
    iteration_count += 1

    data = pd.DataFrame({
        "iterations": list(range(1, iteration_count)),
        **{ eval_set : curve[i] for i, eval_set in enumerate(eval_sets) },
    })

    data = data.melt(id_vars='iterations', var_name='Line', value_name='Y')

    plt.ioff()
    fig, ax = plt.subplots()

    sns.lineplot(x='iterations', y='Y', hue='Line', data=data, ax=ax)

    ax.set_title(f'Learning Curves for {model} using {metric} as Eval Metric')
    ax.set_xlabel('iterations')
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)
    plt.ion()

    return fig
