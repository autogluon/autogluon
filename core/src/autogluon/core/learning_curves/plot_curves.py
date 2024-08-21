import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_curves(learning_curves: dict, model: str, metric: str, return_fig: bool = True) -> Figure | None:
    """
    Plots learning curves across all evaluation sets for specified model-metric pairing.

    Parameters:
    -----------
    learning_curves: Tuple[dict, dict]
        Learning curve data from TabularPredictor's learning_curves() method.
        Note that learning_curves returns a Tuple of (curve_metadata, curve_data).
        You should pass the curve_data object here.
    model: str
        The model to plot curves for. Model must be included in TabularPredictor's
        training process, and thus the associated learning curves.
    metric: str
        The metric to plot curves for. Metric must be specified in TabularPredictor's
        training process, and thus the associated learning curves.
    return_fig: bool
        Whether to return the matplotlib Figure object.
        Relevant for working in jupyter environments.
        See sample Usage below.

    Returns:
    --------
    The matplotlib Figure object with the plotted learning curve data.

    Sample Usage:
    -------------
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

    hyperparameters = {
        "GBM": {},
        "XGB": {},
        "NN_TORCH": {},
    }

    params = {
        "metrics": ["f1", "accuracy", "log_loss"],
    }

    predictor = TabularPredictor(label="class", problem_type="binary")
    predictor = predictor.fit(train_data=train_data, test_data=test_data, learning_curves=params, hyperparameters=hyperparameters)
    metadata, curves = predictor.learning_curves()


    from autogluon.core.learning_curves.plot_curves import plot_curves

    # in jupyter environment, simply call function to view graph
    # note that returning the matplotlib figure object in a jupyter env
    # will cause graph to appear twice, so set return_fig to False
    plot_curves(curves, "GBM", "accuracy", return_fig = False)

    # to save figure to path
    path = "plot.png"
    fig.savefig(path)
    """
    _, model_data = learning_curves
    eval_sets, metrics, data = model_data[model]

    metric_index = metrics.index(metric)
    curve = data[metric_index]

    _, iterations = np.array(curve).shape

    data = pd.DataFrame(
        {
            "iterations": list(range(1, iterations + 1)),
            **{eval_set: curve[i] for i, eval_set in enumerate(eval_sets)},
        }
    )

    data = data.melt(id_vars="iterations", var_name="Line", value_name="Y")

    fig, ax = plt.subplots()

    sns.lineplot(x="iterations", y="Y", hue="Line", data=data, ax=ax)

    ax.set_title(f"Learning Curves for {model} using {metric} as Eval Metric")
    ax.set_xlabel("iterations")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)
    plt.show()

    if return_fig:
        return fig
