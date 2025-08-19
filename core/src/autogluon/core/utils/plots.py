"""Methods to create various plots used throughout AutoGluon.
If matplotlib or bokeh are not installed, simply will print warning message that plots cannot be shown.
"""

import os
import warnings
from collections import OrderedDict

import numpy as np

from .miscs import warning_filter

__all__ = ["plot_performance_vs_trials", "plot_summary_of_models", "plot_tabular_models", "mousover_plot"]


def plot_performance_vs_trials(
    results, output_directory, save_file="PerformanceVsTrials.png", plot_title="", show_plot=True
):
    try:
        import matplotlib.pyplot as plt

        matplotlib_imported = True
    except ImportError:
        matplotlib_imported = False

    if not matplotlib_imported:
        warnings.warn(
            'AutoGluon summary plots cannot be created because matplotlib is not installed. Please do: "pip install matplotlib"'
        )
        return None

    ordered_trials = sorted(list(results["trial_info"].keys()))
    ordered_val_perfs = [results["trial_info"][trial_id][results["reward_attr"]] for trial_id in ordered_trials]
    x = range(1, len(ordered_trials) + 1)
    y = []
    for i in x:
        y.append(
            max([ordered_val_perfs[j] for j in range(i)])
        )  # best validation performance in trials up until ith one (assuming higher = better)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="Completed Trials", ylabel="Best Performance", title=plot_title)
    if output_directory is not None:
        outputfile = os.path.join(output_directory, save_file)
        fig.savefig(outputfile)
        print("Plot of HPO performance saved to file: %s" % outputfile)
    if show_plot:
        plt.show()


def plot_summary_of_models(
    results,
    output_directory,
    save_file="SummaryOfModels.html",
    plot_title="Models produced during fit()",
    show_plot=True,
):
    """Plot dynamic scatterplot summary of each model encountered during fit(), based on the returned Results object."""
    num_trials = len(results["trial_info"])
    attr_color = None
    attr_size = None
    datadict = {"trial_id": sorted(results["trial_info"].keys())}
    datadict["performance"] = [
        results["trial_info"][trial_id][results["reward_attr"]] for trial_id in datadict["trial_id"]
    ]
    datadict["hyperparameters"] = [
        _formatDict(results["trial_info"][trial_id]["config"]) for trial_id in datadict["trial_id"]
    ]
    hidden_keys = []
    # Determine x-axis attribute:
    if "latency" in results["metadata"]:
        datadict["latency"] = [
            results["trial_info"][trial_id]["metadata"]["latency"] for trial_id in datadict["trial_id"]
        ]
        attr_x = "latency"
    else:
        attr_x = list(results["best_config"].keys())[0]
        datadict[attr_x] = [results["trial_info"][trial_id]["config"][attr_x] for trial_id in datadict["trial_id"]]
        hidden_keys.append(attr_x)
    # Determine size attribute:
    if "memory" in results["metadata"]:
        datadict["memory"] = [
            results["trial_info"][trial_id]["metadata"]["memory"] for trial_id in datadict["trial_id"]
        ]
        attr_size = "memory"

    # Determine color attribute:
    if "training_loss" in results:
        datadict["training_loss"] = [
            results["trial_info"][trial_id]["training_loss"] for trial_id in datadict["trial_id"]
        ]
        attr_color = "training_loss"

    save_path = os.path.join(output_directory, save_file) if output_directory else None
    mousover_plot(
        datadict,
        attr_x=attr_x,
        attr_y="performance",
        attr_color=attr_color,
        attr_size=attr_size,
        save_file=save_path,
        plot_title=plot_title,
        hidden_keys=hidden_keys,
        show_plot=show_plot,
    )
    if save_path is not None:
        print("Plot summary of models saved to file: %s" % save_file)


def plot_tabular_models(
    results,
    output_directory=None,
    save_file="SummaryOfModels.html",
    plot_title="Models produced during fit()",
    show_plot=True,
):
    """Plot dynamic scatterplot of every single model trained during tabular_prediction.fit()
    Args:
        results:
            Dict created during TabularPredictor.fit_summary().
            Must at least contain key: 'model_performance'.
    """
    save_path = os.path.join(output_directory, save_file) if output_directory else None
    hidden_keys = []
    model_performancedict = results["model_performance"]
    model_names = list(model_performancedict.keys())
    val_perfs = [model_performancedict[key] for key in model_names]
    model_types = [results["model_types"][key] for key in model_names]
    hidden_keys.append(model_types)
    model_hyperparams = [_formatDict(results["model_hyperparams"][key]) for key in model_names]
    datadict = {
        "performance": val_perfs,
        "model": model_names,
        "model_type": model_types,
        "hyperparameters": model_hyperparams,
    }
    leaderboard = results["leaderboard"].copy()
    leaderboard["fit_time"] = leaderboard["fit_time"].fillna(0)
    leaderboard["pred_time_val"] = leaderboard["pred_time_val"].fillna(0)

    datadict["inference_latency"] = [
        leaderboard["pred_time_val"][leaderboard["model"] == m].values[0] for m in model_names
    ]
    datadict["training_time"] = [leaderboard["fit_time"][leaderboard["model"] == m].values[0] for m in model_names]
    mousover_plot(
        datadict,
        attr_x="inference_latency",
        attr_y="performance",
        attr_color="model_type",
        save_file=save_path,
        plot_title=plot_title,
        hidden_keys=hidden_keys,
        show_plot=show_plot,
    )


def _formatDict(d):
    """Returns dict as string with HTML new-line tags <br> between key-value pairs."""
    s = ""
    for key in d:
        new_s = str(key) + ": " + str(d[key]) + "<br>"
        s += new_s
    return s[:-4]


def mousover_plot(
    datadict,
    attr_x,
    attr_y,
    attr_color=None,
    attr_size=None,
    save_file=None,
    plot_title="",
    point_transparency=0.5,
    point_size=20,
    default_color="#2222aa",
    hidden_keys=[],
    show_plot=False,
):
    """Produces dynamic scatter plot that can be interacted with by mousing over each point to see its label
    Args:
        datadict (dict): keys contain attributes, values of lists of data from each attribute to plot (each list index corresponds to datapoint).
                         The values of all extra keys in this dict are considered (string) labels to assign to datapoints when they are moused over.
                         Apply _formatDict() to any entries in datadict which are themselves dicts.
        attr_x (str): name of column in dataframe whose values are shown on x-axis (eg. 'latency'). Can be categorical or numeric values
        attr_y (str): name of column in dataframe whose values are shown on y-axis (eg. 'validation performance'). Must be numeric values.
        attr_size (str): name of column in dataframe whose values determine size of dots (eg. 'memory consumption'). Must be numeric values.
        attr_color (str): name of column in dataframe whose values determine color of dots  (eg. one of the hyperparameters). Can be categorical or numeric values
        point_labels (list): list of strings describing the label for each dot (must be in same order as rows of dataframe)
        save_file (str): where to save plot to (html) file (if None, plot is not saved)
        plot_title (str): Title of plot and html file
        point_transparency (float): alpha value of points, lower = more transparent
        point_size (int): size of points, higher = larger
        hidden keys (list[str]): which keys of datadict NOT to show labels for.
        show_plot (bool): whether to show plot
    """
    try:
        with warning_filter():
            import bokeh
            from bokeh.models import CategoricalColorMapper, ColorBar, HoverTool, Legend, LegendItem, LinearColorMapper
            from bokeh.palettes import Category20
            from bokeh.plotting import ColumnDataSource, figure, output_file, save, show
    except ImportError:
        warnings.warn(
            'AutoGluon summary plots cannot be created because bokeh is not installed. To see plots, please do: "pip install bokeh==2.0.1"'
        )
        return None

    n = len(datadict[attr_x])
    for key in datadict.keys():  # Check lengths are all the same
        if len(datadict[key]) != n:
            raise ValueError("Key %s in datadict has different length than %s" % (key, attr_x))

    attr_x_is_string = any([type(val) == str for val in datadict[attr_x]])
    if attr_x_is_string:
        attr_x_levels = list(set(datadict[attr_x]))  # use this to translate between int-indices and x-values
        og_x_vals = datadict[attr_x][:]
        attr_x2 = attr_x + "___"  # this key must not already be in datadict.
        hidden_keys.append(attr_x2)
        datadict[attr_x2] = [attr_x_levels.index(category) for category in og_x_vals]  # convert to ints

    legend = None
    if attr_color is not None:
        attr_color_is_string = any([type(val) == str for val in datadict[attr_color]])
        color_datavals = datadict[attr_color]
        if attr_color_is_string:
            attr_color_levels = list(set(color_datavals))
            colorpalette = Category20[20]
            color_mapper = CategoricalColorMapper(
                factors=attr_color_levels,
                palette=[colorpalette[2 * i % len(colorpalette)] for i in range(len(attr_color_levels))],
            )
            legend = attr_color
        else:
            color_mapper = LinearColorMapper(
                palette="Magma256", low=min(datadict[attr_color]), high=max(datadict[attr_color]) * 1.25
            )
        default_color = {"field": attr_color, "transform": color_mapper}

    if attr_size is not None:  # different size for each point, ensure mean-size == point_size
        attr_size2 = attr_size + "____"
        hidden_keys.append(attr_size2)
        og_sizevals = np.array(datadict[attr_size])
        sizevals = point_size + (og_sizevals - np.mean(og_sizevals)) / np.std(og_sizevals) * (point_size / 2)
        if np.min(sizevals) < 0:
            sizevals = -np.min(sizevals) + sizevals + 1.0
        datadict[attr_size2] = list(sizevals)
        point_size = attr_size2

    if save_file is not None:
        output_file(save_file, title=plot_title)
        print("Plot summary of models saved to file: %s" % save_file)

    source = ColumnDataSource(datadict)
    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,hover,save"
    p = figure(title=plot_title, tools=TOOLS)
    if attr_x_is_string:
        circ = p.circle(
            attr_x2,
            attr_y,
            line_color=default_color,
            line_alpha=point_transparency,
            fill_color=default_color,
            fill_alpha=point_transparency,
            size=point_size,
            source=source,
        )
    else:
        circ = p.circle(
            attr_x,
            attr_y,
            line_color=default_color,
            line_alpha=point_transparency,
            fill_color=default_color,
            fill_alpha=point_transparency,
            size=point_size,
            source=source,
        )
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([(key, "@" + key + "{safe}") for key in datadict.keys() if key not in hidden_keys])
    # Format axes:
    p.xaxis.axis_label = attr_x
    p.yaxis.axis_label = attr_y
    if attr_x_is_string:  # add x-ticks:
        p.xaxis.ticker = list(range(len(attr_x_levels)))
        p.xaxis.major_label_overrides = {i: attr_x_levels[i] for i in range(len(attr_x_levels))}

    # Legend additions:
    if attr_color is not None and attr_color_is_string:
        legend_it = []
        for i in range(len(attr_color_levels)):
            legend_it.append(
                LegendItem(
                    label=attr_color_levels[i],
                    renderers=[circ],
                    index=datadict[attr_color].index(attr_color_levels[i]),
                )
            )
        legend = Legend(items=legend_it, location=(0, 0))
        p.add_layout(legend, "right")

    if attr_color is not None and not attr_color_is_string:
        color_bar = ColorBar(
            color_mapper=color_mapper, title=attr_color, label_standoff=12, border_line_color=None, location=(0, 0)
        )
        p.add_layout(color_bar, "right")

    if attr_size is not None:
        p.add_layout(Legend(items=[LegendItem(label='Size of points based on "' + attr_size + '"')]), "below")

    if show_plot:
        show(p)
    elif save_file is not None:
        save(p)
