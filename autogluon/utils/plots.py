""" Methods to create various plots used throughout AutoGluon.
    If matplotlib or bokeh are not installed, simply will print warning message that plots cannot be shown.
"""

import warnings
import numpy as np
from collections import OrderedDict
import pandas as pd

__all__ = ['plot_performance_vs_trials', 'plot_summary_of_models', 'mousover_plot']


def plot_performance_vs_trials(results, output_directory, filename="PerformanceVsTrials.png"):
    try:
        import matplotlib.pyplot as plt
        matplotlib_imported = True
    except ImportError:
        matplotlib_imported = False
    
    if not matplotlib_imported:
        warnings.warn('AutoGluon summary plots cannot be created because matplotlib is not installed. Please do: "pip install matplotlib"')
        return None
    
    ordered_val_perfs = [self.results['trial_info'][trial_id]['validation_perf'] for trial_id in ordered_trials]
    x = range(len(ordered_trials))
    y = []
    for i in x:
        y.append(max([ordered_val_perfs[j] for j in range(i)])) # best validation performance in trials up until ith one (assuming higher = better)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Completed Trials', ylabel='Best Validation Performance')
    if output_directory is not None:
        fig.savefig(output_directory + filename)
    plt.show()

def plot_summary_of_models(results, output_directory, save_file='SummaryOfModels.html', plot_title="Models produced during fit()"):
    """ Plot dynamic scatterplot summary of each model encountered during fit(), based on the returned Results object. 
    """
    num_trials = len(results['trial_info'])
    attr_color = None
    attr_size = None
    datadict = {'trial_id': sorted(results['trial_info'].keys())}
    datadict['validation_performance'] = [results['trial_info'][trial_id]['validation_perf'] for trial_id in datadict['trial_id']]
    datadict['hyperparameters'] = [str(results['trial_info'][trial_id]['config']) for trial_id in datadict['trial_id']]
    
    # Determine x-axis attribute:
    if 'latency' in results['metadata']:
        datadict['latency'] = [results['trial_info'][trial_id]['metadata']['latency'] for trial_id in datadict['trial_id']]
        attr_x = 'latency'
    
    # Determine size attribute:
    if 'memory' in results['metadata']:
        datadict['memory'] = [results['trial_info'][trial_id]['metadata']['memory'] for trial_id in datadict['trial_id']]
        attr_size = 'memory'
    
    # Determine color attribute:
    if 'training_loss' in results:
        datadict['training_loss'] = [results['trial_info'][trial_id]['training_loss'] for trial_id in datadict['trial_id']]
        attr_color = 'training_loss'
    
    mousover_plot(datadict, attr_x=attr_x, attr_y='validation_perf', attr_color=attr_color, 
                  attr_size=attr_size, save_file=output_directory+save_file, plot_title=plot_title)


def mousover_plot(datadict, attr_x, attr_y, attr_color=None, attr_size=None, save_file=None, plot_title="",
                  point_transparency = 0.5, point_size=20, default_color="#2222aa", hidden_keys = []):
    """ Produces dynamic scatter plot that can be interacted with by mousing over each point to see its label
        Args:
            datadict (dict): keys contain attributes, values of lists of data from each attribute to plot (each list index corresponds to datapoint).
                             The values of all extra keys in this dict are considered (string) labels to assign to datapoints when they are moused over.
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
    """
    try:
        import warnings
        # suppress deprecation warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import bokeh
        from bokeh.plotting import output_file, ColumnDataSource, show, figure
        from bokeh.models import HoverTool, CategoricalColorMapper, LinearColorMapper, Legend, LegendItem, ColorBar
        from bokeh.palettes import Category20
        # re-enable deprecation warning
        warnings.filterwarnings("default", category=DeprecationWarning)
        bokeh_imported = True
    except ImportError:
        bokeh_imported = False
    
    if not bokeh_imported:
        warnings.warn('AutoGluon summary plots cannot be created because bokeh is not installed. Please do: "pip install bokeh"')
        return None
    
    n = len(datadict[attr_x])
    for key in datadict.keys(): # Check lengths are all the same
        if len(datadict[key]) != n:
            raise ValueError("Key %s in datadict has different length than %s" % (key, attr_x))
    
    attr_x_is_string = any([type(val)==str for val in datadict[attr_x]])
    if attr_x_is_string:
        attr_x_levels = list(set(datadict[attr_x])) # use this to translate between int-indices and x-values
        og_x_vals = datadict[attr_x][:]
        attr_x2 = attr_x + "___"
        hidden_keys.append(attr_x2)
        datadict[attr_x2] = [attr_x_levels.index(category) for category in og_x_vals] # convert to ints
    
    legend = None
    if attr_color is not None:
        attr_color_is_string = any([type(val)==str for val in datadict[attr_color]])
        color_datavals = datadict[attr_color]
        if attr_color_is_string:
            attr_color_levels = list(set(color_datavals))
            colorpalette = Category20[20]
            color_mapper = CategoricalColorMapper(factors=attr_color_levels, palette=[colorpalette[2*i % len(colorpalette)] for i in range(len(attr_color_levels))])
            legend = attr_color
        else:
            color_mapper = LinearColorMapper(palette='Magma256', low=min(datadict[attr_color]), high=max(datadict[attr_color])*1.25)
        default_color = {'field': attr_color, 'transform': color_mapper}
    
    if attr_size is not None: # different size for each point, ensure mean-size == point_size
        attr_size2 = attr_size + "____"
        hidden_keys.append(attr_size2)
        og_sizevals = np.array(datadict[attr_size])
        sizevals = point_size + (og_sizevals - np.mean(og_sizevals))/np.std(og_sizevals) * (point_size/2)
        if np.min(sizevals) < 0:
            sizevals = -np.min(sizevals) + sizevals + 1.0
        datadict[attr_size2] = list(sizevals)
        point_size = attr_size2
    
    if save_file is not None:
        output_file(save_file, title=plot_title)
    source = ColumnDataSource(datadict)
    TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
    p = figure(title=plot_title, tools=TOOLS)
    if attr_x_is_string:
        circ = p.circle(attr_x2, attr_y, line_color=default_color, line_alpha = point_transparency,
                fill_color = default_color, fill_alpha=point_transparency, size=point_size, source=source)
    else:
        circ = p.circle(attr_x, attr_y, line_color=default_color, line_alpha = point_transparency,
                fill_color = default_color, fill_alpha=point_transparency, size=point_size, source=source)
    hover =p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([(key,'@'+key) for key in datadict.keys() if key not in hidden_keys])
    # Format axes:
    p.xaxis.axis_label = attr_x
    p.yaxis.axis_label = attr_y
    if attr_x_is_string: # add x-ticks:
        p.xaxis.ticker = range(len(attr_x_levels))
        p.xaxis.major_label_overrides = {i: attr_x_levels[i] for i in range(len(attr_x_levels))}
    
    # Legend additions:
    if attr_color is not None and attr_color_is_string:
        legend_it = []
        for i in range(len(attr_color_levels)):
            legend_it.append(LegendItem(label=attr_color_levels[i], renderers = [circ], index=datadict[attr_color].index(attr_color_levels[i])))
        legend = Legend(items=legend_it, location=(0, 0))
        p.add_layout(legend, 'right')
    
    if attr_color is not None and not attr_color_is_string: 
        color_bar = ColorBar(color_mapper=color_mapper, title = attr_color, 
                             label_standoff=12, border_line_color=None, location=(0,0))
        p.add_layout(color_bar, 'right')
    
    if attr_size is not None:
        p.add_layout(Legend(items=[LegendItem(label='Size of points based on "'+attr_size + '"')]), 'below')
    
    show(p)

