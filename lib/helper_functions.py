import logging

logger_matplotlib = logging.getLogger('matplotlib')
logger_matplotlib.setLevel(logging.WARNING)

import io
import textwrap

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from PIL import Image, ImageFile
import PIL.Image
import PIL.ImageFile
import seaborn as sns


def _warning(message, category, filename, lineno, file, line):
    '''Make warnings display only the actual message instead of the long (with context) message'''
    # See https://stackoverflow.com/q/2187269/1071459
    warning_text = f'{category.__name__}: {message} ({lineno} in {filename})'
    warning_text = textwrap.fill(warning_text, width=100)
    print(warning_text)
    # print('=' * 20)
    # traceback.print_stack()
    # print('=' * 20)


def gini(values: list | np.ndarray) -> float:
    """Computes the Gini index (or coefficient).

    Parameters
    ----------
    values : list | np.ndarray
        A list with all the values.

    Returns
    -------
    float
        The Gini index value.
    """
    values = np.array(values, dtype=float).flatten()
    n = len(values)
    sorted_values = np.sort(values)
    cumulative_values = np.cumsum(sorted_values)
    return float((n + 1 - 2 * (np.sum(cumulative_values) / np.sum(values))) / n)


def lorenz_plot(values: list | np.ndarray) -> plt.figure:
    """Returns the plot of the Lorenz curve used to compute the Gini index alongsize a y=x curve from 0 to 1.

    Parameters
    ----------
    values : list | np.ndarray
        A list with all the values.

    Returns
    -------
    plt.figure
        The figure.
    """
    values = sorted(values)
    values_cum_pct = np.cumsum(values) / np.sum(values)
    fig = plt.figure(figsize=(5, 3))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1))
    # Add one other number to make the lorenz curve begin at 0,0
    plt.plot(np.linspace(0, 1, len(values_cum_pct) + 1), np.append([0], values_cum_pct), marker='.')
    plt.tight_layout()
    plt.autoscale(tight=True)
    plt.grid(axis='x', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    return fig


def barplot(
    data: pd.DataFrame,
    title: str = None,
    ylabel: str = None,
    xlabel: str = None,
    show_mean: bool = True,
    show_mean_value: bool = True,
    show_values: bool = False,
    colors: str | list[tuple[float, float, float]] | None = None,
) -> plt.figure:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        The data to be plotted. Axis 0 is the metric to be printed and axis 1 are the values for each component.
    title : str
        The title, by default None.
    ylabel : str, optional
        The ylabel, by default None.
    xlabel : str, optional
        The xlabel, by default None.
    show_mean : str, optional
        Wether to show the mean as a line, by default True.
    show : str, optional
        When showing the mean add the value of the mean in the legend, by default None.
    show_values : bool, optional
        Whether to show values in plot or not, by default False.
    colors : str | list[tuple[float, float, float]] | None, optional
        The colors to use for the bars, by default None. If a string is provided, it will be used as a seaborn palette. If a list of tuples is provided, it will be used as the colors for the bars.

    Returns
    -------
    plt.figure
        The figure.
    """
    if data.index.shape[0] != 1:
        raise ValueError('Data index should be of length 1.')

    fig = plt.figure(figsize=(10, 2.25))
    ax = sns.barplot(
        data=data,
        palette=colors,
        saturation=1, # The colors passed are the ones we want, we don't want the saturation to be changed by the default 0.75
    )

    if show_mean:
        _mean = data.mean(axis=1).iloc[0]
        _mean_legend_txt = f'Average = {round(_mean,1)}' if show_mean_value else 'Average'
        plt.axhline(y=_mean, color='lime', label=_mean_legend_txt)
        plt.legend()

    if show_values:
        # This is a hack to give more space to the label, it should probably be done in another way
        plt.ylim(0, data.iloc[0].max() * 1.12)

        for _container in ax.containers:
            ax.bar_label(
                _container,
                fmt='%.1f',
                # label_type='center',
                # padding=-5,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    return fig


def figure_to_image(figure: matplotlib.figure.Figure) -> PIL.ImageFile.ImageFile:
    """Transorms a figure into a PIL Image.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to transform.

    Returns
    -------
    PIL.ImageFile.ImageFile
        The Image.
    """
    # Save figure to an in-memory bytes buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)  # Important: go to the start of the buffer, TODO: verify if necessary

    # Open as PIL Image directly from memory
    return PIL.Image.open(buf)


def png_to_image(png: bytes) -> PIL.ImageFile.ImageFile:
    """Transforms a png as bytes into a PIL Image.

    Parameters
    ----------
    png : bytes
        The png as bytes.

    Returns
    -------
    PIL.ImageFile.ImageFile
        The Image.
    """
    buf_chord = io.BytesIO(png)
    return PIL.Image.open(buf_chord)
