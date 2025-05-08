import logging

pltlogger = logging.getLogger('matplotlib')
pltlogger.setLevel(logging.WARNING)

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make warnings display only the actual message instead of the long (with context) message
# See https://stackoverflow.com/q/2187269/1071459
def _warning(message, category, filename, lineno, file, line):
    warning_text = f'{category.__name__}: {message} ({lineno} in {filename})'
    warning_text = textwrap.fill(warning_text, width=100)
    print(warning_text)
    # print('=' * 20)
    # traceback.print_stack()
    # print('=' * 20)


def gini(values: list | np.ndarray):
    values = np.array(values, dtype=float).flatten()
    n = len(values)
    sorted_values = np.sort(values)
    cumulative_values = np.cumsum(sorted_values)
    return float((n + 1 - 2 * (np.sum(cumulative_values) / np.sum(values))) / n)


def gini_plot(values: list | np.ndarray):
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


def barplot(data: pd.DataFrame, title: str, ylabel: str, xlabel: str):
    pltlogger = logging.getLogger('matplotlib')
    pltlogger.setLevel(logging.WARNING)

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data,
        palette='viridis',
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    # plt.show()
    # fig.show()

    return fig
