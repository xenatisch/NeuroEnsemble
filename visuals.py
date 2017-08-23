#!/usr/bin/env python3


# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# Python's library
# ----------------------------------------------------------------------------------------------------------------------
from os import path as os_path
from functools import partial

# ----------------------------------------------------------------------------------------------------------------------
# Third party libraries
# ----------------------------------------------------------------------------------------------------------------------
from numpy import (
    arange, linspace, zeros, float32,
    asarray, array, hstack, newaxis,
    int32, abs
)

from pandas import DataFrame

from matplotlib.pyplot import (
    subplots, subplots_adjust, close as close_fig,
    subplot2grid, figure,
    gcf)
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ColorConverter
import seaborn as sns

from numba import jit

# ----------------------------------------------------------------------------------------------------------------------
# Internal
# ----------------------------------------------------------------------------------------------------------------------
from . import results_directory, figure_settings, colormap
from .utils.utils import *

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

__all__ = [
    'plotter', 'plot_collections', 'plot_heatmap', 'plot_windowed_comparison'
]

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

sns.set(style="ticks")


@jit
def _stack_channel(channel: ListOrTuple, index: int,
                   index_size: int, data: ArrayOrDataFrame) -> array:
    return hstack((index[:index_size, newaxis], data[channel][:, newaxis]))


def _edged_colormap(total=100, colors=None):
    """
    
    :param total: 
    :type total: 
    :return: 
    :rtype: 
    """

    def make_colormap(seq):
        """
        :param seq: A sequence of floats and RGB-tuples. The floats should be increasing
                    and in the interval (0,1).
        :type seq:
        :return: a LinearSegmentedColormap
        :rtype:
        """
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}

        for index, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[index - 1]
                r2, g2, b2 = seq[index + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])

        return LinearSegmentedColormap('CustomMap', cdict)

    c = ColorConverter().to_rgb

    if colors is None:
        colors = (
            'red',
            'forestgreen',
            'darkmagenta',
            'brown',
            'slategray',
            'cornflowerblue',
            'maroon',
            'mediumvioletred',
            'olive',
            'blue',
            'darkgoldenrod',
        )

    colors *= total // len(colors)

    _cm = list()
    for cl, v in zip(colors, linspace(0, 1, len(colors))):
        _cm += [c(cl), v]

    return make_colormap(_cm[:-1])


@logger(start='Generating.', end='saved')
def plotter(data: DataFrame, location: str, freq: int, thresh: ThreshType, save=True):
    # fig_format = 'pdf'  # figure_settings['format']
    extension = 'pdf'  # figure_settings['format']

    amplify, linewidth = .5, 0.01 if extension in ('svg', 'pdf') else 0.5

    if not save:
        linewidth = .6

    low, high = thresh
    title = f'Filtered Data {low}-{high}Hz'
    index_size = data.index.size
    index = linspace(0, index_size / freq, index_size, dtype=float32)

    index_max_squared = index.max() * 2

    sns.set(style='ticks')

    fig_width = 60 if index_max_squared > 60 else index_max_squared
    fig_height = data.columns.size / 4
    fig_kws = dict(
        figsize=(fig_width, fig_height)
    )

    fig, ax = subplots(fig_kw=fig_kws)

    data_min, data_max = data.quantile(0.02).mean(), data.quantile(0.98).mean()

    dr = (data_max - data_min) * (1 / amplify)  # Amplify

    func = partial(_stack_channel, index=index, index_size=index_size, data=data)

    # [::-1] is to display the matrix in the actual order.
    segs = map(func, data.columns[::-1])

    offsets = zeros((data.columns.size, 2), dtype=float32)
    offsets[:, 1] = arange(-1, data.columns.size - 1) * dr

    lines = LineCollection(
        segs,
        offsets=offsets,
        transOffset=None,
        linewidth=linewidth,
        cmap=_edged_colormap()
    )
    lines.set_array(arange(0, data.columns.size))
    ax.add_collection(lines)

    # Display settings:
    # ----------------------------------------------------------
    ax.set_title(title, fontsize=14)

    ax.set_xlim([index.min(), index.max()])
    ax.set_ylim([data_min - dr * 2, (data.columns.size - 1) * dr + data_max])

    ax.set_yticks(offsets[:, 1])

    # [::-1] is to display the matrix in the actual order.
    ax.set_yticklabels(data.columns[::-1], fontsize=12)

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Channels [$\mu V$]')

    subplots_adjust(top=0.98, bottom=.02, left=.08, right=.99)

    ax.set_xticks(arange(0, index.max() + 1))
    ax.xaxis.grid(
        b=True,
        which='major',
        color='lightgray',
        linestyle=':',
        linewidth=1,
        alpha=.3,
        antialiased=True
    )

    subplots_adjust(
        top=0.98,
        bottom=.02,
        left=.08,
        right=.99
    )

    sns.despine(right=True, top=True, bottom=True, left=True)

    location = os_path.join(
        location,
        results_directory['general'],
        title.replace(' ', '_').lower() + '.' + extension
    )

    if save:
        fig.savefig(
            location,
            format=extension,
            bbox_inches='tight',
            dpi=figure_settings['resolution']
        )

        close_fig(fig)
        return True

    return fig, ax


@logger(start='Generating the collection', end='saved')
def plot_collections(shifted_data: tuple, thresh: ThreshType, rate: int, location: str):
    normal, delayed = shifted_data
    low, high = thresh
    fig_format = 'pdf'  # figure_settings['format']

    linewidth = 0.01 if fig_format in ('svg', 'pdf') else 0.5

    name = f'time delay {low}-{high}Hz'
    f_name = name.replace(' ', '_')
    file_name = os_path.join(location, results_directory['general'], f'{f_name}.{fig_format}')

    nrows = int(round(delayed.columns.size / 8))

    delay_size = int32(round(rate / (((low + high) / 2) * 4)))
    sns.set(style='white')

    default_cols = 8
    ncols = default_cols if delayed.columns.size > default_cols else delayed.columns.size

    fig, _axes = subplots(
        nrows=nrows if ncols * nrows == delayed.columns.size else nrows + 1,
        ncols=ncols,
        figsize=[16, 20],
        subplot_kw={'aspect': 'equal', 'adjustable': 'box'}
    )
    fig.suptitle(f'{name.title()}\nDelay = {delay_size}', fontsize=12)
    axes = _axes.ravel()

    for ind, col in enumerate(delayed.columns):
        ax = axes[ind]
        ax.plot(normal[col], delayed[col], lw=linewidth)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('Channel {}'.format(col))

    subplots_adjust(top=0.95, bottom=.01, left=.01, right=.99)

    fig.savefig(
        file_name,
        format=fig_format,
        bbox_inches='tight',
        dpi=figure_settings['resolution']
    )

    close_fig(fig)

    return True


@logger(start='Generating heatmap', end='saved', include=('kind',))
def plot_heatmap(matrix: ArrayOrDataFrame, location: str, thresh: ThreshType, kind: str):
    low, high = thresh
    fig_format = figure_settings['format']

    name = f'{kind} heatmap {low}-{high}Hz'

    f_name = name.replace(' ', '_')
    file_name = os_path.join(location, kind.replace(' ', '_'), f'{f_name}.{fig_format}')

    sns.set(style='ticks')

    fig, ax = subplots()

    # col, ind = matrix.columns, matrix.index
    # matrix_arr = fill_diagonal(matrix.as_matrix(), NaN)
    # matrix = DataFrame(matrix_arr, columns=col, index=ind)

    sns.heatmap(
        matrix,
        square=matrix.shape[0] == matrix.shape[1],
        ax=ax,
        cmap=colormap,
        xticklabels=10,
        yticklabels=10,
        robust=True,
        cbar_kws={'shrink': .5}
    )
    ax.set_title(name.title().replace('_', ' '), fontsize=12)

    subplots_adjust(top=0.98, bottom=.02, left=.06, right=.99)

    fig.savefig(
        file_name,
        format=fig_format,
        bbox_inches='tight',
        dpi=figure_settings['resolution']
    )

    close_fig(fig)

    return True


@logger(start='Generating grids', end='saved')
def plot_windowed_comparison(dt: DataFrame, thresh: ThreshType, rate: int, location: str, columns, corr_lag):
    low, high = thresh
    primary_title = f'Moving Windows - Filter threshold {low}-{high}Hz'
    # matrix, sum_of_medians = results
    moving_mean = DataFrame(dt, columns=columns)

    # Setting up the figure -------------------------------------------------------------

    sns.set(style="ticks")
    fig = figure(figsize=[16, 16])
    fig.suptitle(primary_title, fontsize=14)

    ax1 = subplot2grid((2, 5), (0, 0), colspan=4, rowspan=1)

    # Heatmap ---------------------------------------------------------------------------

    sns.heatmap(
        moving_mean.T,
        cmap=colormap,
        ax=ax1,
        xticklabels=rate * 2,
        yticklabels=10,
        robust=True,
        cbar_kws={
            'orientation': 'horizontal',
            'shrink': 0.20,
            'pad': 0.07
        }
    )
    sns.despine(ax=ax1, right=True, top=True)
    ax1.set_title(
        f'Auto-correlation (lag={corr_lag}) of running window (length={SAMPLING_FREQ})',
        fontsize=12
    )
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=60)

    # Horizontal sum of the heatmap -----------------------------------------------------

    ax2 = subplot2grid((2, 5), (0, 4), colspan=1, rowspan=1)
    sns.heatmap(
        asarray([moving_mean.sum(axis=0)] * 2).T,
        cmap=colormap,
        ax=ax2,
        yticklabels=10,
        xticklabels=False,
        robust=True,
        cbar_kws={
            'orientation': 'horizontal',
            'pad': 0.07
        }
    )
    sns.despine(ax=ax2, right=True, top=True)
    ax2.set_title(
        'Sum of medians\nper channel',
        fontsize=12
    )
    ax2.set_yticklabels([])

    # Vertical sum of the heatmap -------------------------------------------------------

    ax3 = subplot2grid((4, 1), (2, 0), colspan=1, rowspan=1)
    time_delta = linspace(0, max(moving_mean.shape) / rate, max(moving_mean.shape))
    ax3.plot(
        time_delta,
        moving_mean.sum(axis=1).tolist(),
        '--',
        lw=.75,
        label='Normal',
        alpha=.5
    )
    ax3_se = ax3.twinx()
    ax3_se.plot(
        time_delta,
        abs(moving_mean).sum(axis=1).tolist(),
        lw=1,
        label='Modulus'
    )
    ax3.legend(loc='upper left')
    ax3_se.legend(loc='upper right')

    ax3.set_title(
        'Sum of medians per window (second)'.format(*thresh),
        fontsize=12
    )
    ax3.set_xlim(time_delta.min(), time_delta.max())

    # Saving the figure -----------------------------------------------------------------

    path = os_path.join(
        location,
        results_directory['windowed'],
        primary_title + '.' + figure_settings['format']
    )

    fig.savefig(
        path,
        format=figure_settings['format'],
        bbox_inches='tight',
        dpi=figure_settings['resolution']
    )

    close_fig(fig)

    return True


def plot(data: DataFrame, fs: int, ax=None):
    # fig_format = 'pdf'  # figure_settings['format']
    axis = ax is not None
    extension = 'pdf'  # figure_settings['format']

    amplify, linewidth = .5, 0.01 if extension in ('svg', 'pdf') else 0.5

    linewidth = .6

    index_size = data.index.size
    index = linspace(0, index_size / fs, index_size, dtype=float32)

    index_max_squared = index.max() * 2

    sns.set(style='ticks')

    if not axis:
        fig, ax = subplots(
            figsize=(60 if index_max_squared > 60 else index_max_squared, data.columns.size // 4)
        )
        axis = False

    data_min, data_max = data.quantile(0.02).mean(), data.quantile(0.98).mean()

    dr = (data_max - data_min) * (1 / amplify)  # Amplify

    func = partial(_stack_channel, index=index, index_size=index_size, data=data)
    # [::-1] is to display the matrix in the actual order.
    segs = map(func, data.columns[::-1])

    offsets = zeros((data.columns.size, 2), dtype=float32)
    offsets[:, 1] = arange(-1, data.columns.size - 1) * dr

    lines = LineCollection(
        segs,
        offsets=offsets,
        transOffset=None,
        linewidth=linewidth,
        cmap=_edged_colormap()
    )
    lines.set_array(arange(0, data.columns.size))
    ax.add_collection(lines)

    ax.set_xlim([index.min(), index.max()])
    ax.set_ylim([data_min - dr * 2, (data.columns.size - 1) * dr + data_max])

    ax.set_yticks(offsets[:, 1])

    # [::-1] is to display the matrix in the actual order.
    ax.set_yticklabels(data.columns[::-1], fontsize=12)

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Channels [$\mu V$]')

    subplots_adjust(top=0.98, bottom=.02, left=.08, right=.99)

    ax.set_xticks(arange(0, index.max() + 1))
    ax.xaxis.grid(
        b=True,
        which='major',
        color='lightgray',
        linestyle=':',
        linewidth=1,
        alpha=.3,
        antialiased=True
    )

    subplots_adjust(
        top=0.98,
        bottom=.02,
        left=.08,
        right=.99
    )

    sns.despine(right=True, top=True, bottom=True, left=True, ax=ax)

    return ax
