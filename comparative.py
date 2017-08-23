#!/usr/bin/env python3

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot, gca
import seaborn as sns

from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
from numpy import linspace, arange, nanmedian, asarray, percentile, NaN
from pandas import DataFrame

from skimage import exposure
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from collections import namedtuple


def _adjust_contrast(im, lower, upper):
    v_min, v_max = percentile(im, (lower, upper))
    resp = exposure.rescale_intensity(im, in_range=(v_min, v_max))
    return resp


def _cluster(X, n_clusters=2, n_neighbors=8):
    X[X == 1] = 0
    knn_graph = kneighbors_graph(X, n_neighbors, include_self=False)

    linkage = 'ward'
    model = AgglomerativeClustering(
        linkage=linkage,
        connectivity=knn_graph,
        n_clusters=n_clusters
    )

    model.fit(X)

    dd = DataFrame()
    dd['X0'] = X[:, 0]
    dd['X1'] = X[:, 1]
    dd['columns_list'] = model.labels_

    return dd


def _smooth_line(y):
    t = arange(0, len(y))
    new_t = linspace(t.min(), t.max(), len(t) * 100)
    spline = make_interp_spline(t, y)
    return spline(new_t)


template = namedtuple(
    'data',
    ['c', 'title', 'ax', 'upper_ax', 'func', 'hist_label']
)


def plot_comparatives(data: DataFrame):
    columns = data.columns

    fig = figure(figsize=[14, 7])

    gs = GridSpec(3, 5, width_ratios=[1.5, 1.5, 1.5, .06, 2.5], height_ratios=[.3, 1, 1])
    gs.update(left=0.05, right=0.95, top=.95, wspace=0.3, hspace=0)

    hmap_original = subplot(gs[1, 0], xticks=[])
    hmap_original_x = subplot(gs[0, 0], yticks=[], xticks=[])

    hmap_blurred = subplot(gs[1, 1], xticks=[], yticks=[])
    hmap_blurred_x = subplot(gs[0, 1], yticks=[], xticks=[])

    hmap_contrast = subplot(gs[1, 2], xticks=[], yticks=[])
    hmap_contrast_x = subplot(gs[0, 2], yticks=[], xticks=[])

    hmap_cbar = subplot(gs[1, 3])

    gs_s = GridSpec(3, 2, width_ratios=[5, 2], height_ratios=[.5, 1, 1])
    gs_s.update(left=0.05, right=0.95, top=.95, wspace=0.5, hspace=0.3)
    hist = subplot(gs_s[0, 1], yticks=[])
    scat = subplot(gs_s[1, 1], xticks=[], yticks=[])

    gs2 = GridSpec(1, 2)
    gs2.update(left=0.05, right=0.95, top=.4, wspace=0.05)

    box1 = subplot(gs2[0, 0], xticks=[])
    box2 = subplot(gs2[0, 1], yticks=[], xticks=[])

    # ------------------------------------------------------------------

    functions = (
        template(
            'r',
            'Modulus of medians for\n2 sec windows Kendall Tau',
            hmap_original,
            hmap_original_x,
            lambda x: x,
            'original'
        ),
        template(
            'g',
            'Modulus of Gaussian for\ncorrelation matrix',
            hmap_blurred,
            hmap_blurred_x,
            lambda x: gaussian_filter(x, sigma=(1, 1), order=0),
            'gaussian'
        ),
        template(
            'b',
            'Modulus of Gaussian\nwith increased contrast',
            hmap_contrast,
            hmap_contrast_x,
            lambda x: _adjust_contrast(x, lower=10, upper=90),
            'high contrast'
        ),
    )

    dm = list()
    for start in data.index[::512]:
        m = data[:][start:start + 512].corr('kendall').abs()
        dm.append(m.as_matrix())

    dm = asarray(dm)
    dm = nanmedian(dm, axis=0)

    hmap_kws = dict(
        xticklabels=10,
        yticklabels=10,
        square=True,
        vmin=0,
        vmax=1
    )

    # HEATMAPS ----------------------------------------------

    dfm = DataFrame(dm, columns=columns, index=columns).abs()
    results = dict()

    for index, item in enumerate(functions):
        dfm = DataFrame(item.func(dfm.as_matrix()), columns=columns, index=columns).abs()
        results[item.hist_label] = dfm.abs()

        sns.heatmap(dfm, ax=item.ax, cbar_ax=hmap_cbar, **hmap_kws)
        item.ax.set_yticklabels(item.ax.get_yticklabels(), rotation=60, fontsize=8)
        item.ax.set_xticklabels(item.ax.get_xticklabels(), rotation=30, fontsize=8)

        item.upper_ax.set_title(item.title)
        sum_dt = _smooth_line(dfm.as_matrix().sum(axis=0))
        item.upper_ax.plot(sum_dt)
        item.upper_ax.set_xlim(0, sum_dt.size)

        d_nan = dfm.abs().copy()
        d_nan[d_nan == 1] = NaN
        sns.kdeplot(d_nan.as_matrix().ravel(), ax=hist, c=item.c, label=item.hist_label, lw=0.8)
        hist.set_xticks([0])
        hist.legend(fontsize=8)

        if index == 0:
            item.upper_ax.set_ylabel('Sum', fontsize=8)
            continue

        item.ax.set_yticks([])
    else:
        hist.set_title('Distributions of the medians\nof 2 sec Kendall kendall')
        sns.despine(left=True, right=True, top=True, bottom=False, offset=5, ax=hist)

        hmap_cbar.set_aspect(10)

    # BOX PLOTS -------------------------------------

    dfm = results['original']
    dfm[dfm == 1] = 0
    sns.boxplot(data=results['original'], ax=box1, linewidth=0.5, fliersize=3)
    box1.set_title('Distribution of each channel (original)')
    sns.despine(left=False, right=True, top=True, bottom=True, ax=box1, offset=5)
    box1.set_yticks([0, 1])
    box1.set_yticklabels([0, 1])
    box1.set_xticks([])
    box1.set_xlabel('Channels', fontsize=8)
    box1.set_ylim(0, 1)

    dfm = DataFrame(dm, columns=columns, index=columns).abs()
    dfm[dfm == 1] = 0
    sns.boxplot(data=results['high contrast'], ax=box2, linewidth=0.5, fliersize=3)
    sns.despine(left=False, right=True, top=True, bottom=True, ax=box2, offset=5)
    box2.set_title('Distribution of each channel (high contrast)')
    box2.set_yticks([0, 1])
    box2.set_yticklabels([])
    box2.set_xticks([])
    box2.set_xlabel('Channels', fontsize=8)
    box2.set_ylim(0, 1)

    dfm = results['high contrast']
    dfm[dfm == 1] = 0
    clustered_dt = _cluster(dfm.as_matrix())
    scat.scatter(clustered_dt['X0'], clustered_dt['X1'], c=clustered_dt['columns_list'], cmap='spectral', s=15)
    scat.set_title('Ward linkage')
    scat.set_xticks([])
    scat.set_yticks([])

    return fig


