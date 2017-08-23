#!/usr/bin/env python3
from matplotlib.colors import ListedColormap
from numba import jit
from matplotlib.gridspec import GridSpec
from numpy import arange, mod, exp, abs, asarray, pi, linspace, rad2deg, angle, real, c_, histogram
from pandas import DataFrame
import seaborn as sns
from matplotlib.pyplot import figure, subplot, subplots, gcf
from scipy.stats.mstats import ks_twosamp
from . import colormap
from .utils import *
from .structure import Signal

pipi = 2 * pi


@jit()
def get_phase_difference(phase_a, phase_b, shift, average=False, as_angle=True):
    res = 1j * (phase_a - shift * phase_b)
    res = exp(res)
    if average:
        if not isinstance(shift, int):
            res = res.mean(axis=1)
        else:
            res = res.mean()
    if as_angle:
        return angle(res)
    return res


@jit()
def phase_core(m, p1, p2):
    arr = arange(0, m, .1).reshape([-1, 1])
    a = asarray([mod(p1, pipi)] * arr.size)
    b = asarray([mod(p2, pipi)] * arr.size)
    return abs(get_phase_difference(a, b, arr, average=True, as_angle=False))


def get_wrapped_phase(phase, shift, degree=False):
    res = mod(shift * phase, pipi)
    if not degree:
        return res - pi
    return rad2deg(res) - 180


class Phase(Signal):
    colors = 'cornflowerblue', 'lightsalmon', 'forestgreen', 'magenta', 'red'

    def __init__(self, signal: DataFrame, fs: int, thresholds: Tuple[Band]):
        super().__init__(signal, fs, signal.columns, signal.index)

        self.thresholds = thresholds

        if not isinstance(thresholds, (tuple, list)):
            raise TypeError(
                f'Expected a tuple of bands for THRESHOLDS, '
                f'got "{type(THRESHOLDS)}" instead.'
            )

        self.filtered_bands = tuple(self.filter_by_freq(band) for band in self.thresholds)

        self.phases = tuple(
            band.hilbert.instantaneous_phase
            for band in self.filtered_bands
        )

    def plot_phases(self, column: str, reference: Band, max_shift: int=10):
        sns.set(style='ticks')

        fig = figure(figsize=[12, 8])
        gs1, gs2 = GridSpec(9, 1), GridSpec(2, 1)
        gs1.update(left=0.1, right=.68, wspace=0.1)
        gs2.update(left=0.75, right=0.98, hspace=0.25)

        ticks_kw = dict(yticks=(), xticks=(), xlim=(0, 1))
        channel_ax = subplot(gs1[0, 0], **ticks_kw)

        phase_kw = ticks_kw.copy()
        ticks_kw.update(dict(yticklabels=()))
        phase_kw.update(dict(yticks=(-pi, 0, pi), ylim=(-pi, pi), yticklabels=('$-\pi$', 0, '$\pi$')))
        axes = tuple(  # (phase, signal)
            (subplot(gs1[item, 0], **phase_kw), subplot(gs1[item + 1, 0], **ticks_kw))
            for item in range(1, 9, 2)
        )

        shifts_ax, box_ax = tuple(subplot(gs2[ind, 0]) for ind in range(2))

        self.data[column][:1].plot(lw=.5, c='k', ax=channel_ax)
        channel_ax.set_ylabel('Signal')
        sns.despine(left=False, right=True, bottom=True, top=True, ax=channel_ax, offset=5)

        iterator = zip(self.phases, self.filtered_bands, self.thresholds, axes)

        for phase, band, th, (phase_ax, signal_ax) in iterator:
            title = th['name']
            band[column][:1].plot(ax=signal_ax, lw=.5, c='k')
            signal_ax.set_ylabel(fr'$\{title}$', fontsize=12, labelpad=35, rotation=0)

            p = mod(phase[column], pipi) - pi
            phase_ax.scatter(phase.index, p, marker='.', s=4, c='k')
            phase_ax.set_ylabel(fr'$\Phi_{{\{title}}}$', fontsize=12, labelpad=5, rotation=0)

            for ax in (signal_ax, phase_ax):
                sns.despine(left=False, right=True, bottom=True, top=True, ax=ax, offset=5)

            if th != self.thresholds[-1]:
                signal_ax.set_xticklabels([])
                continue

            signal_ax.set_xticks([0, 1])
            sns.despine(left=False, right=True, bottom=False, top=True, ax=signal_ax, offset=5)

        mv = DataFrame(index=linspace(0, max_shift, max_shift * 10))

        ref_index = self.thresholds.index(reference)
        ref_name = self.thresholds[ref_index]["name"]

        colors = iter(self.colors)

        for index, phase in enumerate(self.phases):
            if index == ref_index:
                continue

            title = self.thresholds[index]["name"]
            label = fr'$\Phi_{{\{title}}} - \Phi{{\{ref_name}}}$'

            mv[label] = phase_core(max_shift, phase[column], self.phases[ref_index][column])

            mv[label].plot(ax=shifts_ax, c=next(colors), label=label, lw=1)
            shifts_ax.set_xlim(0, max_shift)
            shifts_ax.set_ylim(0, max(mv.max()))
            shifts_ax.legend(fontsize=12)
            shifts_ax.set_yticklabels(['{:.2f}'.format(item) for item in shifts_ax.get_yticks()])
            shifts_ax.set_xticklabels(
                [f'1 : {int(item)}' if item else str() for item in shifts_ax.get_xticks()],
                rotation=-30
            )
            sns.despine(right=True, ax=shifts_ax, offset=5)

        sns.boxplot(
            data=mv,
            ax=box_ax,
            linewidth=2,
            fliersize=3,
            palette=self.colors[:len(self.phases)],
            width=.9
        )
        sns.despine(right=True, ax=box_ax, offset=10)
        box_ax.set_yticklabels([f'{item:.2f}' for item in box_ax.get_yticks()])

        return fig

    def find_optimal_shifts(self, column, reference, max_shift=100):
        ref_index = self.thresholds.index(reference)
        shifts = dict()
        for ind, phase in enumerate(self.phases):
            if ind == ref_index:
                continue
            s = phase_core(max_shift, self.phases[ind][column], self.phases[ref_index][column])
            s_max = s[1:].max()  # Exclude zero.
            # There are 10 items per period.
            shifts[self.thresholds[ind]['name']] = list(s).index(s_max) / 10

        return shifts

    def plot_best_shift(self, column: str, reference: Band, max_shift: int=100):
        t = linspace(0, self.phases[0][column].size / self.fs, self.phases[0][column].size)
        ref_index = self.thresholds.index(reference)
        ref_name = self.thresholds[ref_index]["name"]

        shifts = self.find_optimal_shifts(column, reference, max_shift)

        sns.set(style='ticks')
        figure(figsize=[13, 6])

        thresh_len = len(self.thresholds)
        # This will be rounded up Don't use // instead of /.
        col_len = int(thresh_len / 2)

        gs1 = GridSpec(thresh_len, thresh_len + 2)
        gs1.update(left=0.1, right=.99, wspace=.2, hspace=.3)

        axes = tuple(subplot(gs1[ind, :col_len]) for ind in range(thresh_len))
        axes2_norm_kws = dict(adjustable='box-forced', xlim=(0, 1), ylim=(-pi, pi))
        axes2_polar_kws = dict(polar=True)
        axes2 = (
            (
                subplot(gs1[:col_len, item], yticks=(), xticks=(), **axes2_norm_kws),
                subplot(gs1[col_len:, item], **axes2_polar_kws)
            ) for item in range(2, thresh_len + 1)
        )
        phasecore_ax, box_ax = subplot(gs1[:col_len, -1], xticks=()), subplot(gs1[col_len:, -1])

        for ind, (ax, th) in enumerate(zip(axes, self.thresholds)):
            title = th['name']
            shift = shifts.get(title, 1)

            ax.scatter(t, rad2deg(mod(shift * self.phases[ind][column], pipi)) - 180, marker='.', s=5, c='k')
            ax.set_xlim(0, 1)
            ax.set_ylabel(fr'${shift} \times \Phi_{{\{title}}}$', fontsize=12, labelpad=1)
            ax.set_yticks([-180, 0, 180])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
            sns.despine(left=False, right=True, bottom=True, top=True, ax=ax, offset=5)

        max_shift: float = max(shifts.values())
        max_shift += 2
        min_shift: float = max((min(shifts.values()), 0))
        df = DataFrame(index=arange(0, max_shift, .1))

        for ind, th in enumerate(self.thresholds):
            if ind == ref_index:
                continue

            title = th['name']
            shift: float = shifts.get(title, 1)

            phase_ax, polar_ax = next(axes2)

            phasediff = get_phase_difference(self.phases[ind][column], self.phases[ref_index][column], shift)

            phase_ax.scatter(t, rad2deg(phasediff), marker='.', s=5, c='k')
            phase_ax.set_yticks([-180, 0, 180])
            phase_ax.set_yticks([])
            phase_ax.set_xticks([])

            d = ks_twosamp(
                self.phases[ind][column],
                shift * self.phases[ref_index][column],
                'two-sided'
            )

            phase_ax_ttl = fr'$\Delta\Phi_{{1:{shift}}} = \Phi_{{\{title}}} - {shift}\Phi_{{\{ref_name}}}$'
            phase_ax.set_title(phase_ax_ttl + f'\n$D_{{n, m}} = {d[0]:.3f}$')

            label = fr'$\Phi_{{\{title}}}-{shift}\Phi_{{\{ref_name}}}$'
            df[label] = phase_core(max_shift, self.phases[ind][column], shift * self.phases[ref_index][column])

            r, phi = histogram(phasediff + pi, bins=20)
            theta = c_[phi[:-1], phi[1:]].mean(axis=1)
            phi_probability = (2 * r) / r.sum()

            mean_angle = angle(exp(1j * (self.phases[ind][column] - shift * self.phases[ref_index][column])).mean())
            r_mean = abs(exp(1j * (self.phases[ind][column] - shift * self.phases[ref_index][column])).mean()) + pi
            zm = r_mean * exp(1j * mean_angle) * max(phi_probability)
            polar_ax.plot([0, real(zm)], [0, 1], lw=1, c='r', alpha=.7)

            polar_ax.bar(theta, phi_probability, width=.2, alpha=.7)
            polar_ax.set_yticks([])
            polar_ax.set_xticks([0, pi / 2, pi, (3 * pi) / 2])
            polar_ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
            polar_ax.set_ylim(0, .2)

        y_ticks = 0, max(df.max())
        y_labels = [f'{tick:.2f}' for tick in box_ax.get_yticks()]

        sns.boxplot(data=df, ax=box_ax, palette=self.colors, fliersize=0, linewidth=1, width=.9)
        sns.despine(left=True, right=False, bottom=True, top=True, ax=box_ax)
        box_ax.set_xticks([])
        box_ax.set_yticks(y_ticks)
        box_ax.set_yticklabels(y_labels)

        df.plot(ax=phasecore_ax, lw=1, color=self.colors[:df.columns.size])
        phasecore_ax.set_xticklabels(['1 : %d' % item if item else '' for item in phasecore_ax.get_xticks()])
        sns.despine(left=True, right=False, bottom=False, top=True, ax=phasecore_ax)
        phasecore_ax.set_yticks(y_ticks)
        phasecore_ax.set_yticklabels(y_labels)
        phasecore_ax.set_xlim(min_shift, max_shift+2)

    def plot_phase_shifts(self, column: str, reference: Band, shift: int, max_shift=10):
        t = linspace(0, self.phases[0][column].size / self.fs, self.phases[0][column].size)
        ref_index = self.thresholds.index(reference)
        ref_name = self.thresholds[ref_index]["name"]

        sns.set(style='ticks')
        figure(figsize=[13, 6])

        thresh_len = len(self.thresholds)
        # This will be rounded up Don't use // instead of /.
        col_len = int(thresh_len / 2)

        gs1 = GridSpec(thresh_len, thresh_len + 2)
        gs1.update(left=0.1, right=.99, wspace=.2, hspace=.3)

        axes = tuple(subplot(gs1[ind, :col_len]) for ind in range(thresh_len))
        axes2_norm_kws = dict(adjustable='box-forced', xlim=(0, 1), ylim=(-pi, pi))
        axes2_polar_kws = dict(polar=True)
        axes2 = (
            (
                subplot(gs1[:col_len, item], yticks=(), xticks=(), **axes2_norm_kws),
                subplot(gs1[col_len:, item], **axes2_polar_kws)
            ) for item in range(2, thresh_len + 1)
        )
        phasecore_ax, box_ax = subplot(gs1[:col_len, -1], xticks=()), subplot(gs1[col_len:, -1])

        for ind, (ax, th) in enumerate(zip(axes, self.thresholds)):
            title = th['name']
            ax.scatter(t, rad2deg(mod(shift * self.phases[ind][column], pipi)) - 180, marker='.', s=5, c='k')
            ax.set_xlim(0, 1)
            ax.set_ylabel(fr'${shift} \times \Phi_{{\{title}}}$', fontsize=12, labelpad=1)
            ax.set_yticks([-180, 0, 180])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
            sns.despine(left=False, right=True, bottom=True, top=True, ax=ax, offset=5)

        df = DataFrame(index=linspace(0, max_shift, max_shift * 10))

        for ind, th in enumerate(self.thresholds):
            if ind == ref_index:
                continue

            title = th['name']
            phase_ax, polar_ax = next(axes2)

            # phasediff = angle(exp(1j * (self.phases[ind][column] - shift * self.phases[ref_index][column])))
            phasediff = get_phase_difference(self.phases[ind][column], self.phases[ref_index][column], shift)

            phase_ax.scatter(t, rad2deg(phasediff), marker='.', s=5, c='k')
            phase_ax.set_yticks([-180, 0, 180])
            phase_ax.set_yticks([])
            phase_ax.set_xticks([])

            d = ks_twosamp(
                self.phases[ind][column],
                shift * self.phases[ref_index][column],
                'two-sided'
            )

            phase_ax_ttl = fr'$\Delta\Phi_{{1:{shift}}} = \Phi_{{\{title}}} - {shift}\Phi_{{\{ref_name}}}$'
            phase_ax.set_title(phase_ax_ttl + f'\n$D_{{n, m}} = {d[0]:.3f}$')

            label = fr'$\Phi_{{\{title}}}-{shift}\Phi_{{\{ref_name}}}$'
            df[label] = phase_core(max_shift, self.phases[ind][column], shift * self.phases[ref_index][column])

            r, phi = histogram(phasediff + pi, bins=20)
            theta = c_[phi[:-1], phi[1:]].mean(axis=1)
            phi_probability = (2 * r) / r.sum()

            mean_angle = angle(exp(1j * (self.phases[ind][column] - shift * self.phases[ref_index][column])).mean())
            r_mean = abs(exp(1j * (self.phases[ind][column] - shift * self.phases[ref_index][column])).mean()) + pi
            zm = r_mean * exp(1j * mean_angle) * max(phi_probability)
            polar_ax.plot([0, real(zm)], [0, 1], lw=1, c='r', alpha=.7)

            polar_ax.bar(theta, phi_probability, width=.2, alpha=.7)
            polar_ax.set_yticks([])
            polar_ax.set_xticks([0, pi / 2, pi, (3 * pi) / 2])
            polar_ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
            polar_ax.set_ylim(0, .2)

        y_ticks = 0, max(df.max())
        y_labels = [f'{tick:.2f}' for tick in box_ax.get_yticks()]

        sns.boxplot(data=df, ax=box_ax, palette=self.colors, fliersize=0, linewidth=1, width=.9)
        sns.despine(left=True, right=False, bottom=True, top=True, ax=box_ax)
        box_ax.set_xticks([])
        box_ax.set_yticks(y_ticks)
        box_ax.set_yticklabels(y_labels)

        df.plot(ax=phasecore_ax, lw=1, color=self.colors[:df.columns.size])
        phasecore_ax.set_xticklabels(['1 : %d' % item if item else '' for item in phasecore_ax.get_xticks()])
        sns.despine(left=True, right=False, bottom=False, top=True, ax=phasecore_ax)
        phasecore_ax.set_yticks(y_ticks)
        phasecore_ax.set_yticklabels(y_labels)

    def plot_hist(self, channel, reference):
        ref_index = self.thresholds.index(reference)
        ref_name = self.thresholds[ref_index]["name"]
        shifts = self.find_optimal_shifts(channel, reference)

        sns.set(style='white')
        fig, fig_axes = subplots(nrows=2, ncols=len(self.thresholds)-1, figsize=(14, 7), sharex=True, sharey=True)
        fig.suptitle(fr'Reference: $\{ref_name}$')

        axes = iter(fig_axes.T)
        for index, thresh in enumerate(self.thresholds):
            if ref_index == index:
                continue

            band = thresh['name']
            shift = shifts[band]

            y_ind = self.thresholds.index(thresh)
            phasediff = get_phase_difference(self.phases[y_ind][channel], self.phases[ref_index][channel], shift)
            phase_x = mod(self.phases[ref_index][channel], pipi) - pi
            phase_y = mod(self.phases[y_ind][channel], pipi) - pi

            axes_repo = next(axes)
            sns.kdeplot(phase_y, phase_x, kind="kde", size=5, space=0, shade=True, cmap='gray_r', ax=axes_repo[0])
            sns.kdeplot(phase_y, phase_x, kind="kde", size=5, space=0, cmap='gray_r', ax=axes_repo[0])
            axes_repo[0].set_xlabel(fr'$\Phi_{{\{band}}}$')

            sns.kdeplot(phasediff, shift * phase_x, kind="kde", size=5, space=0, shade=True, cmap='gray_r', ax=axes_repo[1])
            sns.kdeplot(phasediff, shift * phase_x, kind="kde", size=5, space=0, cmap='gray_r', ax=axes_repo[1])
            axes_repo[1].set_xlabel(fr'$\Phi_{{\{band}}} - {shift:.2g}\Phi_{{\{ref_name}}}$')

            for ax in axes_repo:
                ax.set_xlim([-pi, pi])
                ax.set_ylim([-pi, pi])
                ax.set_yticks([-pi, 0, pi])
                ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
                ax.set_xticks([-pi, 0, pi])
                ax.set_xticklabels(['$-\pi$', 0, '$\pi$'])
                ax.set_ylabel(fr'$\Phi_{{\{ref_name}}}$')

    def plot_shift_heatmap(self, max_shift=20):
        sns.set(style='white')
        fig, ax = subplots(
            figsize=[(len(self.thresholds) - 1) * 3, len(self.thresholds) * 3],
            ncols=len(self.thresholds) - 1,
            nrows=len(self.thresholds),
            sharex=True,
            sharey=True
        )

        axes_col = iter(ax)

        for reference in self.thresholds:
            ref_index = self.thresholds.index(reference)
            ref_name = self.thresholds[ref_index]["name"]

            axes_rows = iter(next(axes_col))

            for index, thresh in enumerate(self.thresholds):
                if ref_index == index:
                    continue

                ax = next(axes_rows)
                band = thresh['name']
                mv = DataFrame(
                    (phase_core(max_shift, self.phases[index][ch], self.phases[ref_index][ch]) for ch in self.data.columns),
                    index=self.data.columns,
                    columns=linspace(0, max_shift, max_shift*10)
                )

                ax.imshow(mv, aspect='auto', cmap='gist_gray_r')
                ax.set_xticks(arange(0, max_shift * 10, 5 * 10))
                ax.set_xticklabels([f'{val//10}' for val in ax.get_xticks()])
                ax.set_title(fr'Reference: $\{ref_name}$ v $\{band}$')

    def plot_spectrogram(self, channel, reference):
        ref_index = self.thresholds.index(reference)
        ref_name = self.thresholds[ref_index]["name"]
        shifts = self.find_optimal_shifts(channel, reference)

        sns.set(style='white')
        fig, fig_axes = subplots(nrows=len(self.thresholds)-1, figsize=(16, 28), sharex=True, sharey=True)
        # fig.suptitle(fr'Reference: $\{ref_name}$')

        axes = iter(fig_axes.ravel())
        for index, thresh in enumerate(self.thresholds):
            if ref_index == index:
                continue

            band = thresh['name']
            shift = shifts[band]

            y_ind = self.thresholds.index(thresh)
            phasediff = get_phase_difference(self.phases[y_ind][channel], self.phases[ref_index][channel], shift)
            # phase_x = mod(shift * self.phases[ref_index][channel], pipi) - pi
            # phase_y = mod(shift * self.phases[y_ind][channel], pipi) - pi

            axes_repo = next(axes)
            # spec, freqs, t, cax = axes_repo.specgram(phasediff, Fs=self.SAMPLING_FREQ, cmap=colormap)
            # f, t, Sxx = spectrogram(phasediff, SAMPLING_FREQ=self.SAMPLING_FREQ)
            spec, freqs, t, cax = axes_repo.specgram(phasediff, Fs=self.fs, cmap=colormap)
            # cax = axes_repo.pcolormesh(t, f, Sxx, cmap=colormap)
            axes_repo.set_xlabel('Time')
            axes_repo.set_ylabel('Frequency')

            fig.colorbar(cax, ax=axes_repo)
            # sns.kdeplot(phase_y, phase_x, kind="kde", size=5, space=0, shade=True, cmap='gray_r', ax=axes_repo[0])
            # sns.kdeplot(phase_y, phase_x, kind="kde", size=5, space=0, cmap='gray_r', ax=axes_repo[0])
            # axes_repo[0].set_xlabel(fr'$\Phi_{{\{band}}}$')

            # sns.kdeplot(phasediff, phase_x, kind="kde", size=5, space=0, shade=True, cmap='gray_r', ax=axes_repo[1])
            # sns.kdeplot(phasediff, phase_x, kind="kde", size=5, space=0, cmap='gray_r', ax=axes_repo[1])
            axes_repo.set_title(fr'$\Phi_{{\{band}}} - {shift:.2g}\Phi_{{\{ref_name}}}$')

            # for ax in axes_repo:
            #     ax.set_xlim([-pi, pi])
            #     ax.set_ylim([-pi, pi])
            #     ax.set_yticks([-pi, 0, pi])
            #     ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
            #     ax.set_xticks([-pi, 0, pi])
            #     ax.set_xticklabels(['$-\pi$', 0, '$\pi$'])
            #     ax.set_ylabel(fr'$\Phi_{{\{ref_name}}}$')

    def get_phases(self, as_mod=True):
        """
        
        :return: 
        :rtype: 
        """
        res = dict()

        for index, thresh in enumerate(self.thresholds):
            th_name = thresh['name']

            res[th_name] = (self.phases[index] % pipi) - pi if as_mod else self.phases[index]

        return res

    def plot_phase_heatmap(self, digitize: bool=False, title: str=str(),
                           cmap=None, offset: RealNumber=1, span: bool=True,
                           span_color: str='red', span_alpha=0.15):
        fig = gcf()

        gs = GridSpec(nrows=1, ncols=len(self.thresholds))
        phases = self.get_phases()

        im, axes = None, dict()
        hmap_cmap, dig = cmap or 'gray_r', cmap or str()

        if digitize and not cmap:
            hmap_cmap = ListedColormap(['white', 'lightgray', 'darkgray', 'black'])
            dig = ' | Digitized'

        for index, th in enumerate(self.thresholds):
            band, thresh = th['name'], th['thresh']

            phase = phases[band].data
            channels = list(phase.columns)

            im_kws = {
                'aspect': 'auto',
                'extent': (min(phase.index), max(phase.index), min(phase.shape), 0)
            }

            if not index:
                ax = subplot(
                    gs[:, index],
                    yticks=arange(0, len(channels), 5),
                    ylabel=f'{title}{dig}',
                    xlim=im_kws['extent'][:2]
                )

            else:
                t_m = phase.index.size / asarray(thresh).mean()
                t_m /= self.fs
                t_m += offset
                phase = phase.ix[offset:t_m]

                im_kws['extent'] = min(phase.index), max(phase.index), min(phase.shape), 0

                last_band = self.thresholds[index - 1]['name']

                ax = subplot(
                    gs[:, index],
                    yticks=tuple(),
                    yticklabels=tuple(),
                    xlim=im_kws['extent'][:2]
                )

                if span:
                    axes[last_band].axvspan(offset, t_m, alpha=span_alpha, color=span_color)

            axes[band] = ax

            ax.set_title(fr'$\{band}$')

            im = ax.imshow(
                phase.T,
                vmin=-pi,
                vmax=pi,
                cmap=hmap_cmap,
                **im_kws
            )

            if not index:
                ticks = tuple(str(channels[ind]).strip() for ind in ax.get_yticks().astype(int))
                ax.set_yticklabels(ticks, fontsize=8)

        cax = fig.add_axes([.92, .2, .01, .6])
        cbar = fig.colorbar(im, cax, ticks=[-pi, 0, pi])
        cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
        axes['cbar'] = cax

        return axes
