#!/usr/bin/env python3

from collections import OrderedDict
from functools import wraps, total_ordering
from inspect import isfunction
from typing import TYPE_CHECKING, Callable

from numpy import linspace, asarray, floor, ones, array_equal, where, ones_like
from pandas import DataFrame, Series

from .utils.utils import *


def constructor(func):
    @wraps(func)
    def wrapper(klass, *args, **kwargs):
        res = func(klass, *args, **kwargs)

        if isinstance(res, DataFrame):
            try:
                if list(res.index) == list(res.columns):
                    return Signal(res, klass.fs, res.columns, res.index)
                return Signal(res, klass.fs, klass.channels, klass.time)
            except ValueError:
                if array_equal(res.columns, klass.columns):
                    cols = klass.columns
                else:
                    cols = res.columns
                return Signal(res, klass.fs, cols, res.index)
        return res
    return wrapper


class ConstructorMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases):
        return OrderedDict()

    def __new__(mcs, name, bases, namespace, **kwargs):
        for key, value in namespace.items():
            if isfunction(value) and key != '__init__':
                namespace[key] = constructor(value)

        return type.__new__(mcs, name, bases, dict(namespace))


@total_ordering
class Signal(metaclass=ConstructorMeta):
    """
    
    """

    data: DataFrame = DataFrame()

    def __init__(self, signal: ArrayOrDataFrame, fs: int, channels: Iterable = None, time: Iterable = None):
        """
        
        :param signal: 
        :type signal: 
        :param fs: 
        :type fs: 
        :param channels: 
        :type channels: 
        :param time: 
        :type time: 
        """

        self.data: DataFrame = signal if isinstance(signal, DataFrame) else DataFrame(signal)

        self.fs = fs
        self._hilbert = None

        if channels is not None:
            self.data.columns = channels

        ind_len = max(signal.shape)
        self.data.index = time if time is not None else linspace(0, ind_len / fs, ind_len)

    @classmethod
    def from_edf(cls, path: str, filter_func: Callable=None,
                 min_t: RealNumber=0, max_t: RealNumberOrNone=None, all_channels=False):
        """
        
        :param path: 
        :type path: 
        :param filter_func: 
        :type filter_func: 
        :param min_t: 
        :type min_t: 
        :param max_t: 
        :type max_t: 
        :param all_channels:
        :type all_channels:
        :return: 
        :rtype: 
        """
        from re import compile
        from mne.io.edf.edf import RawEDF

        edf = RawEDF(path, preload=False, montage=None, stim_channel=None, verbose=False)
        edf_channels = edf.ch_names
        ch_names, label_filter = list(), ones_like(edf_channels, dtype=bool)
        pattern = compile(r'(\w+)?.(\w+\d+)?-?(\w+)?')

        _filter_func = lambda x: True

        filter_func = filter_func or _filter_func

        for index, item in enumerate(edf_channels):
            found = pattern.findall(item)[0]

            full_name = None

            if all_channels:
                full_name = f'{found[1].strip()} {found[2].strip() or found[3].strip()}'
            elif found[1] and found[0].lower() in ('eeg', 'ecog', 'meg'):
                full_name = found[1].strip()

            if not full_name or not filter_func(full_name):
                label_filter[index] = False
            else:
                ch_names.append(full_name)

        fs = edf.info['sfreq']

        data, times = edf.get_data(
            picks=where(label_filter)[0],
            start=min_t,
            stop=max_t,
            return_times=True
        )

        return cls(data.T, time=times, fs=fs, channels=ch_names)

    @classmethod
    def from_txt(cls, path: str, fs: int, max_t=None, filter_func: Callable=None):
        """
        
        :param path: 
        :type path: 
        :param fs: 
        :type fs: 
        :param headings: 
        :type headings: 
        :return: 
        :rtype: 
        """
        from pandas import read_csv

        _filter_func = lambda x: True
        filter_func = filter_func or _filter_func

        dt = read_csv(path, sep='\t')
        dt.columns = [c.replace(' ', '_') for c in dt.columns]
        cols = [c for c in dt.columns if filter_func(c)]
        dt = dt[cols][:max_t or dt.index.max()]

        return cls(dt, fs, channels=cols)

    @classmethod
    def from_hdf(cls, path: str, key: str, fs: int):
        """
        
        :param path: 
        :type path: 
        :param key: 
        :type key: 
        :param fs: 
        :type fs: 
        :return: 
        :rtype: 
        """
        from .analyser import _from_hdf5
        dt = _from_hdf5(path, key)[key]
        return cls(dt.compute(optimize_graph=True), fs)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.data, item)

    def __len__(self):
        return self.data.size

    def __call__(self):
        return self.data

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.to_string()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        return self.data.__delitem__(key)

    def __getstate__(self):
        return self.data.__getstate__()

    def __setstate__(self, state):
        return self.data.__setstate__(state)

    def __invert__(self):
        return self.data.__invert__()

    def __iter__(self):
        return self.data.__iter__()

    def __reduce__(self):
        return self.data.__reduce__()

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __add__(self, other):
        return self.data + other

    def __iadd__(self, other):
        return self.data + other

    def __sub__(self, other):
        return self.data - other

    def __isub__(self, other):
        return self.data - other

    def __mul__(self, other):
        return self.data * other

    def __imul__(self, other):
        return self.data * other

    def __divmod__(self, other):
        return self.data // other, self.data % other

    def __mod__(self, other):
        return self.data % other

    def __imod__(self, other):
        return self.data % other

    def __floordiv__(self, other):
        return self.data // other

    def __ifloordiv__(self, other):
        return self.data // other

    def __truediv__(self, other):
        return self.data / other

    def __itruediv__(self, other):
        return self.data / other

    def __pow__(self, power, modulo=None):
        return self.data ** power if modulo is None else (self.data ** power) % modulo

    def __ipow__(self, other):
        return self.data ** other

    def __abs__(self):
        return self.data.abs()

    def __lt__(self, other):
        return self.data < other

    def __eq__(self, other):
        return self.data == other

    def __ne__(self, other):
        return not (self.data == other)

    def __neg__(self):
        return -self.data

    @property
    def len_channels(self):
        """

        :return:
        :rtype:
        """
        return self.data.columns.size

    @property
    def len_time(self):
        """

        :return:
        :rtype:
        """
        return self.data.index.size

    @property
    def channels(self):
        """

        :return:
        :rtype:
        """
        return asarray(self.data.columns)

    @property
    def time(self):
        """

        :return:
        :rtype:
        """
        return asarray(self.data.index)

    @property
    def shape(self):
        """

        :return:
        :rtype:
        """
        return self.data.shape

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return self.data.transpose

    def to_list(self):
        """

        :return:
        :rtype:
        """
        return list(self.data)

    def to_matrix(self):
        return self.data.as_matrix().T

    def to_dict(self):
        return self.data.to_dict()

    def as_dataframe(self):
        """

        :return:
        :rtype:
        """
        return self.data

    def corr(self, method='pearson', min_periods=1):
        return self.data.corr(method, min_periods)

    def cov(self, min_periods=None):
        return self.data.cov(min_periods=min_periods)

    def apply(self, func, axis=0, broadcast=False, raw=False, reduce=None, *args, **kwargs):
        return self.data.apply(func=func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, *args, **kwargs)

    def min(self):
        return self.data.min()

    def max(self):
        return self.data.max()

    def plot(self, *args, collection=True, ax=None, **kwargs):
        """
        Plotting the data as a line collection if ``collection`` is True, else 
        Pandas DataFrame plot method will be called.
        
        :param collection:
        :type collection:
        
        DataFrame Plot: :func:`~DataFrame.plot`

        """

        if collection:
            from .visuals import plot
            ax = plot(self.data, fs=self.fs, ax=ax)
            return ax

        return self.data.plot(ax=ax, *args, **kwargs)

    def filter_by_freq(self, band: Band):
        """
        The signal is transformed using a 4th order Butterworth filter such that it 
        will only contain the frequencies between the lower and the upper THRESHOLDS. 
        
        .. Note::
            Butterworth is a analogue method, and is applied through a 
            forward-backward filter cascaded second-order sections; and 
            the results are returned as a digital signal. Digitization 
            might introduce (a) some minor inaccuracies in both ends of
            the signal, and (b) minor OVERLAP of frequencies above and 
            below the threshold (Butterworth does not cut off 
            instantaneously).
        
        :param band: Threshold - Use ``NeuroEnsemble.bands`` for predefined THRESHOLDS, or define 
                     custom ones as follows: ``{"name": 'alpha', "thresh": (8, 15)}``
        :type band: bands, dict["name": str, "thresh": (float, float)]
        :return: Filtered signal.
        :rtype: Electrogram
        
        Example 
        -------
        
        .. plot::
            :context: close-figs
        
            >>> from NeuroEnsemble.structure import Signal
            >>> from ECoG import bands
            >>> from numpy.random import random
            
            >>> signal_raw = random([2048, 5])
            >>> SAMPLING_FREQ = 512
            >>> signal = Signal(signal=signal_raw, channels=list('abcde'), SAMPLING_FREQ=SAMPLING_FREQ)
            >>> signal.plot()
            
            >>> theta_band = signal.filter_by_freq(bands.THETA)
            >>> theta_band.plot()
            
        """
        from .tools import butterworth_filter

        filtered = self.data.apply(
            butterworth_filter,
            thresh=band['thresh'],
            rate=self.fs
        )

        return filtered

    columns = channels
    index = time
    cov.__doc__ = data.cov.__doc__
    corr.__doc__ = data.corr.__doc__
    min.__doc__ = DataFrame.min.__doc__
    max.__doc__ = DataFrame.max.__doc__
    to_dict.__doc__ = data.to_dict.__doc__
    apply.__doc__ = DataFrame.apply.__doc__
    T.__doc__ = DataFrame.transpose.__doc__
    to_matrix.__doc__ = data.as_matrix.__doc__
    transpose.__doc__ = DataFrame.transpose.__doc__
    __len__.__doc__ = DataFrame.size
    __str__.__doc__ = DataFrame.__str__.__doc__
    __repr__.__doc__ = DataFrame.__repr__.__doc__
    __setstate__.__doc__ = DataFrame.__setstate__.__doc__
    __getstate__.__doc__ = DataFrame.__getstate__.__doc__

    if TYPE_CHECKING:
        from .tools import Hilbert
        from .phase import Phase

    def shuffled_phase_surrogate(self):
        """
        See :func:`~ECoG.tools.shuffled_phase_surrogate` for additional information.

        :return: Signal with shuffled phases. 
        :rtype: BaseStructure
        """
        from .surrogates import shuffled_phase_surrogate
        return Signal(shuffled_phase_surrogate(self.data), fs=self.fs, channels=self.channels, time=self.time)

    def correlated_phase_surrogate(self):
        """
        See :func:`~ECoG.tools.correlated_noise_surrogate` for additional information.

        :return: Signal with correlated surrogate phases. 
        :rtype: BaseStructure
        """
        from .surrogates import correlated_noise_surrogate
        return Signal(correlated_noise_surrogate(self), fs=self.fs, channels=self.channels, time=self.time)

    @property
    def hilbert(self) -> 'Hilbert':
        """
        See :func:`~ECoG.tools.Hilbert` for additional information.

        :return: Hilbert transformed signal.
        :rtype: Hilbert
        """
        from .tools import Hilbert

        if self._hilbert is None:
            self._hilbert = Hilbert(self.data, fs=self.fs)

        return self._hilbert

    def bandpower(self) -> Series:
        """
        Produces the band power for each channel using the trapezoidal integral of 
        the absolute values of the Fourier transformed signals with the resultant 
        spectrum limited between :math:`[0, Fs/4]`.

        See :func:`~ECoG.tools.spectral_power` for additional information.

        :return: Powers of spectrum for each channel.
        :rtype: Series

        Example
        -------

        .. plot::
            :context: close-figs

            >>> from NeuroEnsemble.structure import Signal
            >>> from NeuroEnsemble import bands
            >>> from numpy.random import random

            >>> signal_raw = random([1024, 5])
            >>> SAMPLING_FREQ = 512
            >>> signal = Signal(signal=signal_raw, channels=list('abcde'), SAMPLING_FREQ=SAMPLING_FREQ)
            >>> theta_band = signal.filter_by_freq(bands.THETA)
            >>> powers = theta_band.spectral_power()
            >>> powers.plot('bar')

        """
        from .tools import spectral_power
        return spectral_power(self.data, self.fs)

    def phases(self, thresholds: Tuple[Band]) -> 'Phase':
        """

        :param thresholds:
        :type thresholds:
        :return:
        :rtype:
        """
        from .phase import Phase
        return Phase(self.data, self.fs, thresholds)

    def crosscorr_with(self, other: 'Signal', normed=True,
                       individually: bool=True, mode: str='full',
                       method: str='auto'):
        """
        Cross-correlate two N-dimensional arrays.

        Cross-correlate `in1` and `in2`, with the output size determined by the
        `mode` argument.
        
        The correlation z of two d-dimensional arrays x and y is defined as::

            z[...,k,...] = sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])
    
        This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')`` then
          
        .. math::
    
              z[k] &= (x * y)(k - N + 1) \\
                   &= \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}
    
        for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`
    
        where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
        and :math:`y_m` is 0 when m is outside the range of y.
    
        ``method='fft'`` only works for numerical arrays as it relies on
        `fftconvolve`. In certain cases (i.e., arrays of objects or when
        rounding integers can lose PRECISION), ``method='direct'`` is always used.
    
        :param other: DataFrame to calculate the cross correlation with. Regarded
                     as `int2`.
        :type other: Signal
        
        :param mode: str {'full', 'valid', 'same'}, optional.
                     A string indicating the size of the output:
                    
                    ``full``
                       The output is the full discrete linear cross-correlation
                       of the inputs. 
                    ``valid``
                       The output consists only of those elements that do not
                       rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
                       must be at least as large as the other in every dimension.
                    ``same``
                       The output is the same size as `in1`, centered
                       with respect to the 'full' output. (Default)
        :type mode: str
        
        :param method: str {'auto', 'direct', 'fft'}, optional
                        A string indicating which method to use to calculate the correlation.
                
                    ``direct``
                       The correlation is determined directly from sums, the definition of
                       correlation.
                    ``fft``
                       The Fast Fourier Transform is used to perform the correlation more
                       quickly (only available for numerical arrays.)
                    ``auto``
                       Automatically chooses direct or Fourier method based on an estimate
                       of which is faster (default).  See `convolve` Notes for more detail.
        :type method: str
        
        :return: Cross kendall
        :rtype: Signal
        
        Attributes
        ----------
        Function uses the default [SciPy]_ implementation for cross correlation; 
        see `SciPy documentations`_ for additional information.
        
        .. _Scipy documentations: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        
        .. [SciPy] Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source 
           Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 18/06/2017].
        
        """
        from numpy import arange

        if individually:
            from .tools import xcorr

            res = DataFrame()
            lags = None
            for channel in self.channels:
                res[channel], lags = xcorr(
                    x=self[channel].as_matrix().ravel(),
                    y=other[channel].as_matrix().ravel(),
                    normalize='biased'
                )
            res.index = lags

        else:
            from scipy.signal import correlate
            res = correlate(in1=self, in2=other, mode=mode, method=method)

        dt = DataFrame(res)

        # if normed:
        #     from numpy.linalg import norm
        #
        #     dt = dt/norm(res)

        # if mode == 'full':
        #     dt.index = arange(-max(other.shape), self.time.size + max(other.shape))
        #
        # # ToDo: Index of mode='same' to be defined.

        return dt

    def xcos_with(self, other):
        from .tools import xcosine_similarity
        from numpy import vectorize, float64
        from numpy.linalg import norm

        win = vectorize(xcosine_similarity, signature='(n),(m)->(n)')

        xcos = win(self.as_matrix().T.astype(float64), other[list(self.channels)].as_matrix().T.astype(float64)).T

        xcos = xcos / norm(xcos)

        res = DataFrame(
            xcos,
            columns=self.channels,
            index=self.time
        )

        return res

    def factorize(self, n_components='auto'):
        from sklearn.decomposition import TruncatedSVD
        if n_components == 'auto':
            n_components = self.index // 16

        tsvd = TruncatedSVD(n_components=n_components, tol=self.as_matrix().std())
        results = tsvd.fit_transform(self.as_matrix())

        return Signal(results, fs=self.fs // 16, channels=self.channels)


Electrogram = Signal


def rec_plot(sig, eps=0.1, steps=10):
    from scipy.spatial.distance import pdist, squareform

    d = pdist(sig[:, None], metric='cosine')
    d = floor(d / eps)
    d[d > steps] = steps
    z = squareform(d)
    return z


def moving_average(sig, r=5):
    from scipy.signal import convolve
    return convolve(sig, ones((r,)) / r, mode='valid')
