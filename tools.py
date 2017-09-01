#!/usr/bin/env python3

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# Python's library
# ----------------------------------------------------------------------------------------------------------------------
from typing import NewType
from collections import namedtuple

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# Third party libraries
# ----------------------------------------------------------------------------------------------------------------------
from numpy import (
    array, vectorize, pi, angle, unwrap,
    diff, max, float32, ascontiguousarray,
    flipud, exp, dot, mean, linspace, empty,
    NaN, Inf, arange, isscalar, asarray,
    ndarray, float64, correlate, zeros, divide, abs, issubdtype
)
from numpy.linalg import norm
# from numpy.lib.stride_tricks import as_strided

from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.spatial.distance import cosine as cosine_distance

from pandas import DataFrame, to_timedelta

from numba import jit, int64, float32 as nu_float32

# ----------------------------------------------------------------------------------------------------------------------
# Internal
# ----------------------------------------------------------------------------------------------------------------------
from .utils.utils import *
from .structure import Signal

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

__all__ = [
    'pearson', 'spearman', 'cosine_similarity',
    'butterworth_filter', 'kendall', 'Hilbert',
    'shuffled_phase_surrogate', 'window_comparison',
    'spectral_power', 'as_strides'
]

ElectrogramOrNone = NewType('ElectrogramOrNone', Union[Signal, None])

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

# _pearson = vectorize(pearsonr, signature='(n),(n)->(),()')
# _spearman = vectorize(spearmanr, signature='(n),(n)->(),()')

# _cosine = vectorize(cosine_distance, signature='(n),(n)->()')

# _hilbert_vec = vectorize(_hilbert, signature='(n)->(m)', excluded=['N', 'axis'])


def pearson(this: DataFrame) -> DataFrame:
    """
    
    :param this: 
    :type this: 
    :return: 
    :rtype: 
    """
    return this.corr(method='pearson')


def spearman(this: DataFrame) -> DataFrame:
    """
    
    :param this: 
    :type this: 
    :return: 
    :rtype: 
    """
    return this.corr(method='spearman')


def kendall(this: DataFrame) -> DataFrame:
    """
    
    :param this: 
    :type this: 
    :return: 
    :rtype: 
    """
    return this.corr(method='kendall')


@jit(nopython=True)
def cos_similarity(u, v):
    return dot(u, v) / (norm(u) * norm(v))


@jit(nopython=True)
def naive_covariance(u, v):
    """
    Calculates that naive covariance between two 1d arrays of equal lengths.

    :param u: 1d array
    :type u: array

    :param v: 1d array
    :type v: array 

    :return: Covariance of arrays `u` and `v`.
    :rtype: float32


    Example
    -------
    >>> from numpy import asarray
    >>> from NeuroEnsemble.tools import naive_covariance

    >>> x = asarray([1, 5, 3])
    >>> y = asarray([2, 8, 3])

    >>> naive_covariance(x, y)
    4.0

    """
    n = u.size
    cov = ((u * v).sum() - u.sum() * v.sum() / n) / n
    return float32(cov)


@jit(nopython=True)
def pearson_r(u, v):
    """

    :param u: 
    :type u: 
    :param v: 
    :type v: 
    :return: 
    :rtype: float32


    Example
    -------
    >>> from numpy import asarray
    >>> from NeuroEnsemble.tools import pearson_r

    >>> x = asarray([1, 1, 3])
    >>> y = asarray([-1, -1, 1])

    >>> pearson_r(x, y)
    1.0

    >>> x = asarray([1, 1, 3])
    >>> y = asarray([-1, 0, 1])

    >>> pearson_r(x, y)
    0.86602539

    """
    r = divide(naive_covariance(u, v), float32(u.std() * v.std()))
    return r


@jit(nopython=True)
def xcosine_similarity(u, v):
    results = zeros(u.size)

    for index in arange(0, u.size - v.size):
        u_prime = u[index: index + v.size]
        results[index] = cos_similarity(u_prime, v)

    return results


@jit(nopython=True)
def xpearsons(u, v):
    results = zeros(u.size)

    for index in arange(0, u.size - v.size):
        u_prime = u[index: index + v.size]
        results[index] = pearson_r(u_prime, v)

    return results


def cosine_similarity(this: DataFrame, other: DataFrame) -> array:
    """
    
    :param this: 
    :type this: 
    :param other: 
    :type other: 
    :return: 
    :rtype: 
    """

    results = empty((this.columns.size, other.columns.size), dtype=float32)

    for index, col in enumerate(other.columns):
        results[index] = this.apply(func=cos_similarity, v=other[col])

    return results


@jit
def _butterworth_filter(data: array, thresh: ThreshType, rate: int, order: int=4) -> array:
    """
    
    :param data: 
    :type data: 
    :param thresh: 
    :type thresh: 
    :param rate: 
    :type rate: 
    :param order: 
    :type order: 
    :return: 
    :rtype: 
    """
    nyq = rate / 2
    bandpass = asarray(thresh) / nyq
    sos = butter(order, bandpass, btype='bandpass', analog=False, output='sos')
    y = sosfiltfilt(sos, data)
    return y


butterworth_filter = vectorize(
    _butterworth_filter,
    signature='(n)->(n)',
    excluded=['thresh', 'rate', 'order']
)


# @jit
# def windowed_view(arr: ArrayOrDataFrame, window: int, OVERLAP: int=0, dtype=float64) -> array:
#     """
#
#     :param arr:
#     :type arr:
#     :param window:
#     :type window:
#     :param OVERLAP:
#     :type OVERLAP:
#     :param dtype:
#     :type dtype:
#     :return:
#     :rtype:
#     """
#     window_step = window - OVERLAP
#     new_shape = arr.shape[:-1] + ((arr.shape[-1] - OVERLAP) // window_step, window)
#
#     new_strides = (
#         arr.strides[:-1] +
#         (window_step * arr.strides[-1],) +
#         arr.strides[-1:]
#     )
#     return as_strided(arr, shape=new_shape, strides=new_strides).astype(dtype)


def as_strides(arr: Union[DataFrame, ndarray], win, strides):
    shape_max = max(arr.shape)
    time: Iterable = arange(0, shape_max - win, strides, dtype=int)

    if isinstance(arr, ndarray) and arr.shape[0] != shape_max:
        arr: ndarray = arr.T

        for ind in time:
            yield arr[ind: ind + win, :]

    if isinstance(arr, (DataFrame, Signal)):
        columns: List = list(arr.columns)
        arr = arr.reset_index()[columns]

        for ind in time:
            yield arr[:][ind: ind + win]

# ======================================================================================================================
# Window comparison functions
# ----------------------------------------------------------------------------------------------------------------------


@jit(nu_float32(nu_float32[:], nu_float32[:]), nopython=True)
def _cosine_similarity(a, b):
    resp = dot(a, b) / (norm(a) * norm(b))
    return resp


@jit(nu_float32(nu_float32[:], nu_float32[:]), nopython=True)
def correlation_distance(a, b):
    umu = mean(a)
    um = a - umu
    vmu = mean(b)
    vm = b - vmu
    corr = 1.0 - dot(um, vm) / (norm(um) * norm(vm))
    return corr


# @jit(nu_float32[:](nu_float32[:], uint32, uint32))
# def get_wins(arr, OVERLAP, win_len):
#     dt = asarray(arr, dtype=float64)
#     response = empty(dt.shape, dtype=float64)
#     response[:] = NaN
#     OVERLAP = win_len - OVERLAP
#     size = dt.size
#     similarities = empty(size, dtype=float64) * NaN
#
#     sup_ind = sup_start = 0
#
#     # Sup
#     while (sup_start + win_len) <= size:
#         similarities[:] = NaN
#
#         first = dt[sup_start: win_len + sup_start]
#
#         sub_ind = 0
#         sub_start = 0
#
#         # Sub
#         while (sub_start + win_len) <= size:
#             second = dt[sub_start: sub_start + win_len]
#             similarities[sub_ind] = _cosine_similarity(first, second)
#
#             sub_start += win_len
#             sub_ind += 1
#
#         response[sup_ind] = nanmean(similarities)
#         sup_start += OVERLAP
#         sup_ind += 1
#
#     return response


def _get_wins(arr, overlap, win_len):

    @jit(nu_float32[:, :](nu_float32[:], int64, int64), nopython=True)
    def gw(dt, olap, win):
        response = empty((win, dt.size), dtype=float32)
        response[:] = NaN
        size = dt.size

        # Sup
        for off in arange(win):

            first = dt[off: win + off]

            sub_ind = 0
            sub_start = 0

            # Sub
            while (sub_start + off + win) <= size:
                second = dt[sub_start + off: sub_start + win + off]
                response[off, sub_ind] = cosine_similarity(first, second)

                sub_start += olap + off
                sub_ind += 1
        return response

    return DataFrame(gw(asarray(arr, float32), win_len - overlap, win_len)).median(axis=0).dropna()


@logger('window_comparison')
def window_comparison(dt: DataFrame, rate: int) -> DataFrame:
    """

    :param dt: One-dimensional array.
    :type dt: DataFrame
    :param rate: Frequency SAMPLING_FREQ. 
    :type rate: int
    :return: Window comparison of the one dimensional array, where the length of 
             each window is equal to frequency SAMPLING_FREQ.
    :rtype: DataFrame
    """
    res = dt.apply(
        _get_wins,
        overlap=rate - 1,
        win_len=rate
    )

    t_delta = to_timedelta(
        linspace(0, max(dt.shape) / rate, max(dt.shape)),
        unit='s'
    ).astype('timedelta64[s]')

    res.index = t_delta[:max(res.shape)]
    # new_ind =
    # res.index = new_ind
    return res


# ======================================================================================================================
# Hilbert
# ----------------------------------------------------------------------------------------------------------------------

class Hilbert(Signal):
    """
    Analytical signal calculated using Hilbert transform.

    .. math::
        x_a = F^{−1}(F(x)2U) = x + iy 

    :param signal: 1 or 2D array of one dimensional signal(s).
    :type signal: ArrayOrDataFrame
    :param fs: Frequency SAMPLING_FREQ (:math:`SAMPLING_FREQ`).
    :type fs: int
    """

    def __init__(self, signal: DataFrame, *args, **kwargs):
        self._instantaneous_phase: ElectrogramOrNone = None
        self._instantaneous_frequency: ElectrogramOrNone = None
        dt = signal.apply(hilbert)
        super().__init__(signal=dt, *args, **kwargs)

    def instantaneous_frequency(self) -> Signal:
        """
        Where :math:`x` is a vector of Hilbert instantaneous, phase and :math:`h=1`, 
        and :math:`SAMPLING_FREQ` is the frequency SAMPLING_FREQ:

        .. math::
            \delta _{h}[f](x)=f(x+{\tfrac {1}{2}}h)-f(x-{\tfrac {1}{2}}h)

            \frac{\delta _{h}[f](x)}{2\pi} \times SAMPLING_FREQ

        :return: DataFrame of instantaneous frequencies. 
        :rtype: DataFrame
        """
        if self._instantaneous_frequency is not None:
            return self._instantaneous_frequency

        freq = diff(self.instantaneous_phase(as_mod=False), axis=0) / (2 * pi) * self.fs

        self._instantaneous_frequency = Signal(
            freq,
            fs=self.fs,
            time=list(self.time)[1:],
            channels=self.channels
        )

        return self._instantaneous_frequency

    def instantaneous_phase(self, as_mod: bool=True) -> Signal:
        """
        Produces a vector of instantaneous phases.
        
        .. note::
            Phases are returned **unwrapped**; that is, radian phase `\Phi` by changing 
            absolute jumps greater than `discont` (maximum discontinuity between values,
            in this case :math:`\pi`) to their :math:`2 \times \pi` complement along 
            the given axis.

        :param as_mod: Return :math:`\Phi \mathbin{\%} \pi` (:math:`\mathbin{\%}` 
                       denotes the remainder of division) if **True** [default], 
                       otherwise return :math:`\Phi`.
        :type as_mod: bool
        :return: DataFrame of phases (angles) of complex arguments in radians
                 calculated using Hilbert transform.
        :rtype: DataFrame
        """
        if self._instantaneous_phase is not None:
            res = self._instantaneous_phase
        else:
            res = self._instantaneous_phase = self.apply(lambda x: unwrap(angle(x, False)))
        return res % (2 * pi) if as_mod else res

    def amplitude_envelope(self) -> Signal:
        """
        Returns amplitude envelope by calculating the absolute values from Hilbert 
        transformed signal(s). 

        :return:  DataFrame of absolute values.
        :rtype: DataFrame
        """
        return self.abs()


def spectral_power(data: DataFrame, fs: int):
    """
    Produces the band power for each channel using the trapezoidal integral of 
    the absolute values of the Fourier transformed signals with the resultant 
    spectrum limited between :math:`[0, Fs/4]`.

    The calculation is as follows: 

    .. math::
        P = \int_{0}^{Fs/4} |{\hat{x}}(f)|\, \mathrm{d}f

    where

    .. math::
        {\hat{x}}(f) = \int_{-\infty }^{\infty }e^{-2\pi ift}x(t)\, \mathrm{d}t

    is the Fourier Transform of the signal and :math:`f` is the frequency in :math:`Hz`. 

    .. attention::
        The index for ``data`` **must** be equal to time. 

    :param data: Data, with the columns representing channels, and the index 
                 representing time.
    :type data: DataFrame
    :param fs: Sampling frequency (rate).
    :type fs: int
    :return: Powers of spectrum for each channel.
    :rtype: Series

    Example
    -------

    .. plot::
        :context: close-figs

        >>> from NeuroEnsemble.tools import spectral_power
        >>> from numpy.random import random
        >>> from pandas import DataFrame

        >>> signal_raw = random([1024, 5])
        >>> SAMPLING_FREQ = 512
        >>> signal = DataFrame(signal_raw, columns=list('abcde'))
        >>> powers = spectral_power(data=signal, SAMPLING_FREQ=SAMPLING_FREQ)
        >>> powers.plot('bar')
    """
    from scipy.fftpack import fft
    from scipy.integrate import trapz

    fourier = data.apply(fft).abs()
    bp = fourier[:][:fs // 4].apply(trapz, x=data.index)

    return bp


# ======================================================================================================================
# Peak (extrema) finder
# ----------------------------------------------------------------------------------------------------------------------

@jit(nopython=True)
def _extrema(vector, index, delta):
    vector_size = vector.size
    dim_extrema = (vector_size // 2) + 1, 2
    maxima = empty(dim_extrema) * NaN
    minima = maxima.copy()

    mn, mx = Inf, -Inf
    mn_pos = mx_pos = mx_ind = mn_ind = 0

    max_next = True

    vector_range = arange(vector_size)

    for ind, value in zip(vector_range, vector):
        if value > mx:
            mx, mx_pos = value, index[ind]
        elif value < mn:
            mn, mn_pos = value, index[ind]

        if max_next:
            previous = mx - delta
            if vector[ind] < previous:
                maxima[mx_ind] = mx_pos, mx
                mn_pos, mn, max_next = index[ind], value, False
                mx_ind += 1
        elif vector[ind] > (mn + delta):
            minima[mn_ind] = mn_pos, mn
            mx_pos, mx, max_next = index[ind], value, True
            mn_ind += 1

    return maxima[:mx_ind], minima[:mn_ind]


def find_peaks(arr, delta, index=None):
    """
    Finds the local maxima and minima (peaks) in a one-dimensional vector.

    :param arr: One dimensional array.
    :type arr: array, list, tuple

    :param delta: A point is considered a maximum peak if it has the maximal  
                  value, and was preceded (to the left) by a value smaller 
                  than ``point - delta``. In terms of minimum, the point must be 
                  preceded with a value greater than ``point + delta``. 
    :type delta: int, float

    :param index: Values representing the X-axis of the vector. The indexes
                    in the results (columns 0 of each array) are replaced with 
                    their corresponding values from ``x``. If not given, defaults
                    to ``x = arange(0, len(x))``. This is particularly useful 
                    when plotting the extrema for data with custom values for the 
                    X-axis. 
    :type index: list, tuple, array

    :return: Tuple of 2 arrays (maxima and minima), each with 2 columns. 
             Columns 0 is the index of the peak, and columns 1 is the value 
             thereof.

             The results are returns as a ``namedtuple('peaks', (maxima, minima))``,
             where both minima and maxima are arrays of type ``float64``. 

    :rtype: Tuple[array, array]

    Example
    -------
    >>> data = 0, 0, 1, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0
    >>> peaks = find_peaks(data, delta=0.3)
    >>> peaks.maxima
    array([[  2.,   1.],
           [  5.,   2.],
           [ 13.,   2.]])
    >>> peaks.minima
    array([[  3.,   0.],
           [  9.,  -2.],
           [ 17.,  -2.]])
    >>> minima_values = peaks.minima[:, 1]
    >>> minima_indexes = peaks.minima[:, 0]
    
    
    Attributes
    ----------
    Improved version of the `Python interpretation`_ of ``PEAKDET`` 3.4.05 
    `algorithm for MATLAB`_ produced by Eli Billauer. 
    
    This interpretation introduces a 5-fold improvement on execution speed, and 
    is PEP8 compliant. 
    
    Licence
    -------
    The original function is released into the public domain and is not copyrighted. 
    This Python version, authored by **Pouria Hadjibagheri (2017)**, is licenced 
    under the OSI-aproved MIT free software license. 
    
    Copyright 2017, Pouria Hadjibagheri.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of 
    this software and associated documentation files (the "Software"), to deal in 
    the Software without restriction, including without limitation the rights to use, 
    copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
    Software, and to permit persons to whom the Software is furnished to do so, 
    subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all 
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
    .. _Python interpretation: https://gist.github.com/endolith/250860
    .. _algorithm for MATLAB: http://billauer.co.il/peakdet.html
    .. _Anaconda: https://www.continuum.io/downloads
    .. _Numba: http://numba.pydata.org
    .. _Continuum: https://www.continuum.io
    """

    if not isscalar(delta):
        raise ValueError('Input argument delta must be a scalar.')

    if not delta > 0:
        raise ValueError('Input argument delta must be positive.')

    delta = float64(delta)

    vector = asarray(arr, dtype=float64)
    vector_size = vector.size

    index = index or arange(0, vector_size, dtype=float64)

    if not isinstance(index, ndarray):
        index = asarray(index, dtype=float64)

    index_size = index.size

    if not vector_size == index_size:
        raise ValueError('Input vectors v and x must have same length.')

    template = namedtuple('peaks', ('maxima', 'minima'))

    results = _extrema(
        vector=vector,
        index=index,
        delta=delta
    )

    return template(*results)


# ======================================================================================================================
# Windowed operations
# ----------------------------------------------------------------------------------------------------------------------



@jit(nopython=True)
def windowed_correlation(u, v, unpad=False):
    """
    Calculate Pearson's correlation coefficients of the moving window of array `v` to
    the stationary array `u`, where the length of `u` is greater than or equal to that
    of `v`.
    
    :param u: 1d array to which `v` is to be compared. The length of `u` must 
              be greater than or equal to that of `v`.
    :type u: array
    
    :param v: 1d array to be compared to `u`. The length of `v` must be smaller 
              than or equal to that of `u`.
    :type v: array
    
    :param unpad: Remove zero padding from the results.
    :type unpad: bool
    
    :return: Results, with length equal to that of `u`, padded with zeros 
             from both sides to the length of `v` 
             
    :rtype: array
    
    
    Example
    -------
    >>> from numpy import asarray
    >>> from NeuroEnsemble.tools import windowed_correlation
    
    >>> x = asarray([1, 5, 3, 4, 2, 9, 15, 10, 14, 4])
    >>> y = asarray([2, 8, 3])
    
    >>> windowed_correlation(x, y)
    array([ 0.        ,  0.        ,  0.        , -0.60395717,  0.19921742,
            1.        , -0.9994238 ,  0.        ,  0.        ,  0.        ])

    >>> windowed_correlation(x, y, unpad=True)
    array([-0.60395717,  0.19921742,
            1.        , -0.9994238 ])

    """
    results = zeros(u.size)

    for index in arange(v.size, u.size - v.size):
        u_prime = u[index: index + v.size]
        results[index] = pearson_r(u_prime, v)

    if not unpad:
        return results

    return results[v.size:-v.size]


# =============

"""
.. topic:: Correlation module


    Provides two correlation functions. :func:`CORRELATION` is slower than 
    :func:`xcorr`. However, the output is as expected by some other functions. 
    Ultimately, it should be replaced by :func:`xcorr`.

    For real data, the behaviour of the 2 functions is identical. However, for
    complex data, xcorr returns a 2-sides correlation.


    .. autosummary:: 

        ~spectrum.correlation.CORRELATION
        ~spectrum.correlation.xcorr

    .. codeauthor: Thomas Cokelaer, 2011



"""  # from numpy.fft import fft, ifft
import numpy
from numpy import arange, isrealobj, absolute, sqrt, mean, r_


@jit(nopython=True)
def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    sqrt(mean(absolute(a) ** 2))


def xcorr(x, y=None, max_lags=None, normalize='biased'):
    """
    Cross-correlation using numpy.correlate

    Estimates the cross-correlation (and autocorrelation) sequence of a random
    process of length N. By default, there is no normalize and the output
    sequence of the cross-correlation has a length 2*N+1. 

    :param array x: first data array of length N
    :param array y: second data array of length N. If not specified, computes the 
        autocorrelation. 
    :param int maxlags: compute cross correlation between [-maxlags:maxlags]
        when maxlags is not specified, the range of lags is [-N+1:N-1].
    :param str normalize: normalize in ['biased', 'unbiased', None, 'coef']

    The true cross-correlation sequence is

    .. math:: r_{xy}[m] = E(x[n+m].y^*[n]) = E(x[n].y^*[n-m])

    However, in practice, only a finite segment of one realization of the 
    infinite-length random process is available.

    The correlation is estimated using numpy.correlate(x,y,'full'). 
    Normalisation is handled by this function using the following cases:

        * 'biased': Biased estimate of the cross-correlation function
        * 'unbiased': Unbiased estimate of the cross-correlation function
        * 'coef': Normalizes the sequence so the autocorrelations at zero 
           lag is 1.0.

    :return:
        * a numpy.array containing the cross-correlation sequence (length 2*N-1)
        * lags vector

    .. note:: If x and y are not the same length, the shorter vector is 
        zero-padded to the length of the longer vector.

    .. rubric:: Examples

    .. doctest::

        >>> from NeuroEnsemble.tools import xcorr
        >>> x = [1,2,3,4,5]
        >>> c, l = xcorr(x,x, max_lags=0, norm='biased')
        >>> c
        array([ 11.])

    .. seealso:: :func:`correlate`.  
    """
    if y is None:
        y = x

    y_size = y.size
    if not x.size == y.size:
        y = r_[y, zeros(x.size - y.size)]

    if max_lags is None:
        max_lags = x.size - 1
        lags = arange(0, 2 * x.size - 1)
    elif max_lags > x.size:
        raise ValueError('Size of `x` cannot be smaller than the value of `max_lag`.')
    else:
        lags = arange(x.size - max_lags - 1, x.size + max_lags)

    res = correlate(x, y, mode='full')

    if normalize == 'biased':
        res = res[lags] / x.size  # do not use /= !!
    elif normalize == 'unbiased':
        res = res[lags] / (x.size - abs(arange(-x.size + 1, x.size)))[lags]
    elif normalize == 'coeff':
        rms = rms_flat(x) * rms_flat(y)
        res = res[lags] / rms / x.size
    else:
        res = res[lags]

    lags = arange(-max_lags, max_lags + 1)

    return res[x.size-y_size:], lags[x.size-y_size:]
