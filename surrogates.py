#!/usr/bin/env python3

"""
<Description of the programme>

Programmed in Python 3.5.1-final.

.. [PyUnicorn] J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, 
   J. Runge, Q.-Y. Feng, L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, 
   H.A. Dijkstra, and J. Kurths, Unified functional network and nonlinear 
   time series analysis for complex systems science: The pyunicorn package, 
   Chaos 25, 113101 (2015), doi:10.1063/1.4934554, Preprint: 
   arxiv.org:1507.01571 [physics.data-an]. 
"""

# Imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Python:
from functools import wraps

# 3rd party:
from numpy import (
    ascontiguousarray, asarray, ndarray, newaxis,
    uint16, flipud, arange, exp, pi
)
from numpy.random import shuffle, randn, uniform
from scipy.fftpack import fft, ifft

# Internal:
from .utils.docstrings import interpd, dedent_interpd, dedent_docstring

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__author__ = 'Pouria Hadjibagheri'
__copyright__ = 'Copyright 2017'
__credits__ = ['Pouria Hadjibagheri']
__license__ = 'GPLv.3'
__maintainer__ = 'Pouria Hadjibagheri'
__email__ = 'p.bagheri@ucl.ac.uk'
__date__ = '15/08/2017, 23:55'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

interpd.update(Surrogates=dedent_docstring("""
    **Keyword arguments**:
    
    :param copy: 
    :type copy:
    :param axis:
    :type axis:
    
    :return: Surrogate time series (same dimensions as original)
    :rtype: ndarray
"""))


def _rank_indices(arr, axis):
    return arr.argsort(axis=axis).argsort(axis=axis).astype(uint16)


def surrogate(func):
    @dedent_interpd
    @wraps(func)
    def func_wrapper(arr: ndarray, *args, copy=True, axis=0, **kwargs):
        if copy:
            arr_ = asarray(arr).copy()
        else:
            arr_ = asarray(arr)

        kwargs['axis'] = axis

        return func(arr_, *args, **kwargs)
    return func_wrapper


@surrogate
def shuffled_phase_surrogate(arr: ndarray, **kwargs) -> ndarray:
    """
    Generates surrogates by Fourier transforming the original time series,
    randomising the phases and then applying an inverse Fourier transform.
    
    :param arr: 2 dimensional array.
    :type arr: ndarray
    %(Surrogates)s
    """
    axis = kwargs.pop('axis')
    # ind, col = arr.shape
    row_size, column_size = arr.shape

    offset = 1 + (row_size % 2)
    phase_size = (row_size - offset) // 2

    # Calculate FFT of arr time series
    # The FFT of the arr data has to be calculated only once, so it
    # is stored in self.originalFFT
    surrogates = fft(arr, axis=axis)

    for channel in surrogates.imag.T:
        shuffle(channel)

    # Calculate IFFT and take the real part, the remaining imaginary part
    # is due to numerical errors
    reversed_fft = ifft(surrogates, axis=1).real
    results = ascontiguousarray(reversed_fft)

    return results


@surrogate
def correlated_noise_surrogate(arr: ndarray, **kwargs) -> ndarray:
    """
    Generates surrogates by Fourier transforming the original time series,
    randomizing the phases and then applying an inverse Fourier transform.
    Correlated noise surrogates share their power spectrum and
    auto-correlation function with the original time series.
    
    :param arr: dim. 0 is index of time series, dim. 1 is time    
    :type arr: ndarray
    %(Surrogates)s
    """
    axis = kwargs.pop('axis')
    pipi = 2 * pi

    axis_size, other_size = arr.shape[axis], arr.shape[not axis]

    offset = 1 + (axis_size % 2)  # 2 if not row_size % 2 else 1
    phase_size = (axis_size - offset) // 2

    # print(arr.shape)
    surrogates = fft(arr, axis=axis)

    # Generate random phases uniformly distributed in the interval
    #  [0, 2*Pi]. Guarantee that the phases for positive and negative
    #  frequencies are the same to obtain real surrogates in the end!
    phases = uniform(low=0, high=pipi, size=(other_size, phase_size))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]

    new_phases = exp(1j * phases)

    if not axis:
        surrogates.T[:, 1:phase_size + 1].imag *= new_phases.imag
        surrogates.T[:, offset:phase_size + 1].imag = flipud(surrogates[1:phase_size + 1, :].conjugate()).T.imag
    else:
        surrogates[1:phase_size + 1, :] *= new_phases
        surrogates[offset:phase_size + 1, :] = flipud(surrogates[:, 1:phase_size + 1].conjugate())

    #  Discriminate between even and uneven number of samples
    #  Note that the output of fft has the following form:
    #  - Even sample number: (mean, pos. freq, nyquist freq, neg. freq)
    #  - Odd sample number: (mean, pos. freq, neg. freq)
    # surrogates[:, offset:phase_size + 1] = flipud(surrogates[:, 1:phase_size + 1].conjugate())

    # Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors
    reversed_fft = ifft(surrogates, axis=axis).real
    results = ascontiguousarray(reversed_fft)

    return results


@surrogate
def amplitude_adjusted_surrogate(arr: ndarray, **kwargs) -> ndarray:
    axis = kwargs.pop('axis')
    #  Create sorted Gaussian reference series
    gaussian = randn(*arr.shape)
    # gaussian
    gaussian.sort(axis=axis)
    ranks = _rank_indices(arr, axis=axis)

    row_indices = arange(gaussian.shape[0])
    #  Rescale data to Gaussian distribution
    rescaled_data = gaussian[row_indices[:, newaxis], ranks - 1]

    #  Phase randomise rescaled data
    phase_randomized_data = correlated_noise_surrogate(rescaled_data)

    #  Rescale back to amplitude distribution of original data
    sorted_original = arr
    sorted_original.sort(axis=axis)

    ranks = _rank_indices(phase_randomized_data, axis=axis)

    rescaled_data = sorted_original[row_indices[:, newaxis], ranks - 1]

    return rescaled_data


class UniversalSurrogate:
    """
    ToDo: Implements are a surrogate module wherein random arrays are created once and retained for application on 
          other sets of data.
    """
    pass

