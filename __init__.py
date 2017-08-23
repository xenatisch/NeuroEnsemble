#!/usr/bin/env python3

"""
<Description of the programme>

Programmed in Python 3.6.1 | Anaconda distribution. 

Requires at least Python 3.6 to run.


Bibliography
------------
- Numba: A. Hayashi, J. Zhao, M. Ferguson and V. Sarkar, "LLVM-based communication optimizations for PGAS programs", 
Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC - LLVM '15, 2015.

- SciPy: Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, 
http://www.scipy.org/ [Online; accessed 2017-04-01].

- Numpy: Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient 
Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

- Seaborn: Michael Waskom; Olga Botvinnik; drewokane; Paul Hobson; David; Yaroslav Halchenko; Saulius Lukauskas; 
John B. Cole; Jordi Warmenhoven; Julian de Ruiter; Stephan Hoyer; Jake Vanderplas; Santi Villalba; Gero Kunter; 
Eric Quintero; Marcel Martin; Alistair Miles; Kyle Meyer; Tom Augspurger; Tal Yarkoni; Pete Bachant; Mike Williams; 
Constantine Evans; Clark Fitzgerald; Brian; Daniel Wehner; Gregory Hitz; Erik Ziegler; Adel Qalieh; Antony Lee. 
(2016). seaborn: v0.7.1 (June 2016) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.54844

- Pandas: Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in 
Science Conference, 51-56 (2010)

- Cython: Stefan Behnel, Robert Bradshaw, Craig Citro, Lisandro Dalcin, Dag Sverre Seljebotn and Kurt Smith. 
Cython: The Best of Both Worlds, Computing in Science and Engineering, 13, 31-39 (2011), DOI:10.1109/MCSE.2010.118

- MatPlotLib: John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & 
Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55 
"""

from sys import path as sys_path
from os import path as os_path
from seaborn import cubehelix_palette as _cmap_factory

sys_path.append(os_path.dirname(__file__))

results_directory = {
    'main': 'results',
    'general': 'collections',
    'log': 'log.txt',
    'windowed': 'moving_windows',
    'hilbert_phase': 'hilbert_phase'
}

figure_settings = {
    'format': 'png',  # Available: `pdf`, `eps`, `jpg`, or `png`.
    'resolution': 200  # dpi - only used if the format is `png` or `jpg`.
}

# colormap = _cmap_factory(light=1, as_cmap=True)
colormap = _cmap_factory(as_cmap=True, dark=0, light=1, reverse=False, rot=0.4)

from .structure import Signal, Electrogram
from .utils.utils import bands


__all__ = [
    'figure_settings', 'colormap', 'Signal', 'Electrogram',
    'bands'
]


__author__ = 'Pouria Hadjibagheri'
__copyright__ = 'Copyright 2017'
__credits__ = ['Pouria Hadjibagheri', 'Dr Gerold Baier']
__license__ = 'GPLv.3'
__maintainer__ = 'Pouria Hadjibagheri'
__email__ = 'p.bagheri@ucl.ac.uk'
__date__ = '04/04/2017, 19:18'
__status__ = 'Development'
__version__ = '0.6.1'
