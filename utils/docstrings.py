#!/usr/bin/env python3

"""
<Description of the programme>

Programmed in Python 3.5.1-final.
"""

# Imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Python:
from types import MethodType
from re import compile as re_compile

# 3rd party:

# Internal:
from . import PY3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__author__ = 'Pouria Hadjibagheri'
__copyright__ = 'Copyright 2017'
__credits__ = ['Pouria Hadjibagheri']
__license__ = 'GPLv.2'
__maintainer__ = 'Pouria Hadjibagheri'
__email__ = 'p.bagheri.12@ucl.ac.uk'
__date__ = '19/08/2017, 12:37'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Substitution(object):
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """
    def __init__(self, *args, **kwargs):
        assert not (len(args) and len(kwargs)), "Only positional or keyword args are allowed"
        self.params = args or kwargs

    def __call__(self, func):
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs):
        "Assume self.params is a dict and update it with supplied args"
        self.params.update(*args, **kwargs)

    @classmethod
    def from_params(cls, params):
        """
        In the case where the params is a mutable sequence (list or
        dictionary) and it may change before this class is called, one may
        explicitly use a reference to the params rather than using *args or
        **kwargs which will copy the values and not reference them.
        """
        result = cls()
        result.params = params
        return result


interpd = Substitution()


# A regular expression used to determine the amount of space to
# remove.  It looks for the first sequence of spaces immediately
# following the first newline, or at the beginning of the string.
_find_dedent_regex = re_compile("(?:(?:\n\r?)|^)( *)\S")
# A cache to hold the regexs that actually remove the indent.
_dedent_regex = {}


def dedent_docstring(s):
    """
    Remove excess indentation from docstring *s*.

    Discards any leading blank lines, then removes up to n whitespace
    characters from each line, where n is the number of leading
    whitespace characters in the first line. It differs from
    textwrap.dedent in its deletion of leading blank lines and its use
    of the first non-blank line to determine the indentation.

    It is also faster in most cases.
    """
    # This implementation has a somewhat obtuse use of regular
    # expressions.  However, this function accounted for almost 30% of
    # matplotlib startup time, so it is worthy of optimization at all
    # costs.

    if not s:      # includes case of s is None
        return str()

    match = _find_dedent_regex.match(s)
    if match is None:
        return s

    # This is the number of spaces to remove from the left-hand side.
    nshift = match.end(1) - match.start(1)
    if nshift == 0:
        return s

    # Get a regex that will remove *up to* nshift spaces from the
    # beginning of each line.  If it isn't in the cache, generate it.
    unindent = _dedent_regex.get(nshift, None)
    if unindent is None:
        unindent = re_compile("\n\r? {0,%d}" % nshift)
        _dedent_regex[nshift] = unindent

    result = unindent.sub("\n", s).strip()
    return result


def dedent(func):
    """
    Dedent a docstring (if present)
    """
    func.__doc__ = func.__doc__ and dedent_docstring(func.__doc__)
    return func


def dedent_interpd(func):
    """A special case of the interpd that first performs a dedent on
    the incoming docstring"""
    if isinstance(func, MethodType) and not PY3:
        func = func.im_func
    return interpd(dedent(func))

