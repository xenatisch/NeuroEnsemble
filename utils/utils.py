#!/usr/bin/env python3


from collections import namedtuple

# 3rd party:
from pandas import DataFrame
from numpy import array

# Internal:
from typing import (
    TypeVar, NewType, Iterable, Union,
    List, Iterator, Tuple, Any, Dict,
    Type
)
from functools import partial, wraps
from inspect import signature
from os import path as os_path
from pickle import dump as pickle_dumper
import logging as log
from time import time


__all__ = [
    'ArrayOrDataFrame', 'IntegerOrIterable', 'ListOrTuple',
    'StringOrIterable', 'Location', 'Iterable', 'logger', 'temp',
    'ListOfStringsOrNone', 'ListOfIntegersOrNone', 'Iterator',
    'Tuple', 'Union', 'List', 'ThreshType', 'Any', 'Dict', 'Type',
    'RealNumberOrNone', 'RealNumber', 'Band'
]


# Global type definitions:
ArrayOrDataFrame = TypeVar('ArrayOrDataFrame', array, DataFrame)
IntegerOrIterable = TypeVar('IntegerOrIterable', int, Iterable)
ListOrTuple = TypeVar('ListOrTuple', tuple, list, Iterable)
StringOrIterable = TypeVar('StringOrIterable', str, ListOrTuple)
Location = NewType('Location', str)
ListOfStringsOrNone = NewType('ListOfStringsOrNone', Union[List[str], None])
ListOfIntegersOrNone = NewType('ListOfIntegersOrNone', Union[List[int], None])
ThreshType = NewType('ThreshType',  Tuple[Union[int, float], Union[int, float]])
RealNumber = NewType('RealNumber',  Union[int, float])
RealNumberOrNone = NewType('RealNumberOrNone',  Union[RealNumber, None])
Band = NewType('Band', Dict[str, Tuple[RealNumber, RealNumber]])


def _from_args_kwargs(name, func, args, kwargs):
    sig = signature(func).parameters

    if name in kwargs:
        return kwargs[name]

    if name in sig:
        thresh_idx = tuple(sig.keys()).index(name)
        return args[thresh_idx]

    return None


def logger(start: str, end: str=str(), include: ListOrTuple=None, severity: int=log.INFO):
    if isinstance(include, str):
        raise TypeError(
            f'Logger "include" must be an iterable, not "{type(include)}".'
        )

    def log_wrapper(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            get = partial(_from_args_kwargs, func=func, args=args, kwargs=kwargs)
            thresh = get('thresh')

            if thresh is not None:
                prelude = f'{thresh[0]:>3}-{thresh[1]:<3}Hz | {func.__name__:^24}'
            else:
                prelude = f'{func.__name__:^24}'

            add_on = str()
            if hasattr(include, '__iter__'):
                add_on = ' - ' + str.join(', ', map(lambda item: f'{item}: {get(item)}', include))

            started, finished = 'STARTED', 'FINISHED'

            log.log(level=severity, msg=f'{prelude} | {started:^10} > {start}{add_on}')
            output = None

            tic = time()
            try:
                output = func(*args, **kwargs)
            except Exception as error:
                log.exception(f'{prelude}:\n{error}', exc_info=True)
            toc = time()

            log.log(
                level=severity,
                msg=(
                    f'{prelude} | '
                    f'{finished} in {toc-tic:.2f} sec. '
                    f'< ...{end or start}{add_on} '
                )
            )

            return output
        return func_wrapper
    return log_wrapper


def _as_temp(location, key, data):
    key = key.replace('.', '').replace('-', '_')
    path = os_path.join(location, f".{key}.tmp")

    pickler = open(path, mode='wb')
    pickle_dumper(data, file=pickler)
    pickler.close()

    return True


def temp(prepositions: tuple=tuple(), path: str=None) -> True:
    def temp_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            get = partial(_from_args_kwargs, func=func, args=args, kwargs=kwargs)
            thresh = get('thresh')

            location = path
            location = location or get('location')
            pre = prepositions or get('prepositions') or (func.__name__,)

            if location is None:
                message = (
                    f'Location is neither given, nor passed as an argument '
                    f'to the function entitled <{func.__name__}>.'
                )
                log.critical(message)
                raise ValueError(message)

            response = func(*args, **kwargs)

            data = response
            if not isinstance(data, tuple):
                data = (response,)

            for prepend, item in zip(pre, data):
                key = prepend
                if thresh:
                    key += f'_{thresh[0]}_{thresh[1]}Hz'
                _as_temp(location, key, item)

            return response
        return wrapper
    return temp_wrapper


_thresh_template = namedtuple('THRESHOLDS', ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA'])

bands = _thresh_template(
    DELTA={'thresh': (0.5, 3), 'name': 'delta'},
    THETA={'thresh': (4, 7), 'name': 'theta'},
    ALPHA={'thresh': (8, 15), 'name': 'alpha'},
    BETA={'thresh': (16, 31), 'name': 'beta'},
    GAMMA={'thresh': (32, 50), 'name': 'gamma'},
)
