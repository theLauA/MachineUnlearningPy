"""
Data set cross-folding.
"""

from collections import namedtuple
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from . import util

TTPair = namedtuple('TTPair', ['train', 'test'])
TTPair.__doc__ = 'Train-test pair (named tuple).'
TTPair.train.__doc__ = 'Train data for this pair.'
TTPair.test.__doc__ = 'Test data for this pair.'

_logger = logging.getLogger(__name__)


def partition_rows(data, partitions):
    """
    Partition a frame of ratings or other datainto train-test partitions.  This function does not
    care what kind of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: :py:class:`pandas.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """
    _logger.info('partitioning %d ratings into %d partitions', len(data), partitions)

    # create an array of indexes
    rows = np.arange(len(data))
    # shuffle the indices & split into partitions
    np.random.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        test = data.iloc[ts, :]
        trains = test_sets[:i] + test_sets[(i + 1):]
        train_idx = np.concatenate(trains)
        train = data.iloc[train_idx, :]
        yield TTPair(train, test)


def sample_rows(data, partitions, size, disjoint=True):
    """
    Sample train-test a frame of ratings into train-test partitions.  This function does not care
    what kind of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    We can loop over a sequence of train-test pairs::

        >>> ratings = util.load_ml_ratings()
        >>> for train, test in sample_rows(ratings, 5, 1000):
        ...     print(len(test))
        1000
        1000
        1000
        1000
        1000

    Sometimes for testing, it is useful to just get a single pair::

        >>> train, test = sample_rows(ratings, None, 1000)
        >>> len(test)
        1000
        >>> len(test) + len(train) - len(ratings)
        0

    Args:
        data(pandas.DataFrame):
            Data frame containing ratings or other data to partition.
        partitions(int or None):
            The number of partitions to produce.  If ``None``, produce a _single_ train-test
            pair instead of an iterator or list.
        size(int):
            The size of each sample.
        disjoint(bool):
            If ``True``, force samples to be disjoint.

    Returns:
        iterator: An iterator of train-test pairs.
    """

    if partitions is None:
        test = data.sample(n=size)
        tr_mask = pd.Series(True, index=data.index)
        tr_mask.loc[test.index] = False
        train = data[tr_mask]
        return TTPair(train, test)

    if disjoint and partitions * size >= len(data):
        _logger.warning('wanted %d disjoint splits of %d each, but only have %d rows; partitioning',
                        partitions, size, len(data))
        return partition_rows(data, partitions)

    # create an array of indexes
    rows = np.arange(len(data))

    if disjoint:
        _logger.info('creating %d disjoint samples of size %d', partitions, size)
        ips = _disjoint_sample(rows, partitions, size)

    else:
        _logger.info('taking %d samples of size %d', partitions, size)
        ips = _n_samples(rows, partitions, size)

    return (TTPair(data.iloc[ip.train, :], data.iloc[ip.test, :]) for ip in ips)


def _disjoint_sample(xs, n, size):
    # shuffle the indices & split into partitions
    np.random.shuffle(xs)

    # convert each partition into a split
    for i in range(n):
        start = i * size
        test = xs[start:start + size]
        train = np.concatenate((xs[:start], xs[start + size:]))
        yield TTPair(train, test)


def _n_samples(xs, n, size):
    for i in range(n):
        test = np.random.choice(xs, size, False)
        train = np.setdiff1d(xs, test, assume_unique=True)
        yield TTPair(train, test)


class PartitionMethod(ABC):
    """
    Partition methods select test rows for a user or item.  Partition methods
    are callable; when called with a data frame, they return the test rows.
    """

    @abstractmethod
    def __call__(self, udf):
        """
        Subset a data frame.

        :param udf: The input data frame of rows for a user or item.
        :paramtype udf: :py:class:`pandas.DataFrame`
        :returns: The data frame of test rows, a subset of `udf`.
        """
        pass


class SampleN(PartitionMethod):
    """
    Randomly select a fixed number of test rows per user/item.

    :param n: The number of test items to select.
    :paramtype n: integer
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, udf):
        return udf.sample(n=self.n)


class SampleFrac(PartitionMethod):
    """
    Randomly select a fraction of test rows per user/item.

    :param frac: the fraction of items to select for testing.
    :paramtype frac: double
    """
    def __init__(self, frac):
        self.fraction = frac

    def __call__(self, udf):
        return udf.sample(frac=self.fraction)


class LastN(PartitionMethod):
    """
    Select a fixed number of test rows per user/item, based on ordering by a
    column.

    :param n: The number of test items to select.
    :paramtype n: integer
    :param col: The column to sort by.
    """

    def __init__(self, n, col='timestamp'):
        self.n = n
        self.column = col

    def __call__(self, udf):
        return udf.sort_values(self.column).iloc[-self.n:]


class LastFrac(PartitionMethod):
    """
    Select a fraction of test rows per user/item.

    :param frac: the fraction of items to select for testing.
    :paramtype frac: double
    :param col: The column to sort by.
    """
    def __init__(self, frac, col='timestamp'):
        self.fraction = frac
        self.column = col

    def __call__(self, udf):
        n = round(len(udf) * self.fraction)
        return udf.sort_values(self.column).iloc[-n:]


def partition_users(data, partitions: int, method: PartitionMethod):
    """
    Partition a frame of ratings or other data into train-test partitions user-by-user.
    This function does not care what kind of data is in `data`, so long as it is a Pandas DataFrame
    (or equivalent) and has a `user` column.

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: :py:class:`pandas.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :param method: The method for selecting test rows for each user.
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """

    user_col = data['user']
    users = user_col.unique()
    _logger.info('partitioning %d rows for %d users into %d partitions',
                 len(data), len(users), partitions)

    # create an array of indexes into user row
    rows = np.arange(len(users))
    # shuffle the indices & split into partitions
    np.random.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        # get our users!
        test_us = users[ts]
        # sample the data frame
        ugf = data[data.user.isin(test_us)].groupby('user')
        test = ugf.apply(method)
        # get rid of the group index
        test = test.reset_index(0, drop=True)
        # now test is indexed on the data frame! so we can get the rest
        rest = data.index.difference(test.index)
        train = data.loc[rest]
        yield TTPair(train, test)


def sample_users(data, partitions: int, size: int, method: PartitionMethod, disjoint=True):
    """
    Create train-test partitions by sampling users.
    This function does not care what kind of data is in `data`, so long as it is
    a Pandas DataFrame (or equivalent) and has a `user` column.

    Args:
        data(pandas.DataFrame):
            Data frame containing ratings or other data you wish to partition.
        partitions(int):
            The number of partitions.
        size(int):
            The sample size.
        method(PartitionMethod):
            The method for obtaining user test ratings.

    Returns:
        iterator: An iterator of train-test pairs (as :class:`TTPair` objects).
    """

    user_col = data['user']
    users = user_col.unique()
    if disjoint and partitions * size >= len(users):
        _logger.warning('cannot take %d disjoint samples of size %d from %d users',
                        partitions, size, len(users))
        for p in partition_users(data, partitions, method):
            yield p
        return

    _logger.info('sampling %d users into %d partitions (n=%d)',
                 len(users), partitions, size)

    if disjoint:
        np.random.shuffle(users)

    # generate our samples
    for i in range(partitions):
        # get our test users!
        if disjoint:
            test_us = users[i*size:(i+1)*size]
        else:
            test_us = np.random.choice(users, size, False)

        # sample the data frame
        test = data[data.user.isin(test_us)].groupby('user').apply(method)
        # get rid of the group index
        test = test.reset_index(0, drop=True)
        # now test is indexed on the data frame! so we can get the rest
        rest = data.index.difference(test.index)
        train = data.loc[rest]
        yield TTPair(train, test)
