import numpy as np
import pandas as pd

from pytest import approx

from lenskit.topn import recall


def _test_recall(items, rel):
    recs = pd.DataFrame({'item': items})
    truth = pd.DataFrame({'item': rel}).set_index('item')
    return recall(recs, truth)


def test_recall_empty_zero():
    prec = _test_recall([], [1, 3])
    assert prec == approx(0)


def test_recall_norel_na():
    prec = _test_recall([1, 3], [])
    assert prec is None


def test_recall_simple_cases():
    prec = _test_recall([1, 3], [1, 3])
    assert prec == approx(1.0)

    prec = _test_recall([1], [1, 3])
    assert prec == approx(0.5)

    prec = _test_recall([1, 2, 3, 4], [1, 3])
    assert prec == approx(1.0)

    prec = _test_recall([1, 2, 3, 4], [1, 3, 5])
    assert prec == approx(2.0 / 3)

    prec = _test_recall([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = _test_recall([1, 2, 3, 4], range(4, 9))
    assert prec == approx(0.2)


def test_recall_series():
    prec = _test_recall(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3]), pd.Series([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_set():
    prec = _test_recall(pd.Series([1, 2, 3, 4]), [1, 3, 5, 7])
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), range(4, 9))
    assert prec == approx(0.2)


def test_recall_series_index():
    prec = _test_recall(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_array():
    prec = _test_recall(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), np.arange(4, 9, 1, 'u4'))
    assert prec == approx(0.2)


def test_recall_array():
    prec = _test_recall(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(np.array([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(np.array([1, 2, 3, 4]), np.arange(4, 9, 1, 'u4'))
    assert prec == approx(0.2)
