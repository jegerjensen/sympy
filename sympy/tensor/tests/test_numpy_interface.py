try:
    import numpy
except ImportError:
    disabled = True

from sympy.core import symbols
from sympy.utilities.pytest import raises
from sympy.tensor.indexed import IndexedBase, Idx
from sympy.tensor.numpy_interface import get_ndarray, linfunc


def test_get_ndarray():
    A, x, y = map(IndexedBase, ('A', 'x', 'y'))
    m, n = symbols('m n', integer=True)
    i = Idx('i',m)
    j = Idx('j',n)

    r = get_ndarray(A[i, j], lambda i,j: i + j, {n: 3, m: 5})
    expected = numpy.array(
            [[ 0,  1,  2 ],
             [ 1,  2,  3 ],
             [ 2,  3,  4 ],
             [ 3,  4,  5 ],
             [ 4,  5,  6 ]])
    assert numpy.all(r == expected)

def test_get_ndarray_failure():
    A, x, y = map(IndexedBase, ('A', 'x', 'y'))
    m, n = symbols('m n', integer=True)
    i = Idx('i',m)
    j = Idx('j',n)

    raises(ValueError, 'get_ndarray(A[i, j], lambda i,j: i + j, {n: 3})')
    raises(ValueError, 'get_ndarray(A[i, j], lambda i: 2*i, {n: 3, m: 5})')

def test_linfunc():
    A, x, y = map(IndexedBase, ('A', 'x', 'y'))
    m, n = symbols('m n', integer=True)
    i = Idx('i',m)
    j = Idx('j',n)

    f = linfunc(1, 4, 4)
    assert f(0) == 1
    assert f(3) == 4
    assert f(-2) == -1
    assert f(4) == 5
