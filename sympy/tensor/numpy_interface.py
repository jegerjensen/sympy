"""Module with methods to create numpy arrays out of Indexed objects.

    An Indexed object can be considered a symbolic representation of an array.
    The functions implemented in this module provides a simple user interface for
    construction of numpy arrays that have properties consistent with the symbolic
    objects.

"""

from sympy.core import Symbol, Lambda

def get_ndarray(indexed, function, subs_dict, **kw_args):

    """Convenience function that creates a numpy array based on an Indexed object.

    The construction is forwarded to numpy.fromfunction(func, shape, **kw_args)

    Parameters
    ==========
    indexed -- an instance of the Indexed class
    function -- a callable that takes n integer arguments where n is the number
                of indices on ``indexed''.
    subs_dict  --  An argument that will be passed to indexed.subs() before
                   anything is done.  Here, all remaning symbolic dimensions
                   must be specified with numeric integers.
    kw_args  --  will be passed on to numpy as **kw_args

    >>> from sympy.core import symbols, pi
    >>> from sympy.functions import sin
    >>> from sympy.utilities.lambdify import lambdify
    >>> from sympy.tensor.numpy_interface import get_ndarray, linfunc
    >>> from sympy.tensor import Idx, IndexedBase

    >>> x, y = map(IndexedBase, ['x', 'y'])
    >>> m, n = symbols('m n', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> f = lambdify(i, i**2)
    >>> get_ndarray(x[i], f, {m: 4})    #doctest:+NORMALIZE_WHITESPACE
    [ 0.  1.  4.  9.]

    >>> lf = linfunc(1, 4, 4)
    >>> f = lambdify((i, j), lf(i)*lf(j))
    >>> get_ndarray(x[i, j], f, {m: 4, n: 3})   #doctest:+NORMALIZE_WHITESPACE
    [[  1.   2.   3.]
     [  2.   4.   6.]
     [  3.   6.   9.]
     [  4.   8.  12.]]

    """
    try:
        import numpy
    except:
        raise ImportError("get_ndarray depends on numpy")

    if len(function.func_code.co_varnames) != indexed.rank:
        raise ValueError("Number of function args must match rank of indexed.")

    numindexed = indexed.subs(subs_dict)
    shape = numindexed.shape
    not_numbers = filter(lambda x: not x.is_number, shape)
    if not_numbers:
        raise ValueError("Need integer substitution of %s" %
                ", ".join([str(n) for n in not_numbers]))
    return numpy.fromfunction(function, shape, **kw_args)

def linfunc(start, end, dim, endpoint=True):
    """Returns a function that calculates values in a linspace.

    >>> from sympy.tensor.numpy_interface import linfunc
    >>> f = linfunc(1, 4, 4)
    >>> map(f, [0, 1, 2, 3])
    [1.0, 2.0, 3.0, 4.0]
    >>> f = linfunc(1, 4, 4, endpoint=False)
    >>> map(f, [0, 1, 2, 3])
    [1.0, 1.75, 2.5, 3.25]

    Note that the dim parameter is not a hard limit, it is only used to
    calculate the stepsize:

    >>> f(10), f(-10)
    (8.5, -6.5)

    """
    x = Symbol('x')
    if endpoint:
        return Lambda(x, start + x*(end - start)/(dim-1))
    else:
        return Lambda(x, start + x*(end - start)/dim)
