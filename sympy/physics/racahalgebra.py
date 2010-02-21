"""
Racah Algebra

Module for working with spherical tensors.
"""

from sympy import (
    Basic, Function, var, Mul, sympify, Integer, Add, sqrt,
    Number, Matrix, zeros, Pow, I, S,Symbol, latex, cache
)

from sympy.utilities import deprecated, iff

from sympy.core.cache import cacheit


__all__ = [
        ThreeJsymbol,
        ClebschGordanCoefficient,
        SixJsymbol,
        ]

class ThreeJsymbol(Function):
    """
    class to represent a 3j-symbol
    """
    nargs = 6
    is_commutative=True

class SixJsymbol(Function):
    """
    class to represent a 3j-symbol
    """
    nargs = 6
    is_commutative=True

class ClebschGordanCoefficient(Function):
    """
    class to represent a 3j-symbol
    """
    nargs = 6
    is_commutative=True


