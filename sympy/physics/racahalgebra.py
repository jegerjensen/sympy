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



class SphericalTensor(Basic):
    """
    Represents a spherical tensor(ST), i.e. an object that transforms under
    rotations as defined by the Wigner rotation matrices.

    Every ST has a rank 'k>0' and a projection 'q' such that

    -k <= q <= k

    """

    def __new__(cls, rank, projection, tensor1=None, tensor2=None):
        """
        Creates a new spherical tensor (ST) with the given rank and
        projection. If two spherical tensors are supplied as tensor1 and
        tensor2, we return a CompositeSphericalTensor instead.
        """
        if tensor1 and tensor2:
            return CompositeSphericalTensor(rank, projection, tensor1, tensor2)
        obj = Basic.__new__(cls,rank,projection)
        return obj

    @property
    def rank(self):
        return self.args[0]

    @property
    def projection(self):
        return self.args[1]

    def get_uncoupled_form(self):
        """
        Returns the uncoupled, direct product form of a composite tensor.
        """
        return self

class CompositeSphericalTensor(SphericalTensor):
    """
    Represents a composite spherical tensor(CST), i.e. an object that
    transforms under rotations as defined by the Wigner rotation matrices. And
    is considered the tensor product of two other spherical tensors.

    The CST is the result of a tensor product:

      k                         k
    T    =  [tensor1 x tensor2]
      q                         q

    You may build up a composite tensor with any coupling scheme this way.

    """

    def __new__(cls, rank, projection, tensor1, tensor2):
        """
        Creates a new spherical tensor (ST) with the given rank and
        projection. If two spherical tensors are supplied as tensor1 and
        tensor2, the new ST will be considered as the result of a tensor
        product:

          k                         k
        T    =  [tensor1 x tensor2]
          q                         q

        You may build up a composite tensor with any coupling scheme this way.

        """
        obj = Basic.__new__(cls,rank,projection,tensor1,tensor2)
        return obj

    @property
    def tensor1(self):
        return self.args[2]

    @property
    def tensor2(self):
        return self.args[3]

    def get_uncoupled_form(self, **kw_args):
        """
        Returns the uncoupled, direct product form of a composite tensor.

        If the keyword deep=True is supplied, the uncoupling is applied also to the
        two tensors that make up the composite tensor.

        >>> from sympy.physics.racahalgebra import SphericalTensor, CompositeSphericalTensor
        >>> from sympy import symbols
        >>> k,l,m,n,K = symbols('klmnK')
        >>> q,r,s,t,Q = symbols('qrstQ')

        >>> t1 = SphericalTensor(l,r)
        >>> t2 = SphericalTensor(m,s)
        >>> T = SphericalTensor(k,q,t1,t2)
        >>> T.get_uncoupled_form()
        ClebschGordanCoefficient(l, r, m, s, k, q)*SphericalTensor(l, r)*SphericalTensor(m, s)
        >>> t3 = SphericalTensor(n,t)
        >>> S = SphericalTensor(K,Q,T,t3)
        >>> S.get_uncoupled_form()
        ClebschGordanCoefficient(k, q, n, t, K, Q)*ClebschGordanCoefficient(l, r, m, s, k, q)*SphericalTensor(l, r)*SphericalTensor(m, s)*SphericalTensor(n, t)

        """

        t1 = self.tensor1
        t2 = self.tensor2
        if kw_args.get('deep',False):
            return ClebschGordanCoefficient(
                    t1.rank,t1.projection,
                    t2.rank,t2.projection,
                    self.rank,self.projection) * t1 * t2
        else:
            return (
                    ClebschGordanCoefficient(
                        t1.rank,t1.projection,
                        t2.rank,t2.projection,
                        self.rank,self.projection)
                    * t1.get_uncoupled_form(**kw_args)
                    * t2.get_uncoupled_form(**kw_args)
                )
