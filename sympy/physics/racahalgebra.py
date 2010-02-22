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
        'ThreeJSymbol',
        'ClebschGordanCoefficient',
        'SixJSymbol',
        'SphericalTensor',
        'CompositeSphericalTensor',
        ]

class ThreeJSymbol(Function):
    """
    class to represent a 3j-symbol
    """
    nargs = 6
    is_commutative=True

    @classmethod
    def eval(cls, j1, j2, J, m1, m2, M):
        """
        The 3j-symbol will be brought to canoncial form by its
        many symmetries.

        We define the canonical form as the form where:
            1)   j1 <= j2  <= J
            2)  m1 is written without minus sign,
                or if m1==0, m2 is written without minus sign.


        >>> from sympy.physics.racahalgebra import ThreeJSymbol
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> A,B,C = symbols('ABC')
        >>> ThreeJSymbol(C, A, B, c, a, b)
        ThreeJSymbol(A, B, C, a, b, c)
        """

        # We search for even permuations first, to avoid phases if possible
        if j1 > J:
            return ThreeJSymbol(j2,J,j1,m2,M,m1)

        if j1 > j2:
            phase=pow(S.NegativeOne,j1+j2+J)
            return phase*ThreeJSymbol(j2,j1,J,m2,m1,M)

        if j2 > J:
            phase=pow(S.NegativeOne,j1+j2+J)
            return phase*ThreeJSymbol(j1,J,j2,m1,M,m2)

        coeff, term = m1.as_coeff_terms()
        if coeff is S.NegativeOne:
            phase=pow(S.NegativeOne,j1+j2+J)
            return phase*ThreeJSymbol(j1, j2, J, -m1, -m2, -M)
        elif coeff is S.Zero:
            coeff, term = m2.as_coeff_terms()
            if coeff is S.NegativeOne:
                phase=pow(S.NegativeOne,j1+j2+J)
                return phase*ThreeJSymbol(j1, j2, J, -m1, -m2, -M)
        else:
            return


class SixJSymbol(Function):
    """
    class to represent a 6j-symbol
    """
    nargs = 6
    is_commutative=True

class ClebschGordanCoefficient(Function):
    """
    Class to represent a Clebsch-Gordan coefficient.

    This class is just a convenience wrapper for ThreeJSymbol.  When objects of
    type ClebschGordanCoefficient are evaluated with .doit(), they are
    immediately rewritten to ThreeJSymbol.

    >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient,ThreeJSymbol
    >>> from sympy import symbols
    >>> a,b,c,d = symbols('abcd')
    >>> A,B,C,D = symbols('ABCD')

    >>> ClebschGordanCoefficient(A, a, B, b, C, c)
    ClebschGordanCoefficient(A, a, B, b, C, c)
    >>> ClebschGordanCoefficient(A, a, B, b, C, c).doit()
    (-1)**(A + c - B)*(1 + 2*C)**(1/2)*ThreeJSymbol(A, B, C, a, b, -c)
    """
    nargs = 6
    is_commutative=True

    # @classmethod
    # def eval(cls, j1, m1, j2, m2, J, M):
        # pass

    def doit(self,**hints):
        """
        Rewrites to a 3j-symbol, which is then evaluated.
        """
        if not hints.get('clebsh_gordan'):
            j1, m1, j2, m2, J, M = self.args
            return (pow(S.NegativeOne,j1 - j2 + M)*sqrt(2*J + 1)
                    *ThreeJSymbol(j1, j2, J, m1, m2, -M).doit(**hints))


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
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> t1 = SphericalTensor(A,a)
        >>> t2 = SphericalTensor(B,b)
        >>> T = SphericalTensor(D,d,t1,t2)
        >>> T.get_uncoupled_form()
        ClebschGordanCoefficient(A, a, B, b, D, d)*SphericalTensor(A, a)*SphericalTensor(B, b)
        >>> t3 = SphericalTensor(C,c)
        >>> S = SphericalTensor(E,e,T,t3)
        >>> S.get_uncoupled_form()
        ClebschGordanCoefficient(A, a, B, b, D, d)*ClebschGordanCoefficient(D, d, C, c, E, e)*SphericalTensor(A, a)*SphericalTensor(B, b)*SphericalTensor(C, c)

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
