"""
Racah Algebra

Module for working with spherical tensors.
"""

from sympy import (
    Basic, Function, var, Mul, sympify, Integer, Add, sqrt,
    Number, Matrix, zeros, Pow, I, S,Symbol, latex, cache
)

from sympy.core.cache import cacheit
from sympy.functions import Dij


__all__ = [
        'ThreeJSymbol',
        'ClebschGordanCoefficient',
        'SixJSymbol',
        'SphericalTensor',
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
        >>> ThreeJSymbol(A, C, B, a, c, b)
        (-1)**(A + B + C)*ThreeJSymbol(A, B, C, a, b, c)

        If the phase is applied twice, we take care to remove it immediately:

        >>> ThreeJSymbol(A, C, B, -a, c, b)
        ThreeJSymbol(A, B, C, a, -b, -c)
        """

        # We search for even permuations first, to avoid phases if possible
        if j1 > J:
            return ThreeJSymbol(j2,J,j1,m2,M,m1)

        if j1 > j2:
            phase=pow(S.NegativeOne,j1+j2+J)
            expr = ThreeJSymbol(j2,j1,J,m2,m1,M)
            return cls._determine_phase(phase, expr)

        if j2 > J:
            phase=pow(S.NegativeOne,j1+j2+J)
            expr = ThreeJSymbol(j1,J,j2,m1,M,m2)
            return cls._determine_phase(phase, expr)

        if m1 is S.Zero:
            coeff, term = m2.as_coeff_terms()
            if coeff is S.NegativeOne:
                phase=pow(S.NegativeOne,j1+j2+J)
                expr = ThreeJSymbol(j1, j2, J, -m1, -m2, -M)
                return cls._determine_phase(phase, expr)

        coeff, term = m1.as_coeff_terms()
        if coeff is S.NegativeOne:
            phase=pow(S.NegativeOne,j1+j2+J)
            expr = ThreeJSymbol(j1, j2, J, -m1, -m2, -M)
            return cls._determine_phase(phase, expr)


    @classmethod
    def _determine_phase(cls, phase, tjs):
        # The phase is known to be integer, so it cancels if it appears twice.
        if tjs.has(phase):
            return powsimp(pow(S.NegativeOne,-phase.exp)*tjs)
        else:
            return phase*tjs


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
    (A, a, B, b|C, c)
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

    def _sympystr_(self, *args):
        """
        >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> ClebschGordanCoefficient(A,a,B,b,C,c)
        (A, a, B, b|C, c)
        """

        return "(%s, %s, %s, %s|%s, %s)" % self.args


class SphericalTensor(Basic):
    """
    Represents a spherical tensor(ST), i.e. an object that transforms under
    rotations as defined by the Wigner rotation matrices.

    Every ST has a rank 'k>0' and a projection 'q' such that

    -k <= q <= k

    """
    is_commutative=True

    def __new__(cls, symbol, rank, projection, tensor1=None, tensor2=None):
        """
        Creates a new spherical tensor (ST) with the given rank and
        projection. If two spherical tensors are supplied as tensor1 and
        tensor2, we return a CompositeSphericalTensor instead.
        """
        if tensor1 and tensor2:
            return CompositeSphericalTensor(symbol, rank, projection, tensor1, tensor2)
        else:
            return AtomicSphericalTensor(symbol, rank, projection)

    @property
    def rank(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    @property
    def symbol(self):
        return self.args[0]


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

    def __new__(cls, symbol, rank, projection, tensor1, tensor2):
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
        symbol=sympify(symbol)
        obj = Basic.__new__(cls,symbol,rank,projection,tensor1,tensor2)
        return obj

    @property
    def tensor1(self):
        return self.args[3]

    @property
    def tensor2(self):
        return self.args[4]

    def _sympystr_(self, p, *args):
        """
        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> SphericalTensor('T',C,c,t1,t2)
        T[t1(A, a)*t2(B, b)](C, c)

        """

        tensor_product = "[%s*%s]" %(self.tensor1, self.tensor2)
        rank_projection= "(%s, %s)" %(self.rank, self.projection)
        symbol = p.doprint(self.symbol)

        return symbol+tensor_product+rank_projection

    def get_uncoupled_form(self, **kw_args):
        """
        Returns this composite tensor in terms of the direct product of constituent tensors.

        If the keyword deep=True is supplied, the uncoupling is applied also to the
        two tensors that make up the composite tensor.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> T = SphericalTensor('T',D,d,t1,t2)
        >>> T.get_uncoupled_form()
        Sum(a, b)*t1(A, a)*t2(B, b)*(A, a, B, b|D, d)

        With three coupled tensors we get:

        >>> t3 = SphericalTensor('t3',C,c)
        >>> S = SphericalTensor('S',E,e,T,t3)
        >>> S.get_uncoupled_form()
        Sum(a, b)*Sum(c, d)*t1(A, a)*t2(B, b)*t3(C, c)*(A, a, B, b|D, d)*(D, d, C, c|E, e)

        """

        t1 = self.tensor1
        t2 = self.tensor2
        if kw_args.get('deep',False):

            expr = (ClebschGordanCoefficient(
                    t1.rank,t1.projection,
                    t2.rank,t2.projection,
                    self.rank,self.projection) * t1 * t2
                    )
        else:
            expr = (ClebschGordanCoefficient(
                        t1.rank,t1.projection,
                        t2.rank,t2.projection,
                        self.rank,self.projection)
                    * t1.get_uncoupled_form(**kw_args)
                    * t2.get_uncoupled_form(**kw_args)
                    )
        return ASigma(t1.projection, t2.projection)*expr

    def get_direct_product_ito_self(self, **kw_args):
        """
        Returns the direct product of all constituent tensors in terms of (=ito) self.

        The direct product can be expressed as a sum over composite tensors.
        Note: the returned expression of this method is *not* equal to self.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> T = SphericalTensor('T',D,d,t1,t2)
        >>> T.get_direct_product_ito_self()
        Sum(D, d)*(A, a, B, b|D, d)*T[t1(A, a)*t2(B, b)](D, d)

        With three coupled tensors we get:

        >>> t3 = SphericalTensor('t3',C,c)
        >>> S = SphericalTensor('S',E,e,T,t3)
        >>> S.get_direct_product_ito_self()
        Sum(D, d)*Sum(E, e)*(A, a, B, b|D, d)*(D, d, C, c|E, e)*S[T[t1(A, a)*t2(B, b)](D, d)*t3(C, c)](E, e)

        """

        t1 = self.tensor1
        t2 = self.tensor2
        max_j = t1.rank + t2.rank
        min_j = abs(t1.rank - t2.rank)
        sum_J = (self.rank, min_j, max_j)
        sum_M = (self.projection, -self.rank, self.rank)
        expr = (ClebschGordanCoefficient(
                    t1.rank,t1.projection,
                    t2.rank,t2.projection,
                    self.rank,self.projection)
                * t1.get_direct_product_ito_self(nested=True)
                * t2.get_direct_product_ito_self(nested=True)
                )

        if kw_args.get('nested'):
            return ASigma(self.rank, self.projection)*expr
        else:
            return ASigma(self.rank, self.projection)*expr*self



    def get_ito_other_coupling_order(self,other_coupling):
        """
        Returns an expression that states how this Composite spherical tensor
        can be written in terms of the supplied composite spherical tensor.

        - ``other_coupling``: A composite spherical tensor with the desired
          coupling order.  All non-composite tensors in ``self`` must be present in
          ``other_coupling``.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e,f,g,h = symbols('abcdefgh')
        >>> A,B,C,D,E,F,G,H = symbols('ABCDEFGH')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> t3 = SphericalTensor('t3',C,c)
        >>> T12 = SphericalTensor('T12',D,d,t1,t2)
        >>> T23 = SphericalTensor('T23',E,e,t2,t3)
        >>> S1= SphericalTensor('S1',F,f,T12,t3)
        >>> S2= SphericalTensor('S2',G,g,t1,T23)

        This method tells you how S2 can be expressed in terms of S1:

        >>> S1.get_ito_other_coupling_order(S2)
        Sum(E, e)*Sum(a, b)*Sum(c, d)*(A, a, B, b|D, d)*(A, a, E, e|G, g)*(B, b, C, c|E, e)*(D, d, C, c|F, f)*S2[t1(A, a)*T23[t2(B, b)*t3(C, c)](E, e)](G, g)*Dij(F, G)*Dij(f, g)

        Note how F==G and f==g is expressed with the Kronecker delta, Dij.
        """
        my_tensors = self.atoms(AtomicSphericalTensor)
        assert my_tensors == other_coupling.atoms(AtomicSphericalTensor)

        # Use direct product as a link between coupling schemes
        direct_product = Mul(*my_tensors)
        self_as_direct_product = self.get_uncoupled_form()
        direct_product_ito_other = other_coupling.get_direct_product_ito_self()

        # In the direct product there is a sum over other.rank and
        # other.projection, but for a transformation of coupling scheme the
        # coefficient <(..).:J'M'|.(..);J M> implies that J'==J and M'==M.
        # We solve this by replacing the superfluous summation symbol with
        # Kronecker deltas.
        sumJM = ASigma(other_coupling.rank,other_coupling.projection)
        dij = (Dij(self.rank,other_coupling.rank)*
                Dij(self.projection,other_coupling.projection))
        direct_product_ito_other = direct_product_ito_other.subs(sumJM, dij)

        return self_as_direct_product.subs(direct_product,direct_product_ito_other)


class AtomicSphericalTensor(SphericalTensor):
    """
    Represents a spherical tensor(ST) that is not constructed from other STs
    """

    def __new__(cls, symbol, rank, projection):
        """
        Creates a new spherical tensor (ST) with the given rank and
        projection. If two spherical tensors are supplied as tensor1 and
        tensor2, we return a CompositeSphericalTensor instead.
        """
        symbol=sympify(symbol)
        obj = Basic.__new__(cls,symbol,rank,projection)
        return obj

    def get_uncoupled_form(self):
        """
        Returns the uncoupled, direct product form of a composite tensor.
        """
        return self

    def get_direct_product_ito_self(self,**kw_args):
        """
        Returns the direct product expressed by the composite tensor.
        It is not meaningful for non-composite tensors, we return unity
        to break the recursion.
        """
        return S.One

    def _sympystr_(self, *args):
        """
        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> SphericalTensor('t1',A,a)
        t1(A, a)
        """

        return "%s(%s, %s)" % self.args


class ASigma(Basic):
    """
    Summation symbol.  This object is purely symbolic, and is just used to
    display summation indices.

    """
    is_commutative=True

    def __new__(cls, *indices):
        """
        >>> from sympy.physics.racahalgebra import ASigma
        >>> from sympy import symbols
        >>> a,b = symbols('ab')
        >>> ASigma(b,a)
        Sum(a, b)
        """
        indices = sorted(indices)
        obj = Basic.__new__(cls,*indices)
        return obj

    def _sympystr_(self, p, *args):
        l = [p.doprint(o) for o in self.args]
        return "Sum" + "(%s)"%", ".join(l)

