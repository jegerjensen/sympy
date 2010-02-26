"""
Racah Algebra

Module for working with spherical tensors.
"""

from sympy import (
    Basic, Function, var, Mul, sympify, Integer, Add, sqrt,
    Number, Matrix, zeros, Pow, I, S,Symbol, latex, cache, powsimp
)

from sympy.core.cache import cacheit
from sympy.functions import Dij
from sympy.assumptions import register_handler, remove_handler, Q, ask, Assume, refine
from sympy.assumptions.handlers import AskHandler


__all__ = [
        'ThreeJSymbol',
        'ClebschGordanCoefficient',
        'SixJSymbol',
        'SphericalTensor',
        ]

def initialize_racah():

    class AskHalfIntegerHandler(AskHandler):
        pass

    class ExtendedIntegerHandler(AskHandler):
        """
        Here we determine if Integer taking into account half-integer symbols.

        Return
            - True if expression is Integer
            - False if expression is Half integer
            - None if inconclusive
        """

        @staticmethod
        def Mul(expr, assumptions):
            """
            FIXME: We only consider products of the form
            'number*half_integer'.  For other constellations we
            are inconclusive.

            odd  * half_integer     -> ~integer
            even * half_integer     -> integer
            """

            coeff, factor = expr.as_coeff_terms()
            if len(factor) == 1:
                factor = factor[0]
                if ask(factor,'half_integer',assumptions):
                    if ask(coeff, Q.odd, assumptions):
                        return False
                    if ask(coeff, Q.even, assumptions):
                        return True

    class ExtendedEvenHandler(AskHandler):
        """
        Here we determine even/odd taking into account half-integer symbols.

        Return
            - True if expression is even
            - False if expression is odd
            - None otherwise

        (The Oddhandler is set up to return "not even".)
        """

        @staticmethod
        def Mul(expr, assumptions):
            """
            FIXME: We only consider products of the form
            'number*half_integer'.  For other constellations we
            are inconclusive.

            odd  * 2 * half_integer     -> odd
            even * 2 * half_integer     -> even
            """

            coeff, factor = expr.as_coeff_terms()
            if len(factor) == 1:
                factor = factor[0]
                if ask(factor,'half_integer',assumptions):
                    if ask(coeff/2, Q.odd, assumptions):
                        return False
                    if ask(coeff/2, Q.even, assumptions):
                        return True


    register_handler('half_integer',AskHalfIntegerHandler)
    register_handler(Q.even, ExtendedEvenHandler)
    register_handler(Q.integer, ExtendedIntegerHandler)

    x = Symbol('x')
    assert ask(x,'half_integer', Assume(x,'half_integer')) == True
    assert ask(x,'half_integer', Assume(x,'half_integer', False)) == False
    assert ask(2*x,Q.even, Assume(x,'half_integer')) == False
    assert ask(2*x,Q.odd, Assume(x,'half_integer')) == True



initialize_racah()

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

    @property
    def projections(self):
        """
        The projection values must sum up to zero
        """
        return self.args[3:]

    @property
    def magnitudes(self):
        """
        The magnitude quantum numbers of angular momentum must obey a triangular inequality.
        """
        return self.args[:-3]

    def get_magnitude_projection_dict(self):
        """
        Returns a dict with magnitudes and projections stored as key, value pairs.
        """
        result = dict([])
        for i in range(3):
            J = self.magnitudes[i]
            M = self.projections[i]
            result[J] = M
        return result


    def get_projection(self,J):
        """
        Returns the projection associated with the angular momentum magnitude J.

        >>> from sympy.physics.racahalgebra import ThreeJSymbol
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> A,B,C = symbols('ABC')
        >>> ThreeJSymbol(A, B, C, a, b, 2*c).get_projection(C)
        2*c
        >>> ThreeJSymbol(A, B, C, a, b, 1+A-c).get_projection(C)
        1 + A - c
        >>> ThreeJSymbol(A, B, C, a, b, -c).get_projection(C)
        -c
        """
        args = self.args
        for i in range(3):
            if J == args[i]:
                return args[i+3]

    def get_projection_symbol(self,J):
        """
        Returns the symbol associated with the angular momentum magnitude J.

        Note: the symbol is returned without sign or coefficient, and for more
        complicated expressions we return None.  This is in contrast to
        .get_projection().

        >>> from sympy.physics.racahalgebra import ThreeJSymbol
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> A,B,C = symbols('ABC')
        >>> ThreeJSymbol(A, B, C, a, b, 2*c).get_projection_symbol(C)
        c
        >>> ThreeJSymbol(A, B, C, a, b, 1-c).get_projection_symbol(C)
        >>> ThreeJSymbol(A, B, C, a, b, -c).get_projection_symbol(C)
        c
        """
        args = self.args
        for i in range(3):
            if J == args[i]:
                M = args[i+3]
                if isinstance(M, Symbol):
                    return M
                elif isinstance(M, Mul):
                    sign,M = M.as_coeff_terms()
                    if len(M) == 1 and isinstance(M[0], Symbol):
                        return M[0]
        return None

    def get_as_ClebschGordanCoefficient(self):
        """
        Returns the equivalent C-G
        """
        j1,j2,J,m1,m2,M = self.args
        factor = (-1)**(j1-j2-M)*Pow(sqrt(2*J+1),-1)
        return factor * ClebschGordanCoefficient(j1,m1,j2,m2,J,-M)



class SixJSymbol(Function):
    """
    class to represent a 6j-symbol
    """
    nargs = 6
    is_commutative=True

    @classmethod
    def eval(cls, j1, j2, J12, j3, J, J23):
        """
        The 6j-symbol will be brought to canoncial form by its symmetries.
        (permute any two columns, or permute rows in any two columns.)

        We define the canonical form as the 6j-symbol that has

            1)  The largest element in position J
            2a) The smallest element in position j1
                                or
            2b)  if j2 is smallest, we want the next smallest in position j1


        Position J is often used for the total angular momentum, when a
        6j-symbol is used to change coupling order of 3 angular momenta,
        (Edmonds, Brink & Satchler).  This convention can be
        enforced  by choosing a symbol for the total angular momentum that
        is 'large' to the < operator, e.g. Z.

        >>> from sympy.physics.racahalgebra import SixJSymbol
        >>> from sympy import symbols
        >>> a,b,c,d,e,f = symbols('abcdef')
        >>> A,B,C,D,E,F = symbols('ABCDEF')
        >>> sjs = SixJSymbol(A, B, E, D, C, F); sjs
        SixJSymbol(A, E, B, D, F, C)



        """
        args = [j1, j2, J12, j3, J, J23]
        maxind = args.index(max(args))

        # get maxJ in correct column
        if maxind != 4 and maxind != 1:
            maxcol = maxind %3
            return SixJSymbol(*SixJSymbol._swap_col(1,maxcol,args))

        minind = args.index(min((j1, J12, j3, J23)))

        # get minJ in correct column
        if minind != 0 and minind != 3:
            return SixJSymbol(*SixJSymbol._swap_col(0,2,args))

        # get both in correct row
        elif minind == 3 and maxind == 1:
            return SixJSymbol(*SixJSymbol._swap_row(0,1,args))

        # move j1, keep J
        elif minind == 3:
            return SixJSymbol(*SixJSymbol._swap_row(0,2,args))

        # keep j1, move J
        elif maxind == 1:
            return SixJSymbol(*SixJSymbol._swap_row(1,2,args))




    def get_ito_ThreeJSymbols(self,projection_labels, **kw_args):
        """
        Returns the 6j-symbol expressed with 4 3j-symbols.

        In order to simplify the phase upfront, we introduce global
        assumptions about the angular momenta in the expressions.  For
        a fermionic system, we will have A,a,B,b,D,d as half-integers.
        Combinations of those angular momenta, denoted E,e,F and f, are
        integers, while the total C,c is again half integer.
        This information can be entered as:

        >>> from sympy.physics.racahalgebra import SixJSymbol
        >>> from sympy import symbols
        >>> a,b,c,d,e,f = symbols('abcdef')
        >>> A,B,C,D,E,F = symbols('ABCDEF')

        >>> from sympy import global_assumptions, Q, Assume
        >>> global_assumptions.add( Assume(a, 'half_integer') )
        >>> global_assumptions.add( Assume(b, 'half_integer') )
        >>> global_assumptions.add( Assume(d, 'half_integer') )
        >>> global_assumptions.add( Assume(A, 'half_integer') )
        >>> global_assumptions.add( Assume(B, 'half_integer') )
        >>> global_assumptions.add( Assume(D, 'half_integer') )
        >>> global_assumptions.add( Assume(e, Q.integer) )
        >>> global_assumptions.add( Assume(f, Q.integer) )
        >>> global_assumptions.add( Assume(E, Q.integer) )
        >>> global_assumptions.add( Assume(F, Q.integer) )
        >>> global_assumptions.add( Assume(C, 'half_integer') )
        >>> global_assumptions.add( Assume(c, 'half_integer') )

        >>> sjs = SixJSymbol(A, B, E, D, C, F);
        >>> sjs.get_ito_ThreeJSymbols((a,b,e,d,c,f))
        (-1)**(E + F - a - c - e)*Sum(e, f)*Sum(a, b, d)*ThreeJSymbol(A, B, E, a, -b, -e)*ThreeJSymbol(A, C, F, a, -c, f)*ThreeJSymbol(B, D, F, b, d, f)*ThreeJSymbol(C, D, E, c, d, -e)
        """

        (j1, j2, J12, j3, J, J23) = self.args
        (m1, m2, M12, m3, M, M23) = projection_labels

        definition = kw_args.get('definition') or 'brink_satchler'

        if definition == 'edmonds':
            # phase = pow(S.NegativeOne,j1+m1+j2+m2+j3+m3+J12+M12+J23+M23+J+M)
            phase = pow(S.NegativeOne,j1+m1+j2+j3+J12+M12+J23+J+M)
            # phase = pow(S.NegativeOne,j1+j2+j3+m3+J12+J23+M23+J+M)
            expr = (ThreeJSymbol(j1,j2,J12,m1,m2,M12)*
                    ThreeJSymbol(j1,J,J23,-m1,M,-M23)*
                    ThreeJSymbol(j3,j2,J23,-m3,-m2,M23)*
                    ThreeJSymbol(j3,J,J12,m3,-M,-M12))



        elif definition == 'brink_satchler':
            phase = pow(S.NegativeOne,j1+J12+J-m1-M12-M)
            expr = ( ThreeJSymbol(j1,J23,J, m1,M23,-M)*
                    ThreeJSymbol(J,j3,J12,M, m3,-M12)*
                    ThreeJSymbol(J12,j2,j1,M12,m2,-m1)*
                    ThreeJSymbol(j2,j3,J23,m2,m3,M23))

        elif definition == 'cgc':
            phase= Pow(S.NegativeOne, j1+j2+j3+J)
            hats = Pow(sqrt(2*J12+1)*sqrt(2*J23+1)*(2*J+1), -1)
            cgc_list = (ClebschGordanCoefficient(j1,m1,j2,m2,J12,M12),
                    ClebschGordanCoefficient(J12,M12,j3,m3,J,M),
                    ClebschGordanCoefficient(j2,m2,j3,m3,J23,M23),
                    ClebschGordanCoefficient(j1,m1,J23,M23,J,M))
            tjs_list = [ cgc.get_as_ThreeJSymbol() for cgc in cgc_list ]
            expr = phase*hats*Mul(*tjs_list)

        expr =  refine(powsimp(phase*expr))

        summations = ASigma(m1,m2,m3)*ASigma(M12,M23)



        return summations*expr



    @classmethod
    def _swap_col(cls,i,j,arg_list):
        rowlen = 3
        new_args = list(arg_list)
        new_args[i] = arg_list[j]
        new_args[j] = arg_list[i]
        new_args[i+rowlen] = arg_list[j+rowlen]
        new_args[j+rowlen] = arg_list[i+rowlen]
        return new_args



    @classmethod
    def _swap_row(cls,i,j,arg_list):
        rowlen = 3
        new_args = list(arg_list)
        new_args[i] = arg_list[i+rowlen]
        new_args[j] = arg_list[j+rowlen]
        new_args[i+rowlen] = arg_list[i]
        new_args[j+rowlen] = arg_list[j]
        return new_args


class ClebschGordanCoefficient(Function):
    """
    Class to represent a Clebsch-Gordan coefficient.

    This class is mostly a convenience wrapper for ThreeJSymbol.  In contrast
    to objects of type ThreeJSymbol, the C-G coeffs are not rewritten to
    canonical form upon creation.  This may be convenient in the sense that the
    objects on screen correspond to the input you provided.  For internal use,
    all C-G coefficients are rewritten as tjs before calculations.

    >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient,ThreeJSymbol
    >>> from sympy import symbols
    >>> a,b,c = symbols('abc')
    >>> A,B,C = symbols('ABC')

    >>> ClebschGordanCoefficient(A, a, B, b, C, c)
    (A, a, B, b|C, c)
    >>> ClebschGordanCoefficient(A, a, B, b, C, c).get_as_ThreeJSymbol()
    (-1)**(A + c - B)*(1 + 2*C)**(1/2)*ThreeJSymbol(A, B, C, a, b, -c)
    """
    nargs = 6
    is_commutative=True

    # @classmethod
    # def eval(cls, j1, m1, j2, m2, J, M):
        # pass

    def get_as_ThreeJSymbol(self):
        """
        Rewrites to a 3j-symbol on canonical form
        """
        j1, m1, j2, m2, J, M = self.args
        return (Pow(S.NegativeOne,j1 - j2 + M)*sqrt(2*J + 1)
                *ThreeJSymbol(j1, j2, J, m1, m2, -M))

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

def convert_tjs2cgc(expr):
    subslist = []
    threejs = expr.atoms(ThreeJSymbol)
    for tjs in threejs:
        subslist.append((tjs,tjs.get_as_ClebschGordanCoefficient()))
    return expr.subs(subslist)

def convert_cgc2tjs(expr):
    subslist = []
    cgcoeffs= expr.atoms(ClebschGordanCoefficient)
    for cgc in cgcoeffs:
        subslist.append((cgc,cgc.get_as_ThreeJSymbol()))
    return expr.subs(subslist)

