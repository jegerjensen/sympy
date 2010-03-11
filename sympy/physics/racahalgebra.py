"""
Racah Algebra

Module for working with spherical tensors.
"""

from sympy import (
        Basic, Function, Mul, sympify, Integer, Add, sqrt, Pow, S, Symbol, latex,
        cache, powsimp
        )

from sympy.core.cache import cacheit
from sympy.functions import Dij
from sympy.assumptions import (
        register_handler, remove_handler, Q, ask, Assume, refine
        )
from sympy.assumptions.handlers import AskHandler


__all__ = [
        'ThreeJSymbol',
        'ClebschGordanCoefficient',
        'SixJSymbol',
        'SphericalTensor',
        'refine_tjs2sjs',
        'refine_phases',
        'convert_cgc2tjs',
        'convert_tjs2cgc',
        'combine_ASigmas',
        'remove_summation_indices',
        ]

LOCAL_CACHE = []

def clear_local_cache():
    """clear cache content"""
    for item, cache in LOCAL_CACHE:
        if not isinstance(cache, tuple):
            cache = (cache,)

        for kv in cache:
            kv.clear()


def locally_cacheit(func):
    """caching decorator.

       important: the result of cached function must be *immutable*


       Example
       -------

       @cacheit
       def f(a,b):
           return a+b


       @cacheit
       def f(a,b):
           return [a,b] # <-- WRONG, returns mutable object


       to force cacheit to check returned results mutability and consistency,
       set environment variable SYMPY_USE_CACHE to 'debug'
    """

    func._cache_it_cache = func_cache_it_cache = {}
    LOCAL_CACHE.append((func, func_cache_it_cache))

    def wrapper(*args, **kw_args):
        if kw_args:
            keys = kw_args.keys()
            keys.sort()
            items = [(k+'=',kw_args[k]) for k in keys]
            k = args + tuple(items)
        else:
            k = args
        try:
            return func_cache_it_cache[k]
        except KeyError:
            pass
        func_cache_it_cache[k] = r = func(*args, **kw_args)
        return r

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__

    return wrapper

class ThreeJSymbolsNotCompatibleWithSixJSymbol(Exception):
    pass

class UnableToComplyWithForbiddenAndMandatorySymbols(Exception):
    pass

def initialize_racah():

    class AskHalfIntegerHandler(AskHandler):
        @staticmethod
        def Add(expr, assumptions):
            """
            We only consider sums of the form 'half_integer + integer'
            For other constellations we are inconclusive.

            half_integer + integer      -> half_integer
            half_integer + half_integer -> integer
            else                        -> None
            """
            _result = False
            for arg in expr.args:
                if ask(arg, Q.integer, assumptions):
                    pass
                elif ask(arg, 'half_integer', assumptions):
                    _result = not _result
                else: break
            else:
                return _result

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

        @staticmethod
        def Add(expr, assumptions):
            """
            We only consider sums of the form 'half_integer + half_integer'
            For other constellations we are inconclusive.

            half_integer + half_integer     -> integer
            else                            -> None
            """
            _result = True
            for arg in expr.args:
                if ask(arg, Q.integer, assumptions):
                    pass
                elif ask(arg, 'half_integer', assumptions):
                    _result = not _result
                else: break
            else:
                return _result


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

initialize_racah()

class AngularMomentumSymbol(Function):
    """
    Base class for 3j, 6j, and 9j symbols
    """
    pass

class ThreeJSymbol(AngularMomentumSymbol):
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
            if coeff.is_negative:
                phase=pow(S.NegativeOne,j1+j2+J)
                expr = ThreeJSymbol(j1, j2, J, -m1, -m2, -M)
                return cls._determine_phase(phase, expr)

        coeff, term = m1.as_coeff_terms()
        if coeff.is_negative:
            phase=pow(S.NegativeOne,j1+j2+J)
            expr = ThreeJSymbol(j1, j2, J, -m1, -m2, -M)
            return cls._determine_phase(phase, expr)


    @classmethod
    def _determine_phase(cls, phase, tjs):
        # The phase is known to be integer, so it cancels if it appears twice.
        if tjs.has(phase):
            return tjs.subs(phase, S.One)
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

    def get_triangular_inequalities(self):
        """
        Returns the triangular inequality implied by this 3j-symbol.

        >>> from sympy.physics.racahalgebra import ThreeJSymbol
        >>> from sympy import symbols
        >>> A,B,C,a,b,c = symbols('ABCabc')
        >>> ThreeJSymbol(A, B, C, a, b, c).get_triangular_inequalities()
        set([TriangularInequality(A, B, C)])

        """
        return set([TriangularInequality(*self.magnitudes)])


class SixJSymbol(AngularMomentumSymbol):
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

        >>> from sympy import global_assumptions, Q, Assume
        >>> from sympy.physics.racahalgebra import SixJSymbol
        >>> from sympy import symbols
        >>> a,b,c,d,e,f = symbols('abcdef')
        >>> A,B,C,D,E,F = symbols('ABCDEF')

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
        (-1)**(C + D + F - a - c - e)*Sum(a, b, c, d, e, f)*ThreeJSymbol(A, B, E, a, -e, -b)*ThreeJSymbol(A, C, F, a, f, -c)*ThreeJSymbol(B, D, F, e, -d, -c)*ThreeJSymbol(C, D, E, f, d, b)

        >>> global_assumptions.clear()
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

        # FIXME: sum over M ? Yes -- Heyde claims that the sjs is defined with
        # a sum over *all* projections.  In an expression where the sum over M
        # is missing you can insert a sum and divide by (2J + 1) since the
        # recoupling coefficients are independent of M.
        expr =  refine(powsimp(phase*expr))
        summations = ASigma(m1,m2,m3,M12,M23, M)
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

    def get_triangular_inequalities(self):
        """
        Returns a set containing the triangular conditions implied by this 6j-symbol.

        >>> from sympy.physics.racahalgebra import SixJSymbol
        >>> from sympy import symbols
        >>> A,B,C,D,E,F = symbols('ABCDEF')
        >>> trineq = SixJSymbol(A, E, B, D, F, C).get_triangular_inequalities()
        >>> sorted(trineq)
        [TriangularInequality(A, B, E), TriangularInequality(A, C, F), TriangularInequality(B, D, F), TriangularInequality(C, D, E)]

        """
        j1, j2, j3, j4, j5, j6 = self.args
        triag = set([])
        triag.add( TriangularInequality(j1, j2, j3) )
        triag.add( TriangularInequality(j1, j5, j6) )
        triag.add( TriangularInequality(j4, j5, j3) )
        triag.add( TriangularInequality(j4, j2, j6) )
        return triag


class ClebschGordanCoefficient(AngularMomentumSymbol):
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

    @property
    def projections(self):
        """
        The projection values must sum up to zero
        """
        args = self.args
        return args[1],args[3],args[5]

    @property
    def magnitudes(self):
        """
        The magnitude quantum numbers of angular momentum must obey a triangular inequality.
        """
        args = self.args
        return args[0],args[2],args[4]

    def get_projection_symbol(self,J):
        """
        Returns the symbol associated with the angular momentum magnitude J.

        Note: the symbol is returned without sign or coefficient, and for more
        complicated expressions we return None.  This is in contrast to
        .get_projection().

        >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> A,B,C = symbols('ABC')
        >>> ClebschGordanCoefficient(A,a,B,b,C,c).get_projection_symbol(A)
        a
        >>> ClebschGordanCoefficient(A,a,B,b,C,1+c).get_projection_symbol(C)
        >>> ClebschGordanCoefficient(A,a,B,b,C,2*c).get_projection_symbol(C)
        c
        """
        allJ = self.magnitudes
        allM = self.projections
        for i in range(3):
            if J == allJ[i]:
                M = allM[i]
                if isinstance(M, Symbol):
                    return M
                elif isinstance(M, Mul):
                    sign,M = M.as_coeff_terms()
                    if len(M) == 1 and isinstance(M[0], Symbol):
                        return M[0]
        return None

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

def get_spherical_tensor(symbol, rank, projection, tensor1=None, tensor2=None):
    """
    Creates a new spherical tensor (ST) with the given rank and
    projection. If two spherical tensors are supplied as tensor1 and
    tensor2, we return a CompositeSphericalTensor and if not, we return an
    AtomicSphericalTensor.
    """

    if tensor1 and tensor2:
        return CompositeSphericalTensor(
                symbol, rank, projection, tensor1, tensor2)
    else:
        return AtomicSphericalTensor(symbol, rank, projection)


class SphericalTensor(Basic):
    """
    Represents a spherical tensor(ST), i.e. an object that transforms under
    rotations as defined by the Wigner rotation matrices.

    Every ST has a rank 'k>0' and a projection 'q' such that

    -k <= q <= k

    """
    is_commutative=True

    def __new__(cls, symbol, *args, **kw_args):
        if isinstance(symbol, str):
            # sympify may return unexpected stuff from a string
            symbol = Symbol(symbol)
        else:
            symbol = sympify(symbol)

        if cls == SphericalTensor:
            return get_spherical_tensor(symbol, *args)
        else:
            return Basic.__new__(cls, symbol, *args, **kw_args)

    @property
    def rank(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    def get_rank_proj(self):
        return self.args[1:3]

    @property
    def symbol(self):
        return self.args[0]

    def as_direct_product(self, **kw_args):
        return Mul(*self._eval_as_coeff_direct_product(**kw_args))

    def as_coeff_direct_product(self, **kw_args):
        return self._eval_as_coeff_direct_product(**kw_args)

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

    def _str_drop_projection_(self, *args):
        rank= "%s" %(self.rank,)
        symbol = str(self.symbol)

        return symbol,rank


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
        assert isinstance(tensor1, SphericalTensor)
        assert isinstance(tensor2, SphericalTensor)
        obj = Basic.__new__(cls,symbol,rank,projection,tensor1,tensor2)
        return obj

    @property
    def tensor1(self):
        return self.args[3]

    @property
    def tensor2(self):
        return self.args[4]

    def _str_drop_projection_(self, p, *args):
        tup = ( self.tensor1._str_drop_projection_(p,*args) +
                self.tensor2._str_drop_projection_(p,*args) )
        tensor_product = "[%s(%s)*%s(%s)]" %tup
        rank= "%s" %(self.rank,)
        symbol = p.doprint(self.symbol)

        return symbol+tensor_product,rank

    def _sympystr_(self, p, *args):
        """
        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> SphericalTensor('T',C,c,t1,t2)
        T[t1(A)*t2(B)](C, c)

        """

        tup = ( self.tensor1._str_drop_projection_(p, *args) +
                self.tensor2._str_drop_projection_(p, *args) )
        tensor_product = "[%s(%s)*%s(%s)]" %tup
        rank_projection= "(%s, %s)" %(self.rank, self.projection)
        symbol = p.doprint(self.symbol)

        return symbol+tensor_product+rank_projection

    def _eval_as_coeff_direct_product(self, **kw_args):
        """
        Returns this composite tensor in terms of the direct product of constituent tensors.

        If the keyword deep=False is supplied, the uncoupling is not applied
        to the tensors that make up this composite tensor.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> T = SphericalTensor('T',D,d,t1,t2)
        >>> T.as_direct_product()
        Sum(a, b)*t1(A, a)*t2(B, b)*(A, a, B, b|D, d)
        >>> T.as_direct_product(drop_tensors=True)
        Sum(a, b)*(A, a, B, b|D, d)

        With three coupled tensors we get:

        >>> t3 = SphericalTensor('t3',C,c)
        >>> S = SphericalTensor('S',E,e,T,t3)
        >>> S.as_direct_product()
        Sum(a, b, c, d)*t1(A, a)*t2(B, b)*t3(C, c)*(A, a, B, b|D, d)*(D, d, C, c|E, e)

        """

        t1 = self.tensor1
        t2 = self.tensor2
        coeffs = (ClebschGordanCoefficient(
                t1.rank,t1.projection,
                t2.rank,t2.projection,
                self.rank,self.projection)
                *ASigma(t1.projection, t2.projection))

        if kw_args.get('deep',True):
            c1, t1 = t1._eval_as_coeff_direct_product(**kw_args)
            c2, t2 = t2._eval_as_coeff_direct_product(**kw_args)
            coeffs *= c1*c2

        return combine_ASigmas(coeffs),t1*t2

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
        Sum(D, d)*(A, a, B, b|D, d)*T[t1(A)*t2(B)](D, d)

        With three coupled tensors we get:

        >>> t3 = SphericalTensor('t3',C,c)
        >>> S = SphericalTensor('S',E,e,T,t3)
        >>> S.get_direct_product_ito_self()
        Sum(D, d)*Sum(E, e)*(A, a, B, b|D, d)*(D, d, C, c|E, e)*S[T[t1(A)*t2(B)](D)*t3(C)](E, e)

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
                * t1.get_direct_product_ito_self(drop_self=True)
                * t2.get_direct_product_ito_self(drop_self=True)
                )

        if kw_args.get('drop_self'):
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

        >>> S2.get_ito_other_coupling_order(S1)
        Sum(D, a, b, c, d, e)*(A, a, B, b|D, d)*(A, a, E, e|G, g)*(B, b, C, c|E, e)*(D, d, C, c|F, f)*S1[T12[t1(A)*t2(B)](D)*t3(C)](F, f)*Dij(F, G)*Dij(f, g)

        Note how F==G and f==g is expressed with the Kronecker delta, Dij.

        >>> S1.get_ito_other_coupling_order(S2)
        Sum(E, a, b, c, d, e)*(A, a, B, b|D, d)*(A, a, E, e|G, g)*(B, b, C, c|E, e)*(D, d, C, c|F, f)*S2[t1(A)*T23[t2(B)*t3(C)](E)](G, g)*Dij(F, G)*Dij(f, g)

        """
        my_tensors = self.atoms(AtomicSphericalTensor)
        assert my_tensors == other_coupling.atoms(AtomicSphericalTensor)

        # Use direct product as a link between coupling schemes
        direct_product = Mul(*my_tensors)
        self_as_direct_product = self.as_direct_product()
        direct_product_ito_other = other_coupling.get_direct_product_ito_self()

        # In the direct product there is a sum over other.rank and
        # other.projection, but for a transformation of coupling scheme the
        # coefficient <(..).:J'M'|.(..);J M> implies that J'==J and M'==M.
        # We correct this by replacing the superfluous summation symbol with
        # Kronecker deltas.
        j,m = (other_coupling.rank,other_coupling.projection)
        dij = Dij(self.rank,j)*Dij(self.projection,m)
        direct_product_ito_other = (
                dij* remove_summation_indices(direct_product_ito_other,j,m)
                )

        return combine_ASigmas(self_as_direct_product.subs(
            direct_product,direct_product_ito_other))


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
        obj = Basic.__new__(cls,symbol,rank,projection)
        return obj

    def _eval_as_coeff_direct_product(self, **kw_args):
        """
        Returns the uncoupled, direct product form of a composite tensor.
        """
        if kw_args.get('drop_tensors'):
            return S.One, S.One
        else:
            return S.One, self

    def get_direct_product_ito_self(self,**kw_args):
        """
        Returns the direct product expressed by the composite tensor.
        """
        if kw_args.get('drop_self'):
            return S.One
        else:
            return self

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
        >>> a,b,c = symbols('abc')
        >>> ASigma(b,a)
        Sum(a, b)
        """
        unsigned = []
        for i in indices:
            c,t = i.as_coeff_terms()
            if len(t)==1:
                unsigned.append(t[0])
            else:
                raise ValueError("ASigma doesn't accept products of symbols: %s"%i)
        unsigned.sort()
        obj = Basic.__new__(cls,*unsigned)
        return obj

    def combine(self, other):
        """
        >>> from sympy.physics.racahalgebra import ASigma
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> ASigma(b,a).combine(ASigma(c))
        Sum(a, b, c)
        >>> ASigma(b,a).combine(ASigma(b, c))
        Sum(a, b, b, c)
        """
        assert isinstance(other, ASigma)
        return ASigma(*(self.args + other.args))

    def remove_indices(self, indices):
        """
        Returns the ASigma with the requested indices removed.

        >>> from sympy.physics.racahalgebra import ASigma
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> ASigma(a,b,c).remove_indices([a,c])
        Sum(b)
        >>> ASigma(a,b,c).remove_indices([a,b,c])
        1
        >>> s = ASigma(a,b)
        >>> t = s.remove_indices([])
        >>> s is t
        True

        """
        newargs = list(self.args)
        for i in indices:
            newargs.remove(i)
        if not newargs:
            return S.One
        elif len(newargs) < len(self.args):
            return ASigma(*newargs)
        else:
            return self

    def _sympystr_(self, p, *args):
        l = [p.doprint(o) for o in self.args]
        return "Sum" + "(%s)"%", ".join(l)

class TriangularInequality(Function):
    nargs = 3

    @classmethod
    def eval(cls, j1,j2,j3):
        """
        Represents the triangular inequality between j1, j2 and j3

        |j1-j2| <= j3 <= j1 + j2

        If the relation holds, it holds for all permutations of j1, j2, j3.
        The arguments are sorted such that j1 <= j2 <= j3 for a canonical form.

        If the arguments are numbers, this function evaluates to 1 if
        j1,j2,j3>=0 and the inequality is satisfied.  If any of these
        conditions are violated we return 0.

        >>> from sympy.physics.racahalgebra import TriangularInequality
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> TriangularInequality(c,b,a)
        TriangularInequality(a, b, c)
        >>> TriangularInequality(2,1,1)
        1
        >>> TriangularInequality(4,1,1)
        0

        """
        if j1 > j2: return TriangularInequality(j2, j1, j3)
        if j2 > j3: return TriangularInequality(j1, j3, j2)
        if j1.is_Number and j2.is_Number and j3.is_Number:
            if (abs(j1-j2) <= j3 <= j1+j2) and (j1>=0 and j2>=0 and j3>=0):
                return S.One
            return S.Zero



def refine_phases(expr, forbidden=[], mandatory=[], assumptions=True, **kw_args):
    """
    Simplifies and standardizes expressions containing 3j and 6j symbols.

    The racah algebra produces a lot of factors (-1)**(x + y + ...), a.k.a.
    phases.  This function standardizes the expression by rewriting the phase
    to an equivalent form, simplifying it if possible.

    ``forbidden`` -- iterable containing symbols that cannot be in the phase
    ``mandatory`` -- iterable containing symbols that must be in the phase

    If there is a conflict, or if the algorithm do not succed, the exception
    ``UnableToComplyWithForbiddenAndMandatorySymbols`` is raised.

    To rewrite the expression, we use information from these sources:
        - the information present in the expression
        - information stored as global_assumptions
        - assumptions supplied as an optional argument to this function.
        - expressions supplied in list with the keyword identity_sources=
        - expressions stored in a local cache. (Only if  keep_local_cache==True)

    Powers of (-1) can be simplified a lot if we have assumptions about symbols
    being integers or half-integers.

    Identities applied in this function
    ===================================

    1) The angular momenta involved in triangular equalities, will always sum
    up to an integer, so if the symbols imply |A - B| <= C <= |A + B| we know
    that

        (-1)**(2*A + 2*B + 2*C) == 1.

    2) The Projection quantum numbers in a 3j-symbol will always sum to 0, so
    if a,b,c are projections in a 3j-symbol, we know that

        (-1)**(a + b + c) == 1.

    3) We also know that quantum mechanical angular momenta have projection
    numbers that differ from the magnitude by an integer.  If A and a are
    magnitude and projection respectively, we know that

        (-1)**(2*A - 2*a) == 1.

    >>> from sympy.physics.racahalgebra import SixJSymbol, ThreeJSymbol, refine_phases
    >>> from sympy import symbols, global_assumptions
    >>> a,b,c,d,e,f = symbols('abcdef')
    >>> A,B,C,D,E,F = symbols('ABCDEF')
    >>> global_assumptions.clear()

    >>> expr = (-1)**(a+b+c)*ThreeJSymbol(A,B,C,a,b,c)
    >>> refine_phases(expr, [a, b, c, A, B, C])
    ThreeJSymbol(A, B, C, a, b, c)

    >>> expr = (-1)**(a+b+c)*ThreeJSymbol(A,B,C,a,b,c)
    >>> refine_phases(expr, [a,b,c], [A,B,C])
    (-1)**(-4*A - 4*B - 4*C)*ThreeJSymbol(A, B, C, a, b, c)
    >>> expr = (-1)**(a+b+c)*ThreeJSymbol(A,B,C,a,b,-c)
    >>> refine_phases(expr, [a, b, c, A, B], [C])
    (-1)**(2*C)*ThreeJSymbol(A, B, C, a, b, -c)
    >>> global_assumptions.clear()

    """
    forbidden = set(forbidden)
    mandatory = set(mandatory)
    if forbidden & mandatory: raise UnableToComplyWithForbiddenAndMandatorySymbols

    # fetch the phase
    expr = refine(powsimp(expr), assumptions)
    phase = S.Zero
    for orig_phase_pow in expr.atoms(Pow):
        if orig_phase_pow.base == S.NegativeOne:
            phase = orig_phase_pow.exp
            break
    if phase is S.Zero:
        orig_phase_pow = S.One
        pow_atoms = set([])
    else:
        pow_atoms = phase.atoms(Symbol)

    # determine what should be done
    to_remove = forbidden & pow_atoms
    to_insert = mandatory - pow_atoms
    if not (to_remove or to_insert):
        return expr
    to_keep = mandatory & pow_atoms

    # determine what can be done and setup identities as sympy expressions
    projections = set([])
    triags = set([])
    jm_pairs = set([])
    identity_sources = set(kw_args.get('identity_sources', []))
    identity_sources.update(expr.atoms(AngularMomentumSymbol))
    for njs in identity_sources:
        triags.update(njs.get_triangular_inequalities())
        if isinstance(njs, ThreeJSymbol):
            projections.add(Add(*njs.projections))
            jm = njs.get_magnitude_projection_dict()
            jm_list = [ Add(2*j,2*m) for j,m in jm.items() ]
            jm_pairs.update(jm_list)

            # These are not needed if we can rely on refinement based on assumptions:
            # jm_list = [ Add(2*j,-2*m) for j,m in jm.items() ]
            # jm_pairs.update(jm_list)

    triags = set([2*Add(*t.args) for t in triags])


    # organize information around the forbidden and mandatory symbols
    #
    # the dict known_identities have two kind of keys:
    #   <symbol>: <relevant identities>
    #  <identity: <count how many times the identity has been applied>
    #
    # the count is used to break the recursion, the symbol-identities relation
    # should be used in an intelligent algorithm. FIXME:  we work with
    # addition modulo 2, so can we apply some group theory to speed this up?

    known_identities =dict([])
    for symbol in forbidden | mandatory:
        known_identities[symbol] = []
        for identity in projections:
            if symbol in identity:
                # FIXME: for the brutal approach we skip the symbol keys
                # known_identities[symbol].append(identity)
                known_identities[identity]=0

        for identity in triags | jm_pairs:
            if symbol in identity:
                # FIXME: for the brutal approach we skip the symbol keys
                # known_identities[symbol].append(identity)
                known_identities[identity]=0


    # Since global cache doesn't account for global assumptions, we use
    # a local cache that we reset before recursion:
    if not kw_args.get('keep_local_cache'):
        clear_local_cache()

    better_phase = _brutal_search_for_simple_phase(
            phase, known_identities, forbidden, mandatory)

    if orig_phase_pow is S.One:
        return expr*Pow(-1,better_phase)
    else:
        return expr.subs(orig_phase_pow, Pow(-1,better_phase))




def _brutal_search_for_simple_phase(phase, known_identities,
        forbidden, mandatory, recursion_limit = 1, start=0):
    """
    Tries all combinations of known_identities in order to rewrite the phase.

    The goal is to obtain a phase without forbidden symbols, and with all
    mandatory symbols.  We apply all identities at most once, and if the goal
    is not acheived, we raise an exception.

    Brutal approach  (FIXME)
    """

    # simplify first
    phase = _simplify_Add_modulo2(phase, mandatory)

    current_symbols = phase.atoms(Symbol)
    missing = mandatory - current_symbols
    to_remove = forbidden & current_symbols

    # break recursion if we are done
    if not (missing | to_remove):
        return phase

    id_list = known_identities.keys()
    for i in range(start, len(known_identities)):
        identity = id_list[i]
        if 0 <= known_identities[identity] < recursion_limit:
            known_identities[identity] +=1
            try:
                return _brutal_search_for_simple_phase(
                        phase + identity,
                        known_identities,
                        forbidden, mandatory,
                        recursion_limit,
                        i
                        )
            except UnableToComplyWithForbiddenAndMandatorySymbols:
                known_identities[identity] -= 1
        if 0 >= known_identities[identity] > -recursion_limit:
            known_identities[identity] -=1
            try:
                return _brutal_search_for_simple_phase(
                        phase - identity,
                        known_identities,
                        forbidden, mandatory,
                        recursion_limit,
                        i
                        )
            except UnableToComplyWithForbiddenAndMandatorySymbols:
                known_identities[identity] +=1


    # if we come here, we have tried everything up to recursion_limit
    # without success
    raise UnableToComplyWithForbiddenAndMandatorySymbols

def _simplify_Add_modulo2(add_expr, leave_alone=None):
    """
    We use assumptions to simplify addition modulo 2

    All odd terms simplify to 1
    All even terms simplify to 0

    >>> from sympy.physics.racahalgebra import _simplify_Add_modulo2, clear_local_cache
    >>> from sympy import symbols, global_assumptions, Assume
    >>> a,b,c = symbols('abc')
    >>> A,B,C = symbols('ABC')
    >>> global_assumptions.add( Assume(a, 'half_integer') )
    >>> global_assumptions.add( Assume(b, 'half_integer') )
    >>> global_assumptions.add( Assume(c, 'half_integer') )
    >>> global_assumptions.add( Assume(A, 'integer') )
    >>> global_assumptions.add( Assume(B, 'integer') )
    >>> global_assumptions.add( Assume(C, 'integer') )
    >>> clear_local_cache()

    >>> _simplify_Add_modulo2(2*A)
    0
    >>> _simplify_Add_modulo2(2*a)
    1
    >>> _simplify_Add_modulo2(2*a+2*A+c)
    1 + c
    >>> _simplify_Add_modulo2( b + c + 2*A + 3*a )
    b + c - a

    """
    if add_expr.is_Add:
        # discard even, count odd and collect others
        odd =  0
        others = []
        for arg in add_expr.args:
            if leave_alone and arg in leave_alone:
                # others.append(arg)
                others.append(_standardize_coeff(arg))
            elif _ask_odd(arg):
                odd += 1
            elif _ask_even(arg):
                pass
            else:
                others.append(_standardize_coeff(arg))
        if odd % 2:
            others.append(S.One)
        return Add(*others)
    elif add_expr.is_Mul:
        # trick it into an Add
        return _simplify_Add_modulo2(add_expr+2)
    else:
        return add_expr

def _standardize_coeff(expr):
    """
    make sure that coeffs don't grow big
    that will only happen for half_integer with odd coeff
    we standardise on +/- 1
    """
    c,t = expr.as_coeff_terms()
    if _ask_half_integer(Mul(*t)):
        if c >= 3:
            c = -(c%4 - 2)
        elif c < -1:
            c = c%4
        return Mul(c,t[0])
    elif _ask_integer(Mul(*t)) and (c > 2 or c < -2):
        c = c%2
        if not c: c=2  #if expr didn't vanish already it was not meant to happen!
        return Mul(c,t[0])
    else:
        return expr

@locally_cacheit
def _ask_integer(expr):
    return ask(expr,Q.integer)

@locally_cacheit
def _ask_half_integer(expr):
    return ask(expr,'half_integer')

@locally_cacheit
def _ask_even(expr):
    return ask(expr,Q.even)

@locally_cacheit
def _ask_odd(expr):
    return ask(expr,Q.odd)



def refine_tjs2sjs(expr):
    """
    Tries to rewrite four 3j-symbols to a 6j-symbol.

    >>> from sympy.physics.racahalgebra import SixJSymbol, refine_tjs2sjs
    >>> from sympy import global_assumptions, Q, Assume
    >>> from sympy import symbols
    >>> a,b,z,d,e,f = symbols('abzdef')
    >>> A,B,Z,D,E,F = symbols('ABZDEF')
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
    >>> global_assumptions.add( Assume(Z, 'half_integer') )
    >>> global_assumptions.add( Assume(z, 'half_integer') )

    >>> expr = SixJSymbol(A, B, E, D, Z, F).get_ito_ThreeJSymbols((a,b,e,d,z,f))
    >>> refine_tjs2sjs(expr)
    SixJSymbol(A, B, E, D, Z, F)

    >>> expr = SixJSymbol(A, B, E, D, Z, F).get_ito_ThreeJSymbols((a,b,e,d,z,f), definition='edmonds')
    >>> refine_tjs2sjs(expr)
    SixJSymbol(A, B, E, D, Z, F)

    >>> global_assumptions.clear()

    FIXME: need to call refine_phases() before returning.
    """

    expr = combine_ASigmas(expr)

    for permut in _iter_tjs_permutations(expr):

        summations, phases, factors, threejs, ignorables = permut

        sjs = _identify_SixJSymbol(threejs)

        # expand 6j-symbol i.t.o. 3j-symbols and try to match with original
        M_symbols = []
        for J in sjs.args:
            for tjs in threejs:
                M = tjs.get_projection_symbol(J)
                if M:
                    M_symbols.append(M)
                    break
        assert len(M_symbols)==6
        new_tjs_expr = sjs.get_ito_ThreeJSymbols(M_symbols)

        # There is only one permutation here but we want to split new_tjs_expr:
        for permut2 in _iter_tjs_permutations(new_tjs_expr):
            summations2, phases2, factors2, threejs2, ignorables2 = permut2

        # we can already simplify some parts
        phases = refine(powsimp(phases/phases2))
        factors = factors/factors2

        if threejs2 != threejs:
            # We need to bring the 3js to same form by changing sign of
            # summation indices (projection symbols that are summed over).
            new_tjs = dict([])
            projdict = dict([])
            for tjs in threejs2:
                new_tjs[tjs.magnitudes] = tjs
            for old in threejs:
                new = new_tjs[old.magnitudes]
                projdict[new] =  _find_projections_to_invert(old, new)

            # now we need to process the list of alternative projection inversions
            # in order to find combinations that are consistent for all 3js
            # simultaneously
            alternative_phases = _get_phase_subslist_dict(projdict)

            # choose the simplest phase
            orig = phases
            for alt in alternative_phases:
                test_phase = refine(powsimp(orig*alt))
                if test_phase is S.One or test_phase is S.NegativeOne:
                    phases = test_phase
                    break
                elif orig == phases:
                    phases = test_phase
                elif len(test_phase.exp.args) < len(phases.exp.args):
                    phases = test_phase


        # make sure there is summation over all projections
        for m in [ m for m in M_symbols if not (m in summations.args)]:
            # ... {}{}{}{} ... => ...( sum_abcdef {}{}{}{}/C ) ...
            # C is a factor to compensate for the summations we introduce.
            #
            # Since we are searching for a 6j symbol, the expresion with 4 3j symbols
            # cannot depend on the projections.  (If they do, it means they cannot be
            # rewritten to 6j, so we will fail at a later point.)
            #
            # This means that the factor C is just (2j + 1) for every new sum.
            j = sjs.args[ M_symbols.index(m) ]
            factors = factors/(2*j + 1)

        # remove all projection summations
        summations = summations.remove_indices(
                [m for m in M_symbols if (m in summations.args)])


        # sjs == phases2*factors2*(tjs)^4  =>  (tjs)^4 == sjs/factors2/phases2
        # expr = phases*factors*(tjs)^4 == phases/phases2 * factors/factors2 * sjs

        expr = Mul(summations, phases, sjs, factors, *ignorables)

        # get rid of any projection symbols in the phase
        try:
            expr = refine_phases(expr, M_symbols, identity_sources=threejs)
        except UnableToComplyWithForbiddenAndMandatorySymbols:
            raise ThreeJSymbolsNotCompatibleWithSixJSymbol


    return expr

def _get_phase_subslist_dict(projection_dict):
    """
    Determines possible combinations of projection inversion and corresponding phase.

    ``projection_dict`` must have tjs as keys, and projection lists (from
    _find_projections_to_invert) as values.

    With four 3j-symbols, and two possible inversions for each, we get
    2**4 possible combinations.  A brutal approach may work for rewriting a 6j-
    symbol, but will not scale very well to a 9j-symbol, so we need a better approach.

    If there is any tjs with empty projection list, we know that we must either
    invert none of those projections, or all of them.  This can provide a guide
    when we step through the list of possible combinations.

    But even without any empty projection list, the alternatives for each
    symbol are exclusive.  So that when we step through the alternatives, we
    make a choice sometimes, but often we will be guided by the choices we
    made at earlier steps.

    This is implemnted by recursion.
    """

    def _recurse_into_projections(projlist, will_invert=set([]), must_keep=set([])):

        if will_invert & must_keep:
            raise ThreeJSymbolsNotCompatibleWithSixJSymbol

        if not projlist:
            # break recursion here
            return will_invert, must_keep

        this_step = projlist[0]
        alt1 = this_step[0]
        alt2 = this_step[1]

        # Assume first that we cannot choose at this step
        for p in alt1:
            if p in will_invert:
                will_invert.update(alt1)
                must_keep.update(alt2)
                return _recurse_into_projections(projlist[1:], will_invert, must_keep)
            if p in must_keep:
                will_invert.update(alt2)
                must_keep.update(alt1)
                return _recurse_into_projections(projlist[1:], will_invert, must_keep)

        for p in alt2:
            if p in will_invert:
                will_invert.update(alt2)
                must_keep.update(alt1)
                return _recurse_into_projections(projlist[1:], will_invert, must_keep)
            if p in must_keep:
                will_invert.update(alt1)
                must_keep.update(alt2)
                return _recurse_into_projections(projlist[1:], will_invert, must_keep)

        # If we reach here we have a choice, so we do both

        to_invert = will_invert.copy()
        to_keep = must_keep.copy()
        to_invert.update(alt1)
        to_keep.update(alt2)
        try:
            result1 = _recurse_into_projections(projlist[1:], to_invert, to_keep)
        except ThreeJSymbolsNotCompatibleWithSixJSymbol:
            result1 = None

        to_invert = will_invert.copy()
        to_keep = must_keep.copy()
        to_invert.update(alt2)
        to_keep.update(alt1)
        try:
            result2 = _recurse_into_projections(projlist[1:], to_invert, to_keep)
        except ThreeJSymbolsNotCompatibleWithSixJSymbol:
            if result1 == None: raise
            result2 = None

        # collect both alternatives
        result = []
        if isinstance(result1, list): result.extend(result1)
        elif result1: result.append(result1)
        if isinstance(result2, list): result.extend(result2)
        elif result2: result.append(result2)

        return result


    # determine possible inversions
    inversions = _recurse_into_projections(projection_dict.values())

    # determine phases by doing the inversions
    phase_inversion_dict = dict([])
    tjs_expr = Mul(*projection_dict.keys())
    for inv in inversions:
        to_invert = inv[0]
        subslist = [ (m, -m) for m in to_invert ]
        tjs_expr = powsimp(tjs_expr.subs(subslist))
        eliminate_tjs = [(tjs,S.One) for tjs in tjs_expr.atoms(ThreeJSymbol)]
        phase = refine(tjs_expr.subs(eliminate_tjs))

        phase_inversion_dict[phase] = subslist
    return phase_inversion_dict.keys()


def _substitute_tjs2sjs(threej_atoms, expression, sjs, sjs_ito_tjs):
    """
    Substitutes threej_atoms in expression with sjs.

    ``sjs_ito_tjs`` -- the 6j-symbol expressed in terms of 3j-symbols
    """
    assert len(threej_atoms) == 4
    subsdict = dict([])
    for tjs in threej_atoms:
        subsdict[tjs] = S.One
    sjs_divided_with_phases = sjs/ (sjs_ito_tjs.subs(subsdict))

    subsdict[tjs] = sjs_divided_with_phases
    return powsimp(expression.subs(subsdict))


def _find_projections_to_invert(start, goal):
    """
    Identify ways to harmonize two ThreeJSymbol objects.

    Returns a list of projection symbols that can be inverted so that ``start``
    will end up as a phase times ``goal``.

    Since the 3j-symbol generates a canonical form immediately, there are always
    two possible inversion alternatives. We return a list of lists of the form:
        [ [a, b], [c] ]
    The interpretation is that
        (a and b) or c
    could be inverted, and both alternatives produce a tjs idential to ``goal``.
    Both alternatives work, but they differ by a phase.  The phase-generating
    alternative is always the second element in the list.

    Examples
    ========

    >>> from sympy.physics.racahalgebra import ThreeJSymbol, _find_projections_to_invert
    >>> from sympy import symbols
    >>> a,b,c = symbols('abc')
    >>> A,B,C = symbols('ABC')
    >>> goal = ThreeJSymbol(A, B, C, a, b, c);
    >>> start = ThreeJSymbol(A, B, C, a, b, -c);
    >>> _find_projections_to_invert(start, goal)
    [[c], [a, b]]

    First alternative is without phase:

    >>> start.subs(c, -c)
    ThreeJSymbol(A, B, C, a, b, c)

    The second generates a phase:

    >>> start.subs([(a, -a), (b, -b)])
    (-1)**(A + B + C)*ThreeJSymbol(A, B, C, a, b, c)
    """
    assert start.magnitudes == goal.magnitudes

    # we cannot invert non-atomic projections reliably
    M = tuple([ goal.get_projection_symbol(j) for j in goal.magnitudes ])
    assert not [ e for e in M if e is None ]

    identical = []
    for i in range(3):
        identical.append(start.projections[i]==goal.projections[i])

    identical = tuple(identical)
    alternatives = []

    # (a -b -c)
    if identical == (True, False, False):
        alternatives.append([ M[1], M[2] ])
        alternatives.append([ M[0] ])

    # (a  b -c)
    elif identical == (True, True, False):
        alternatives.append([ M[2] ])
        alternatives.append([ M[0], M[1] ])

    # (a -b  c)
    elif identical == (True, False, True):
        alternatives.append([ M[1] ])
        alternatives.append([ M[0], M[2] ])

    # (a  b  c)
    else:
        alternatives.append([ ])
        alternatives.append([ M[0], M[1], M[2] ])


    return alternatives


def _iter_tjs_permutations(expr):
    """
    Iterates over possible candidates for  4*tjs -> sjs in a Mul

    returns a tuple of lists:
        summations, phases, factors, threejs, ignorables

    summations -- ASigma() containing all summation indices
    phases -- Pow, containing all phases.
    threejs -- list of 4 tjs
    factors -- list of that could be relevant in search for 6j expression
    ignorables -- list of factors that are deemed irrelevant for 6j algorithm.

    The original expression can always be reconstructed as

        expr == Mul(*(threejs + factors + ignorables))

    FIXME!!  Must check:
        - that there are 4 tjs containing only 6 J values
        - that the selected 4 tjs are independent of any remaining tjs
        - that there are summation symbols over the involved projections.


    """
    def combinations(iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        #
        # NOTE: copy-paste from http://docs.python.org/library/itertools.html
        #
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = range(r)
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)


    # First map out expression
    if expr.is_Mul:
        summations = []
        phases = []
        threejs = []
        factors = []
        ignorables = []

        for arg in expr.args:
            if isinstance(arg, ThreeJSymbol):
                threejs.append(arg)
            elif arg.is_Pow and isinstance(arg.base, ThreeJSymbol):
                if arg.exp.is_Integer and arg.exp.is_positive:
                    for i in range(arg.exp):
                        threejs.append(arg.base)
                else:
                    raise ValueError("Confused by %s"%arg)
            elif arg.is_Pow and arg.base is S.NegativeOne:
                phases.append(arg)
            elif isinstance(arg, ASigma):
                summations.append(arg)
            elif isinstance(arg, SphericalTensor):
                ignorables.append(arg)
            else:
                factors.append(arg)
    else:
        raise ValueError("Expected a Mul")

    if not summations: raise ThreeJSymbolsNotCompatibleWithSixJSymbol("No sums!")
    if len(threejs)<4: raise ThreeJSymbolsNotCompatibleWithSixJSymbol("Too few 3js!")

    summations = combine_ASigmas(Mul(*summations))
    phases = powsimp(Mul(*phases))
    factors = Mul(*factors)

    for tjs in combinations(threejs, 4):

        # FIXME: check sensibility of tjs!

        yield tuple([summations, phases, factors, tjs, ignorables])


def _identify_SixJSymbol(threejs):
    """
    Identify and construct the 6j-symbol available for the 3j-symbols.

    ``threejs`` is an iterable containing 4 objects of type ThreeJSymbol.
    """

    def _find_connections(J, threejs):
        conn = set([])
        for tjs in threejs:
            if J in tjs.magnitudes:
                conn.update(tjs.magnitudes)
        return conn

    def _next_to_min(set1):
        min_el = min(set1)
        set1.remove(min_el)
        result = min(set1)
        set1.add(min_el)
        return result

    keys_J = set([])
    connections = dict([])
    for tjs in threejs:
        keys_J.update(tjs.magnitudes)

    # j1 and J are given by canonical form of SixJSymbol
    totJ  = maxJ = max(keys_J)
    j1 = minJ = min(keys_J)
    for j in maxJ, minJ:
        connections[j] = _find_connections(j, threejs)

    if not maxJ in connections[minJ]:
        # this correspond to having minJ and maxJ in the same column
        # we must use the next to minimal J as minJ in order to determine the 6j

        del connections[minJ]
        j1 = minJ = _next_to_min(keys_J)
        connections[minJ] = _find_connections(minJ, threejs)

        # check that sjs is actually possible
        if not maxJ in connections[minJ]:
            raise ThreeJSymbolsNotCompatibleWithSixJSymbol

    # Two elements in a column of the sjs never appear in the same 3j-symbol
    j3 = keys_J - connections[  j1]; j3 = j3.pop()
    j2 = keys_J - connections[totJ]; j2 = j2.pop()

    # J12 is always in a 3j-symbol together with j1 and j2.  J23 ditto.
    J12, J23 = None, None
    for tjs in threejs:
        magnitudes = tjs.magnitudes
        if j1 in magnitudes and j2 in magnitudes:
            if J12 is not None:
                raise ThreeJSymbolsNotCompatibleWithSixJSymbol
            J12 = [ el for el in magnitudes if (el != j1 and el != j2)]
            J12 = J12[0]
        elif j2 in magnitudes and j3 in magnitudes:
            if J23 is not None:
                raise ThreeJSymbolsNotCompatibleWithSixJSymbol
            J23 = [ el for el in magnitudes if (el != j2 and el != j3)]
            J23 = J23[0]

    return  SixJSymbol(j1,j2,J12,j3,totJ,J23)

def invert_clebsch_gordans(expr):
    """
    Replaces every sum over m1, m2 with a sum over J, M.
    (or the other way around.)

    Inverts any other factors.

    >>> from sympy import symbols
    >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient
    >>> from sympy.physics.racahalgebra import invert_clebsch_gordans, ASigma
    >>> a,b,c,A,B,C = symbols('a b c A B C')
    >>> cgc = ClebschGordanCoefficient(A,a,B,b,C,c)
    >>> invert_clebsch_gordans(cgc*ASigma(a,b))
    Sum(C, c)*(A, a, B, b|C, c)
    >>> invert_clebsch_gordans(cgc*ASigma(C,c)*a)
    Sum(a, b)*(A, a, B, b|C, c)/a

    """
    coeff,cgcs = expr.as_coeff_terms(ClebschGordanCoefficient)
    sums = coeff.atoms(ASigma)
    if len(sums) == 0 or len(cgcs) == 0:
        return expr
    coeff = coeff.subs([(s,S.One) for s in sums])
    if len(sums) > 1: indices = set(combine_ASigmas(Mul(*sums)))
    else: indices = set(sums.pop().args)
    for cg in cgcs:
        m1 = cg.get_projection_symbol(cg.args[0])
        m2 = cg.get_projection_symbol(cg.args[2])
        J = cg.args[4]
        M = cg.get_projection_symbol(J)
        m1m2 = set((m1, m2))
        JM = set((J, M))
        if m1m2 <= indices:
            indices -= m1m2
            indices |= JM
        elif JM <= indices:
            indices -= JM
            indices |= m1m2
        else:
            raise CannotInvertClebschGordan(cg)
    return ASigma(*indices)*Mul(*cgcs)/coeff

def combine_ASigmas(expr):
    """
    Combines multiple ASigma factors to one ASigma

    >>> from sympy.physics.racahalgebra import ASigma, combine_ASigmas
    >>> from sympy import symbols
    >>> a,b,c = symbols('abc')
    >>> expr = ASigma(b,a)*ASigma(c);
    >>> combine_ASigmas(expr)
    Sum(a, b, c)

    """
    if isinstance(expr, Add):
        return Add(*[combine_ASigmas(term) for term in expr.args])
    sigmas = expr.atoms(ASigma)
    if sigmas:
        subslist = [ (s,S.One) for s in sigmas ]
        new = sigmas.pop()
        for s in sigmas:
            new = new.combine(s)
        return new*expr.subs(subslist)
    else:
        return expr

def remove_summation_indices(expr, *indices):
    """
    Locates all ASigma in ``expr`` and removes requested indices.

    Note: if you try to remove an index which is not there, you get a ValueError

    >>> from sympy.physics.racahalgebra import ASigma, remove_summation_indices
    >>> from sympy import symbols, Function
    >>> a,b,c = symbols('abc')
    >>> f = Function('f')
    >>> expr = ASigma(a,b)*f(a,b) + ASigma(a)*f(a,c)
    >>> remove_summation_indices(expr, a)
    Sum(b)*f(a, b) + f(a, c)

    """
    if expr.is_Add:
        return Add(*[ remove_summation_indices(arg, *indices) for arg in expr.args ])
    expr = combine_ASigmas(expr)
    sigma = expr.atoms(ASigma).pop()
    new = sigma.remove_indices(indices)
    return expr.subs(sigma,new)

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

