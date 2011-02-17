"""
Racah Algebra

Module for working with spherical tensors.
"""

from sympy import (
        Expr, Function, Mul, sympify, Integer, Add, sqrt, Pow, S, Symbol, latex,
        cache, powsimp, ratsimp, simplify, sympify
        )

from sympy.core.cache import cacheit
from sympy.functions import Dij
from sympy.printing.pretty.stringpict import  prettyForm
from sympy.assumptions import (
        register_handler, remove_handler, Q, ask, Assume, refine
        )
from sympy.assumptions.handlers import CommonHandler
from sympy.logic.boolalg import conjuncts

def pretty_hash(expr):
    return expr._pretty_key_()

sortkey = pretty_hash

__all__ = [
        'ThreeJSymbol',
        'ClebschGordanCoefficient',
        'SixJSymbol',
        'NineJSymbol',
        'SphericalTensor',
        'refine_tjs2sjs',
        'refine_phases',
        'convert_cgc2tjs',
        'convert_tjs2cgc',
        'combine_ASigmas',
        'evaluate_sums',
        'apply_deltas',
        'apply_orthogonality',
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

    class AskHalfIntegerHandler(CommonHandler):

        @staticmethod
        def Expr(expr, assumptions):
            assumps = conjuncts(assumptions)
            if ~Q.half_integer(expr) in assumps:
                return False
            elif Q.half_integer(expr) in assumps:
                return True
            elif Q.integer(expr) in assumps:
                return False

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

    class ExtendedIntegerHandler(CommonHandler):
        """
        Here we determine if Integer taking into account half-integer symbols.

        Return
            - True if expression is Integer
            - False if expression is Half integer
            - None if inconclusive
        """

        @staticmethod
        def Expr(expr, assumptions):
            assumps = conjuncts(assumptions)
            if Q.half_integer(expr) in assumps:
                return False

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


    class ExtendedEvenHandler(CommonHandler):
        """
        Here we determine even/odd taking into account half-integer symbols.

        Return
            - True if expression is even
            - False if expression is odd
            - None otherwise

        (The Oddhandler is set up to return "not even".)
        """

        @staticmethod
        def Expr(expr, assumptions):
            assumps = conjuncts(assumptions)
            if Q.half_integer(expr) in assumps:
                return False

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

        If one of the angular momenta is zero, we immediately simplify to deltafunctions.

        >>> ThreeJSymbol(A, B, 0, a, b, 0)
        (-1)**(A - a)*Dij(A, B)*Dij(a, -b)/(1 + 2*A)**(1/2)
        """

        # We search for even permuations first, to avoid phases if possible
        if sortkey(j1) > sortkey(J):
            return ThreeJSymbol(j2,J,j1,m2,M,m1)

        if sortkey(j1) > sortkey(j2):
            phase=pow(S.NegativeOne,j1+j2+J)
            expr = ThreeJSymbol(j2,j1,J,m2,m1,M)
            return cls._determine_phase(phase, expr)

        if sortkey(j2) > sortkey(J):
            phase=pow(S.NegativeOne,j1+j2+J)
            expr = ThreeJSymbol(j1,J,j2,m1,M,m2)
            return cls._determine_phase(phase, expr)

        # If any j is zero, it is now in position j1
        if j1 is S.Zero:
            return (-1)**(j2-m2)*(2*j2 + 1)**(S.NegativeOne/2)*Dij(j2, J)*Dij(m2, -M)*Dij(m1, S.Zero)

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

    def _latex(self, p, exp=None):
        """
        >>> from sympy.physics.racahalgebra import ThreeJSymbol
        >>> from sympy import symbols, latex
        >>> A,B,C,a,b,c = symbols('ABCabc')
        >>> print latex(ThreeJSymbol(A, B, C, a, b, c))
        %
        \\left(
          \\begin{array}{ccc}
           A & B & C \\\\
           a & b & c
          \\end{array}
        \\right)
        """

        magn = " & ".join([ p._print(J) for J in self.magnitudes  ])
        proj = " & ".join([ p._print(J) for J in self.projections ])
        res = "%%\n\\left(\n  \\begin{array}{ccc}\n   %s \\\\\n   %s\n  \\end{array}\n\\right)" % (magn, proj)

        if exp:
            res += "^{%s}" % exp

        return res


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
        maxind = args.index(max(args, key=sortkey))

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
        if isinstance(projection_labels, dict):
            d = projection_labels
            projection_labels = [ d[key] for key in self.args ]

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
            (A, B, E, D, C, F) = self.args
            (a, b, e, d, c, f) = projection_labels
            phase = pow(S.NegativeOne, A+E+C-a-e-c)
            expr = (ThreeJSymbol(A, F, C, a, f,-c)*
                    ThreeJSymbol(C, D, E, c, d,-e)*
                    ThreeJSymbol(E, B, A, e, b,-a)*
                    ThreeJSymbol(B, D, F, b, d, f))

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

    def _latex(self, p, exp=None):
        """
        >>> from sympy.physics.racahalgebra import SixJSymbol
        >>> from sympy import symbols, latex
        >>> A,B,C,D,E,F = symbols('ABCDEF')
        >>> print latex(SixJSymbol(A, E, B, D, F, C))
        %
        \\left\\{
          \\begin{array}{ccc}
           A & E & B \\\\
           D & F & C
          \\end{array}
        \\right\\}
        """
        magn = " & ".join([ p._print(J) for J in self.args[:3]])
        proj = " & ".join([ p._print(J) for J in self.args[3:]])
        res = "%%\n\\left\\{\n  \\begin{array}{ccc}\n   %s \\\\\n   %s\n  \\end{array}\n\\right\\}" % (magn, proj)
        if exp:
            res += "^{%s}" % exp

        return res

class NineJSymbol(AngularMomentumSymbol):
    """
    class to represent a 9j-symbol
    """
    nargs = 9
    is_commutative=True

    @classmethod
    def eval(cls, j1, j2, j3, j4, j5, j6, j7, j8, j9):
        pass

    def canonical_form(self):
        """
        Rewrites this 9j-symbol to the canonical form.

        We define the canonical form of the 9j-symbol depending on the order of
        the elements when they are sorted according to SymPy.

        The canonicalization proceeds through the following steps

           1. The `smallest' element is placed in the top left corner

           2. The 4 elements in the lower right corner are arranged such that
              the `largest' of them is in the lower right corner

           3. If the element in the upper right corner is `larger' than the
              element in the lower left corner, we invoke a reflection over the
              main diagonal.
        """
        expr = self._place_largest()
        njs = expr.atoms(NineJSymbol).pop()
        expr = expr.subs(njs, njs._place_smallest())
        njs = expr.atoms(NineJSymbol).pop()
        j = njs.args
        if j[2] == sorted([j[2], j[6]]).pop():
            expr = expr.subs(njs, njs.reflect_major_diagonal())
        return expr

    def _place_smallest(self):
        args = list(self.args)
        lower_square = [args[4], args[5], args[7], args[8]]
        j8 = sorted(lower_square).pop()
        pos = args.index(j8)
        col = self.col_of_arg_number(pos)
        row = self.row_of_arg_number(pos)
        result1 = self.permute_rows(2, row)
        njs = result1.atoms(NineJSymbol).pop()
        result2 = njs.permute_columns(2, col)
        if (col*row == 1):
            # the phase cancels if it appears twice
            return result2.atoms(NineJSymbol).pop()
        else:
            return result1.subs(njs, result2)

    def _place_largest(self):
        j1 = sorted(self.args)[0]
        args = list(self.args)
        pos = args.index(j1)
        col = self.col_of_arg_number(pos)
        row = self.row_of_arg_number(pos)
        result1 = self.permute_rows(0, row)
        njs = result1.atoms(NineJSymbol).pop()
        result2 = njs.permute_columns(0, col)
        if (col and row):
            # the phase cancels if it appears twice
            return result2.atoms(NineJSymbol).pop()
        else:
            return result1.subs(njs, result2)

    @classmethod
    def row_of_arg_number(cls, arg_number):
        """Returns the row corresponding to argument number <arg_number>
        """
        if 0 <= arg_number < 3:
            return 0
        if 3 <= arg_number < 6:
            return 1
        if 6 <= arg_number < 9:
            return 2

    @classmethod
    def col_of_arg_number(cls, arg_number):
        """Returns the column corresponding to argument number <arg_number>
        """
        if arg_number in [0, 3, 6]:
            return 0
        if arg_number in [1, 4, 7]:
            return 1
        if arg_number in [2, 5, 8]:
            return 2

    def reflect_major_diagonal(self):
        """Returns an equivalent 9j-symbol, reflected across the main diagonal"""
        j1, j2, j3, j4, j5, j6, j7, j8, j9 = self.args
        return NineJSymbol(j1, j4, j7, j2, j5, j8, j3, j6, j9)

    def reflect_minor_diagonal(self):
        """Returns an equivalent 9j-symbol, reflected across the secondary diagonal"""
        j1, j2, j3, j4, j5, j6, j7, j8, j9 = self.args
        return NineJSymbol(j9, j6, j3, j8, j5, j2, j7, j4, j1)

    def permute_rows(self, row1, row2):
        """Rewrites self by permuting row1 and row2"""
        assert 0 <= row1 <= 2
        assert 0 <= row2 <= 2
        if row1 == row2:
            return self
        phase = (-1)**(Add(*self.args))
        r0 = self.args[0:3]
        r1 = self.args[3:6]
        r2 = self.args[6:9]
        if row1 > row2:
            row1, row2 = row2, row1
        if (row1, row2) == (0, 1):
            args = r1 + r0 + r2
        elif (row1, row2) == (0, 2):
            args = r2 + r1 + r0
        elif (row1, row2) == (1, 2):
            args = r0 + r2 + r1
        return phase*NineJSymbol(*args)

    def permute_columns(self, col1, col2):
        """Rewrites self by permuting col1 and col2"""
        reflected = self.reflect_major_diagonal()
        permuted = reflected.permute_rows(col1, col2)
        njs = permuted.atoms(NineJSymbol).pop()
        return permuted.subs(njs, njs.reflect_major_diagonal())

    def _latex(self, p, exp=None):
        """
        >>> from sympy.physics.racahalgebra import NineJSymbol
        >>> from sympy import symbols, latex
        >>> a,b,c,d,e,f,g,h,i = symbols('abcdefghi')
        >>> print latex(NineJSymbol(a,b,c,d,e,f,g,h,i))
        %
        \\left(
          \\begin{array}{ccc}
           a & b & c \\\\
           d & e & f \\\\
           g & h & i
          \\end{array}
        \\right)
        """

        row1 = " & ".join([p._print(J) for J in self.args[0:3]])
        row2 = " & ".join([p._print(J) for J in self.args[3:6]])
        row3 = " & ".join([p._print(J) for J in self.args[6:9]])
        template = """%%
\\left(
  \\begin{array}{ccc}
    %s \\\\
    %s \\\\
    %s
  \\end{array}
\\right)"""
        result = template % (row1, row2, row3)
        if exp:
            result += "^{%s}" % exp
        return result

    def get_ito_ThreeJSymbols(self, projection_labels):
        """Returns the 3j symbol expression for this 9j-symbol

        >>> from sympy.physics.racahalgebra import NineJSymbol
        >>> from sympy import global_assumptions, Q, assume_all
        >>> from sympy import symbols
        >>> a,b,c,d,e,f,g,h,i = symbols('abcdefghi')
        >>> A,B,C,D,E,F,G,H,I = symbols('ABCDEFGHI', nonnegative=True)
        >>> global_assumptions.add(*assume_all([a,b,d,e], 'half_integer'))
        >>> global_assumptions.add(*assume_all([A,B,D,E], 'half_integer'))
        >>> global_assumptions.add(*assume_all([c,f,g,h,i], 'integer'))
        >>> global_assumptions.add(*assume_all([C,F,G,H,I], 'integer'))

        >>> NineJSymbol(A,B,C,D,E,F,G,H,I).get_ito_ThreeJSymbols([a,b,c,d,e,f,g,h,i])
        Sum(a, b, c, d, e, f, g, h, i)*ThreeJSymbol(A, B, C, a, b, c)*ThreeJSymbol(A, D, G, a, d, g)*ThreeJSymbol(B, E, H, b, e, h)*ThreeJSymbol(C, F, I, c, f, i)*ThreeJSymbol(D, E, F, d, e, f)*ThreeJSymbol(G, H, I, g, h, i)

        >>> global_assumptions.clear()
        """
        j1, j2, J12, j3, j4, J34, J13, J24, J = self.args
        if isinstance(projection_labels, dict):
            d = projection_labels
            projection_labels = [ d[key] for key in self.args ]
        m1, m2, M12, m3, m4, M34, M13, M24, M = projection_labels

        return  (ThreeJSymbol(j1, j2, J12, m1, m2, M12)*
                ThreeJSymbol(j3, j4, J34, m3, m4, M34)*
                ThreeJSymbol(j1, j3, J13, m1, m3, M13)*
                ThreeJSymbol(j2, j4, J24, m2, m4, M24)*
                ThreeJSymbol(J12, J34, J, M12, M34, M)*
                ThreeJSymbol(J13, J24, J, M13, M24, M)*
                ASigma(*projection_labels))


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

    >>> ClebschGordanCoefficient(A, a, B, b, 0, c).get_as_ThreeJSymbol()
    (-1)**(A - a)*Dij(0, c)*Dij(A, B)*Dij(a, -b)/(1 + 2*A)**(1/2)
    >>> ClebschGordanCoefficient(A, a, 0, b, C, c).get_as_ThreeJSymbol()
    Dij(0, b)*Dij(A, C)*Dij(a, c)
    >>> ClebschGordanCoefficient(0, a, B, b, C, c).get_as_ThreeJSymbol()
    Dij(0, a)*Dij(B, C)*Dij(b, c)

    """
    nargs = 6
    is_commutative=True

    def get_as_ThreeJSymbol(self):
        """
        Rewrites to a 3j-symbol on canonical form

        Check if we have a zero components first.
        """
        j1, m1, j2, m2, J, M = self.args
        if j1 is S.Zero:
            return Dij(j2, J)*Dij(m2, M)*Dij(m1, S.Zero)
        if j2 is S.Zero:
            return Dij(j1, J)*Dij(m1, M)*Dij(m2, S.Zero)
        if J  is S.Zero:
            return ((-1)**(j1-m1)*(2*j1+1)**(S.NegativeOne/2)
                    *Dij(j1, j2)*Dij(m1, -m2)*Dij(M, S.Zero))
        return (Pow(S.NegativeOne,j1 - j2 + M)*sqrt(2*J + 1)
                *ThreeJSymbol(j1, j2, J, m1, m2, -M))

    def canonical_form(self):
        """
        Rewrites self to a canonical form

        We define the canonical form similarily to the tjs
        """
        projs = self.projections
        if projs[0] == S.Zero:
            projs = projs[1:]
        c, t = projs[0].as_coeff_terms()
        if c.is_negative:
            return self.invert_projections()
        else:
            return self

    def invert_projections(self):
        """
        Returns the C-G coefficient with all projections inverted.

        This generates a phase (-1)**(j1 + j2 - J)

        >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient,ThreeJSymbol
        >>> from sympy import symbols
        >>> a,b,c = symbols('abc')
        >>> A,B,C = symbols('ABC')
        >>> ClebschGordanCoefficient(A, a, B, b, C, c).invert_projections()
        (-1)**(A + B - C)*(A, -a, B, -b|C, -c)
        """
        args = list(self.args)
        for i in range(3):
            args[2*i + 1] = -args[2*i + 1]
        phase = Pow(-1, args[0] + args[2] - args[4])
        return phase*ClebschGordanCoefficient(*args)

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

    def get_triangular_inequalities(self):
        """
        Returns the triangular inequality implied by this ClebschGordanCoefficient

        >>> from sympy.physics.racahalgebra import ThreeJSymbol
        >>> from sympy import symbols
        >>> A,B,C,a,b,c = symbols('ABCabc')
        >>> ThreeJSymbol(A, B, C, a, b, c).get_triangular_inequalities()
        set([TriangularInequality(A, B, C)])

        """
        return set([TriangularInequality(*self.magnitudes)])

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

    def _latex(self, p, exp=None):
        symbs = [p._print(a) for a in self.args]
        symbs = [a if a[0] != '-' else '{'+a+'}' for a in symbs]
        res = "\\left(%s %s %s %s\\middle|%s %s\\right)" % tuple(symbs)
        if exp:
            res += "^{%s}" % exp

        return res

    def _sympystr(self, *args):
        """
        >>> from sympy.physics.racahalgebra import ClebschGordanCoefficient
        >>> from sympy import symbols
        >>> a,b,c,d,e = symbols('abcde')
        >>> A,B,C,D,E = symbols('ABCDE')

        >>> ClebschGordanCoefficient(A,a,B,b,C,c)
        (A, a, B, b|C, c)
        """

        return "(%s, %s, %s, %s|%s, %s)" % self.args

    def _pretty_(self, p, *args):
        return prettyForm(self._sympystr())

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


class SphericalTensor(Expr):
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
        args = map(sympify, args)

        if cls == SphericalTensor:
            return get_spherical_tensor(symbol, *args)
        else:
            return Expr.__new__(cls, symbol, *args, **kw_args)

    @property
    def rank(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    def get_rank_proj(self):
        return self.rank, self.projection

    @property
    def symbol(self):
        return self.args[0]

    def as_coeff_tensor(self):
        """Returns the coefficient and tensor that characterizes a rotation

        Derived classes may encapsulate rotational properties such that a
        phase and a different tensor is needed in coupling expressions.
        """
        return S.One, self

    def _dagger_(self):
        """
        Hermitian conjugate of a SphericalTensor.

        We follow the definition of Edmonds (1974).

        >>> from sympy.physics.braket import SphericalTensor, Dagger
        >>> from sympy import symbols
        >>> k,q,T = symbols('k q T')
        >>> SphericalTensor(T, k, q)
        T(k, q)
        >>> Dagger(SphericalTensor(T, k, q))
        (-1)**(k + q)*T(k, -q)

        """
        k = self.rank
        q = self.projection
        T = self.symbol
        cls = type(self)
        return (-1)**(k + q)*self.__new__(cls, T, k, -q)

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
        Sum(_D, _d)*(A, a, B, b|_D, _d)*T[t1(A)*t2(B)](_D, _d)

        If you know that you will not need to distinguish the summation indices
        and would prefer an expression without dummy symbols, you can supply
        the keyword argument use_dummies=False.

        >>> T.get_direct_product_ito_self(use_dummies=False)
        Sum(D, d)*(A, a, B, b|D, d)*T[t1(A)*t2(B)](D, d)

        With three coupled tensors we get:

        >>> t3 = SphericalTensor('t3',C,c)
        >>> S = SphericalTensor('S',E,e,T,t3)
        >>> S.get_direct_product_ito_self()
        Sum(_D, _E, _d, _e)*(A, a, B, b|_D, _d)*(_D, _d, C, c|_E, _e)*S[T[t1(A)*t2(B)](_D)*t3(C)](_E, _e)

        """
        coeff = self._eval_coeff_for_direct_product_ito_self(**kw_args)
        if kw_args.get('use_dummies', True):
            return convert_sumindex2dummy(coeff*self)
        else:
            return coeff*self

    def as_direct_product(self, **kw_args):
        """
        Returns this tensor in terms of a direct product of constituent tensors.

        If the keyword deep=False is supplied, only the top-level composite
        tensor (i.e. self) is uncoupled.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> t1 = SphericalTensor('t1', 'A', 'a')
        >>> t2 = SphericalTensor('t2', 'B', 'b')
        >>> T = SphericalTensor('T', 'D', 'd', t1, t2)
        >>> T.as_direct_product()
        Sum(_a, _b)*t1(A, _a)*t2(B, _b)*(A, _a, B, _b|D, d)

        If you know that you will not need to distinguish the summation indices
        and would prefer an expression without dummy symbols, you can supply
        the keyword argument use_dummies=False.

        >>> T.as_direct_product(use_dummies=False)
        Sum(a, b)*t1(A, a)*t2(B, b)*(A, a, B, b|D, d)

        With three coupled tensors we get:

        >>> t3 = SphericalTensor('t3', 'C', 'c')
        >>> S = SphericalTensor('S', 'E', 'e', T, t3)
        >>> S.as_direct_product()
        Sum(_a, _b, _c, _d)*t1(A, _a)*t2(B, _b)*t3(C, _c)*(A, _a, B, _b|D, _d)*(D, _d, C, _c|E, e)

        """
        return Mul(*self.as_coeff_direct_product(**kw_args))

    def as_coeff_direct_product(self, **kw_args):
        """
        Returns a tuple with the coupling coefficient and direct product.

        If the keyword deep=False is supplied, only the top-level composite
        tensor (i.e. self) is uncoupled.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> t1 = SphericalTensor('t1', 'A', 'a')
        >>> t2 = SphericalTensor('t2', 'B', 'b')
        >>> T = SphericalTensor('T', 'D', 'd', t1, t2)
        >>> T.as_coeff_direct_product()
        (Sum(_a, _b)*(A, _a, B, _b|D, d), t1(A, _a)*t2(B, _b))

        """
        coeff, dirprod = self._eval_as_coeff_direct_product(**kw_args)
        if kw_args.get('use_dummies', True):
            coeff = convert_sumindex2dummy(coeff)
            subsdict = extract_symbol2dummy_dict(coeff)
            dirprod = dirprod.subs(subsdict)
        return coeff, dirprod

    def _sympystr(self, *args):
        """
        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy import symbols
        >>> SphericalTensor('t1', 'A', 'a')
        t1(A, a)
        """

        return "%s(%s, %s)" % self.args

    def _str_drop_projection_(self, *args):
        rank= "%s" %(self.rank,)
        symbol = str(self.symbol)

        return symbol,rank

    def _latex(self, p):
        return "%s(%s, %s)" % tuple([p._print(a) for a in self.args])

    def _latex_drop_projection(self, p):
        return "%s(%s)" % tuple([p._print(a) for a in self.args[:2]])


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
        obj = Expr.__new__(cls,symbol,rank,projection,tensor1,tensor2)
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

    def _sympystr(self, p, *args):
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
        c1, t1 = self.tensor1.as_coeff_tensor()
        c2, t2 = self.tensor2.as_coeff_tensor()
        coeffs = (c1*c2*ClebschGordanCoefficient(
                t1.rank,t1.projection,
                t2.rank,t2.projection,
                self.rank,self.projection)
                *ASigma(t1.projection, t2.projection))

        if kw_args.get('deep', True):
            c1, t1 = t1._eval_as_coeff_direct_product(**kw_args)
            c2, t2 = t2._eval_as_coeff_direct_product(**kw_args)
            coeffs *= c1*c2

        return combine_ASigmas(coeffs),t1*t2

    def _eval_coeff_for_direct_product_ito_self(self, **kw_args):
        c1, t1 = self.tensor1.as_coeff_tensor()
        c2, t2 = self.tensor2.as_coeff_tensor()
        expr = (ClebschGordanCoefficient(
                    t1.rank,t1.projection,
                    t2.rank,t2.projection,
                    self.rank,self.projection) / c1 / c2
                * t1._eval_coeff_for_direct_product_ito_self()
                * t2._eval_coeff_for_direct_product_ito_self()
                * ASigma(self.rank, self.projection)
                )

        return combine_ASigmas(expr)



    def as_other_coupling(self, other_coupling, **kw_args):
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

        >>> S2.as_other_coupling(S1)
        Sum(_D, _a, _b, _c, _d, _e)*(A, _a, B, _b|_D, _d)*(A, _a, E, _e|G, g)*(B, _b, C, _c|E, _e)*(_D, _d, C, _c|G, g)*S1[T12[t1(A)*t2(B)](_D)*t3(C)](G, g)

        Note how S1(F, f) has been replaced with S1(G, g) because only those
        values of rank and projection can contribute when S2 is expressed
        i.t.o. S1.  Here, there are no dummy variables with conflicting names,
        so we can also ask for an expression without dummies:

        >>> S2.as_other_coupling(S1, use_dummies=False)
        Sum(D, a, b, c, d, e)*(A, a, B, b|D, d)*(A, a, E, e|G, g)*(B, b, C, c|E, e)*(D, d, C, c|G, g)*S1[T12[t1(A)*t2(B)](D)*t3(C)](G, g)

        The result has now been filtered through convert_sumindex2nondummy().
        If there is a name conflict during the substitution, an exception is
        raised.  If we ask for the other relation, S1 expressed in terms of S2,
        we see that the only S2(F, f) can contribute:

        >>> S1.as_other_coupling(S2, use_dummies=False)
        Sum(E, a, b, c, d, e)*(A, a, B, b|D, d)*(A, a, E, e|F, f)*(B, b, C, c|E, e)*(D, d, C, c|F, f)*S2[t1(A)*T23[t2(B)*t3(C)](E)](F, f)

        """
        my_tensors = self.atoms(AtomicSphericalTensor)
        assert my_tensors == other_coupling.atoms(AtomicSphericalTensor)

        # Use direct product as a link between coupling schemes
        self_as_direct_product = self.as_direct_product()
        direct_product_ito_other = other_coupling.get_direct_product_ito_self()

        # In the direct product there is a sum over other.rank and
        # other.projection, but for a transformation of coupling scheme the
        # coefficient <(..).:J'M'|.(..);J M> implies that J'==J and M'==M.
        # We can evaluate those sums immediately, i.e. only return the
        # surviving term. So we remove the summations and substitute with the
        # correct rank, projection.
        j = self.rank
        m = self.projection
        direct_product_ito_other = convert_sumindex2nondummy(
                direct_product_ito_other,
                [(other_coupling.rank, j), (other_coupling.projection, m)]
                )
        direct_product_ito_other = remove_summation_indices(
                direct_product_ito_other, (j, m)
                )

        # self_as_direct_product is a sum over dummy symbols.  To substitute
        # correctly, these dummies must be inserted in the substitution target,
        # and in direct_product_ito_other as well:
        subsdict = extract_symbol2dummy_dict(self_as_direct_product)
        direct_product = Mul(*my_tensors).subs(subsdict)
        direct_product_ito_other = direct_product_ito_other.subs(subsdict)

        expr = combine_ASigmas(
                self_as_direct_product.subs(direct_product, direct_product_ito_other)
                )
        if kw_args.get('use_dummies', True):
            return expr
        else:
            return convert_sumindex2nondummy(expr)

    def _latex(self, p):
        return "%s\left[%s \otimes %s\\right]^{%s}_{%s}" % tuple([p._print(a) for a in self.args])

    def _latex_drop_projection(self, p):
        return "%s\left[%s \otimes %s\\right]^{%s}" % tuple([p._print(a) for a in self.args[:-1]])


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
        obj = Expr.__new__(cls,symbol,rank,projection)
        return obj

    def _eval_as_coeff_direct_product(self, **kw_args):
        """
        Returns the uncoupled, direct product form of a composite tensor.
        """
        return S.One, self

    def _eval_coeff_for_direct_product_ito_self(self,**kw_args):
        """
        Returns the direct product expressed by the composite tensor.
        """
        return S.One

class ASigma(Expr):
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
        >>> ASigma(0,a)
        Sum(a)
        >>> ASigma(0,1)
        1
        """
        indices = map(sympify, indices)
        unsigned = []
        for i in indices:
            c,t = i.as_coeff_terms()
            if len(t)==1:
                unsigned.append(t[0])
            elif len(t)>1:
                raise ValueError("ASigma doesn't accept products of symbols: %s"%i)
        unsigned.sort()
        if unsigned:
            obj = Expr.__new__(cls,*unsigned)
        else:
            obj = S.One
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
        if len(newargs) < len(self.args):
            return ASigma(*newargs)
        else:
            return self

    def _sympystr(self, p, *args):
        l = [p.doprint(o) for o in self.args]
        return "Sum" + "(%s)"%", ".join(l)

    def _latex(self, p):
        labels = " ".join([ p._print(i) for i in self.args ])
        return r"\sum_{%s}" % labels

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



__all_phases_seen_in_search = set()

def refine_phases(expr, forbidden=[], mandatory=[], assumptions=True, **kw_args):
    """
    Simplifies and standardizes expressions containing 3j and 6j symbols.

    The racah algebra produces a lot of factors (-1)**(x + y + ...), a.k.a.
    phases.  This function standardizes the expression by rewriting the phase
    to an equivalent form, simplifying it if possible.

    ``forbidden`` -- iterable containing symbols that cannot be in the phase
    ``mandatory`` -- iterable containing symbols that must be in the phase

    If there are conflicting requirements, the exception
    ``UnableToComplyWithForbiddenAndMandatorySymbols`` is raised.
    If you supply the keyword, strict=True, this exception is also raised if
    the algorithm do not succed.

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
    (-1)**(-2*A - 2*B - 2*C)*ThreeJSymbol(A, B, C, a, b, c)
    >>> expr = (-1)**(a+b+c)*ThreeJSymbol(A,B,C,a,b,-c)
    >>> refine_phases(expr, [a, b, c, A, B], [C])
    (-1)**(2*C)*ThreeJSymbol(A, B, C, a, b, -c)
    >>> global_assumptions.clear()

    """
    forbidden = set(forbidden)
    mandatory = set(mandatory)
    if forbidden & mandatory: raise UnableToComplyWithForbiddenAndMandatorySymbols

    # fetch the phase
    expr = powsimp(expr)
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


    # Since the cache doesn't account for global assumptions, we use
    # a local cache that we reset before starting the recursions:
    if not kw_args.get('keep_local_cache'):
        clear_local_cache()

    # determine what should be done
    to_remove = forbidden & pow_atoms
    to_insert = mandatory - pow_atoms
    if not (to_remove or to_insert):
        if orig_phase_pow is S.One:
            return expr
        else:
            phase = _simplify_Add_modulo2(phase)
            return expr.subs(orig_phase_pow, Pow(S.NegativeOne, phase))

    # determine what can be done and setup identities as sympy expressions
    projections = set([])
    triags = set([])
    jm_pairs = set([])
    identity_sources = set(kw_args.get('identity_sources', []))
    identity_sources.update(expr.atoms(AngularMomentumSymbol))
    for njs in identity_sources:
        if isinstance(njs, (ThreeJSymbol, ClebschGordanCoefficient)):
            jm = njs.get_magnitude_projection_dict()
            jm_list = [ Add(2*j,2*m) for j,m in jm.items() ]
            jm_pairs.update(jm_list)
        if isinstance(njs, ThreeJSymbol):
            projections.add(Add(*njs.projections))
        elif isinstance(njs, ClebschGordanCoefficient):
            projections.add(Add(*(njs.projections[0:2] + (-njs.projections[2],))))
        triags.update(njs.get_triangular_inequalities())

    triags = set([2*Add(*t.args) for t in triags])

    # Now it is a good idea to remove redundant symbols from the identities
    # Even terms are identities on their own, so they will not change the phase
    all_identities = triags | jm_pairs | projections
    for identity in  triags | jm_pairs | projections:
        not_even = [ arg for arg in identity.args if not _ask_even(arg) ]
        if len(not_even) < len(identity.args):
            all_identities.remove(identity)
            if not_even:
                all_identities.add(Add(*not_even))

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
        # known_identities[symbol] = []
        for identity in all_identities:
            if symbol in identity:
                # FIXME: for the brutal approach we skip the symbol keys
                # known_identities[symbol].append(identity)
                known_identities[identity]=0

    #
    # What we really want to do, is to express the forbidden, f, in terms of the
    # mandatory, m, so that:
    #
    #  (-1)**(f + m0 + x0) ==> (-1)**(m + x1)
    #
    # where x1 contains no forbidden symbols.

    # separate f and m0 from x0
    x0 = []; fm0 = []
    for arg in phase.args:
        if arg.atoms() & (forbidden|mandatory):
            fm0.append(arg)
        else:
            x0.append(arg)

    phase = Add(*fm0)

    __all_phases_seen_in_search.clear()
    try:
        better_phase = _brutal_search_for_simple_phase(
                phase, known_identities, forbidden, mandatory)
        better_phase = _simplify_Add_modulo2(better_phase+Add(*x0), mandatory)
    except UnableToComplyWithForbiddenAndMandatorySymbols:
        if kw_args.get('strict'):
            raise
        else:
            tmpset = set()
            for p in __all_phases_seen_in_search:
                tmpset.add(_simplify_Add_modulo2(p+Add(*x0), mandatory))
            __all_phases_seen_in_search.clear()
            __all_phases_seen_in_search.update(tmpset)
            better_phase = _determine_best_phase(forbidden, mandatory)

    if orig_phase_pow is S.One:
        return expr*Pow(-1,better_phase)
    else:
        return expr.subs(orig_phase_pow, Pow(-1,better_phase))



def _determine_best_phase(forbidden, mandatory):
    best_phase = __all_phases_seen_in_search.pop()
    best_symbs = best_phase.atoms(Symbol)
    best_forbidden = best_symbs & forbidden
    best_mandatory = best_symbs & mandatory

    while __all_phases_seen_in_search:
        phase = __all_phases_seen_in_search.pop()
        symbs = phase.atoms(Symbol)
        if len(symbs & forbidden) > len(best_forbidden): continue
        if len(symbs & mandatory) < len(best_mandatory): continue
        # fewer symbols is most important
        if len(symbs) > len(best_symbs): continue
        # then fewer terms
        if len(phase.args) < len(best_phase.args):
            best_phase = phase
            best_forbidden = symbs & forbidden
            best_mandatory = symbs & mandatory
            best_symbs = symbs
    return best_phase



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
    phase = _simplify_Add_modulo2(phase, mandatory, True)
    __all_phases_seen_in_search.add(phase)

    current_symbols = set([ s for s in phase.atoms() if s.is_Symbol ])
    missing = mandatory - current_symbols
    to_remove = forbidden & current_symbols
    current_symbols = missing | to_remove


    # break recursion if we are done
    if not current_symbols:
        return _simplify_Add_modulo2(phase, mandatory, True)

    id_list = known_identities.keys()
    for i in range(start, len(known_identities)):
        identity = id_list[i]
        # skip this identity if it does not contain symbols we need
        if not current_symbols & set([s for s in identity.atoms() if s.is_Symbol]):
            continue
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

def _simplify_Add_modulo2(add_expr, leave_alone=None, standardize_coeff=True):
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
                if standardize_coeff:
                    others.append(_standardize_coeff(arg, True))
                else:
                    others.append(arg)
            elif _ask_odd(arg):
                odd += 1
            elif _ask_even(arg):
                pass
            else:
                if standardize_coeff:
                    others.append(_standardize_coeff(arg, False))
                else:
                    others.append(arg)
        if odd % 2:
            others.append(S.One)
        return Add(*others)
    elif add_expr.is_Mul:
        # trick it into an Add
        return _simplify_Add_modulo2(add_expr+2)
    else:
        return add_expr

def _standardize_coeff(expr, skip_obvious = False):
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
        if skip_obvious and not c: c=4  #if expr didn't vanish already it was not meant to happen!
        return Mul(c,t[0])
    elif _ask_integer(Mul(*t)) and (c > 2 or c <= -2):
        c = c%2
        if skip_obvious and not c: c=2  #if expr didn't vanish already it was not meant to happen!
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


def refine_tjs2njs(expr, **kw_args):
    """
    Tries to rewrite six 3j-symbols to a 9j-symbol.

    >>> from sympy.physics.racahalgebra import NineJSymbol, refine_tjs2njs
    >>> from sympy import global_assumptions, Q, assume_all
    >>> from sympy import symbols
    >>> a,b,c,d,e,f,g,h,i = symbols('abcdefghi')
    >>> A,B,C,D,E,F,G,H,I = symbols('ABCDEFGHI', nonnegative=True)
    >>> global_assumptions.add(*assume_all([a,b,d,e], 'half_integer'))
    >>> global_assumptions.add(*assume_all([A,B,D,E], 'half_integer'))
    >>> global_assumptions.add(*assume_all([c,f,g,h,i], 'integer'))
    >>> global_assumptions.add(*assume_all([C,F,G,H,I], 'integer'))

    >>> expr = NineJSymbol(A, B, C, D, E, F, G, H, I).get_ito_ThreeJSymbols((a,b,c,d,e,f,g,h,i))
    >>> refine_tjs2njs(expr)
    NineJSymbol(A, B, C, D, E, F, G, H, I)

    >>> global_assumptions.clear()
    """

    expr = convert_cgc2tjs(expr)
    expr = combine_ASigmas(expr)
    expr = canonicalize(expr)
    expr = _process_tjs_permutations(expr, 6, **kw_args)
    return refine_phases(expr, keep_local_cache=True)


def refine_tjs2sjs(expr, **kw_args):
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

    """

    expr = convert_cgc2tjs(expr)
    expr = combine_ASigmas(expr)
    expr = canonicalize(expr)
    expr = _process_tjs_permutations(expr, 4, **kw_args)
    return refine_phases(expr, keep_local_cache=True)


def _process_tjs_permutations(expr, number_to_rewrite, **kw_args):
    """Rewrites `number_to_rewrite' 3j-symbols into a 6j- or 9j- symbol.
    """
    for permut in _iter_tjs_permutations(expr, number_to_rewrite, ThreeJSymbol):
        summations, phases, factors, threejs, ignorables = permut

        # find 6j-symbol, expand as 3j-symbols and try to match with original
        try:
            if number_to_rewrite == 4:
                Njs = _identify_SixJSymbol(threejs, **kw_args)
            elif number_to_rewrite == 6:
                Njs = _identify_NineJSymbol(threejs, **kw_args)
        except ThreeJSymbolsNotCompatibleWithSixJSymbol, e:
            if kw_args.get('verbose'):
                print "Could not find 6j symbol for", threejs
                print e
            continue

        # Determine projection symbols for expansion
        M_symbols = {}
        conflicts = []
        for J in Njs.args:
            for tjs in threejs:
                M = tjs.get_projection_symbol(J)
                if M:
                    if J not in M_symbols:
                        M_symbols[J] = M
                    elif M_symbols[J] != M:
                        conflicts.append((tjs, J))

        # if expr has different projection symbols for the same J, it cannot
        # be identified as equal to the 6js.  We must rewrite it first.
        for tjs, J in conflicts:
            M = M_symbols[J]
            alt = tjs.get_projection_symbol(J)
            raise ValueError("FIXME: conflicting projections: %s, %s" %(M, alt))
            # FIXME: we could have continued with a Kronecker delta on the
            # offending projection.

        new_tjs_expr = Njs.get_ito_ThreeJSymbols(M_symbols, **kw_args)

        # There is only one permutation here but we want to split new_tjs_expr:
        for permut2 in _iter_tjs_permutations(new_tjs_expr):
            summations2, phases2, factors2, threejs2, ignorables2 = permut2


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

            # We need to process the list of alternative projection inversions
            # in order to find combinations that are consistent for all 3js
            # simultaneously
            try:
                proj_inversions = _get_projection_inversion_dict(projdict)
            except ThreeJSymbolsNotCompatibleWithSixJSymbol:
                if kw_args.get('verbose'):
                    print projdict
                raise

            # We do the inversions on the 6j symbol expression
            phases2 = phases2.subs(proj_inversions)
            factors2 = factors2.subs(proj_inversions)
            for tjs in threejs2:
                compatible_tjs = tjs.subs(proj_inversions)
                coeff, junk = compatible_tjs.as_coeff_terms(AngularMomentumSymbol)
                phases2 *= coeff



        # make sure there is summation over all projections
        M_symbols = [ M_symbols[key] for key in Njs.args ]
        for m in [ m for m in M_symbols if not (m in summations.args)]:
            # ... {}{}{}{} ... => ...( sum_abcdef {}{}{}{}/C ) ...
            # C is a factor to compensate for the summations we introduce.
            #
            # Since we are searching for a 6j symbol, the expresion with 4 3j symbols
            # cannot depend on the projections.  (If they do, it means they cannot be
            # rewritten to 6j, so we will fail at a later point.)
            #
            # This means that the factor C is just (2j + 1) for every new sum.
            j = Njs.args[ M_symbols.index(m) ]
            factors = factors/(2*j + 1)

        # remove all projection summations
        summations = summations.remove_indices(
                [m for m in M_symbols if (m in summations.args)])


        # sjs == phases2*factors2*(tjs)^4  =>  (tjs)^4 == sjs/factors2/phases2
        # expr = phases*factors*(tjs)^4 == phases/phases2 * factors/factors2 * sjs

        phases = refine(powsimp(phases/phases2))
        factors = factors/factors2

        expr = Mul(summations, phases, Njs, factors, *ignorables)

        # get rid of any projection symbols in the phase
        try:
            expr = refine_phases(expr, M_symbols, strict=True, identity_sources=threejs)
            break
        except UnableToComplyWithForbiddenAndMandatorySymbols:
            if kw_args.get('verbose'):
                print "Unable to remove all projection symbols from phase"
                print "result was ",expr
            if kw_args.get('let_pass'):
                print "WARNING: Unable to remove all projection symbols from phase"
                break
    else:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol
    return expr


def canonicalize(expr):
    """
    Driver routine to rewrite an expression to canonical form

    The canonicalization starts at the deepest level of the expression tree.
    """

    if not expr.is_Atom:
        expr = expr.func(*[canonicalize(arg) for arg in expr.args])
    try:
        return expr.canonical_form()
    except AttributeError:
        return expr

def is_equivalent(expr1, expr2, verbosity=0):
    """
    Tries hard to verify that expr1 == expr2.
    """
    expr1 = canonicalize(expr1)
    expr2 = canonicalize(expr2)

    for permut in _iter_tjs_permutations(expr1):
        summations1, phases1, factors1, njs1, ignorables1 = permut
    for permut in _iter_tjs_permutations(expr2):
        summations2, phases2, factors2, njs2, ignorables2 = permut

    fails = {}
    if njs1 != njs2:
        if verbosity:
            fails['AngularMomentumSymbols'] = (njs1,njs2)
        else:
            return False
    if summations1 != summations2:
        if verbosity:
            fails['summations'] = (summations1, summations2)
        else:
            return False
    if phases1 != phases2:
        ratio = refine(powsimp(phases1/phases2))
        if ratio is S.One:
            pass
        elif ratio is S.NegativeOne:
            if verbosity:
                fails['phaseratio'] = ratio
            else:
                return False
        else:
            ratio = refine_phases(ratio, ratio.exp.atoms(Symbol), [],
                    identity_sources=njs1 + njs2)
            if not ratio is S.One:
                if verbosity:
                    fails['phaseratio'] = ratio
                else:
                    return False
    if factors1 != factors2:
        if not ratsimp(factors1/factors2) is S.One:
            if verbosity:
                fails['factors'] = (factors1,factors2)
            else:
                return False
    if ignorables1 != ignorables2:
        if not simplify(Add(*ignorables2) - Add(*ignorables1)) is S.Zero:
            if verbosity:
                fails['other'] = (ignorables1, ignorables2)
            else:
                return False

    if fails:
        print "failing matches are:",fails
        return False
    else:
        return True


def _get_projection_inversion_dict(projection_dict):
    """
    Checks combinations of projection inversion and returns a subsdict.

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
            raise ThreeJSymbolsNotCompatibleWithSixJSymbol(
                    "Problem with projections: %s" %(will_invert & must_keep))

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

    # determine best inversion as the quickest (fewest substitutions)
    best = inversions[0][0]
    for inv in inversions[1:]:
        if len(inv[0]) < len(best): best = inv[0]

    subsdict = {}
    for M in best:
        subsdict[M] = -M

    return subsdict


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


def _iter_tjs_permutations(expr, how_many=None, which_objects=AngularMomentumSymbol):
    """
    Iterates over possible candidates for  4*tjs -> sjs in a Mul

    ``how_many`` -- number of objects in each combination
    ``which_objects`` -- class of objects to collect

    If how_many is left out, we return only the combination containing all objects.
    If ``which_objects`` is left out we return all onstances of AngularMomentumSymbol.

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
            if isinstance(arg, which_objects):
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

    if how_many and len(threejs)<how_many:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol("Too few 3js!")

    summations = combine_ASigmas(Mul(*summations))
    phases = powsimp(Mul(*phases))
    factors = Mul(*factors)

    if how_many:
        for tjs in combinations(threejs, how_many):
            yield tuple([summations, phases, factors, tjs, ignorables])
    else:
        yield tuple([summations, phases, factors, threejs, ignorables])


def _identify_SixJSymbol(threejs, **kw_args):
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
    if len(keys_J) != 6:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol("Number of J is %s"%len(keys_J))

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
            raise ThreeJSymbolsNotCompatibleWithSixJSymbol(
                    "connections not satisfied for s%"%maxJ)

    # Two elements in a column of the sjs never appear in the same 3j-symbol
    j2 = keys_J - connections[totJ]  # j2 and totJ are in same column
    j3 = keys_J - connections[  j1]  # j3 and j1   are in same column
    if not (len(j3) == len(j2) == 1):
        if kw_args.get('verbose'):
            print "possible j2:", j2
            print "possible j3:", j3
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol(
                "Found no unique j2 and j3 for 6j symbol.")
    j2 = j2.pop()
    j3 = j3.pop()

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

    if J12 is None or J23 is None:
        if kw_args.get('verbose'):
            print "failed to identify 6j symbol"
            print "Closest match was: SixJSymbol(", j1,j2,J12, j3,totJ, J23,')'
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol


    return  SixJSymbol(j1,j2,J12,j3,totJ,J23)

def _identify_NineJSymbol(threejs, **kw_args):
    """
    Identify and construct the 9j-symbol available for the 3j-symbols.

    ``threejs`` is an iterable containing 6 objects of type ThreeJSymbol.
    """

    def _find_connections(J, threejs):
        conn = set([])
        for tjs in threejs:
            if J in tjs.magnitudes:
                conn.update(tjs.magnitudes)
        conn.remove(J)
        return conn

    keys_J = set([])
    connections = dict([])
    for tjs in threejs:
        keys_J.update(tjs.magnitudes)
    if len(keys_J) != 9:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol("Number of J is %s"%len(keys_J))

    for j in keys_J:
        connections[j] = _find_connections(j, threejs)

    # j1 and j9 are given by canonical form of 9j-symbol
    j1 = sorted(keys_J)[0]
    j9 = sorted(keys_J - connections[j1]).pop()

    # j3 and j7 are connected with both j1 and j9
    common19 = connections[j1] & connections[j9]
    if len(common19) != 2:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol(
                "Unable to determine j3 and j6 from %s" %common19)
    # canonical form requires that j3 < j7
    j3, j7 = sorted(common19)

    # the element in the middle, j5, is disconnected from all corners
    j5 = (keys_J - connections[j1] - connections[j3]
            - connections[j7] - connections[j9])
    if len(j5) != 1:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol(
                "Unable to determine j5, got %s" %j5)
    j5 = j5.pop()

    try:
        j2 = (connections[j1] & connections[j3] & connections[j5]).pop()
        j4 = (connections[j1] & connections[j7] & connections[j5]).pop()
        j6 = (connections[j3] & connections[j9] & connections[j5]).pop()
        j8 = (connections[j7] & connections[j9] & connections[j5]).pop()
    except KeyError, e:
        raise ThreeJSymbolsNotCompatibleWithSixJSymbol(
                "Unable to determine j2, j4, j6 or j8")

    return  NineJSymbol(j1, j2, j3, j4, j5, j6, j7, j8, j9)

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

def remove_summation_indices(expr, indices):
    """
    Locates all ASigma in ``expr`` and removes requested indices.

    If you try to remove an index which is not there, you get a ValueError

    >>> from sympy.physics.racahalgebra import ASigma, remove_summation_indices
    >>> from sympy import symbols, Function
    >>> a,b,c = symbols('abc')
    >>> f = Function('f')
    >>> expr = ASigma(a,b)*f(a,b) + ASigma(a)*f(a,c)
    >>> remove_summation_indices(expr, [a])
    Sum(b)*f(a, b) + f(a, c)

    """
    if expr.is_Add:
        return Add(*[ remove_summation_indices(arg, indices) for arg in expr.args ])
    expr = combine_ASigmas(expr)
    sigma = expr.atoms(ASigma)
    if sigma:
        sigma = sigma.pop()
    else:
        return expr
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

def extract_symbol2dummy_dict(expr):
    """
    Returns a dict mapping nondummies to dummies in expr.

    >>> from sympy.physics.racahalgebra import extract_symbol2dummy_dict
    >>> from sympy import symbols
    >>> a,b = symbols('a b')
    >>> expr = a.as_dummy() + b.as_dummy()
    >>> extract_symbol2dummy_dict(expr)
    {a: _a, b: _b}

    """
    Dummy = type(Symbol('x', dummy=True))
    dummies = list(expr.atoms(Dummy))
    nondummies ={}
    for k,v in [ (d.as_nondummy(), d) for d in dummies ]:
        if k in nondummies:
            raise ValueError("No unique mapping for symbols to dummies")
        else:
            nondummies[k] = v
    return nondummies

def extract_dummy2symbol_dict(expr):
    """
    Returns a dict mapping all dummies in expr to nondummies.

    In case the nondummy is present in expr, we map to a primed nondummy in
    order to differentiate the symbols.

    >>> from sympy.physics.racahalgebra import extract_dummy2symbol_dict
    >>> from sympy import symbols
    >>> a,b = symbols('a b')
    >>> expr = a.as_dummy() + b.as_dummy() + b.as_dummy() + b
    >>> extract_dummy2symbol_dict(expr)                  # doctest: +SKIP
    {_a: a, _b: b__', _b: b__''}

    """
    Dummy = type(Symbol('x', dummy=True))
    dummies = expr.atoms(Dummy)
    nondummies = expr.atoms(Symbol) - dummies
    subsdict = {}
    for nond, dum in [ (d.as_nondummy(), d) for d in dummies ]:
        n = 0
        name = nond.name
        while nond in nondummies:
            n = n+1
            if n <= 3: nond = Symbol(name + "__" + "'"*n)
            else: nond = Symbol(name+"__(%s)"%n)
        subsdict[dum] = nond
        nondummies.add(nond)
    return subsdict

def convert_sumindex2dummy(expr):
    """
    Replaces all summation Symbols with dummies.
    """
    expr = combine_ASigmas(expr)
    try:
        sums = expr.atoms(ASigma).pop()
    except KeyError:
        return expr

    subslist = []
    Dummy = type(Symbol('x', dummy=True))
    for x in sums.args:
        if not isinstance(x, Dummy):
            subslist.append((x, x.as_dummy()))
    return (expr).subs(subslist)

def convert_sumindex2nondummy(expr, subslist=[]):
    """
    Replaces summation dummies according to recipe in subslist.

     - if ``subslist'' is not present: every dummy is replaced

     - if ``subslist'' is present it should contain tuples of symbols (old, new)

    If old is of type Dummy, the substitution is performed directly from the
    supplied subslist.

    If old is of type Symbol, and there is a unique summation dummy symbol that
    corresponds to ``old'', it will be substituted.  If there is more than one
    summation dummy that corresponds to ``old'' an exception is raised.

    """
    Dummy = type(Symbol('x', dummy=True))
    expr = combine_ASigmas(expr)
    try:
        sums = expr.atoms(ASigma).pop()
    except KeyError:
        return expr

    dummies = [ d for d in sums.args if isinstance(d, Dummy)]
    nondummies = extract_symbol2dummy_dict(sums)

    new_subslist = []
    if not subslist:
        for x in dummies:
            new_subslist.append((x, x.as_nondummy()))
    for old, new in subslist:
        if old in dummies:
            new_subslist.append(old, new)
        elif old in nondummies:
                new_subslist.append((nondummies[old], new))

    return (expr).subs(new_subslist)

def apply_deltas(expr, only_deltas=[], **kw_args):
    """
    Applies the KroneckerDeltas without removing them (by default)

    Any summation indices will not be substituted.

    If you supply an iterable ``only_deltas'' every other delta wil be ignored.

    If you want to get rid of particular symbols, you can supply a list of them
    with the keyword remove_all=[].  If the symbol is present in a delta function,
    it is substituted throughout the expression, removing the delta
    function as well.

    >>> from sympy.physics.racahalgebra import apply_deltas
    >>> from sympy import symbols, Function, Dij

    >>> a,b,c,d = symbols('a b c d')
    >>> f = Function('f')
    >>> apply_deltas(f(c)*f(a)*Dij(a,c))
    f(a)**2*Dij(a, c)
    >>> apply_deltas(f(c)*f(a)*Dij(a,-c), remove_all=[a])
    f(c)*f(-c)
    """
    expr, summations = expr.as_coeff_terms(ASigma)
    only_deltas = set(only_deltas)
    deltas = expr.atoms(Dij)
    if only_deltas:
        deltas = deltas & only_deltas
    for s in kw_args.get('remove_all', []):
        for d in deltas:
            if d.has(s):
                i, j = d.args
                ci, i = i.as_coeff_terms(); i = Mul(*i)
                cj, j = j.as_coeff_terms(); j = Mul(*j)
                if s == i:
                    expr = expr.subs(i, cj*j/ci)
                elif s == j:
                    expr = expr.subs(j, ci*i/cj)

    deltas = expr.atoms(Dij)
    if only_deltas:
        deltas = deltas & only_deltas

    d2dum = {}
    dum2d = {}
    for d in deltas:
        dum = Symbol('x', dummy=True)
        d2dum[d] = dum
        dum2d[dum] = d
    expr = expr.subs(d2dum)
    for d in deltas:
        i, j = d.args
        ci, i = i.as_coeff_terms(); i = Mul(*i)
        cj, j = j.as_coeff_terms(); j = Mul(*j)
        if sortkey(i) > sortkey(j):
            expr = expr.subs(i, cj*j/ci)
        else:
            expr = expr.subs(j, ci*i/cj)
    expr = expr.subs(dum2d)*Mul(*summations)
    return expr

def evaluate_sums(expr, **kw_args):
    """
    Tries to evaluate the sums using any KroneckerDeltas.

    >>> from sympy.physics.racahalgebra import evaluate_sums, ASigma, Dij
    >>> from sympy import symbols, Function

    >>> a,b,c,d = symbols('a b c d')
    >>> f = Function('f')
    >>> evaluate_sums(ASigma(a, b)*f(a)*Dij(a,c))
    Sum(b)*f(c)
    >>> evaluate_sums(ASigma(a, b)*f(c)*Dij(a,c))
    Sum(b)*f(c)
    >>> evaluate_sums(ASigma(a, b)*f(a)*Dij(a,b))
    Sum(a)*f(a)
    >>> evaluate_sums(ASigma(a, b)*f(a)*Dij(b,c))
    Sum(a)*f(a)

    If you know that the total expression is independent of a summation
    variable, and know the summation limits, the summation can be replaced with
    a factor.  E.g. a summation over m=-j, j, can be replaced with (2j+1).  You
    can supply pairs of (summation_index, replacement_factor) as key,value
    pairs in a dictionary with the keyword independent_of={}

    >>> evaluate_sums(ASigma(a, b)*f(a, b, c), independent_of={a:3})
    3*Sum(b)*f(a, b, c)


    """

    def get_coeff_index(index):
        c,t = index.as_coeff_terms()
        if t:
            i = t[0]
        else:
            i = S.One
        return c, i


    expr = combine_ASigmas(expr)
    expr, summations = expr.as_coeff_terms(ASigma)
    if not summations:
        return expr
    assert len(summations) == 1
    summations = summations[0]

    replace_dict = kw_args.get('independent_of', {})
    for i in replace_dict:
        try:
            summations = summations.remove_indices([i])
            expr = expr*replace_dict[i]
            expr = apply_deltas(expr, remove_all=[i])
        except ValueError:
            pass

    expr, deltas = expr.as_coeff_terms(Dij)

    # expr is now stripped of both summations and deltas

    deltas = sorted(deltas)
    remaining_deltas = []
    while deltas:
        d = deltas.pop()
        if isinstance(d, Pow):
            d = d.base
        assert isinstance(d, Dij), "Expected Dij, got %s"%type(d)
        i,j = d.args
        c1, i = get_coeff_index(i)
        c2, j = get_coeff_index(j)
        if j in summations.args:
            subsexpr = (j, i*c1/c2)
            summations = summations.remove_indices([j])
        elif i in summations.args:
            summations = summations.remove_indices([i])
            subsexpr = (i, j*c2/c1)
        elif kw_args.get('all_deltas'):
            # the delta has been stripped and must be inserted again
            expr = apply_deltas(expr*d, [d])
            subsexpr = None
        else:
            subsexpr = None

        if subsexpr:
            expr = expr.subs(*subsexpr)
            deltas = [ delta.subs(*subsexpr) for delta in deltas ]
            # drop any deltas that are now evaluated
            deltas = filter(lambda x: x.has(Dij), deltas)
        else:
            remaining_deltas.append(d)

    return summations*expr*Mul(*remaining_deltas)


def apply_orthogonality(expr, summations, **kw_args):
    """
    Tries to simplify by applying orthogonality relations of angular momentum symbols.

    ``summations'' -- list of summation symbols to evaluate.

    >>> from sympy.physics.racahalgebra import (ThreeJSymbol,
    ...         ClebschGordanCoefficient, apply_orthogonality, ASigma)
    >>> from sympy import symbols, global_assumptions, Assume, Q
    >>> A,B,C,D,E,F,a,b,c,d,e,f = symbols('A B C D E F a b c d e f')
    >>> global_assumptions.add( Assume(a, 'half_integer') )
    >>> global_assumptions.add( Assume(b, 'half_integer') )
    >>> global_assumptions.add( Assume(c, 'half_integer') )
    >>> global_assumptions.add( Assume(d, 'half_integer') )
    >>> global_assumptions.add( Assume(A, 'half_integer') )
    >>> global_assumptions.add( Assume(B, 'half_integer') )
    >>> global_assumptions.add( Assume(C, 'half_integer') )
    >>> global_assumptions.add( Assume(D, 'half_integer') )
    >>> global_assumptions.add( Assume(e, Q.integer) )
    >>> global_assumptions.add( Assume(f, Q.integer) )
    >>> global_assumptions.add( Assume(E, Q.integer) )
    >>> global_assumptions.add( Assume(F, Q.integer) )

    >>> expr = ClebschGordanCoefficient(A, a, B, b, E, e)*ClebschGordanCoefficient(A, a, B, b, F, f)
    >>> apply_orthogonality(expr*ASigma(a, b), [a, b])
    Dij(E, F)*Dij(e, f)
    >>> expr = ClebschGordanCoefficient(A, a, B, b, E, e)*ClebschGordanCoefficient(A, c, B, d, E, e)
    >>> apply_orthogonality(expr*ASigma(E, e), [E, e])
    Dij(a, c)*Dij(b, d)

    We canonize the symbols, so that we can recognize

    >>> expr = ClebschGordanCoefficient(A, -a, B, -b, E, e)*ClebschGordanCoefficient(A, a, B, b, F, f)
    >>> apply_orthogonality(expr*ASigma(a, b), [a, b])
    (-1)**(E - A - B)*Dij(E, F)*Dij(e, -f)

    We can also treat orthogonality of the ThreeJSymbol:

    >>> expr = (2*C+1)*ThreeJSymbol(B, A, C, -b, -a, c)*ThreeJSymbol(A, B, D, a, b, d)
    >>> apply_orthogonality(expr*ASigma(a, b), [a, b])
    Dij(C, D)*Dij(d, -c)

    Trivial cases are also handled correctly:

    >>> expr = ThreeJSymbol(A, B, E, a, b, e)**2
    >>> apply_orthogonality(expr*ASigma(a, b), [a, b])
    1/(1 + 2*E)
    >>> expr = ClebschGordanCoefficient(A, a, B, b, E, e)**2
    >>> apply_orthogonality(expr*ASigma(a, b), [a, b])
    1

    If the list of summation symbols is empty, all summation variables
    in the expression is considered.

    """
    summations = map(sympify, summations)
    sumlabels = []
    sigmas = expr.atoms(ASigma)
    for s in sigmas:
        sumlabels.extend(s.args)

    summations = summations or sumlabels

    valids = [ s for s in summations if s in sumlabels ]
    if not valids:
        return expr


    angmoms = expr.atoms(AngularMomentumSymbol)

    # Which AngularMomentumSymbol are relevant for the summations?
    indices = {}
    for njs in angmoms:
        hits = [ s for s in valids if njs.has(s) ]
        if len(hits) >= 2:
            for k in hits:
                if k in indices:
                    indices[k].add(njs)
                else:
                    indices[k] = set([njs])

    # Relations must involve exactly two angular momentum symbols
    items = list(indices.iteritems())
    for k,v in items:
        if len(v) == 1:
            ams = v.pop()
            if expr.has(ams**2):
                indices[k] = (ams, ams)
            else:
                del indices[k]
        elif len(v) != 2:
            del indices[k]

    # indices maps indices -> njs
    # we need to find two indices that map to the same two nj-symbols
    candidates = []
    for label1, njs1 in indices.iteritems():
        for label2, njs2 in indices.iteritems():
            if label1 != label2 and njs1 == njs2:
                candidates.append((label1, label2, njs1))


    sum_removals = []
    subsdict = {}
    for s1, s2, njs in candidates:
        key1, key2 = njs1, njs2 = tuple(njs)
        # substitution of squared symbol must replace also the exponent
        if key1 == key2:
            key1 = njs1**2
            key2 = key1
        if key1 in subsdict: continue
        if key2 in subsdict: continue
        coeff = S.One
        if isinstance(njs1, ClebschGordanCoefficient):
            c1, njs1 = njs1.get_as_ThreeJSymbol().as_coeff_terms(ThreeJSymbol)
            njs1 = njs1[0]
            coeff *= c1
        if isinstance(njs2, ClebschGordanCoefficient):
            c2, njs2 = njs2.get_as_ThreeJSymbol().as_coeff_terms(ThreeJSymbol)
            njs2 = njs2[0]
            coeff *= c2

        if isinstance(njs1, ThreeJSymbol) and isinstance(njs2, ThreeJSymbol):

            ranks1 = list(njs1.magnitudes)
            ranks2 = list(njs2.magnitudes)
            matching_ranks = set(ranks1) & set(ranks2)

            if len(matching_ranks) == 3 and not kw_args.get('mode')=="projections":

                # (jmjnJM)(jkjlJM) -> d_mk d_nl

                nosum_ranks = []
                for r in matching_ranks:
                    if r == s1:
                        j = s1
                        m = s2
                    elif r == s2:
                        j = s2
                        m = s1
                    else:
                        nosum_ranks.append(r)

                if len(nosum_ranks) == 2 and njs1.get_projection_symbol(j) == m:
                    ratio = njs1.get_projection(j)/njs2.get_projection(j)
                    j1, j2 = nosum_ranks
                    m11, m21 = njs1.get_projection(j1), njs1.get_projection(j2)
                    m12, m22 = njs2.get_projection(j1), njs2.get_projection(j2)
                    if ratio is S.One:
                        sum_removals.append(s1)
                        sum_removals.append(s2)
                        subsdict[key1] = S.One
                        subsdict[key2] = Dij(m11, m12)*Dij(m21, m22)/(2*j + 1)*coeff
                        continue
                    elif ratio is S.NegativeOne:
                        sum_removals.append(s1)
                        sum_removals.append(s2)
                        subsdict[key1] = S.One
                        subsdict[key2] = Dij(m11, -m12)*(-1)**(-Add(*ranks1))*Dij(m21, -m22)/(2*j + 1)*coeff
                        continue

            if len(matching_ranks) >= 2:

                # (jmjmIM)(jmjmJN) -> d_IJ d_MN

                hit_ranks = []
                for r in matching_ranks:
                    m1 = njs1.get_projection_symbol(r)
                    m2 = njs2.get_projection_symbol(r)

                    if m1 == m2 == s1:
                        hit_ranks.append(r)
                    elif m1 == m2 == s2:
                        hit_ranks.append(r)

                if len(hit_ranks) == 2:

                    # get rank,proj for delta functions
                    J1 = [ r for r in ranks1 if r not in hit_ranks][0]
                    M1 = njs1.get_projection(J1)
                    J2 = [ r for r in ranks2 if r not in hit_ranks][0]
                    M2 = njs2.get_projection(J2)

                    # 3j-symbols must have contractable ranks in same columns.
                    # Canonical ordering ensures that the contractable ranks
                    # appear in the same order in both, while the nonmatching
                    # ranks can be in any column.
                    i1 = ranks1.index(J1)
                    i2 = ranks2.index(J2)
                    if i1 == i2 or (i1, i2) == (0,2) or (i1, i2) == (2,0):
                        c_permut = S.One
                    else:
                        c_permut = (-1)**(Add(*ranks1))



                    ratio1 = njs1.get_projection(hit_ranks[0])/njs2.get_projection(hit_ranks[0])
                    ratio2 = njs1.get_projection(hit_ranks[1])/njs2.get_projection(hit_ranks[1])
                    if ratio1 == ratio2 == S.One:
                        sum_removals.append(s1)
                        sum_removals.append(s2)
                        subsdict[key1] = S.One
                        subsdict[key2] = Dij(J1, J2)*Dij(M1, M2)/(2*J1 + 1)*coeff*c_permut
                        continue
                    if ratio1 == ratio2 == S.NegativeOne:
                        sum_removals.append(s1)
                        sum_removals.append(s2)
                        subsdict[key1] = S.One
                        subsdict[key2] = Dij(J1, J2)*Dij(M1, -M2)*(-1)**(-Add(*ranks1))/(2*J1 + 1)*coeff*c_permut
                        continue

        else:
            raise NotImplementedError

    if subsdict:
        expr = remove_summation_indices(expr, sum_removals)
        # We must be able to clean out all evaluated summation indices from phases
        # else, the orthogonality cannot be applied
        expr = refine_phases(expr, sum_removals, identity_sources=angmoms,
                strict=True)
        expr = expr.subs(subsdict)
        expr = apply_deltas(expr)
        expr = refine_phases(expr, identity_sources=angmoms)

    return expr


def apply_identity_tjs(expr):
    """Applies known 3j-symbol identities to the expression

    We apply

    >>> from sympy.physics.racahalgebra import ThreeJSymbol, ASigma
    >>> from sympy.physics.racahalgebra import apply_identity_tjs
    >>> from sympy import symbols
    >>> a,b,c = symbols('abc')
    >>> A,B,C = symbols('ABC')
    >>> expr = ASigma(a)*(-1)**(A+a)*ThreeJSymbol(A, A, C, a, -a, c)
    >>> apply_identity_tjs(expr)
    (1 + 2*A)**(1/2)*Dij(0, C)/(1 + 2*C)**(1/2)
    """

    threejs = expr.atoms(ThreeJSymbol)
    subslist = []
    for tjs in threejs:
        Js = tjs.magnitudes
        Ms = tjs.projections
        for i,j in [(0,1), (0,2), (1,2)]:
            if Js[i] == Js[j] and Ms[i] == -Ms[j]:
                zero = (set([0,1,2]) - set([i,j])).pop()
                J0 = Js[zero]
                new_tjs = sqrt(2*Js[i]+1)/sqrt(2*J0+1)*Dij(J0, 0)

                junk, m = Ms[i].as_coeff_terms()

                try:
                   new_expr = remove_summation_indices(expr, m)
                except ValueError:
                    continue

                new_expr = new_expr/(-1)**(Js[i] + m[0])
                new_expr = refine_phases(new_expr, forbidden=m, strict=0)
                new_expr = new_expr.subs(tjs, new_tjs)

                if (i,j) == (0,2):
                    # we need uneven permutation
                    new_expr *= (-1)**(Add(*Js))

                # it is now safe to modify original expr
                expr = new_expr
                break

    return expr
