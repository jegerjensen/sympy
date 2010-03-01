from sympy import (
        S, Basic, Mul, global_assumptions, sympify, Symbol, Assume, ask, Q
        )
from sympy.physics.racahalgebra import (
        ThreeJSymbol, ClebschGordanCoefficient, SixJSymbol, SphericalTensor,
        refine_tjs2sjs, refine_phases, convert_cgc2tjs, convert_tjs2cgc,
        combine_ASigmas, remove_summation_indices, ASigma,
        )
from sympy.physics.secondquant import (
        Dagger, AntiSymmetricTensor
        )

class MatrixElement(Basic):
    """
    Base class for all matrix elements

    Is responsible for spawning the correct subclass upon construction.

    >>> from sympy.physics.racahalgebra import SphericalTensor
    >>> from sympy.physics.braket import MatrixElement
    >>> from sympy import symbols
    >>> a,b,c,d,e,f,g,h = symbols('abcdefgh')
    >>> A,B,C,D,E,F,G,H = symbols('ABCDEFGH')

    >>> t1 = SphericalTensor('t1',A,a)
    >>> t2 = SphericalTensor('t2',B,b)
    >>> t3 = SphericalTensor('t3',C,c)
    >>> Op = SphericalTensor('Op',D,d)
    >>> MatrixElement(t1,Op,t3)
    <t1(A, a)| Op(D, d) |t3(C, c)>
    >>> MatrixElement(t1,Op,t3).__class__
    ThreeTensorMatrixElement
    >>> ReducedMatrixElement(t1,Op,t3).get_direct_product_ito_self()
    (C, c, D, d|A, a)*<t1(A)|| Op(D) ||t3(C)>
    >>> MatrixElement(t1,Op,t3, reduced=True)
    <t1(A)|| Op(D) ||t3(C)>

    >>> T = SphericalTensor('T',E,e,t1,t2)
    >>> ReducedMatrixElement(T,Op,t3).get_direct_product_ito_self()
    Sum(E, e)*(A, a, B, b|E, e)*(C, c, D, d|E, e)*<T[t1(A)*t2(B)](E)|| Op(D) ||t3(C)>
    """
    def _sympystr_(self,*args):
        return "<%s| %s |%s>" %self.args

    def __new__(cls,left, operator, right, **kw_args):
        f1,t1 = cls._is_tensor(left)
        f2,t2 = cls._is_tensor(operator)
        f3,t3 = cls._is_tensor(right)
        if f1 and f2 and f3:
            if kw_args.get('reduced') is False:
                return f1*f2*f3*ThreeTensorMatrixElement(t1, t2, t3)
            else:
                return f1*f2*f3*ReducedTensorMatrixElement(t1, t2, t3)
        else:
            if kw_args.get('reduced'):
                raise ValueError("Reduced matrix element needs three tensors")

            if not (isinstance(left, tuple) and isinstance(right, tuple)):
                c,t = left.as_coeff_terms(SphericalTensor)
                if len(t) > 1: raise ValueError("")


                raise ValueError("DirectMatrixElement takes two tuples")
            return DirectMatrixElement(left, operator, right)

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[2]

    @property
    def operator(self):
        return self.args[1]

    @classmethod
    def _is_tensor(cls, tensor):
        if isinstance(tensor, SphericalTensor):
            return S.One, tensor
        if isinstance(tensor, Mul):
            try:
                return _as_coeff_tensor(tensor)
            except ValueError:
                pass
        return False, False


class ReducedMatrixElement(MatrixElement):
    nargs = 3

    def _sympystr_(self,p,*args):
        tup = tuple(["%s(%s)"%e._str_drop_projection_(p,*args) for e in self.args])
        return "<%s|| %s ||%s>" %tup

    def __new__(cls,left, op, right):
        obj = Basic.__new__(cls, left,op,right)
        return obj

    def get_direct_product_ito_self(self):
        """
        Returns the direct product of all involved spherical tensors i.t.o
        the reduced matrix element.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy.physics.braket import ReducedMatrixElement
        >>> from sympy import symbols
        >>> a,b,c,d,e,f,g,h = symbols('abcdefgh')
        >>> A,B,C,D,E,F,G,H = symbols('ABCDEFGH')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> t3 = SphericalTensor('t3',C,c)
        >>> Op = SphericalTensor('Op',D,d)
        >>> ReducedMatrixElement(t1,Op,t3)
        <t1(A)|| Op(D) ||t3(C)>
        >>> ReducedMatrixElement(t1,Op,t3).get_direct_product_ito_self()
        (C, c, D, d|A, a)*<t1(A)|| Op(D) ||t3(C)>

        >>> T = SphericalTensor('T',E,e,t1,t2)
        >>> ReducedMatrixElement(T,Op,t3).get_direct_product_ito_self()
        Sum(E, e)*(A, a, B, b|E, e)*(C, c, D, d|E, e)*<T[t1(A)*t2(B)](E)|| Op(D) ||t3(C)>
        """
        left = self.left.get_direct_product_ito_self()
        c_left, t_left = left.as_coeff_terms(SphericalTensor)
        operator = self.operator.get_direct_product_ito_self()
        c_operator, t_operator = operator.as_coeff_terms(SphericalTensor)
        right = self.right.get_direct_product_ito_self()
        c_right, t_right = right.as_coeff_terms(SphericalTensor)

        return (
                c_left*c_operator*c_right*self
                *ClebschGordanCoefficient(
                    self.right.rank, self.right.projection,
                    self.operator.rank, self.operator.projection,
                    self.left.rank, self.left.projection
                    )
                )


class DirectMatrixElement(MatrixElement, AntiSymmetricTensor):
    """
    Holds matrix elements corresponding to the direct product of any number of tensors.
    """
    nargs=3
    def __new__(cls,left, op, right):
        obj = AntiSymmetricTensor.__new__(op, left,right)
        return obj

    def use_wigner_eckardt(self, **kw_args):
        raise WignerEckardDoesNotApply

class ThreeTensorMatrixElement(MatrixElement):
    """
    Holds reducable matrix element consisting of 3 spherical tensors. (direct product)
    """
    nargs=3

    def __new__(cls,left, op, right):
        obj = Basic.__new__(cls,left,op,right)
        return obj


    def use_wigner_eckardt(self,  **kw_args):
        """
        Applies the Wigner-Eckard theorem to write the supplied direct matrix element
        on the form

                    k                            k
            < J M| T  |J'M'> = (J'M'kq|JM) <J|| T ||J'>
                    q
        where the reduced matrix element <.||.||.> is independent of the projections.

        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy.physics.braket import MatrixElement
        >>> from sympy import symbols
        >>> a,b,c,d,e,f,g,h = symbols('abcdefgh')
        >>> A,B,C,D,E,F,G,H = symbols('ABCDEFGH')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> t3 = SphericalTensor('t3',C,c)
        >>> Op = SphericalTensor('Op',D,d)
        >>> T = SphericalTensor('T',E,e,t1,t2)
        >>> MatrixElement(T, Op, t3).use_wigner_eckardt()
        (C, c, D, d|E, e)*<T[t1(A)*t2(B)](E)|| Op(D) ||t3(C)>

        """
        redmat = ReducedMatrixElement(self.left, self.operator, self.right)

        return (
                ClebschGordanCoefficient(
                    self.right.rank, self.right.projection,
                    self.operator.rank, self.operator.projection,
                    self.left.rank, self.left.projection
                    )
                * redmat
                )

    def get_direct_product_ito_self(self, **kw_args):
        """
        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy.physics.braket import MatrixElement
        >>> from sympy import symbols
        >>> a,b,c,d,e,f,g,h = symbols('abcdefgh')
        >>> A,B,C,D,E,F,G,H = symbols('ABCDEFGH')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> t3 = SphericalTensor('t3',C,c)
        >>> Op = SphericalTensor('Op',D,d)
        >>> T = SphericalTensor('T',E,e,t1,t2)
        >>> MatrixElement(T, Op, t3).get_direct_product_ito_self()
        Sum(E, e)*(A, a, B, b|E, e)*<T[t1(A)*t2(B)](E, e)| Op(D, d) |t3(C, c)>
        >>> MatrixElement(T, Op, t3).get_direct_product_ito_self(wigner_eckardt=True)
        Sum(E, e)*(A, a, B, b|E, e)*(C, c, D, d|E, e)*<T[t1(A)*t2(B)](E)|| Op(D) ||t3(C)>

        """
        if kw_args.get('wigner_eckardt',False):
            matrix = self.use_wigner_eckardt()
        else:
            matrix = self

        left = self.left.get_direct_product_ito_self(dual=True)
        c_left, t_left = left.as_coeff_terms(SphericalTensor)
        operator = self.operator.get_direct_product_ito_self()
        c_operator, t_operator = operator.as_coeff_terms(SphericalTensor)
        right = self.right.get_direct_product_ito_self()
        c_right, t_right = right.as_coeff_terms(SphericalTensor)

        return combine_ASigmas(c_left*c_operator*c_right)*matrix

    def get_self_ito_direct_product(self, **kw_args):
        """
        >>> from sympy.physics.racahalgebra import SphericalTensor
        >>> from sympy.physics.braket import MatrixElement
        >>> from sympy import symbols
        >>> a,b,c,d,e,f,g,h = symbols('abcdefgh')
        >>> A,B,C,D,E,F,G,H = symbols('ABCDEFGH')

        >>> t1 = SphericalTensor('t1',A,a)
        >>> t2 = SphericalTensor('t2',B,b)
        >>> t3 = SphericalTensor('t3',C,c)
        >>> Op = SphericalTensor('Op',D,d)
        >>> T = SphericalTensor('T',E,e,t1,t2)
        >>> MatrixElement(T, Op, t3).get_self_ito_direct_product()
        Sum(a, b)*(A, a, B, b|E, e)*<t1(A, a)*t2(B, b)| Op(D, d) |t3(C, c)>

        To get the reduced matrix element in terms of the direct product:
        >>> MatrixElement(T, Op, t3).get_self_ito_direct_product(wigner_eckardt=True)
        Sum(a, b, c, d)*(A, a, B, b|E, e)*(C, c, D, d|E, e)*<t1(A, a)*t2(B, b)| Op(D, d) |t3(C, c)>


        """
        if kw_args.get('wigner_eckardt',False):
            redmat = self.use_wigner_eckardt()
            cgc = redmat.atoms(ClebschGordanCoefficient)
            if len(cgc) == 1:
                cgc = cgc.pop()
            else:
                raise ValueError("Wigner-Eckardt should produce 1 clebsch-gordan")

            #    <A||O||B> (B,b,O,o|A,a) = <Aa|Oo|Bb>
            # <A||O||B> (B,b,O,o|A,a)**2 = <Aa|Oo|Bb> (B,b,O,o|A,a)
            #                  <A||O||B> = sum_bo <Aa|Oo|Bb> (B,b,O,o|A,a)
            cgc = cgc*ASigma(cgc.args[1],cgc.args[3])

        else:
            cgc = 1

        left = self.left.get_uncoupled_form()
        c_left, t_left = left.as_coeff_terms(SphericalTensor)
        operator = self.operator.get_uncoupled_form()
        c_operator, t_operator = operator.as_coeff_terms(SphericalTensor)
        right = self.right.get_uncoupled_form()
        c_right, t_right = right.as_coeff_terms(SphericalTensor)

        return (
                combine_ASigmas(c_left*c_operator*c_right*cgc)
                *MatrixElement(Mul(*t_left), Mul(*t_operator), Mul(*t_right))
                )

def apply_wigner_eckardt(expr):
    """
    Applies the Wigner-Eckardt theorem to all instances of ThreeTensorMatrixElement
    """
    ttme = expr.atoms(ThreeTensorMatrixElement)
    subslist= []
    for m in ttme:
        subslist.append((m,m.use_wigner_eckardt()))
    return expr.subs(subslist)

def rewrite_as_direct_product(expr, **kw_args):
    """
    Rewrites all Three-tensor matrix-elements in expr into a direct product form.
    """
    matrices = expr.atoms(ThreeTensorMatrixElement)
    subslist= []
    for m in matrices:
        subslist.append((m,m.get_self_ito_direct_product(**kw_args)))
    return expr.subs(subslist)

def _as_coeff_tensor(tensor):
        c,t = tensor.as_coeff_terms(SphericalTensor)
        if len(t) == 1:
            return c, t[0]
        else:
            raise ValueError("not a tensor")


