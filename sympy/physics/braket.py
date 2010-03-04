from sympy import (
        S, Basic, Mul, global_assumptions, sympify, Symbol, Assume, ask, Q
        )
from sympy.physics.racahalgebra import (
        ThreeJSymbol, ClebschGordanCoefficient, SixJSymbol, SphericalTensor,
        refine_tjs2sjs, refine_phases, convert_cgc2tjs, convert_tjs2cgc,
        combine_ASigmas, remove_summation_indices, ASigma,
        )
from sympy.physics.secondquant import (
        Dagger, AntiSymmetricTensor, _sort_anticommuting_fermions
        )

braket_assumptions = global_assumptions

class WignerEckardDoesNotApply(Exception): pass

class QuantumState(Basic): pass

class QuantumOperator(Basic):
    """
    Base class for all objects representing a quantum operator.
    """
    is_commutative=False

class SphericalTensorOperator(QuantumOperator, SphericalTensor):
    """
    An Operator that transforms like a spherical tensor

    fulfills certain commutation relations with J_+- and J_z.
    """
    def __new__(cls, symbol, rank, projection):
        return SphericalTensor.__new__(cls, symbol, rank, projection)

    def _dagger_(self):
        """
        Hermitian conjugate of a SphericalTensorOperator.

        We follow the definition of Edmonds (1974).

        >>> from sympy.physics.braket import SphericalTensorOperator, Dagger
        >>> from sympy import symbols
        >>> k,q,T = symbols('k q T')
        >>> SphericalTensorOperator(T, k, q)
        T(k, q)
        >>> Dagger(SphericalTensorOperator(T, k, q))
        (-1)**(k + q)*T(k, -q)

        """
        k = self.rank
        q = self.projection
        T = self.symbol
        cls = type(self)
        return (-1)**(k + q)*self.__new__(cls, T, k, -q)

class BosonState(QuantumState):
    @property
    def spin_assume(self):
        """
        >>> from sympy.physics.braket import BosonState
        >>> BosonState('a').spin_assume
        'integer'
        """
        return Q.integer

    @classmethod
    def _sort_states(cls,states):
        return sorted(states), S.Zero

class FermionState(QuantumState):
    @property
    def spin_assume(self):
        """
        >>> from sympy.physics.braket import FermionState
        >>> FermionState('a').spin_assume
        'half_integer'
        """
        return 'half_integer'

    @classmethod
    def _sort_states(cls,states):
        return _sort_anticommuting_fermions(states)

class BraKet(QuantumState):

    is_commutative = False

    def __str__(self, *args):
        return self.left_braket+self._str_nobraket_(*args)+self.right_braket

    def _str_nobraket_(self, *args):
        str_args = []
        for s in self.args:
            try:
                str_args.append(s._str_nobraket_(*args))
            except AttributeError:
                str_args.append(str(s))
        return ", ".join([ str(s) for s in str_args])

    _sympystr_ = __str__

class Ket(BraKet):
    left_braket = '|'
    right_braket = '>'

class Bra(BraKet):
    left_braket = '<'
    right_braket = '|'

class QuantumState(Basic):
    """
    Base class for all objects representing a quantum state.

    - single-particle or many-body
    - coupled or uncoupled
    - bra-ket or a function in continuous basis

    etc.

    The common ground is the ability to be modified by a QuantumOperator

    """


    @property
    def symbol(self):
        return self.args[0]

    def get_operator_result(self, operator):
        """
        Calculates the resulting state as determined by the supplied operator.
        """
        assert isinstance(operator, QuantumOperator)
        return self._eval_operator(operator)

    def _eval_operator(self, operator):
        raise NotImplemented

    def _dagger_(self):
        """
        Dagger(|abc>) = <abc|

        >>> from sympy.physics.braket import SphFermKet, Dagger
        >>> A = SphFermKet('a')
        >>> Dagger(A)
        <a|
        >>> Dagger(Dagger(A))
        |a>
        >>> B = SphFermKet('b')
        >>> C = SphFermKet('c',A,B)
        >>> Dagger(C)
        <c(a, b)|

        >>> C.single_particle_states
        (|a>, |b>)
        >>> Dagger(C).single_particle_states
        (<a|, <b|)
        """
        new_args = [self.symbol]
        for arg in self.args[1:]:
            new_args.append(Dagger(arg))
        return self._hermitian_conjugate(*new_args)

    @property
    def single_particle_states(self):
        """
        A tuple containing all s.p. states of this QuantumState
        """
        return tuple(self[1:])

class RegularQuantumState(QuantumState):
    """
    A quantum state that has the properties of ket-states

    - Operators that act must be to the left of a RegularQuantumState.
    - Any operators acting on a RegularQuantumState, acts unmodified.
    """
    pass

class DualQuantumState(QuantumState):
    """
    A quantum state that has the properties of bra-states

    - Operators that act must be to the right of a DualQuantumState.
    - Any operators acting on a DualQuantumState, acts like its hermitian
    conjugate.
    """
    pass

class DirectQuantumState(QuantumState):
    """
    Base class for general quantum states of fermions or bosons.

    This is mainly a container for single particle states.  The states will be
    ordered canonically, according to the method ._sort_states() which must be
    defined.  This method is available if the subclass also inherits
    FermionState or BosonState as appropriate.
    """
    def __new__(cls, *args, **kw_args):
        """
        Will order the states canonically

        >>> from sympy.physics.braket import DirectQuantumState, FermionState, BosonState
        >>> from sympy import symbols
        >>> a,b,c,d = symbols('a b c d')
        >>> class Fermions(DirectQuantumState, FermionState):
        ...     pass
        >>> Fermions(b,a,c,d)
        -Fermions(a, b, c, d)
        >>> class Bosons(DirectQuantumState, BosonState):
        ...     pass
        >>> Bosons(b,a,c,d)
        Bosons(a, b, c, d)
        """
        new_args,sign = cls._sort_states(args)
        obj = QuantumState.__new__(cls, *new_args, **kw_args)
        return (-1)**sign*obj


class SphericalQuantumState(QuantumState):
    """
    Base class for a quantum state that transforms like a spherical tensor.

    We access SphericalTensor through aggregation rather than inheritance, as
    that will shield the tensor methods, preventing errors related to the
    tensorial properties of dual states.

    The constructor of spherical states could setup an instance variable
    spherical tensor, which is returned whenever necessary. (from
    _eval_as_tensor() for instance)

    Spherical states are eigenfunctions of the J**2 and J_z operators.

    They have well defined properties when operated on by a J_+ or J_-
    operator.
    """

    def __new__(cls, symbol, state1=None, state2=None):
        """
        Creates either a single particle spherical state or a
        coupled state constructed from state1 and state2.

        Given a symbol, 'i', this constructor creates two new symbols
        j_i and m_i, and register the appropriate assumptions.

        Coupled states have capital J, M letters denoting rank and projection.
        """
        if isinstance(symbol, str): symbol = Symbol(symbol)
        else: symbol = sympify(symbol)

        if state1 and state2:
            obj = QuantumState.__new__(cls, symbol, state1, state2)
            obj.state1 = state1
            obj.state2 = state2
            obj.is_coupled = True
            obj._j = Symbol("J_"+str(symbol))
            obj._m = Symbol("M_"+str(symbol))
        else:
            obj = QuantumState.__new__(cls, symbol)
            obj.is_coupled = False
            obj._j = Symbol("j_"+str(symbol))
            obj._m = Symbol("m_"+str(symbol))

        # register spin assumptions
        if obj.is_coupled:
            if ask(state1._j + state2._j,Q.integer):
                spin_assume = Q.integer
            elif ask(state1._j + state2._j,'half_integer'):
                spin_assume = 'half_integer'
            else:
                raise ValueError("Couldn't determine spin assumptions")
        else:
            spin_assume = obj.spin_assume
        braket_assumptions.add(Assume(obj._j,spin_assume))
        braket_assumptions.add(Assume(obj._m,spin_assume))


        return obj

    def _str_nobraket_(self, *args):
        """
        Coupling must show in represetantion.
        """
        if self.is_coupled:
            return "%s(%s, %s)"%(self.symbol,
                    self.state1._str_nobraket_(),
                    self.state2._str_nobraket_()
                    )
        else:
            return str(self.symbol)

    def as_coeff_tensor(self, **kw_args):
        """
        Returns the coefficient and tensor corresponding to this state.

        If this state is composed of other spherical states, and the keyword
        deep=True is supplied, we return the full decoupling of the state.

        >>> from sympy.physics.braket import SphFermKet, SphFermBra
        >>> a = SphFermKet('a')
        >>> a.as_coeff_tensor()
        (1, t(j_a, m_a))

        The rotation properties of a dual state implies that the corresponding
        tensor expression is different:

        >>> b = SphFermBra('b')
        >>> b.as_coeff_tensor()
        ((-1)**(j_b - m_b), t(j_b, -m_b))

        Constructing a two-particle ket state 'ab', we may ask for the tensor
        properties of ab:

        >>> ab = SphFermKet('ab',a,b); ab
        |ab(a, b)>
        >>> ab.as_coeff_tensor()
        (1, T(J_ab, M_ab))

        Note that the returned tensor is *not* a composite tensor, even if
        |ab> do correspond to a composite tensor.  This is done because the
        coeff in the above result is related to |ab>, but by definition
        completely ignorant of the internal tensors 'a' and 'b'.  In this
        example, the tensor of the dual state <b| have a phase (-1)**(j_b -
        m_b) that is essential to its rotational properties.  The user should
        not be mislead to believe that a subsequent decomposition of a
        composite tensor T[t(a)*t(b)] would be correct.

        To get the full tensor decomposition, you can supply the keyword
        deep=True:

        >>> full = ab.as_coeff_tensor(deep=True)
        >>> full[0]
        (-1)**(j_b - m_b)*Sum(m_a, m_b)*(j_a, m_a, j_b, -m_b|J_ab, M_ab)
        >>> full[1]
        t(j_a, m_a)*t(j_b, -m_b)

        Here we see that the dual state <b| is represented properly as
        (-1)**(j_b-m_b)*T(j_b,-m_b).

        """
        return self._eval_as_coeff_tensor(**kw_args)

    def as_direct_product(self, **kw_args):
        c,p = self._eval_as_coeff_sp_states(**kw_args)
        return Mul(c,*p)

    def as_coeff_sp_states(self, **kw_args):
        return self._eval_as_coeff_sp_states(**kw_args)

    @property
    def single_particle_states(self):
        """
        A tuple containing all s.p. states of this QuantumState
        """
        if self.is_coupled:
            return (self.state1.single_particle_states
                    + self.state2.single_particle_states)
        else:
            return self,


class SphericalRegularQuantumState(SphericalQuantumState, RegularQuantumState):
    """
    A quantum state that has the properties of spherical tensors.

    a state |jm> transforms like the spherical tensor T^j_m
    """
    def _eval_as_coeff_tensor(self, **kw_args):
        if self.is_coupled and kw_args.get('deep'):
                c1, t1 = self.state1.as_coeff_tensor()
                c2, t2 = self.state2.as_coeff_tensor()
                c, t = SphericalTensor('T', self._j, self._m, t1, t2
                        ).as_coeff_direct_product(**kw_args)
                return c1*c2*c, t
        elif self.is_coupled:
            return S.One, SphericalTensor('T', self._j, self._m)
        else:
            return S.One, SphericalTensor('t', self._j, self._m)

    def _eval_as_coeff_sp_states(self, **kw_args):
        """
        If we are a sp-states, there is no uncoupling to do,
        so the tensor properties are not relevant.
        """
        if not self.is_coupled:
            return S.One, (self,)
        args = { 'deep':True }
        args.update(kw_args)
        c,t = self.as_coeff_tensor(**args)
        states = self.single_particle_states
        return c,states


class SphericalDualQuantumState(SphericalQuantumState, DualQuantumState):
    """
    A quantum state that has the properties of spherical tensors.

    a state <jm| transforms like the spherical tensor (-1)**(j-m) T^j_-m
    """

    def _eval_as_coeff_tensor(self, **kw_args):
        """
        >>> from sympy.physics.braket import SphFermKet, SphFermBra
        >>> a = SphFermKet('a')
        >>> b = SphFermBra('b')
        >>> ab = SphFermBra('ab',a,b); ab
        <ab(a, b)|
        >>> ab.as_coeff_tensor()
        ((-1)**(J_ab - M_ab), T(J_ab, -M_ab))

        >>> full = ab.as_coeff_tensor(deep=True)
        >>> full[0]
        (-1)**(j_b - m_b)*(-1)**(J_ab - M_ab)*Sum(m_a, m_b)*(j_a, m_a, j_b, -m_b|J_ab, -M_ab)
        >>> full[1]
        t(j_a, m_a)*t(j_b, -m_b)
        """
        c = (-1)**(self._j - self._m)
        if self.is_coupled and kw_args.get('deep'):
            c1, t1 = self.state1.as_coeff_tensor()
            c2, t2 = self.state2.as_coeff_tensor()
            cgc, t = SphericalTensor('T', self._j, -self._m, t1, t2
                    ).as_coeff_direct_product(**kw_args)
            return c1*c2*c*cgc, t
        elif self.is_coupled:
            return c, SphericalTensor('T', self._j, -self._m)
        else:
            return c, SphericalTensor('t', self._j, -self._m)

    def _eval_as_coeff_sp_states(self, **kw_args):
        """
        A coupled bra consisting of two bra states, has the same decoupling
        as if everything was kets.  We try to do that simplification, in
        order to simplify as much as posisble.

        If we are a sp-states, there is no uncoupling to do,
        so the tensor properties are not relevant.
        """
        if not self.is_coupled:
            return S.One, (self,)

        args = { 'deep':True }
        args.update(kw_args)
        if (self.is_coupled and not kw_args.get('strict_bra_coupling')
                and isinstance(self.state1, DualQuantumState)
                and isinstance(self.state2, DualQuantumState)):
            c,t = Dagger(self).as_coeff_tensor(**args)
        else:
            c,t = self.as_coeff_tensor(**args)
        states = self.single_particle_states
        return c,states


class SphFermKet(SphericalRegularQuantumState, FermionState, Ket):
    """
    Represents a spherical fermion ket.

    >>> from sympy.physics.braket import SphFermKet
    >>> from sympy import symbols
    >>> a,b,c = symbols('a b c')
    >>> SphFermKet(a)
    |a>
    >>> SphFermKet(a)._j
    j_a
    >>> SphFermKet(a)._m
    m_a
    >>> A, B = SphFermKet('a'), SphFermKet('b')
    >>> C = SphFermKet('c',A,B); C
    |c(a, b)>

    Single particle states return tensors with symbol 't', coupled states 'T'


    """
    pass



class SphFermBra(SphericalDualQuantumState, FermionState, Bra):
    """
    Represents a spherical fermion bra.

    >>> from sympy.physics.braket import SphFermBra
    >>> from sympy import symbols
    >>> a,b,c = symbols('a b c')
    >>> SphFermBra(a)
    <a|
    >>> SphFermBra(a)._j
    j_a
    >>> SphFermBra(a)._m
    m_a
    >>> A, B = SphFermBra('a'), SphFermBra('b')
    >>> C = SphFermBra('c',A,B); C
    <c(a, b)|

    Single particle states return tensors with symbol 't', coupled states 'T'

    >>> A.as_coeff_tensor()
    ((-1)**(j_a - m_a), t(j_a, -m_a))
    >>> C.as_coeff_tensor()
    ((-1)**(J_c - M_c), T(J_c, -M_c))

    >>> C.as_direct_product()
    Sum(m_a, m_b)*(j_a, m_a, j_b, m_b|J_c, M_c)*<a|*<b|
    >>> C.as_direct_product(strict_bra_coupling=True)
    (-1)**(j_a - m_a)*(-1)**(j_b - m_b)*(-1)**(J_c - M_c)*Sum(m_a, m_b)*(j_a, -m_a, j_b, -m_b|J_c, -M_c)*<a|*<b|

    """
    _hermitian_conjugate = SphFermKet
SphFermKet._hermitian_conjugate = SphFermBra


class FermKet(RegularQuantumState, DirectQuantumState, Ket, FermionState):
    """
    Holds a direct product ket state of fermions.

    >>> from sympy.physics.braket import FermKet
    >>> from sympy import symbols
    >>> a,b,c = symbols('a b c')
    >>> FermKet(a)
    |a>
    >>> FermKet(a, b)
    |a, b>
    >>> FermKet(b, a)
    -|a, b>
    """
    pass

class FermBra(DualQuantumState, DirectQuantumState, Bra, FermionState):
    """
    Holds a dual direct product bra state of fermions.

    >>> from sympy.physics.braket import FermBra
    >>> from sympy import symbols
    >>> a,b,c = symbols('a b c')
    >>> FermBra(a)
    <a|
    >>> FermBra(a, b)
    <a, b|
    >>> FermBra(b, a)
    -<a, b|
    """
    _hermitian_conjugate = FermKet
FermKet._hermitian_conjugate = FermBra

def _get_matrix_element(left, operator, right, **kw_args):
    """
    Is responsible for spawning the correct matrix-class based on the
    arguments
    """
    if (    isinstance(left,SphericalTensor) and
            isinstance(operator, SphericalTensor) and
            isinstance(right, SphericalTensor)
            ):
        if kw_args.get('reduced'):
            return ReducedMatrixElement(left,operator, right, **kw_args)
        else:
            return ThreeTensorMatrixElement(left, operator, right, **kw_args)
    else:
        if kw_args.get('reduced'):
            raise ValueError("Reduced matrix element needs three tensors")
        return DirectMatrixElement(left, operator, right, **kw_args)


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

        assert isinstance(left, DualQuantumState)
        assert isinstance(operator, QuantumOperator)
        assert isinstance(right, RegularQuantumState)

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


