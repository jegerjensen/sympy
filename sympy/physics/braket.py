from sympy import (
        S, Basic, Mul, global_assumptions, sympify, Symbol, Assume, ask, Q
        )
from sympy.physics.racahalgebra import (
        ThreeJSymbol, ClebschGordanCoefficient, SixJSymbol, SphericalTensor,
        refine_tjs2sjs, refine_phases, convert_cgc2tjs, convert_tjs2cgc,
        combine_ASigmas, remove_summation_indices, ASigma,
        invert_clebsch_gordans, AngularMomentumSymbol
        )
from sympy.physics.secondquant import (
        Dagger, AntiSymmetricTensor, _sort_anticommuting_fermions
        )

braket_assumptions = global_assumptions
blank_symbol = Symbol("[blank symbol]")
default_redmat_definition = 'wikipedia'

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

    def __str__(self):
        return self.left_braket+self._str_nobraket_()+self.right_braket

    def _str_nobraket_(self, contained_in=None):
        if contained_in and not contained_in.is_dual is None:
            if self.is_dual != contained_in.is_dual:
                return str(self)
        str_args = []
        for s in self.args:
            if s is blank_symbol: continue
            try:
                str_args.append(s._str_nobraket_(self))
            except AttributeError:
                str_args.append(str(s))
        return ", ".join([ str(s) for s in str_args])

    def _sympystr_(self, p, *args):
        return str(self)

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
    is_dual=None

    def __new__(cls, symbol, *args, **kw_args):

        # interpret negative symbols as antiparticles
        if isinstance(symbol, str):
            if symbol[0] == "-":
                symbol = -Symbol(symbol[1:])
            else:
                symbol = Symbol(symbol)
        elif symbol:
            symbol = sympify(symbol)
            c, t = symbol.as_coeff_terms()
            if c.is_negative:
                if isinstance(t[0], QuantumState):
                    return t[0].get_antiparticle()
        else:
            symbol = blank_symbol


        if kw_args.get('hole'):
            symbol = -symbol
        #
        obj = Basic.__new__(cls, symbol, *args, **kw_args)
        return obj

    def _hashable_content(self):
        return Basic._hashable_content(self) + (self.is_hole,)

    def get_antiparticle(self):
        if len(self.args)>1: raise ValueError("Only single particle states can be anti-particles (FIXME?)")
        return type(self)(-self.symbol)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def label(self):
        c,t = self.symbol.as_coeff_terms()
        return t[0]

    @property
    def is_hole(self):
        if self.symbol is blank_symbol: return None
        c,t = self.symbol.as_coeff_terms()
        return c is S.NegativeOne

    @property
    def symbol_str(self):
        if self.symbol is blank_symbol: return ""
        return str(self.symbol)

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
        if self.symbol is blank_symbol: new_args = []
        else: new_args = [self.symbol]
        for arg in self.args[1:]:
            new_args.append(Dagger(arg))
        return self._hermitian_conjugate(*new_args)

    @property
    def single_particle_states(self):
        """
        A tuple containing all s.p. states of this QuantumState
        """
        return tuple(self.args[1:])

class RegularQuantumState(QuantumState):
    """
    A quantum state that has the properties of ket-states

    - Operators that act must be to the left of a RegularQuantumState.
    - Any operators acting on a RegularQuantumState, acts unmodified.
    """
    is_dual=False

class DualQuantumState(QuantumState):
    """
    A quantum state that has the properties of bra-states

    - Operators that act must be to the right of a DualQuantumState.
    - Any operators acting on a DualQuantumState, acts like its hermitian
    conjugate.
    """
    is_dual=True

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
        -Fermions([blank symbol], a, b, c, d)
        >>> Fermions(b,a,Fermions(c,d))
        -Fermions([blank symbol], a, b, c, d)
        >>> Fermions(Fermions(b,a),Fermions(c,d))
        -Fermions([blank symbol], a, b, c, d)
        >>> class Bosons(DirectQuantumState, BosonState):
        ...     pass
        >>> Bosons(b,a,c,d)
        Bosons([blank symbol], a, b, c, d)

        The first argument None, is the symbol given to QuantumState.  The Bra and Ket
        superclasses define prettier printting where the None symbol is skipped.

        Note: If only one state is supplied, a SphericalQuantumState, is
        returned directly.

        If no arguments are supplied we return a vacuum state

        """
        if len(args) == 0:
            return QuantumVacuum(cls, **kw_args)

        if len(args) == 1:
            if isinstance(args[0], QuantumState):
                if cls.is_dual is None: return args[0]
                if cls.is_dual == args[0].is_dual: return args[0]
                raise ValueError("Cannot have kets in a direct-product bra")
            else:
                if cls.is_dual is None: return SphericalQuantumState(args[0], **kw_args)
                if cls.is_dual: return SphFermBra(args[0], **kw_args)
                else: return SphFermKet(args[0], **kw_args)

        new_args = []
        coeff = S.One
        for arg in args:
            if arg is blank_symbol:
                continue
            if isinstance(arg, str):
                if cls.is_dual is True:
                    arg = SphFermBra(arg)
                elif cls.is_dual is False:
                    arg = SphFermKet(arg)
            c, t = arg.as_coeff_terms()
            if isinstance(t[0], QuantumState):
                if not t[0].is_dual is None:
                    if cls.is_dual != t[0].is_dual:
                        raise ValueError("Cannot have kets in a direct-product bra")
                if isinstance(t[0], QuantumVacuum):
                    continue
                if isinstance(t[0], DirectQuantumState):
                    if len(t[0].args) > 1:
                        new_args.extend(t[0].args[1:])
                        coeff *= c
                        continue
            if c.is_negative:
                new_args.append(t[0].get_antiparticle())
            else:
                new_args.append(arg)

        new_args,sign = cls._sort_states(new_args)

        obj = QuantumState.__new__(cls, None, *new_args, **kw_args)
        return coeff*(-1)**sign*obj


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

    def __new__(cls, symbol=None, state1=None, state2=None, **kw_args):
        """
        Creates either a single particle spherical state or a
        coupled state constructed from state1 and state2.

        Given a symbol, 'i', this constructor creates two new symbols
        j_i and m_i, and register the appropriate assumptions.

        Coupled states have capital J, M letters denoting rank and projection.

        Note: First argument is interpreted as a symbol characteristic for this state.
        """
        if not symbol: return QuantumVacuum(cls)

        if isinstance(symbol, QuantumState):
            raise ValueError("Quantum state cannot act as symbol")

        if state1 and state2:
            if isinstance(state1, str):
                state1 = cls(state1)
            if isinstance(state2, str):
                state2 = cls(state2)
            c,t = state1.as_coeff_terms()
            if c.is_negative: state1 = t[0].get_antiparticle()
            c,t = state2.as_coeff_terms()
            if c.is_negative: state2 = t[0].get_antiparticle()

            obj = QuantumState.__new__(cls, symbol, state1, state2, **kw_args)
            if obj.is_hole:
                raise ValueError("Only single particle states can be anti-particles (FIXME?)")
            obj.state1 = state1
            obj.state2 = state2
            obj.is_coupled = True
            obj._j = Symbol("J_"+str(obj.label))
            obj._m = Symbol("M_"+str(obj.label))
        else:
            obj = QuantumState.__new__(cls, symbol, **kw_args)
            obj.is_coupled = False
            obj._j = Symbol("j_"+str(obj.label))
            obj._m = Symbol("m_"+str(obj.label))

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

    def _str_nobraket_(self, contained_in=None):
        """
        Coupling and hole states must show in represetantion.
        """

        if contained_in and not contained_in.is_dual is None:
            if contained_in.is_dual != self.is_dual:
                return str(self)
        if self.is_coupled:
            return "%s(%s, %s)"%(self.symbol_str,
                    self.state1._str_nobraket_(self),
                    self.state2._str_nobraket_(self)
                    )
        else:
            return self.symbol_str

    def as_coeff_tensor(self, **kw_args):
        """
        Returns the coefficient and tensor corresponding to this state.

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
        |ab(a, <b|)>
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

        To get the full decomposition of the state, use .as_coeff_sp_states()
        """
        return self._eval_as_coeff_tensor(**kw_args)


    def _eval_as_coeff_tensor(self, **kw_args):
        """
        >>> from sympy.physics.braket import SphFermKet, SphFermBra
        >>> a = SphFermKet('a')
        >>> b = SphFermBra('b')
        >>> ab = SphFermBra('ab',a,b); ab
        <ab(|a>, b)|
        >>> ab.as_coeff_tensor()
        ((-1)**(J_ab - M_ab), T(J_ab, -M_ab))

        """
        phase = self._tensor_phase
        m = self._tensor_proj
        j = self._j
        if self.is_coupled:
            return phase, SphericalTensor('T', j, m)
        else:
            return phase, SphericalTensor('t', j, m)


    def _eval_as_coeff_sp_states(self, **kw_args):

        if not self.is_coupled:
            return S.One, (self,)

        phase = self._tensor_phase
        m = self._tensor_proj
        j = self._j

        # get top-level coupling
        c1, t1 = self.state1.as_coeff_tensor()
        c2, t2 = self.state2.as_coeff_tensor()
        cgc, t = SphericalTensor('T', j, m, t1, t2
                ).as_coeff_direct_product(deep=True)
        cgc = cgc*c1*c2

        # call deep to get coefficients for internal structure
        c1, s1 = self.state1._eval_as_coeff_sp_states(**kw_args)
        c2, s2 = self.state2._eval_as_coeff_sp_states(**kw_args)
        return phase*combine_ASigmas(c1*c2*cgc), s1 + s2

    def as_direct_product(self, **kw_args):
        c,p = self._eval_as_coeff_sp_states(**kw_args)
        return Mul(c,*p)

    def as_coeff_sp_states(self, **kw_args):
        """
        Returns the uncoupled form of self as a tuple (coeff, tuple(a,b,c,...))

        a,b,c are the single particle states from which this state is built.

        >>> from sympy.physics.braket import SphFermKet, SphFermBra, Dagger
        >>> a = SphFermKet('a')
        >>> b = SphFermBra('b')
        >>> a.as_coeff_sp_states()
        (1, (|a>,))
        >>> b.as_coeff_sp_states()
        (1, (<b|,))

        >>> coeff, states = SphFermKet('ab',a,b).as_coeff_sp_states()
        >>> coeff
        (-1)**(j_b - m_b)*Sum(m_a, m_b)*(j_a, m_a, j_b, -m_b|J_ab, M_ab)
        >>> states
        (|a>, <b|)

        If a,b are coupled to form a dual state, this is reflected in the
        coupling coefficient:

        >>> bra_ab = SphFermBra('ab',a,b)
        >>> coeff, states = bra_ab.as_coeff_sp_states()
        >>> coeff
        (-1)**(j_b - m_b)*(-1)**(J_ab - M_ab)*Sum(m_a, m_b)*(j_a, m_a, j_b, -m_b|J_ab, -M_ab)
        >>> states
        (|a>, <b|)

        But if two bra states are coupled to form a new bra state, the coupling
        coefficient simplifies by symmetry relations of the Clebsch-Gordan
        coefficients.

        >>> bra_ab = SphFermBra('ab',Dagger(a), b)
        >>> coeff, states = bra_ab.as_coeff_sp_states()
        >>> coeff
        Sum(m_a, m_b)*(j_a, m_a, j_b, m_b|J_ab, M_ab)
        >>> states
        (<a|, <b|)

        Nested couplings are also decomposed:

        >>> abc = SphFermKet('abc', SphFermKet('ab', a,b), 'c'); abc
        |abc(ab(a, <b|), c)>
        >>> coeff_abc, states = abc.as_coeff_sp_states()
        >>> coeff_abc
        (-1)**(j_b - m_b)*Sum(M_ab, m_a, m_b, m_c)*(J_ab, M_ab, j_c, m_c|J_abc, M_abc)*(j_a, m_a, j_b, -m_b|J_ab, M_ab)
        >>> states
        (|a>, <b|, |c>)


        """
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
    def __new__(cls, *args, **kw_args):
        """
        >>> from sympy.physics.braket import SphFermKet, SphFermBra
        >>> SphFermKet('a').as_coeff_tensor()
        (1, t(j_a, m_a))
        >>> SphFermKet('a',hole=True).as_coeff_tensor()
        ((-1)**(j_a - m_a), t(j_a, -m_a))
        """
        obj = SphericalQuantumState.__new__(cls, *args, **kw_args)
        if obj.is_hole:
            obj._tensor_phase = (-1)**(obj._j - obj._m)
            obj._tensor_proj = Mul(S.NegativeOne, obj._m)
        else:
            obj._tensor_phase = S.One
            obj._tensor_proj = obj._m
        return obj


class SphericalDualQuantumState(SphericalQuantumState, DualQuantumState):
    """
    A quantum state that has the properties of spherical tensors.

    a state <jm| transforms like the spherical tensor (-1)**(j-m) T^j_-m
    """
    def __new__(cls, *args, **kw_args):
        """
        >>> from sympy.physics.braket import SphFermBra
        >>> SphFermBra('a').as_coeff_tensor()
        ((-1)**(j_a - m_a), t(j_a, -m_a))
        >>> SphFermBra('a',hole=True).as_coeff_tensor()
        ((-1)**(2*j_a), t(j_a, m_a))
        """
        obj = SphericalQuantumState.__new__(cls, *args, **kw_args)
        if obj.is_hole:
            obj._tensor_phase = (-1)**(2*obj._j)
            obj._tensor_proj = obj._m
        else:
            obj._tensor_phase = (-1)**(obj._j - obj._m)
            obj._tensor_proj = Mul(S.NegativeOne, obj._m)
        return obj


    def _eval_as_coeff_sp_states(self, **kw_args):
        """
        A coupled bra consisting of two bra states, has the same decoupling
        as if everything was kets.  We try to do that simplification, in
        order to simplify as much as posisble.
        """
        if (self.is_coupled and not kw_args.get('strict_bra_coupling')
                and isinstance(self.state1, DualQuantumState)
                and isinstance(self.state2, DualQuantumState)):
            c,s = Dagger(self)._eval_as_coeff_sp_states(**kw_args)
            return c, tuple(map(Dagger, s))
        else:
            return SphericalQuantumState._eval_as_coeff_sp_states(self, **kw_args)


class SphFermKet(SphericalRegularQuantumState, FermionState, Ket):
    """
    Represents a spherical fermion ket.

    >>> from sympy.physics.braket import SphFermKet, Dagger
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
    >>> SphFermKet('z','x','y')
    |z(x, y)>
    >>> SphFermKet('z','x','y').state1
    |x>
    >>> SphFermKet('z','x','y').state2
    |y>

    Single particle states return tensors with symbol 't', coupled states 'T'

    >>> C = SphFermKet('c',A,Dagger(B)); C
    |c(a, <b|)>

    """
    pass


class SphFermBra(SphericalDualQuantumState, FermionState, Bra):
    """
    Represents a spherical fermion bra.

    >>> from sympy.physics.braket import SphFermBra, SphericalTensor, SphericalQuantumState
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
    >>> isinstance(C, SphericalTensor)
    False
    >>> isinstance(C, SphericalQuantumState)
    True
    >>> SphFermBra('z','x','y')
    <z(x, y)|

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

    >>> from sympy.physics.braket import FermKet, SphFermKet
    >>> from sympy import symbols
    >>> a,b,c = symbols('a b c')
    >>> FermKet(a)
    |a>
    >>> FermKet(a, b)
    |a, b>
    >>> FermKet(b, a)
    -|a, b>

    >>> FermKet(SphFermKet(a), SphFermKet(b))
    |a, b>
    >>> FermKet(SphFermKet(b), SphFermKet(a))
    -|a, b>
    """
    pass

class FermBra(DualQuantumState, DirectQuantumState, Bra, FermionState):
    """
    Holds a dual direct product bra state of fermions.

    >>> from sympy.physics.braket import FermBra, SphFermBra
    >>> from sympy import symbols
    >>> a,b,c = symbols('a b c')
    >>> FermBra(a)
    <a|
    >>> FermBra(a, b)
    <a, b|
    >>> FermBra(b, a)
    -<a, b|

    >>> FermBra(SphFermBra(a), SphFermBra(b))
    <a, b|
    >>> FermBra(SphFermBra(b), SphFermBra(a))
    -<a, b|

    >>> a = SphFermBra('a')
    >>> b = SphFermBra('b')
    >>> FermBra(a, -b)
    <a, -b|
    >>> FermBra(a, -b).single_particle_states[0].is_hole
    False
    >>> FermBra(a, -b).single_particle_states[1].is_hole
    True
    >>> FermBra('a', hole=True)
    <-a|
    """
    _hermitian_conjugate = FermKet
FermKet._hermitian_conjugate = FermBra

class QuantumVacuum(DirectQuantumState, SphericalQuantumState):
    """
    Represents the quantum vacuum.
    """

    def __new__(cls, template_cls=None, **kw_args):
        """
        Spawns the correct subclass upon creation.

        If template_cls is given, we examine the provided class and try
        to return a vacuum that matches.  (Bra or Ket)

        >>> from sympy.physics.braket import QuantumVacuum, FermKet, FermBra
        >>> QuantumVacuum(FermKet)
        |>
        >>> QuantumVacuum(FermBra)
        <|
        """
        if cls is QuantumVacuum:
            if template_cls:
                if template_cls.is_dual is True:
                    return QuantumVacuumBra()
                if template_cls.is_dual is False:
                    return QuantumVacuumKet()
            raise ValueError("Failed to guess which Vacuum you want")
        else:
            # Note: Do not call SphericalQuantumState.__new__() from here. It
            # creates an infinite loop!
            obj = QuantumState.__new__(cls, None)
            obj.is_coupled = False
            obj._j = S.Zero
            obj._m = S.Zero
            obj._tensor_proj = S.Zero
            obj._tensor_phase = S.One
            return obj

    @property
    def single_particle_states(self):
        """
        A tuple containing all s.p. states of this QuantumState
        """
        return tuple([])

    @property
    def spin_assume(self):
        # vacuum has spin 0
        return 'integer'

class QuantumVacuumKet(QuantumVacuum, Ket, RegularQuantumState):
    pass

class QuantumVacuumBra(QuantumVacuum, Bra, DualQuantumState):
    _hermitian_conjugate = QuantumVacuumKet
QuantumVacuumKet._hermitian_conjugate = QuantumVacuumBra



def _get_matrix_element(left, operator, right, **kw_args):
    """
    Is responsible for spawning the correct matrix-class based on the
    arguments
    """

    if (    isinstance(left,SphericalQuantumState) and
            isinstance(operator, SphericalTensor) and
            isinstance(right, SphericalQuantumState)
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
    Base class for all matrix elements.

    Responsible for spawning the correct subclass depending on the input
    parameters.

    >>> from sympy.physics.braket import SphFermBra, SphFermKet, FermBra
    >>> from sympy.physics.braket import MatrixElement, SphericalTensorOperator
    >>> from sympy.physics.braket import ThreeTensorMatrixElement
    >>> from sympy.physics.braket import ReducedMatrixElement
    >>> from sympy.physics.braket import DirectMatrixElement
    >>> from sympy import symbols
    >>> a,b,c,d,e,f,g,h,D = symbols('abcdefghD')

    >>> bra_a = SphFermBra(a)
    >>> ket_b = SphFermKet(b)
    >>> Op = SphericalTensorOperator('Op',D,d)
    >>> m = MatrixElement(bra_a,Op,ket_b); m
    <a| Op(D, d) |b>
    >>> isinstance(m, ThreeTensorMatrixElement)
    True

    >>> m = MatrixElement(bra_a,Op,ket_b, reduced=True); m
    <a|| Op(D) ||b>
    >>> isinstance(m, ReducedMatrixElement)
    True

    >>> bra_ac = FermBra(bra_a, SphFermBra(c))
    >>> m = MatrixElement(bra_ac,Op,ket_b); m
    <a, c| Op(D, d) |b>
    >>> isinstance(m, DirectMatrixElement)
    True
    """

    is_commutative=True

    def __new__(cls,left, operator, right, **kw_args):

        coeff = S.One
        if not left : left = FermBra()
        elif isinstance(left, (tuple,list)): left = FermBra(*left)
        elif isinstance(left, str): left = SphFermBra(left)
        if not right: right = FermKet()
        elif isinstance(right,(tuple,list)): right = FermKet(*right)
        elif isinstance(right, str): right = SphFermKet(right)
        if isinstance(left, Mul):
            c,t = left.as_coeff_terms()
            coeff *= c
            left = t[0]
        if isinstance(right, Mul):
            c,t = right.as_coeff_terms()
            coeff *= c
            right = t[0]

        if cls is MatrixElement:
            return coeff*_get_matrix_element(left, operator, right, **kw_args)
        else:
            assert isinstance(left, DualQuantumState), "not dual: %s" %left
            assert isinstance(operator, QuantumOperator)
            assert isinstance(right, RegularQuantumState)
            obj = Basic.__new__(cls, left, operator, right, **kw_args)
            return coeff*obj

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[2]

    @property
    def operator(self):
        return self.args[1]


    def __str__(self):
        return "%s %s %s" %self.args[:4]

    def _sympystr_(self, p, *args):
        return str(self)


class ReducedMatrixElement(MatrixElement):
    """
    The reduced matrix element <.||.||.> is defined in terms of three
    SphericalTensors, but is independent of all three projections.
    The relation with the direct product matrix element is on the form

                k                            k
        < J M| T  |J'M'> = G(J'M'kqJM) <J|| T ||J'>
                q

    where G(...) is a purely geometrical factor depending on ranks and
    projections of the involved spherical tensors.

    The exact form of the geometrical factor defines the reduced matrix
    element, and there are different versions available.  A few definitions
    are supplied in this module, and the section "Alternative definitions"
    below provides details and examples.

    >>> from sympy.physics.braket import (
    ...         ReducedMatrixElement, SphFermKet, SphFermBra,
    ...         SphericalTensorOperator, default_redmat_definition
    ...         )
    >>> from sympy import symbols
    >>> k,q = symbols('k q')
    >>> bra = SphFermBra('a')
    >>> ket = SphFermKet('b')
    >>> T = SphericalTensorOperator('T',k,q)
    >>> ReducedMatrixElement(bra, T, ket)
    <a|| T(k) ||b>

    The geometrical factor is available through the method
    ._get_reduction_factor():

    >>> ReducedMatrixElement(bra, T, ket)._get_reduction_factor()
    (j_b, m_b, k, q|j_a, m_a)

    You can also formulate the direct product (corresponding to a full decomposition
    of all states) in terms of (ito) the ReducedMatrixElement:
    Note that those expressions are not equal to the ReducedMatrixElement,
    but rather to the corresponding uncoupled matrix element.

    >>> ReducedMatrixElement(bra, T, ket).get_direct_product_ito_self()
    (j_b, m_b, k, q|j_a, m_a)*<a|| T(k) ||b>
    >>> ReducedMatrixElement(bra, T, SphFermKet('bc',ket,'c')).get_direct_product_ito_self()
    Sum(J_bc, M_bc)*(J_bc, M_bc, k, q|j_a, m_a)*(j_b, m_b, j_c, m_c|J_bc, M_bc)*<a|| T(k) ||bc(b, c)>

    The ReducedMatrixElement can also express itself in terms of a direct product.

    >>> ReducedMatrixElement(bra, T, ket).as_direct_product()
    Sum(m_b, q)*(j_b, m_b, k, q|j_a, m_a)*<a| T(k, q) |b>
    >>> ReducedMatrixElement(bra, T, SphFermKet('bc',ket,'c')).as_direct_product()
    Sum(M_bc, m_b, m_c, q)*(J_bc, M_bc, k, q|j_a, m_a)*(j_b, m_b, j_c, m_c|J_bc, M_bc)*<a| T(k, q) |b, c>


    Alternative definitions
    =======================

    The default definition is specified with the global variable
    default_redmat_definition:

    >>> default_redmat_definition
    'wikipedia'
    >>> type(ReducedMatrixElement(bra, T, ket)).__name__
    'ReducedMatrixElement_wikipedia'

    Other definitions currently available are 'edmonds' and 'brink_satchler':

    >>> ReducedMatrixElement(bra, T, ket, 'edmonds')._get_reduction_factor()
    (-1)**(j_a - m_a)*ThreeJSymbol(j_a, j_b, k, m_a, -m_b, -q)
    >>> ReducedMatrixElement(bra, T, ket, 'brink_satchler')._get_reduction_factor()
    (-1)**(2*k)*(j_b, m_b, k, q|j_a, m_a)

    If you want to introduce yet another definition, all you have to do is
    to subclass ReducedMatrixElement and overload the method
    ReducedMatrixElement._get_reduction_factor(self).  The convention is to
    name the subclass ReducedMatrixElement_<definition_string>.  (Future
    functionality may depend on that.)

    """

    nargs = 3
    _definition = None


    def __new__(cls,left, op, right, *opt_args, **kw_args):

        if cls is ReducedMatrixElement:
            if len(opt_args)==1:
                definition = opt_args[0]
            else:
                definition = kw_args.get('definition', default_redmat_definition)
            if definition == 'edmonds':
                return ReducedMatrixElement_edmonds(left, op, right, **kw_args)
            elif definition == 'brink_satchler':
                return ReducedMatrixElement_brink_satchler(left, op, right, **kw_args)
            elif definition == 'wikipedia':
                return ReducedMatrixElement_wikipedia(left, op, right, **kw_args)
            else:
                raise ValueError(
                        "Incomprehensible reduced matrix definition: %s"%definition)
        else:
            obj = MatrixElement.__new__(cls, left,op,right, **kw_args)
            return obj

    def _get_reduction_factor(self, **kw_args):
        """
        Returns the ClebschGordanCoefficient that relates this reduced
        matrix element to the corresponding direct matrix element.

        >>> from sympy.physics.braket import (
        ...         ReducedMatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> ReducedMatrixElement(bra, T, ket)._get_reduction_factor()
        (j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, definition='brink_satchler')._get_reduction_factor()
        (-1)**(2*k)*(j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, 'edmonds')._get_reduction_factor()
        (-1)**(j_a - m_a)*ThreeJSymbol(j_a, j_b, k, m_a, -m_b, -q)
        """
        left,op,right = self.args
        c_ket, t_ket = right.as_coeff_tensor()
        j1, m1 = t_ket.get_rank_proj()
        j2, m2 = op.get_rank_proj()
        J, M = left._j, left._m
        if self.definition == 'wikipedia':
            factor = c_ket*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)
        elif self.definition == 'edmonds':
            factor = c_ket*(-1)**(J-M)*ThreeJSymbol(J, j2, j1, -M, m2, m1)
        elif self.definition == 'brink_satchler':
            factor = (-1)**(2*j2)*c_ket*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)

        if kw_args.get('3j'):
            return refine_phases(convert_cgc2tjs(factor))
        else:
            return factor

    def _get_inverse_reduction_factor(self, **kw_args):
        """
        Returns the inverted ClebschGordanCoefficient that relates this reduced
        matrix element to the corresponding direct matrix element.

        The inversion is done with orthogonality relations, employing the fact
        that the reduced matrix element is independent of the projections.

        >>> from sympy.physics.braket import (
        ...         ReducedMatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> ReducedMatrixElement(bra, T, ket)._get_inverse_reduction_factor()
        Sum(m_b, q)*(j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, definition='brink_satchler')._get_inverse_reduction_factor()
        (-1)**(-2*k)*Sum(m_b, q)*(j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, 'edmonds')._get_inverse_reduction_factor()
        (-1)**(m_a - j_a)*(1 + 2*j_a)*Sum(m_b, q)*ThreeJSymbol(j_a, j_b, k, m_a, -m_b, -q)
        """
        left,op,right = self.args
        c_ket, t_ket = right.as_coeff_tensor()
        j1, m1 = t_ket.get_rank_proj()
        j2, m2 = op.get_rank_proj()
        J, M = left._j, left._m
        if self.definition == 'wikipedia':
            factor = ASigma(m1, m2)*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)/c_ket
        elif self.definition == 'edmonds':
            factor = (-1)**(M-J)*ASigma(m1, m2)*(2*J+1)*ThreeJSymbol(J, j2, j1, -M, m2, m1)/c_ket
        elif self.definition == 'brink_satchler':
            factor = (-1)**(-2*j2)*ASigma(m1,m2)*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)/c_ket

        if kw_args.get('3j'):
            return refine_phases(convert_cgc2tjs(factor))
        else:
            return factor

    def _get_ThreeTensorMatrixElement(self):
        """
        Returns the direct matrix element that is related to this
        reduced matrix element by a reduction factor (a Clebsch-Gordan
        coefficient).

        >>> from sympy.physics.braket import (
        ...         ReducedMatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> ReducedMatrixElement(bra, T, ket)._get_ThreeTensorMatrixElement()
        <a| T(k, q) |b>
        """
        return ThreeTensorMatrixElement(*self.args)

    def __str__(self):
        return "%s| %s |%s" %(
                self.left,
                "%s(%s)"%self.operator._str_drop_projection_(),
                self.right
                )

    @property
    def definition(self):
        """
        Returns the a string describing the definition associated with this RedMat.
        """
        return self._definition


    def get_direct_product_ito_self(self, **kw_args):
        """
        Returns the direct product of all involved spherical tensors i.t.o
        the reduced matrix element.

        >>> from sympy.physics.braket import (
        ...         ReducedMatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> ReducedMatrixElement(bra, T, ket).get_direct_product_ito_self()
        (j_b, m_b, k, q|j_a, m_a)*<a|| T(k) ||b>

        """
        cgc = self._get_reduction_factor(**kw_args)
        matel = self._get_ThreeTensorMatrixElement()
        dirprod = matel.get_direct_product_ito_self()
        return cgc * dirprod.subs(matel, self)

    def as_direct_product(self, **kw_args):
        """
        Returns the reduced matrix element in terms of direct product
        matrices.
        """
        invcgc = self._get_inverse_reduction_factor(**kw_args)
        matel = self._get_ThreeTensorMatrixElement()

        return combine_ASigmas(matel.as_direct_product()*invcgc)

class ReducedMatrixElement_edmonds(ReducedMatrixElement):
    _definition = 'edmonds'
class ReducedMatrixElement_brink_satchler(ReducedMatrixElement):
    _definition = 'brink_satchler'
class ReducedMatrixElement_wikipedia(ReducedMatrixElement):
    _definition = 'wikipedia'

class DirectMatrixElement(MatrixElement):
    """
    Holds matrix elements corresponding to the direct product of any number of tensors.

    The sp states are typically SphFermKet, but they may be FermKet as well

    >>> from sympy.physics.braket import SphFermBra, SphFermKet
    >>> from sympy.physics.braket import DirectMatrixElement, SphericalTensorOperator
    >>> from sympy import symbols
    >>> a,b,c,d,e,f,g,h,D = symbols('abcdefghD')

    >>> bra_a = SphFermBra(a)
    >>> ket_b = SphFermKet(b)
    >>> ket_c = SphFermKet(c)
    >>> Op = SphericalTensorOperator('Op',D,d)
    >>> DirectMatrixElement(bra_a,Op,(ket_b,ket_c))
    <a| Op(D, d) |b, c>
    >>> DirectMatrixElement(bra_a,Op,(ket_c,ket_b))
    -<a| Op(D, d) |b, c>
    """
    nargs=3
    def __new__(cls,left, op, right, **kw_args):


        obj = MatrixElement.__new__(cls, left, op, right, **kw_args)

        return obj

    def use_wigner_eckardt(self, **kw_args):
        raise WignerEckardDoesNotApply

    def as_only_particles(self):
        """
        Returns this matrix element in a vacuum s.t. self has only particles.

        Returns an equivalent DirectMatrixElement where all hole states have been
        reexpressed as pareticle states in a different vacuum.

        """
        holes = [ h for h in (self.left.single_particle_states
            + self.right.single_particle_states) if h.is_hole ]
        return self.shift_vacuum(holes)

    def shift_vacuum(self, states):
        """
        We rewrite this matrix element by a change of vacuum.

        ``states`` -- iterable containing single particle QuantumStates

        For each of the supplied states we absorb it into the vacuum, or
        separate it from the vacuum depending on the initial vacuum.
        We return this DirectMatrix element expressed in the shifted vacuum.

        >>> from sympy.physics.braket import DirectMatrixElement, SphFermKet, SphFermBra
        >>> from sympy.physics.braket import SphericalTensorOperator
        >>> a = SphFermBra('a')
        >>> b = SphFermBra('b')
        >>> c = SphFermKet('c')
        >>> d = SphFermKet('d')
        >>> Op = SphericalTensorOperator('T','k','q')

        >>> m = DirectMatrixElement((a,b),Op,(c,d)); m
        <a, b| T(k, q) |c, d>

        >>> m.shift_vacuum([d])
        <a, b, -d| T(k, q) |c>

        """
        shifted = self
        coeff = S.One
        for s in states:
            c, shifted = shifted._eval_coeff_vacuumshifted(s)
            coeff *= c
        return coeff*shifted

    def _eval_coeff_vacuumshifted(self, state):
        sign = 0
        left = list(self.left.single_particle_states)
        right = list(self.right.single_particle_states)
        for i in range(len(left)):
            if left[-1-i] == state:
                sign = i
                left.remove(state)
                right.append(Dagger(state.get_antiparticle()))
                break
        else:
            for i in range(len(right)):
                if right[-1-i] == state:
                    sign = i
                    right.remove(state)
                    left.append(Dagger(state.get_antiparticle()))
                    break
            else:
                return S.One, self

        left = FermBra(*left)
        right = FermKet(*right)
        matrix = type(self)(left, self.operator, right)
        c,t = matrix.as_coeff_terms()
        return (-1)**(sign)*c,t[0]


class ThreeTensorMatrixElement(MatrixElement):
    """
    Holds reducable matrix element consisting of 3 spherical tensors. (direct product)
    """
    nargs=3

    def __new__(cls,left, op, right):
        obj = MatrixElement.__new__(cls,left,op,right)
        return obj


    def use_wigner_eckardt(self, definition=default_redmat_definition, **kw_args):
        """
        Applies the Wigner-Eckard theorem to write the supplied direct matrix
        element on the form

                    k                            k
            < J M| T  |J'M'> = (J'M'kq|JM) <J|| T ||J'>
                    q
        where the reduced matrix element <.||.||.> is independent of the
        projections.

        >>> from sympy.physics.braket import (
        ...         MatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> MatrixElement(bra, T, ket).use_wigner_eckardt()
        (j_b, m_b, k, q|j_a, m_a)*<a|| T(k) ||b>
        >>> MatrixElement(bra, T, ket).use_wigner_eckardt('brink_satchler')
        (-1)**(2*k)*(j_b, m_b, k, q|j_a, m_a)*<a|| T(k) ||b>
        >>> MatrixElement(bra, T, ket).use_wigner_eckardt('edmonds')
         (-1)**(j_a - m_a)*<a|| T(k) ||b>*ThreeJSymbol(j_a, j_b, k, m_a, -m_b, -q)

        """
        redmat = ReducedMatrixElement(self.left, self.operator, self.right, definition)
        return redmat._get_reduction_factor(**kw_args)*redmat

    def get_direct_product_ito_self(self, **kw_args):
        """
        >>> from sympy.physics.braket import (
        ...         MatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('c')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> MatrixElement(bra, T, ket).get_direct_product_ito_self()
        <a| T(k, q) |c>

        >>> a = SphFermBra('a')
        >>> b = SphFermBra('b')
        >>> bra_ab = SphFermBra('ab',a,b)
        >>> MatrixElement(bra_ab, T, ket).get_direct_product_ito_self()
        Sum(J_ab, M_ab)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*<ab(a, b)| T(k, q) |c>

        >>> c = SphFermKet('c')
        >>> d = SphFermKet('d')
        >>> ket_cd = SphFermKet('cd',c,d)
        >>> MatrixElement(bra_ab, T, ket_cd).get_direct_product_ito_self()
        Sum(J_ab, J_cd, M_ab, M_cd)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*(j_c, m_c, j_d, m_d|J_cd, M_cd)*<ab(a, b)| T(k, q) |cd(c, d)>

        """
        if kw_args.get('wigner_eckardt'):
            matrix = self.use_wigner_eckardt(**kw_args)
        else:
            matrix = self

        coeffs = self.as_direct_product(only_coeffs=True)
        return invert_clebsch_gordans(coeffs)*matrix


    def as_direct_product(self, **kw_args):
        """
        >>> from sympy.physics.braket import (
        ...         MatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('c')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> MatrixElement(bra, T, ket).as_direct_product()
        <a| T(k, q) |c>

        >>> a = SphFermBra('a')
        >>> b = SphFermBra('b')
        >>> bra_ab = SphFermBra('ab',a,b)
        >>> MatrixElement(bra_ab, T, ket).as_direct_product()
        Sum(m_a, m_b)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*<a, b| T(k, q) |c>

        >>> MatrixElement(bra_ab, T, ket).as_direct_product(strict_bra_coupling=1)
        (-1)**(j_a - m_a)*(-1)**(j_b - m_b)*(-1)**(J_ab - M_ab)*Sum(m_a, m_b)*(j_a, -m_a, j_b, -m_b|J_ab, -M_ab)*<a, b| T(k, q) |c>

        >>> c = SphFermKet('c')
        >>> d = SphFermKet('d')
        >>> ket_cd = SphFermKet('cd',c,d)
        >>> MatrixElement(bra_ab, T, ket_cd).as_direct_product()
        Sum(m_a, m_b, m_c, m_d)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*(j_c, m_c, j_d, m_d|J_cd, M_cd)*<a, b| T(k, q) |c, d>

        The keyword wigner_eckardt=True gives you an expression for the reduced
        matrix element in terms of the direct product matrix.  To this end, we
        use the ClebschGordanCoefficient orthogonality to rewrite

        <(ab)JM|T(k,q)|(cd)JM> == (J_cd, M_cd, k, q|J_ab, M_ab) <J(ab)||T(k)||J(cd)>

        as

        <J(ab)||T(k)||J(cd)> ==
            Sum(M_cd, q) (J_cd, M_cd, k, q|J_ab, M_ab) <(ab)JM|T(k,q)|(cd)JM>

        For the above expression we obtain:

        >>> MatrixElement(bra_ab, T, ket_cd).as_direct_product(wigner_eckardt=True)
        Sum(M_cd, m_a, m_b, m_c, m_d, q)*(J_cd, M_cd, k, q|J_ab, M_ab)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*(j_c, m_c, j_d, m_d|J_cd, M_cd)*<a, b| T(k, q) |c, d>

        >>> MatrixElement(bra_ab, T, ket_cd).as_direct_product(wigner_eckardt=True, definition='brink_satchler')
        (-1)**(-2*k)*Sum(M_cd, m_a, m_b, m_c, m_d, q)*(J_cd, M_cd, k, q|J_ab, M_ab)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*(j_c, m_c, j_d, m_d|J_cd, M_cd)*<a, b| T(k, q) |c, d>
        """
        if kw_args.get('only_particle_states'):
            matrix = self.get_related_direct_matrix(only_particle_states=True)
        else:
            matrix = self.get_related_direct_matrix()

        cbra, bra = self.left.as_coeff_sp_states(**kw_args)
        cket, ket = self.right.as_coeff_sp_states(**kw_args)

        if kw_args.get('wigner_eckardt'):
            cgc = ReducedMatrixElement(self.left, self.operator, self.right,
                    **kw_args)._get_reduction_factor()
            # inversion of cgc is best done with orthogonality:
            c, t = cgc.as_coeff_terms(AngularMomentumSymbol)
            cgc = ASigma(self.operator.projection, self.right._m)*t[0]/c
        else:
            cgc = S.One

        if kw_args.get('only_coeffs'):
            return combine_ASigmas(cbra*cket*cgc)
        else:
            return combine_ASigmas(cbra*cket*cgc)*matrix


    def get_related_direct_matrix(self, **kw_args):
        """
        Returns the direct product matrix that corresponds to this matrix.

        >>> from sympy.physics.braket import (
        ...         MatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> a = SphFermBra('a')
        >>> b = SphFermBra('b')
        >>> bra = SphFermBra('ab',a,b)
        >>> c = SphFermKet('c')
        >>> d = SphFermKet('d')
        >>> ket = SphFermKet('cd',c,d)
        >>> m=MatrixElement(bra, T, ket); m
        <ab(a, b)| T(k, q) |cd(c, d)>
        >>> m.get_related_direct_matrix()
        <a, b| T(k, q) |c, d>
        """

        sp_states = list(self.left.single_particle_states +
                self.right.single_particle_states)
        bra = FermBra(*[ sp for sp in sp_states
            if isinstance(sp, DualQuantumState) ])
        ket = FermKet(*[ sp for sp in sp_states
            if isinstance(sp, RegularQuantumState) ])

        if kw_args.get('only_particle_states'):
            m = DirectMatrixElement(bra, self.operator, ket)
            c,t = m.as_coeff_terms(DirectMatrixElement)
            return c*t[0].as_only_particles()

        return DirectMatrixElement(bra, self.operator, ket)

    def as_other_coupling(self, other, **kw_args):
        """
        Expresses self in terms of other
        """
        assert isinstance(other, ThreeTensorMatrixElement)
        assert self.operator == other.operator
        self_as_direct = self.as_direct_product(only_particle_states=True, **kw_args)
        direct_as_other = other.get_direct_product_ito_self(**kw_args)

        others_direct = other.get_related_direct_matrix(only_particle_states=True)

        # if other_direct matrix comes with a sign, the substitution would fail
        c,t = others_direct.as_coeff_terms()
        if len(t) != 1: raise Error

        if not self_as_direct.has(t[0]):
            raise ValueError("The matrices are not compatible: %s, %s" %(t[0],self.get_related_direct_matrix()))

        result =  self_as_direct.subs(t[0],c*direct_as_other)
        return combine_ASigmas(result)


def apply_wigner_eckardt(expr, **kw_args):
    """
    Applies the Wigner-Eckardt theorem to all instances of ThreeTensorMatrixElement
    """
    ttme = expr.atoms(ThreeTensorMatrixElement)
    subslist= []
    for m in ttme:
        subslist.append((m,m.use_wigner_eckardt(**kw_args)))
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

class InjectionFailureException(Exception):
    def __init__(self, failures):
        self.failures = failures

protected_symbols = set((
    blank_symbol,
    ))
def inject_every_symbol_globally(expr, force=False, quiet=False, strict=False):
    """
    Extracts all symbols in expr, and injects them all into the global namespace.

    If a name is already in the global namespace, we don't overwrite it.  If there
    is a conflict, we raise an exception, but inject everything else first.
    """
    import inspect

    def _report():
        if quiet: return
        if force:
            print "injected", injections
            print "*** Replaced ***", replaced_items
        else:
            print  "injected", injections
            print  "Failed on", failures

    frame = inspect.currentframe().f_back
    try:
        s = expr.atoms(Symbol)
        s -= protected_symbols

        if not s:
            return None
        failures = {}
        injections = []
        for t in s:
            old = frame.f_globals.get(t.name)
            if old and old != t:
                failures[t] = old
            elif not old:
                frame.f_globals[t.name] = t
                injections.append(t)
        if failures and not force:
            _report()
            if strict:
                raise InjectionFailureException(failures)
        elif failures and force:
            replaced_items = []
            for k,v in failures.items():
                frame.f_globals[k.name] = k
                injections.append(k)
                replaced_items.append(v)
            _report()
            return replaced
        return injections
    finally:
        # we should explicitly break cyclic dependencies as stated in inspect doc
        del frame


