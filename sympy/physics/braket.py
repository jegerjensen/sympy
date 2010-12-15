from sympy import (
        S, Expr, Mul, global_assumptions, sympify, Symbol, Assume, ask, Q
        )
from sympy.physics.racahalgebra import (
        ThreeJSymbol, ClebschGordanCoefficient, SixJSymbol, SphericalTensor,
        refine_tjs2sjs, refine_phases, convert_cgc2tjs, convert_tjs2cgc,
        combine_ASigmas, remove_summation_indices, ASigma,
        invert_clebsch_gordans, AngularMomentumSymbol,
        extract_symbol2dummy_dict, convert_sumindex2dummy
        )
from sympy.physics.secondquant import (
        Dagger, AntiSymmetricTensor, _sort_anticommuting_fermions
        )

__all__ = [
        'SphericalTensorOperator',
        'DualSphericalTensorOperator',
        'SphFermKet',
        'SphFermBra',
        'ReducedMatrixElement',
        'DirectMatrixElement',
        'ThreeTensorMatrixElement',
        ]


braket_assumptions = global_assumptions
blank_symbol = Symbol("[blank symbol]")
default_redmat_definition = 'wikipedia'

class WignerEckardDoesNotApply(Exception): pass

class QuantumState(Expr): pass

class QuantumOperator(Expr):
    """
    Base class for all objects representing a quantum operator.
    """
    is_commutative=False
    def __new__(cls, *args, **kw_args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args, **kw_args)

class SphericalTensorOperator(QuantumOperator, SphericalTensor):
    """
    An Operator that transforms like a spherical tensor T(j,m)

    fulfills certain commutation relations with J_+- and J_z.
    """
    def __new__(cls, symbol, rank, projection, **kw_args):
        symbol, rank, projection = map(sympify, (symbol, rank, projection))
        obj = SphericalTensor.__new__(cls, symbol, rank, projection, **kw_args)
        obj._tensor_proj = projection
        obj._tensor_phase = S.One
        return obj

    def as_coeff_tensor(self, **kw_args):
        tensor = SphericalTensor(self.symbol, self.rank, self._tensor_proj)
        return self._tensor_phase, tensor

    @property
    def projection(self):
        raise TypeError("Use: SphericalTensorOperator.as_coeff_tensor()")

    def _dagger_(self):
        return DualSphericalTensorOperator(*self.args)

class DualSphericalTensorOperator(SphericalTensorOperator):
    """
    An Operator that transforms like a spherical tensor (-1)**(j-m)*T(j,-m)

    fulfills certain commutation relations with J_+- and J_z.


    FIXME: Should this object transform like -1**(j + m) or -1**(j - m) ?
    It depends on interpretation of this object.  Currently the implementation
    is that DualSphericalTensorOperator represent an operator that transforms
    like (-1)**(j-m)T(j,-m).  That is: the DualSphericalTensorOperator object
    is not actually a spherical tensor, but represents a conjugated
    SphericalTensorOperator.

    >>> from sympy.physics.braket import SphericalTensorOperator
    >>> from sympy.physics.braket import DualSphericalTensorOperator
    >>> SphericalTensorOperator('T','k','q').as_coeff_tensor()
    (1, T(k, q))
    >>> DualSphericalTensorOperator('T','k','q').as_coeff_tensor()
    ((-1)**(k - q), T(k, -q))
    """

    def __new__(cls, symbol, rank, projection, **kw_args):
        symbol, rank, projection = map(sympify, (symbol, rank, projection))
        obj = SphericalTensorOperator.__new__(cls, symbol, rank, projection, **kw_args)
        obj._tensor_proj = -projection
        obj._tensor_phase = S.NegativeOne**(rank - projection)
        return obj

    def _dagger_(self):
        return SphericalTensorOperator(*self.args)

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

    def _sympystr(self, p, *args):
        return str(self)

    def _latex_nobraket_(self, p, contained_in=None):
        if contained_in and not contained_in.is_dual is None:
            if self.is_dual != contained_in.is_dual:
                return p._print(self)
        str_args = []
        for s in self.args:
            if s is blank_symbol: continue
            try:
                str_args.append(s._latex_nobraket_(p, self))
            except AttributeError:
                str_args.append(p._print(s))
        return ", ".join([ s for s in str_args])


class Ket(BraKet):
    left_braket = '|'
    right_braket = '>'

    def _latex(self, p):
        return r"\left| %s \right\rangle" % self._latex_nobraket_(p)

class Bra(BraKet):
    left_braket = '<'
    right_braket = '|'

    def _latex(self, p):
        return r"\left\langle %s \right|" % self._latex_nobraket_(p)

class QuantumState(Expr):
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
        obj = Expr.__new__(cls, symbol, *args, **kw_args)
        return obj

    def _hashable_content(self):
        return Expr._hashable_content(self) + (self.is_hole,)

    def get_antiparticle(self):
        if len(self.args)>1: raise ValueError("Only single particle states can be anti-particles (FIXME?)")
        obj = type(self)(-self.symbol)
        obj._j = self._j
        obj._m = self._m
        obj.is_coupled = self.is_coupled
        return obj

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
        <c(a --> b)|

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
            if kw_args.get('reverse'):
                obj.is_coupled = -1
            else:
                obj.is_coupled = 1
            obj._j = Symbol("J_"+str(obj.label), nonnegative=True)
            obj._m = Symbol("M_"+str(obj.label))
        else:
            obj = QuantumState.__new__(cls, symbol, **kw_args)
            obj.is_coupled = 0
            obj._j = Symbol("j_"+str(obj.label), nonnegative=True)
            obj._m = Symbol("m_"+str(obj.label))

        cls._register_spin_assumptions(obj)

        return obj

    @classmethod
    def _register_spin_assumptions(cls, obj):
        if obj.is_coupled:
            if ask(obj.state1._j + obj.state2._j,Q.integer):
                spin_assume = Q.integer
            elif ask(obj.state1._j + obj.state2._j,'half_integer'):
                spin_assume = 'half_integer'
            else:
                raise ValueError("Couldn't determine spin assumptions of %s and %s"% (obj.state1._j, obj.state2._j))
        else:
            spin_assume = obj.spin_assume
        braket_assumptions.add(Assume(obj._j,spin_assume))
        braket_assumptions.add(Assume(obj._m,spin_assume))


    def _eval_subs(self, old, new):
        if self == old: return new
        obj = QuantumState._eval_subs(self, old, new)
        obj.is_coupled = self.is_coupled
        obj._j = self._j._eval_subs(old, new)
        obj._m = self._m._eval_subs(old, new)
        self._register_spin_assumptions(obj)
        return obj

    def _hashable_content(self):
        return QuantumState._hashable_content(self) + (self._j, self._m, self.is_coupled)

    def _dagger_(self):
        obj = QuantumState._dagger_(self)
        obj.is_coupled = self.is_coupled
        obj._j = self._j
        obj._m = self._m
        return obj

    def _str_nobraket_(self, contained_in=None):
        """
        Coupling and hole states must show in represetantion.
        """

        if contained_in and not contained_in.is_dual is None:
            if contained_in.is_dual != self.is_dual:
                return str(self)
        if self.is_coupled == 1:
            return "%s(%s --> %s)"%(self.symbol_str,
                    self.state1._str_nobraket_(self),
                    self.state2._str_nobraket_(self)
                    )
        if self.is_coupled == -1:
            return "%s(%s <-- %s)"%(self.symbol_str,
                    self.state1._str_nobraket_(self),
                    self.state2._str_nobraket_(self)
                    )
        else:
            return self.symbol_str

    def _latex_nobraket_(self, p, contained_in=None):
        """
        Coupling and hole states must show in represetantion.
        """

        if contained_in and not contained_in.is_dual is None:
            if contained_in.is_dual != self.is_dual:
                return p._print(self)
        if self.is_coupled == 1:
            return r"%s(%s \curvearrowright %s)"%(self.symbol_str,
                    self.state1._latex_nobraket_(p, self),
                    self.state2._latex_nobraket_(p, self)
                    )
        if self.is_coupled == -1:
            return r"%s(%s \curvearrowleft %s)"%(self.symbol_str,
                    self.state1._latex_nobraket_(p, self),
                    self.state2._latex_nobraket_(p, self)
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
        |ab(a --> <b|)>
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
        <ab(|a> --> b)|
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
        if self.is_coupled == 1:
            t = SphericalTensor('T', j, m, t1, t2)
        elif self.is_coupled == -1:
            t = SphericalTensor('T', j, m, t2, t1)

        cgc, t = t.as_coeff_direct_product(deep=True, **kw_args)
        cgc = phase*cgc*c1*c2

        # call deep to get coefficients for internal structure
        c1, s1 = self.state1._eval_as_coeff_sp_states(**kw_args)
        c2, s2 = self.state2._eval_as_coeff_sp_states(**kw_args)

        if kw_args.get('use_dummies', True):
            s2d = extract_symbol2dummy_dict(cgc)
            c1 = c1.subs(s2d)
            c2 = c2.subs(s2d)
            cgc = cgc.subs(s2d)
            s1 = [ s.subs(s2d) for s in s1]
            s2 = [ s.subs(s2d) for s in s2]

        return combine_ASigmas(c1*c2*cgc), tuple(s1 + s2)

    def as_direct_product(self, **kw_args):
        c,p = self._eval_as_coeff_sp_states(**kw_args)
        result = Mul(c,*p)
        if kw_args.get('tjs'):
            result = refine_phases(convert_cgc2tjs(result))
        return result

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
        (-1)**(j_b - _m_b)*Sum(_m_a, _m_b)*(j_a, _m_a, j_b, -_m_b|J_ab, M_ab)
        >>> states
        (|a>, <b|)

        Although it does not show in the representation, also the states
        contain dummy symbols if they are subject to summation:

        >>> [ s._m for s in states ]
        [_m_a, _m_b]

        If you know what you are doing and would prefer an expression without dummy
        symbols, you can supply the keyword use_dummies=False.

        >>> coeff, states = SphFermKet('ab',a,b).as_coeff_sp_states(use_dummies=False)
        >>> coeff
        (-1)**(j_b - m_b)*Sum(m_a, m_b)*(j_a, m_a, j_b, -m_b|J_ab, M_ab)
        >>> [ s._m for s in states ]
        [m_a, m_b]

        If |a> and <b| are coupled to form a dual state, this is reflected in the
        coupling coefficient:

        >>> bra_ab = SphFermBra('ab',a,b)
        >>> coeff, states = bra_ab.as_coeff_sp_states(use_dummies=False)
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
        Sum(_m_a, _m_b)*(j_a, _m_a, j_b, _m_b|J_ab, M_ab)
        >>> states
        (<a|, <b|)

        Nested couplings are also decomposed:

        >>> abc = SphFermKet('abc', SphFermKet('ab', a,b), 'c'); abc
        |abc(ab(a --> <b|) --> c)>
        >>> coeff_abc, states = abc.as_coeff_sp_states()
        >>> coeff_abc
         (-1)**(j_b - _m_b)*Sum(_M_ab, _m_a, _m_b, _m_c)*(J_ab, _M_ab, j_c, _m_c|J_abc, M_abc)*(j_a, _m_a, j_b, -_m_b|J_ab, _M_ab)
        >>> [ s._m for s in states ]
        [_m_a, _m_b, _m_c]


        """
        return self._eval_as_coeff_sp_states(**kw_args)

    @property
    def single_particle_states(self):
        """
        A tuple containing all s.p. states of this QuantumState

        The states are ordered as the state was created from left to right
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
        return obj

    @property
    def _tensor_phase(self):
        if self.is_hole:
            return (-1)**(self._j - self._m)
        else:
            return S.One

    @property
    def _tensor_proj(self):
        if self.is_hole:
            return Mul(S.NegativeOne, self._m)
        else:
            return self._m

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
        return obj

    @property
    def _tensor_phase(self):
        if self.is_hole:
            return (-1)**(2*self._j)
        else:
            return (-1)**(self._j - self._m)

    @property
    def _tensor_proj(self):
        if self.is_hole:
            return self._m
        else:
            return Mul(S.NegativeOne, self._m)


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
    >>> from sympy import symbols, latex
    >>> a,b,c = symbols('a b c')
    >>> SphFermKet(a)
    |a>
    >>> SphFermKet(a)._j
    j_a
    >>> SphFermKet(a)._m
    m_a
    >>> A, B = SphFermKet('a'), SphFermKet('b')
    >>> C = SphFermKet('c',A,B); C
    |c(a --> b)>
    >>> SphFermKet('z','x','y')
    |z(x --> y)>
    >>> SphFermKet('z','x','y').state1
    |x>
    >>> SphFermKet('z','x','y').state2
    |y>

    Single particle states return tensors with symbol 't', coupled states 'T'

    >>> C = SphFermKet('c',A,Dagger(B)); C
    |c(a --> <b|)>
    >>> latex(C)
    '\\\\left| c(a \\\\rightarrow \\\\left< b \\\\right|) \\\\right>'

    >>> C = SphFermKet('c',A,B, reverse=True); C
    |c(a <-- b)>
    >>> latex(C)
    '\\\\left| c(a \\\\leftarrow b) \\\\right>'

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
    <c(a --> b)|
    >>> isinstance(C, SphericalTensor)
    False
    >>> isinstance(C, SphericalQuantumState)
    True
    >>> SphFermBra('z','x','y')
    <z(x --> y)|

    Single particle states return tensors with symbol 't', coupled states 'T'

    >>> A.as_coeff_tensor()
    ((-1)**(j_a - m_a), t(j_a, -m_a))
    >>> C.as_coeff_tensor()
    ((-1)**(J_c - M_c), T(J_c, -M_c))

    >>> C.as_direct_product()
    Sum(_m_a, _m_b)*(j_a, _m_a, j_b, _m_b|J_c, M_c)*<a|*<b|
    >>> C.as_direct_product(strict_bra_coupling=True)
    (-1)**(j_a - _m_a)*(-1)**(j_b - _m_b)*(-1)**(J_c - M_c)*Sum(_m_a, _m_b)*(j_a, -_m_a, j_b, -_m_b|J_c, -M_c)*<a|*<b|

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
            obj.is_coupled = 0
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


class MatrixElement(Expr):
    """
    Base class for all matrix elements.

    Responsible for spawning the correct subclass depending on the input
    parameters.

    >>> from sympy.physics.braket import SphFermBra, SphFermKet, FermBra
    >>> from sympy.physics.braket import MatrixElement, SphericalTensorOperator
    >>> from sympy.physics.braket import ThreeTensorMatrixElement
    >>> from sympy.physics.braket import ReducedMatrixElement
    >>> from sympy.physics.braket import DirectMatrixElement
    >>> from sympy import symbols, latex
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

    >>> latex(MatrixElement(bra_a,Op,ket_b))
    '\\\\left\\\\langle a \\\\middle| Op(D, d) \\\\middle| b \\\\right\\\\rangle'

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
        if isinstance(operator, Mul):
            c,t = operator.as_coeff_terms(QuantumOperator)
            coeff *= c
            operator = t[0]

        if cls is MatrixElement:
            return coeff*_get_matrix_element(left, operator, right, **kw_args)
        else:
            assert isinstance(left, DualQuantumState), "not dual: %s" %left
            assert isinstance(operator, QuantumOperator), "not operator: %s" %operator
            assert isinstance(right, RegularQuantumState)
            obj = Expr.__new__(cls, left, operator, right, **kw_args)
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

    def _dagger_(self):
        l = Dagger(self.right)
        op = Dagger(self.operator)
        r = Dagger(self.left)
        return type(self)(l,op,r)

    def __str__(self):
        return "%s %s %s" %self.args[:4]

    def _sympystr(self, p, *args):
        return str(self)

    def _latex(self, p):
        left = self.left._latex_nobraket_(p)
        right = self.right._latex_nobraket_(p)
        op = p._print(self.operator)
        return r"\left\langle %s \middle| %s \middle| %s \right\rangle" %(
                left, op, right )

    def as_other_coupling(self, other, **kw_args):
        """
        Expresses the MatrixElement ``self'' in terms of ``other''

        This is done by rewriting self in terms of a direct product matrix,
        and then expressing the direct product matrix in terms of the other.

        >>> from sympy.physics.braket import (
        ...         MatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator, S
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> a = SphFermBra('a')
        >>> b = SphFermBra('b')
        >>> c = SphFermKet('c')
        >>> d = SphFermKet('d')

        A straightforward coupling of two-particle states gives the following
        ThreeTensorMatrixElement:

        >>> bra_ab = SphFermBra('ab',a,b)
        >>> ket_cd = SphFermKet('cd',c,d)
        >>> M1 = MatrixElement(bra_ab, T, ket_cd); M1
        <ab(a --> b)| T(k, q) |cd(c --> d)>
        >>> M1.as_direct_product()
        Sum(_m_a, _m_b, _m_c, _m_d)*(j_a, _m_a, j_b, _m_b|J_ab, M_ab)*(j_c, _m_c, j_d, _m_d|J_cd, M_cd)*<a, b| T(k, q) |c, d>

        But we can also create ``cross-coupled'' matrix elements.
        (See Kuo & al. 1981, eq.(20))

        >>> bra_ca = SphFermBra('ca',c,a)
        >>> ket_db = SphFermKet('db',d,b)
        >>> M2 = MatrixElement(bra_ca, T, ket_db); M2
        <ca(|c> --> a)| T(k, q) |db(d --> <b|)>
        >>> expr = M2.as_direct_product(); expr
        (-1)**(j_a - _m_a)*(-1)**(J_ca - M_ca)*(-1)**(j_b - _m_b)*Sum(_m_a, _m_b, _m_c, _m_d)*(j_c, _m_c, j_a, -_m_a|J_ca, -M_ca)*(j_d, _m_d, j_b, -_m_b|J_db, M_db)*<a, b| T(k, q) |c, d>

        Note that the cross coupled ThreeTensorMatrixElement M2 associates
        itself with the same DirectMatrixElement as M1 above.  Now, in order to
        get M2 in terms of M1, we need to put together the above expression with
        the direct product matrix expressed in terms of M1:

        >>> M1.get_direct_product_ito_self()
        Sum(_J_ab, _J_cd, _M_ab, _M_cd)*(j_a, m_a, j_b, m_b|_J_ab, _M_ab)*(j_c, m_c, j_d, m_d|_J_cd, _M_cd)*<ab(a --> b)| T(k, q) |cd(c --> d)>

        Inserted we get the long expression:  (where strict bra coupling have been used.)

        >>> expr = M2.as_other_coupling(M1); expr
        (-1)**(J_ca - M_ca - _J_ab + _M_ab)*Sum(_J_ab, _J_cd, _M_ab, _M_cd, _m_a, _m_b, _m_c, _m_d)*(j_a, -_m_a, j_b, -_m_b|_J_ab, -_M_ab)*(j_c, _m_c, j_a, -_m_a|J_ca, -M_ca)*(j_c, _m_c, j_d, _m_d|_J_cd, _M_cd)*(j_d, _m_d, j_b, -_m_b|J_db, M_db)*<ab(a --> b)| T(k, q) |cd(c --> d)>

        If T happens to be a rank 0 tensor (k=q=0), we can write the relation
        between the reduced matrix elements in terms of a 6j-symbol.  In order
        to compare with the result of Kuo & al. we use Edmonds definition of
        reduced matrix elements.

        >>> from sympy.physics.racahalgebra import (
        ...         refine_tjs2sjs, convert_cgc2tjs,
        ...         refine_phases, evaluate_sums,
        ...         SixJSymbol )
        >>> M1 = M1.subs({k: S(0), q: S(0)})
        >>> M2 = M2.subs({k: S(0), q: S(0)})

        >>> expr = M2.as_other_coupling(M1, wigner_eckardt=True, definition='edmonds'); expr
        (-1)**(J_db + M_ca + _J_cd + _M_ab)*(1 + 2*J_ca)**(1/2)*Sum(_J_ab, _J_cd, _M_ab, _M_cd, _M_db, _m_a, _m_b, _m_c, _m_d)*(j_a, -_m_a, j_b, -_m_b|_J_ab, -_M_ab)*(j_c, _m_c, j_a, -_m_a|J_ca, -M_ca)*(j_c, _m_c, j_d, _m_d|_J_cd, _M_cd)*(j_d, _m_d, j_b, -_m_b|J_db, _M_db)*Dij(J_ca, J_db)*Dij(M_ca, _M_db)*Dij(_J_ab, _J_cd)*Dij(_M_ab, _M_cd)*<ab(a --> b)|| T(0) ||cd(c --> d)>/(1 + 2*_J_ab)**(1/2)

        >>> expr = convert_cgc2tjs(expr)
        >>> expr = refine_phases(expr)
        >>> expr = evaluate_sums(expr, all_deltas=1); expr
        (-1)**(1 + M_ca + j_a + j_d + 2*J_ca + _M_ab)*(1 + 2*J_ca)**(3/2)*(1 + 2*_J_ab)**(1/2)*Sum(_J_ab, _M_ab, _m_a, _m_b, _m_c, _m_d)*<ab(a --> b)|| T(0) ||cd(c --> d)>*ThreeJSymbol(J_ca, j_a, j_c, M_ca, -_m_a, _m_c)*ThreeJSymbol(J_ca, j_b, j_d, M_ca, _m_b, -_m_d)*ThreeJSymbol(_J_ab, j_a, j_b, _M_ab, -_m_a, -_m_b)*ThreeJSymbol(_J_ab, j_c, j_d, _M_ab, -_m_c, -_m_d)

        >>> expr = refine_tjs2sjs(expr); expr
        (-1)**(1 + J_ca + j_a + j_d - _J_ab)*(1 + 2*J_ca)**(1/2)*(1 + 2*_J_ab)**(1/2)*Sum(_J_ab)*<ab(a --> b)|| T(0) ||cd(c --> d)>*SixJSymbol(J_ca, j_a, j_c, _J_ab, j_d, j_b)

        Kuo & al. arrive at another 6j symbol, but we can easily check the equivalence:

        >>> (j_c, j_a, J_ca, j_d, j_b, J_ab) = map(S, "j_c j_a J_ca j_d j_b J_ab".split())
        >>> SixJSymbol(j_c, j_a, J_ca, j_b, j_d, J_ab.as_dummy())
        SixJSymbol(J_ca, j_a, j_c, _J_ab, j_d, j_b)

        """
        if not isinstance(other, MatrixElement):
            raise ValueError("argument ``other'' must be a MatrixElement object")
        if not self.operator == other.operator:
            raise ValueError("MatrixElements not compatible")

        self_as_direct = self.as_direct_product(only_particle_states=True, strict_bra_coupling=0, **kw_args)
        my_direct = self_as_direct.atoms(DirectMatrixElement).pop()
        subsdict = extract_symbol2dummy_dict(self_as_direct)

        other = other.subs(subsdict)
        direct_as_other = other.get_direct_product_ito_self(only_particle_states=True, strict_bra_coupling=0, **kw_args)
        others_direct = other.get_related_direct_matrix(only_particle_states=True)

        # if other_direct matrix comes with a sign, the substitution would fail
        c,t = others_direct.as_coeff_terms(MatrixElement)
        if len(t) != 1: raise Error

        if not self_as_direct.has(t[0]):
            raise ValueError("The matrices are not compatible: %s, %s" %(t[0],self.get_related_direct_matrix()))

        assert  t[0] == my_direct

        result =  self_as_direct.subs(t[0],c*direct_as_other)
        result = refine_phases(result)
        return combine_ASigmas(result)

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
    >>> a = SphFermBra('a')
    >>> b = SphFermKet('b')
    >>> T = SphericalTensorOperator('T',k,q)
    >>> ReducedMatrixElement(a, T, b)
    <a|| T(k) ||b>

    The geometrical factor is available through the method
    .get_reduction_factor():

    >>> ReducedMatrixElement(a, T, b).get_reduction_factor()
    (j_b, m_b, k, q|j_a, m_a)

    You can also formulate the direct product (corresponding to a full
    decomposition of all states) in terms of (ito) the ReducedMatrixElement:
    Note that those expressions are not equal to the ReducedMatrixElement, but
    rather to the corresponding uncoupled matrix element.

    >>> ReducedMatrixElement(a, T, b).get_direct_product_ito_self()
    (j_b, m_b, k, q|j_a, m_a)*<a|| T(k) ||b>
    >>> ReducedMatrixElement(a, T, SphFermKet('bc',b,'c')).get_direct_product_ito_self()
    Sum(_J_bc, _M_bc)*(j_b, m_b, j_c, m_c|_J_bc, _M_bc)*(_J_bc, _M_bc, k, q|j_a, m_a)*<a|| T(k) ||bc(b --> c)>

    The ReducedMatrixElement can also express itself in terms of a direct
    product.

    >>> ReducedMatrixElement(a, T, b).as_direct_product()
    Sum(_m_b, _q)*(j_b, _m_b, k, _q|j_a, m_a)*<a| T(k, _q) |b>
    >>> ReducedMatrixElement(a, T, SphFermKet('bc',b,'c')).as_direct_product()
    Sum(_M_bc, _m_b, _m_c, _q)*(J_bc, _M_bc, k, _q|j_a, m_a)*(j_b, _m_b, j_c, _m_c|J_bc, _M_bc)*<a| T(k, _q) |b, c>

    Note that in the above expression, the summation over _M_bc and _q
    originates from the inverted clebsch gordan coefficient, and the sum over
    _m_b and _m_c from the coupling in the state |bc(b, c)>.  The angular
    momenta of |bc(b, c)> are not dummy symbols, hence not subject to the
    summation.


    Alternative definitions
    =======================

    The default definition is specified with the global variable
    default_redmat_definition:

    >>> default_redmat_definition
    'wikipedia'
    >>> type(ReducedMatrixElement(a, T, b)).__name__
    'ReducedMatrixElement_wikipedia'

    Other definitions currently available are 'edmonds' and 'brink_satchler':

    >>> ReducedMatrixElement(a, T, b, 'edmonds').get_reduction_factor()
    (-1)**(j_a - m_a)*ThreeJSymbol(j_a, j_b, k, m_a, -m_b, -q)
    >>> ReducedMatrixElement(a, T, b, 'brink_satchler').get_reduction_factor()
    (-1)**(2*k)*(j_b, m_b, k, q|j_a, m_a)

    If you want to introduce yet another definition, all you have to do is
    to subclass ReducedMatrixElement and overload the method
    ReducedMatrixElement.get_reduction_factor(self).  The convention is to
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

            return eval( "ReducedMatrixElement_%s(left, op, right, **kw_args)" % definition)

        else:
            obj = MatrixElement.__new__(cls, left,op,right, **kw_args)
            return obj

    def get_reduction_factor(self, **kw_args):
        """
        Returns the ClebschGordanCoefficient that relates this reduced
        matrix element to the corresponding direct matrix element.

        >>> from sympy.physics.braket import (
        ...         ReducedMatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator, DualSphericalTensorOperator
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> ReducedMatrixElement(bra, T, ket).get_reduction_factor()
        (j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, definition='brink_satchler').get_reduction_factor()
        (-1)**(2*k)*(j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, 'edmonds').get_reduction_factor()
        (-1)**(j_a - m_a)*ThreeJSymbol(j_a, j_b, k, m_a, -m_b, -q)

        If the spherical tensor operator has tensor properties related to dual
        states, this will be treated correctly in the coupling coefficient:

        >>> L = DualSphericalTensorOperator('L',k,q)
        >>> ReducedMatrixElement(bra, L, ket).get_reduction_factor()
        (-1)**(k - q)*(j_b, m_b, k, -q|j_a, m_a)
        """
        factor = self._eval_reduction_factor()

        if kw_args.get('tjs'):
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
        Sum(_m_b, _q)*(j_b, _m_b, k, _q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, definition='brink_satchler')._get_inverse_reduction_factor(use_dummies=False)
        (-1)**(-2*k)*Sum(m_b, q)*(j_b, m_b, k, q|j_a, m_a)
        >>> ReducedMatrixElement(bra, T, ket, 'edmonds')._get_inverse_reduction_factor()
        (-1)**(m_a - j_a)*(1 + 2*j_a)*Sum(_m_b, _q)*ThreeJSymbol(j_a, j_b, k, m_a, -_m_b, -_q)
        """
        left,op,right = self.args
        c_ket, t_ket = right.as_coeff_tensor()
        j1, m1 = t_ket.get_rank_proj()
        c_op, t_op = op.as_coeff_tensor()
        j2, m2 = t_op.get_rank_proj()
        J, M = left._j, left._m
        if self.definition == 'wikipedia':
            factor = ASigma(m1, m2)*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)/c_ket/c_op
        elif self.definition == 'edmonds':
            factor = (-1)**(M-J)*ASigma(m1, m2)*(2*J+1)*ThreeJSymbol(J, j2, j1, -M, m2, m1)/c_ket/c_op
        elif self.definition == 'brink_satchler':
            factor = (-1)**(-2*j2)*ASigma(m1,m2)*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)/c_ket/c_op

        if kw_args.get('use_dummies', True):
            factor = convert_sumindex2dummy(factor)
        if kw_args.get('tjs'):
            factor = refine_phases(convert_cgc2tjs(factor))
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

    def get_related_direct_matrix(self, **kw_args):
        ttm =  self._get_ThreeTensorMatrixElement()
        return ttm.get_related_direct_matrix(**kw_args)

    def __str__(self):
        return "%s| %s |%s" %(
                self.left,
                "%s(%s)"%self.operator._str_drop_projection_(),
                self.right
                )

    def _latex(self, p):
        left = self.left._latex_nobraket_(p)
        right = self.right._latex_nobraket_(p)
        op = self.operator._latex_drop_projection(p)
        return r"\left\langle %s \middle\| %s \middle\| %s \right\rangle" %(
                left, op, right )

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
        >>> ReducedMatrixElement(bra, T, SphFermKet('bc','b','c')).get_direct_product_ito_self()
        Sum(_J_bc, _M_bc)*(j_b, m_b, j_c, m_c|_J_bc, _M_bc)*(_J_bc, _M_bc, k, q|j_a, m_a)*<a|| T(k) ||bc(b --> c)>

        """
        matel = self._get_ThreeTensorMatrixElement()
        kw = kw_args.copy()
        kw['wigner_eckardt']=False
        dirprod = matel.get_direct_product_ito_self(**kw)
        subsdict = extract_symbol2dummy_dict(dirprod)

        matel = matel.subs(subsdict)
        redmat = self.subs(subsdict)
        cgc = redmat.get_reduction_factor(**kw_args)

        result = cgc * dirprod.subs(matel, redmat)

        if kw_args.get('tjs'):
            result = refine_phases(convert_cgc2tjs(result))
        return result

    def as_direct_product(self, **kw_args):
        """
        Returns the reduced matrix element in terms of direct product
        matrices.

        >>> from sympy.physics.braket import (
        ...         ReducedMatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator, MatrixElement
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')

        >>> bra = SphFermBra('a')
        >>> ket = SphFermKet('b')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> m = ReducedMatrixElement(bra, T, ket).as_direct_product(); m
        Sum(_m_b, _q)*(j_b, _m_b, k, _q|j_a, m_a)*<a| T(k, _q) |b>
        >>> m = m.atoms(MatrixElement).pop()
        >>> m.right._m
        _m_b
        >>> m = ReducedMatrixElement(bra, T, SphFermKet('bc', 'b', 'c')).as_direct_product(); m
        Sum(_M_bc, _m_b, _m_c, _q)*(J_bc, _M_bc, k, _q|j_a, m_a)*(j_b, _m_b, j_c, _m_c|J_bc, _M_bc)*<a| T(k, _q) |b, c>
        """
        invcgc = self._get_inverse_reduction_factor(**kw_args)
        matel = self._get_ThreeTensorMatrixElement()

        subsdict = extract_symbol2dummy_dict(invcgc)
        matel = matel.subs(subsdict)
        new_kw = kw_args.copy()
        new_kw['wigner_eckardt'] = False
        matel = matel.as_direct_product(**new_kw)

        result = combine_ASigmas(matel*invcgc)
        if kw_args.get('tjs'):
            result = refine_phases(convert_cgc2tjs(result))
        return result


class ReducedMatrixElement_edmonds(ReducedMatrixElement):
    _definition = 'edmonds'
    def _eval_reduction_factor(self):
        left,op,right = self.args
        c_ket, t_ket = right.as_coeff_tensor()
        j1, m1 = t_ket.get_rank_proj()
        c_op, t_op = op.as_coeff_tensor()
        j2, m2 = t_op.get_rank_proj()
        J, M = left._j, left._m
        return c_op*c_ket*(-1)**(J-M)*ThreeJSymbol(J, j2, j1, -M, m2, m1)

class ReducedMatrixElement_brink_satchler(ReducedMatrixElement):
    _definition = 'brink_satchler'
    def _eval_reduction_factor(self):
        left,op,right = self.args
        c_ket, t_ket = right.as_coeff_tensor()
        j1, m1 = t_ket.get_rank_proj()
        c_op, t_op = op.as_coeff_tensor()
        j2, m2 = t_op.get_rank_proj()
        J, M = left._j, left._m
        return (-1)**(2*j2)*c_op*c_ket*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)

class ReducedMatrixElement_wikipedia(ReducedMatrixElement):
    _definition = 'wikipedia'
    def _eval_reduction_factor(self):
        left,op,right = self.args
        c_ket, t_ket = right.as_coeff_tensor()
        j1, m1 = t_ket.get_rank_proj()
        c_op, t_op = op.as_coeff_tensor()
        j2, m2 = t_op.get_rank_proj()
        J, M = left._j, left._m
        return c_op*c_ket*ClebschGordanCoefficient(j1, m1, j2, m2, J, M)

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

    def get_direct_product_ito_self(self, **kw_args):
        return self

    def as_direct_product(self, **kw_args):
        return self.get_related_direct_matrix(**kw_args)

    def get_related_direct_matrix(self, **kw_args):
        if kw_args.get('only_particle_states'):
            return self.as_only_particles()
        return self

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
        redmat = self.get_related_redmat(definition=definition)
        return redmat.get_reduction_factor(**kw_args)*redmat

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
        Sum(_J_ab, _M_ab)*(j_a, m_a, j_b, m_b|_J_ab, _M_ab)*<ab(a --> b)| T(k, q) |c>
        >>> MatrixElement(bra_ab, T, ket).get_direct_product_ito_self(strict_bra_coupling=True)
        (-1)**(m_a - j_a)*(-1)**(m_b - j_b)*(-1)**(-_J_ab + _M_ab)*Sum(_J_ab, _M_ab)*(j_a, -m_a, j_b, -m_b|_J_ab, -_M_ab)*<ab(a --> b)| T(k, q) |c>

        The matrix element also contains dummy symbols now,
        >>> m = MatrixElement(bra_ab, T, ket)
        >>> m_dummy = m.get_direct_product_ito_self()
        >>> m_dummy = m_dummy.atoms(MatrixElement).pop()
        >>> m_dummy == m
        False
        >>> m_dummy.left._j
        _J_ab
        >>> m_dummy.left._m
        _M_ab

        Now a cross coupled matrix element:

        >>> c = SphFermKet('c')
        >>> ket_bc = SphFermKet('bc',b,c)
        >>> MatrixElement(a, T, ket_bc).get_direct_product_ito_self()
        (-1)**(m_b - j_b)*Sum(_J_bc, _M_bc)*(j_b, -m_b, j_c, m_c|_J_bc, _M_bc)*<a| T(k, q) |bc(<b| --> c)>

        ...and in combination with the Wigner-Eckard theorem:

        >>> MatrixElement(a, T, ket_bc).get_direct_product_ito_self(wigner_eckardt=True)
        (-1)**(m_b - j_b)*Sum(_J_bc, _M_bc)*(j_b, -m_b, j_c, m_c|_J_bc, _M_bc)*(_J_bc, _M_bc, k, q|j_a, m_a)*<a|| T(k) ||bc(<b| --> c)>
        """
        if kw_args.get('wigner_eckardt'):
            redmat = self.get_related_redmat(**kw_args)
            return redmat.get_direct_product_ito_self(**kw_args)

        kw = kw_args.copy()
        kw['use_dummies'] = False
        kw['tjs'] = False
        coeffs_inv = self.as_direct_product(only_coeffs=True, **kw)
        coeffs = invert_clebsch_gordans(coeffs_inv)
        matrix = self

        if kw_args.get('use_dummies', True):
            coeffs = convert_sumindex2dummy(coeffs)
            subsdict = extract_symbol2dummy_dict(coeffs)
            matrix = matrix.subs(subsdict)
        if kw_args.get('tjs'):
            coeffs = refine_phases(convert_cgc2tjs(coeffs))

        return coeffs*matrix


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
        Sum(_m_a, _m_b)*(j_a, _m_a, j_b, _m_b|J_ab, M_ab)*<a, b| T(k, q) |c>

        >>> MatrixElement(bra_ab, T, ket).as_direct_product(strict_bra_coupling=1)
        (-1)**(j_a - _m_a)*(-1)**(J_ab - M_ab)*(-1)**(j_b - _m_b)*Sum(_m_a, _m_b)*(j_a, -_m_a, j_b, -_m_b|J_ab, -M_ab)*<a, b| T(k, q) |c>

        The matrix element also contain dummy symbols:

        >>> m = MatrixElement(bra_ab, T, ket).as_direct_product()
        >>> m = m.atoms(MatrixElement).pop(); m
        <a, b| T(k, q) |c>
        >>> [ s._m for s in m.left.single_particle_states ]
        [_m_a, _m_b]

        The keyword wigner_eckardt=True gives you an expression for the reduced
        matrix element in terms of the direct product matrix.
        For the above expression we obtain:

        >>> MatrixElement(bra_ab, T, ket).as_direct_product(wigner_eckardt=True, use_dummies=False)
        Sum(m_a, m_b, m_c, q)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*(j_c, m_c, k, q|J_ab, M_ab)*<a, b| T(k, q) |c>
        >>> MatrixElement(a, T, SphFermKet('cd','c','d')).as_direct_product(wigner_eckardt=True)
        Sum(_M_cd, _m_c, _m_d, _q)*(J_cd, _M_cd, k, _q|j_a, m_a)*(j_c, _m_c, j_d, _m_d|J_cd, _M_cd)*<a| T(k, _q) |c, d>

        To express everything with a different ReducedMatrixElement definition
        supply the definition with a keyword:

        >>> MatrixElement(bra_ab, T, ket).as_direct_product(wigner_eckardt=True, definition='brink_satchler', use_dummies=False)
        (-1)**(-2*k)*Sum(m_a, m_b, m_c, q)*(j_a, m_a, j_b, m_b|J_ab, M_ab)*(j_c, m_c, k, q|J_ab, M_ab)*<a, b| T(k, q) |c>
        """

        if kw_args.get('wigner_eckardt'):
            return self.get_related_redmat(**kw_args).as_direct_product(**kw_args)

        matrix = self.get_related_direct_matrix(**kw_args)
        cbra, bra = self.left.as_coeff_sp_states(**kw_args)
        cket, ket = self.right.as_coeff_sp_states(**kw_args)

        if kw_args.get('use_dummies', True):
            subsdict = extract_symbol2dummy_dict(cbra*cket)
            matrix = matrix.subs(subsdict)

        if kw_args.get('only_coeffs'):
            result = combine_ASigmas(cbra*cket)
        else:
            result =  combine_ASigmas(cbra*cket)*matrix

        if kw_args.get('tjs'):
            result = refine_phases(convert_cgc2tjs(result))

        return result


    def get_related_direct_matrix(self, **kw_args):
        """
        Returns the direct product matrix that corresponds to this matrix.

        >>> from sympy.physics.braket import (
        ...         MatrixElement, SphFermKet, SphFermBra,
        ...         SphericalTensorOperator, FermKet
        ...         )
        >>> from sympy import symbols
        >>> k,q = symbols('k q')
        >>> T = SphericalTensorOperator('T',k,q)
        >>> a = SphFermBra('a')
        >>> b = SphFermBra('b')
        >>> c = SphFermKet('c')
        >>> d = SphFermKet('d')

        >>> bra = SphFermBra('ab',a,b)
        >>> ket = SphFermKet('cd',c,d)
        >>> m=MatrixElement(bra, T, ket); m
        <ab(a --> b)| T(k, q) |cd(c --> d)>
        >>> m.get_related_direct_matrix()
        <a, b| T(k, q) |c, d>

        >>> bra = SphFermBra('ab',a,-b)
        >>> ket = SphFermKet('cd',c,-d)
        >>> m=MatrixElement(bra, T, ket); m
        <ab(a --> -b)| T(k, q) |cd(c --> -d)>
        >>> m.get_related_direct_matrix()
        <a, -b| T(k, q) |c, -d>
        >>> m.get_related_direct_matrix(only_particle_states=True)
        <a, d| T(k, q) |b, c>

        For cross coupled matrix elements the direct product matrix is
        determined by placing all s.p. kets in a direct broduct ket, and all
        s.p. bras in a direct product bra.

        >>> m = MatrixElement(SphFermBra('ac',a, c), T, SphFermKet('bd', b, d)); m
        <ac(a --> |c>)| T(k, q) |bd(<b| --> d)>
        >>> m.get_related_direct_matrix()
        <a, b| T(k, q) |c, d>

        The direct product states are ordered strictly as the s.p. states
        appear in the coupled matrix element, so if we swap c and d, the related
        direct matrix element is

        >>> m = MatrixElement(SphFermBra('ad',a, d), T, SphFermKet('bc', b, c)); m
        <ad(a --> |d>)| T(k, q) |bc(<b| --> c)>
        >>> m.get_related_direct_matrix()
        -<a, b| T(k, q) |c, d>

        The sign comes form the canonical form of the direct state |d, c>
        >>> FermKet(d, c)
        -|c, d>

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


    def get_related_redmat(self, **kw_args):
        return ReducedMatrixElement(self.left, self.operator, self.right, **kw_args)


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

def rewrite_coupling(expr, other, **kw_args):
    """
    Tries to rewrite every MatrixElement in terms of the MatrixElement ``other''.
    """
    if isinstance(other, (list, dict)):
        for o in other:
            expr = rewrite_coupling(expr, o, **kw_args)
        return expr

    junk, t = other.as_coeff_terms(MatrixElement)
    assert len(t) == 1;
    other = t[0]
    matels = expr.atoms(MatrixElement)
    subsdict = {}
    for m in matels:
        try:
            subsdict[m] = m.as_other_coupling(other)
        except ValueError:
            if kw_args.get('verbose'):
                print "ValueError:", m, other
            if kw_args.get('strict'):
                raise
    if subsdict:
        return expr.subs(subsdict)
    else:
        return expr

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
            return replaced_items
        return injections
    finally:
        # we should explicitly break cyclic dependencies as stated in inspect doc
        del frame


