from sympy import symbols, C, S, global_assumptions, Q

from sympy.physics.braket import (
        SphericalTensorOperator, Dagger, BosonState, FermionState, Ket, Bra,
        QuantumState, SphFermKet, SphFermBra, FermKet, FermBra, ASigma,
        DirectQuantumState, SphericalTensor, ClebschGordanCoefficient,
        MatrixElement, SphericalTensorOperator, ThreeTensorMatrixElement,
        ReducedMatrixElement, DirectMatrixElement, SphericalQuantumState,
        BraKet, WignerEckardDoesNotApply, QuantumVacuumKet, QuantumVacuumBra,
        ThreeJSymbol, DualSphericalTensorOperator
        )
from sympy.physics.racahalgebra import is_equivalent

from sympy.utilities import raises
from sympy.utilities.pytest import XFAIL


def test_SphericalTensorOperator():
    k,q,T = symbols('k q T')
    assert Dagger(SphericalTensor(T,k,q))==(-1)**(k+q)*SphericalTensor(T, k, -q)
    assert Dagger(SphericalTensorOperator(T,k,q))==DualSphericalTensorOperator(T, k, q)

def test_Fermion_Boson():
    assert BosonState('a').spin_assume == Q.integer
    assert FermionState('a').spin_assume == Q.half_integer

def test_braket():
    assert str(Ket('a')) == "|a>"
    assert str(Bra('a')) == "<a|"

def test_QuantumState():
    assert QuantumState('a',hole=True).is_hole == True
    assert QuantumState('a').is_hole == False
    assert QuantumState('a').get_antiparticle().is_hole == True

    assert QuantumState('-a',hole=True).is_hole == False
    assert QuantumState('-a').is_hole == True
    assert QuantumState('-a').get_antiparticle().is_hole == False

    a = SphFermKet('a'); assert a.func(*a.args) == a
    a = SphFermKet('-a'); assert a.func(*a.args) == a
    a = SphFermKet('a',hole=True); assert a.func(*a.args) == a
    a = SphFermKet('a',hole=False); assert a.func(*a.args) == a
    a = SphFermKet(-a); assert a.func(*a.args) == a
    a = FermKet('a'); assert a.func(*a.args) == a
    a = FermKet('-a'); assert a.func(*a.args) == a
    a = FermKet('a',hole=True); assert a.func(*a.args) == a
    a = FermKet('a',hole=False); assert a.func(*a.args) == a
    a = FermKet(-a); assert a.func(*a.args) == a

def test_Dagger():
    assert Dagger(FermKet('a')) == FermBra('a')
    assert Dagger(FermBra('a')) == FermKet('a')
    assert Dagger(FermBra(FermBra('a'),FermBra('b'))) == FermKet(FermKet('a'),FermKet('b'))
    assert Dagger(SphFermKet('a')) == SphFermBra('a')
    assert Dagger(SphFermBra('a')) == SphFermKet('a')
    assert Dagger(SphFermBra('c',SphFermBra('a'),SphFermBra('b'))) == SphFermKet('c',SphFermKet('a'),SphFermKet('b'))

def test_antiparticle_Dagger_commutation():
    assert SphFermKet('a',hole=True) != SphFermKet('a',hole=False)
    assert Dagger(SphFermKet('a',hole=True)) == SphFermBra('a',hole=True)
    assert Dagger(SphFermKet('a').get_antiparticle()) == SphFermBra('a').get_antiparticle()
    assert Dagger(SphFermBra('a').get_antiparticle()) == SphFermKet('a').get_antiparticle()

    assert FermKet('a',hole=True) != FermKet('a',hole=False)
    assert Dagger(FermKet('a',hole=True)) == FermBra('a',hole=True)
    assert Dagger(FermKet('a').get_antiparticle()) == FermBra('a').get_antiparticle()
    assert Dagger(FermBra('a').get_antiparticle()) == FermKet('a').get_antiparticle()

def test_DirectQuantumState():
    a,b,c,d = map(QuantumState, 'abcd')
    class Fermions(DirectQuantumState, FermionState):
        pass
    class Bosons(DirectQuantumState, BosonState):
        pass
    assert Fermions(b,a,c,d) == -Fermions(a, b, c, d)
    assert Bosons(b,a,c,d) == Bosons(a, b, c, d)
    assert DirectQuantumState(SphFermKet('a')) == SphFermKet('a')
    assert DirectQuantumState(SphFermBra('a')) == SphFermBra('a')

def test_FermKet_FermBra():
    assert FermBra('a') == SphFermBra('a')
    assert FermKet('c') == SphFermKet('c')
    assert FermKet(-SphFermKet('a')) == SphFermKet('a', hole=True)
    raises(ValueError, 'FermKet(FermBra("a"))')
    raises(ValueError, 'FermBra(FermKet("c"))')
    assert FermKet('a',hole=True).is_hole


    a = SphFermKet('a')
    b = SphFermKet('b')
    c = SphFermKet('c')
    d = SphFermKet('d')
    bh = SphFermKet('b', hole=True)
    ch = SphFermKet('c', hole=True)
    assert FermKet(a,-b,-c,d) == FermKet(a, bh, ch, d)
    assert FermKet(-b) == FermKet(bh)

def test_QuantumVacuum():

    assert isinstance(FermKet(),QuantumVacuumKet)
    assert isinstance(FermBra(),QuantumVacuumBra)
    assert isinstance(SphFermKet(),QuantumVacuumKet)
    assert isinstance(SphFermBra(),QuantumVacuumBra)
    assert FermKet().single_particle_states ==tuple([])
    assert FermKet()._j is S.Zero
    assert FermKet()._m is S.Zero
    assert FermKet()._tensor_proj is S.Zero
    assert FermKet()._tensor_phase is S.One

def test_SphericalQuantumState():

    j_a, m_a = symbols('j_a m_a')
    J_c, M_c = symbols('J_c M_c')

    assert SphFermKet('a')._j == j_a
    assert SphFermKet('a')._m == m_a
    assert SphFermKet('a').is_hole == False
    assert SphFermKet('a', hole=True).is_hole == True

    assert SphFermKet('c', SphFermKet('a'), SphFermKet('b'))._j == J_c
    assert SphFermKet('c', SphFermKet('a'), SphFermKet('b'))._m == M_c
    raises(ValueError,"SphFermKet('c', SphFermKet('a'), SphFermKet('b'), hole=True)")
    assert SphFermKet('a') < SphFermKet('b')

    assert SphFermKet(-SphFermKet('a')) == SphFermKet('a', hole=True)

    assert SphFermKet('a').subs(j_a, J_c)._j == J_c

@XFAIL
def test_SphFermKet_func():
    # FIXME: this should work.
    a_mod = SphFermKet('a').subs(j_a, J_c)
    assert a_mod.func(*a_mod.args) == a_mod

def test_as_coeff_tensor():
    global_assumptions.clear()
    t, T, j_a, m_a = symbols('t T j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    J_c, M_c = symbols('J_c M_c')

    # sp states
    assert SphFermKet('a').as_coeff_tensor() == (1, SphericalTensor(t,j_a,m_a))
    assert SphFermBra('a').as_coeff_tensor() == ((-1)**(j_a-m_a), SphericalTensor(t,j_a,-m_a))

    # hole states
    assert SphFermBra('a', hole=True).as_coeff_tensor() == ((-1)**(2*j_a), SphericalTensor(t,j_a,m_a))
    assert SphFermKet('a', hole=True).as_coeff_tensor() == ((-1)**(j_a-m_a), SphericalTensor(t,j_a,-m_a))

    # coupled states
    assert SphFermKet('c', SphFermKet('a'), SphFermKet('b')).as_coeff_tensor() == (1, SphericalTensor(T, J_c, M_c))
    assert SphFermBra('c', SphFermBra('a'), SphFermBra('b')).as_coeff_tensor() == ((-1)**(J_c-M_c), SphericalTensor(T, J_c, -M_c))

def test_as_coeff_sp_states():
    global_assumptions.clear()
    t, T = symbols('t T')
    j_a, j_b, J_c = symbols('j_a j_b J_c', nonnegative=True)
    m_a, m_b, M_c = symbols('m_a m_b M_c')

    # sp states
    assert SphFermKet('a').as_coeff_sp_states() == (1, (SphFermKet('a'),))
    assert SphFermBra('a').as_coeff_sp_states() == (1, (SphFermBra('a'),))

    # hole states
    assert SphFermKet('a', hole=True).as_coeff_sp_states() == (1, (SphFermKet('a', hole=True),))
    assert SphFermBra('a', hole=True).as_coeff_sp_states() == (1, (SphFermBra('a', hole=True),))

    # coupled states
    a = SphFermKet('a')
    b = SphFermKet('b')
    assert SphFermKet('c', a, b).as_coeff_sp_states(use_dummies=False) == (ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_c,M_c), (a,b))
    a = SphFermBra('a')
    b = SphFermBra('b')
    assert SphFermBra('c', a, b).as_coeff_sp_states(use_dummies=False) == (ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_c,M_c), (a,b))

    # strict bra coupling
    assert SphFermBra('c', a, b).as_coeff_sp_states(strict_bra_coupling=True, use_dummies=False) == ((-1)**(J_c-M_c)*(-1)**(j_a - m_a)*(-1)**(j_b - m_b)*ASigma(m_a, m_b)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_c, -M_c), (a,b))

    # hole couplings
    a = SphFermKet('a', hole=True)
    b = SphFermKet('b')
    assert SphFermKet('c', a, b).as_coeff_sp_states(use_dummies=False) == ((-1)**(j_a-m_a)*ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,-m_a,j_b,m_b,J_c,M_c), (a,b))
    a = SphFermBra('a', hole=True)
    b = SphFermBra('b')
    assert SphFermBra('c', a, b).as_coeff_sp_states(use_dummies=False) == ((-1)**(j_a-m_a)*ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,-m_a,j_b,m_b,J_c,M_c), (a,b))

def test_nested_coupling():
    global_assumptions.clear()
    j_a, m_a = symbols('j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    j_c, m_c = symbols('j_c m_c')
    J_ab, M_ab = symbols('J_ab M_ab')
    J_abc, M_abc = symbols('J_abc M_abc')

    Ket = SphFermKet
    a,b = Ket('a'), Ket('b')
    c,d = Ket('c'), Ket('d')
    assert Ket('ab',a,b).as_coeff_sp_states(use_dummies=False) == (ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_ab,M_ab), (a,b))
    assert Ket('abc',Ket('ab',a,b),c).as_coeff_sp_states(use_dummies=False) == (ASigma(M_ab, m_c, m_a,m_b)*ClebschGordanCoefficient(J_ab,M_ab,j_c,m_c,J_abc,M_abc)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_ab,M_ab), (a,b,c))

    Bra = SphFermBra
    a,b = Bra('a'), Bra('b')
    c,d = Bra('c'), Bra('d')
    assert Bra('ab',a,b).as_coeff_sp_states(use_dummies=False) == (ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_ab,M_ab), (a,b))
    assert Bra('abc',Bra('ab',a,b),c).as_coeff_sp_states(use_dummies=False) == (ASigma(M_ab, m_c, m_a,m_b)*ClebschGordanCoefficient(J_ab,M_ab,j_c,m_c,J_abc,M_abc)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_ab,M_ab), (a,b,c))

def test_MatrixElement_construction():
    global_assumptions.clear()
    bra_a = SphFermBra('a')
    ket_b = SphFermKet('b')
    bra_ac = FermBra(bra_a, SphFermBra('c'))
    Op = SphericalTensorOperator('T','D','d')
    assert isinstance(MatrixElement(bra_a,Op,ket_b), ThreeTensorMatrixElement)
    assert isinstance(MatrixElement(bra_a,Op,ket_b,reduced=True), ReducedMatrixElement)
    assert isinstance(MatrixElement(bra_ac,Op,ket_b), DirectMatrixElement)

    # test difference antiparticle and negative coefficient of state:
    # negative states in tuple => antiparticle state
    # negative state without tuple => sign extracted out and in front of matrix element
    assert MatrixElement(bra_a,Op,(-ket_b,))==MatrixElement(bra_a,Op,ket_b.get_antiparticle())
    assert MatrixElement(bra_a,Op,-ket_b) == -MatrixElement(bra_a,Op,ket_b)
    assert MatrixElement(-bra_a,Op,ket_b,reduced=True)== -ReducedMatrixElement(bra_a,Op,ket_b)
    assert MatrixElement((-bra_a,),Op,ket_b,reduced=True)==ReducedMatrixElement(bra_a.get_antiparticle(),Op,ket_b)
    assert MatrixElement(bra_ac,Op,-ket_b) == -MatrixElement(bra_ac,Op,ket_b)
    assert MatrixElement(bra_ac,Op,(-ket_b,)) == MatrixElement(bra_ac,Op,ket_b.get_antiparticle())

def test_MatrixElement_subs():
    global_assumptions.clear()
    j_a, m_a = symbols('j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    j_c, m_c = symbols('j_c m_c')
    j_d, m_d = symbols('j_d m_d')
    k, q = symbols('k q')
    J_ab, M_ab = symbols('J_ab M_ab')
    J_cd, M_cd = symbols('J_cd M_cd')
    J_ad, M_ad = symbols('J_ad M_ad')
    J_cb, M_cb = symbols('J_cb M_cb')
    Op = SphericalTensorOperator('T',k,q)
    Bra = SphFermBra
    Ket = SphFermKet
    a,b = Bra('a'), Bra('b')
    c,d = Ket('c'), Ket('d')
    x, y, z = symbols('x y z')
    M = MatrixElement(Bra('ab', a, b), Op, Ket('cd', c, d))
    assert M.subs(M_ab, x).left._m == x
    assert M.subs(J_ab, x).left._j == x
    assert M.subs(m_a, x).left.state1._m == x
    assert M.subs(j_a, x).left.state1._j == x
    assert M.subs(m_b, x).left.state2._m == x
    assert M.subs(j_b, x).left.state2._j == x
    assert M.subs(q, x).operator.as_coeff_tensor() == (S.One, SphericalTensor('T', k, x))
    assert M.subs(k, x).operator.rank == x
    assert M.subs(k, x).operator.as_coeff_tensor() == (S.One, SphericalTensor('T', x, q))
    assert M.subs(M_cd, x).right._m == x
    assert M.subs(J_cd, x).right._j == x
    assert M.subs(m_c, x).right.state1._m == x
    assert M.subs(j_c, x).right.state1._j == x
    assert M.subs(m_d, x).right.state2._m == x
    assert M.subs(j_d, x).right.state2._j == x

def test_ReducedMatrixElement():
    global_assumptions.clear()
    k, q, j_a, m_a = symbols('k q j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    j_c, m_c = symbols('j_c m_c')
    J_ac, M_ac = symbols('J_ac M_ac')
    bra_a = SphFermBra('a')
    ket_b = SphFermKet('b')
    Op = SphericalTensorOperator('T','k','q')
    assert ReducedMatrixElement(bra_a, Op, ket_b).get_reduction_factor()==ClebschGordanCoefficient(j_b,m_b,k,q,j_a,m_a)
    assert ReducedMatrixElement(bra_a, Op, ket_b)._get_ThreeTensorMatrixElement()==ThreeTensorMatrixElement(bra_a, Op, ket_b)

    bra_ac = SphFermBra('ac', bra_a, SphFermBra('c'))
    assert ReducedMatrixElement(bra_ac, Op, ket_b).get_reduction_factor()==ClebschGordanCoefficient(j_b,m_b,k,q,J_ac,M_ac)
    assert ReducedMatrixElement(bra_ac, Op, ket_b)._get_ThreeTensorMatrixElement()==ThreeTensorMatrixElement(bra_ac, Op, ket_b)

    assert ReducedMatrixElement(bra_ac, Op, ket_b).as_direct_product(use_dummies=False)==ASigma(m_b, q, m_a, m_c)*ClebschGordanCoefficient(j_b,m_b,k,q,J_ac,M_ac)*ClebschGordanCoefficient(j_a, m_a, j_c, m_c, J_ac, M_ac)*DirectMatrixElement((bra_a, SphFermBra('c')), Op, ket_b)
    assert ReducedMatrixElement(bra_ac, Op, ket_b).get_direct_product_ito_self(use_dummies=False)==ASigma(J_ac, M_ac)*ClebschGordanCoefficient(j_b,m_b,k,q,J_ac,M_ac)*ClebschGordanCoefficient(j_a, m_a, j_c, m_c, J_ac, M_ac)*ReducedMatrixElement(bra_ac, Op, ket_b)

    redmat = ReducedMatrixElement(bra_a,Op,ket_b,'brink_satchler')
    assert redmat.definition == 'brink_satchler'
    assert redmat.func(*redmat.args) == redmat
    assert redmat.func(*redmat.args).definition == 'brink_satchler'
    assert redmat.get_reduction_factor()==ClebschGordanCoefficient(j_b,m_b,k,q,j_a,m_a)*(-1)**(2*k)

    redmat = ReducedMatrixElement(bra_a,Op,ket_b,'edmonds')
    assert redmat.definition == 'edmonds'
    assert redmat.func(*redmat.args).definition == 'edmonds'
    assert redmat.get_reduction_factor()==ThreeJSymbol(j_a,k,j_b,-m_a,q,m_b)*(-1)**(j_a-m_a)


def test_DirectMatrixElement():
    global_assumptions.clear()
    Op = SphericalTensorOperator('T','k','q')
    a = SphFermBra('a')
    b = SphFermBra('b')
    c = SphFermKet('c')
    d = SphFermKet('d')
    raises(WignerEckardDoesNotApply, 'DirectMatrixElement(a, Op, c).use_wigner_eckardt()')

    DME = DirectMatrixElement

    # symmetries
    M = DME((a,b),Op,(c,d))
    assert M ==-DME((a,b), Op, (d,c))
    assert M ==-DME((b,a), Op, (c,d))
    assert M == DME((b,a), Op, (d,c))

    # vacuum shifting
    assert M.shift_vacuum([b])==DME((a), Op, (c, d,-Dagger(b)))
    assert M.shift_vacuum([d])==DME((a, b,-Dagger(d)), Op, (c))
    assert M.shift_vacuum([b,d])==-DME((a, -Dagger(d)), Op, (c, -Dagger(b)))
    assert M.shift_vacuum([a,b])==DME(None, Op, (c, d, -Dagger(b),-Dagger(a)))
    assert M.shift_vacuum([c,d])==DME((a, b, -Dagger(d),-Dagger(c)), Op, None)
    assert M.shift_vacuum([b,a,d,c])==DME((-Dagger(d),-Dagger(c)), Op, (-Dagger(b), -Dagger(a)))

    assert M.shift_vacuum([b,d])==M.shift_vacuum([d,b])
    assert M.shift_vacuum([a,b,c,d])==M.shift_vacuum([b,a,c,d])

def test_ThreeTensorMatrixElement():
    global_assumptions.clear()
    j_a, j_b, j_c, j_d = symbols('j_a j_b j_c j_d', nonnegative=True)
    m_a, m_b, m_c, m_d = symbols('m_a m_b m_c m_d')
    J_ab, J_cd, J_ca, J_db = symbols('J_ab J_cd J_ca J_db', nonnegative=True)
    M_ab, M_cd, M_ca, M_db = symbols('M_ab M_cd M_ca M_db')
    k, q = symbols('k q')
    Op = SphericalTensorOperator('T',k,q)
    Bra = SphFermBra
    Ket = SphFermKet
    a,b = Bra('a'), Bra('b')
    c,d = Ket('c'), Ket('d')

    # trivial tests
    assert ThreeTensorMatrixElement(a,Op,c) == ThreeTensorMatrixElement(a,Op,c).get_direct_product_ito_self()
    assert DirectMatrixElement(a,Op,c) == ThreeTensorMatrixElement(a,Op,c).as_direct_product()

    # test coupling and decoupling
    assert DirectMatrixElement((a,b),Op,(c,d)) == ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).get_related_direct_matrix()
    assert ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).get_direct_product_ito_self(use_dummies=False) == ASigma(J_ab, J_cd, M_ab, M_cd)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d))
    assert ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).as_direct_product(use_dummies=False) == ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*DirectMatrixElement((a,b),Op,(c,d))

    #test wigner_eckardt
    assert ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).get_direct_product_ito_self(wigner_eckardt=True, use_dummies=False) == ASigma(J_ab, J_cd, M_ab, M_cd)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).use_wigner_eckardt()
    assert  ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).as_direct_product(wigner_eckardt=True, use_dummies=False) == ASigma(M_cd, m_a, m_b, m_c, m_d, q)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*ClebschGordanCoefficient(J_cd, M_cd, k, q,J_ab, M_ab)*DirectMatrixElement((a,b),Op,(c,d))
    assert  ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).as_direct_product(wigner_eckardt=True, definition='brink_satchler', use_dummies=False) == ASigma(M_cd, m_a, m_b, m_c, m_d, q)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*ClebschGordanCoefficient(J_cd, M_cd, k, q,J_ab, M_ab)*DirectMatrixElement((a,b),Op,(c,d))*(-1)**(-2*k)



def test_coupling_order():
    global_assumptions.clear()
    j_a, j_b, j_c, j_d = symbols('j_a j_b j_c j_d', nonnegative=True)
    m_a, m_b, m_c, m_d = symbols('m_a m_b m_c m_d')
    J_ab, J_cd, J_ad, J_cb, J_ca, J_db = symbols('J_ab J_cd J_ad J_cb J_ca J_db', nonnegative=True)
    M_ab, M_cd, M_ad, M_cb, M_ca, M_db = symbols('M_ab M_cd M_ad M_cb M_ca M_db')
    k, q = symbols('k q')
    Op = SphericalTensorOperator('T',k,q)
    Bra = SphFermBra
    Ket = SphFermKet
    a,b = Bra('a'), Bra('b')
    c,d = Ket('c'), Ket('d')

    braAB = Bra('ab',a,b)
    braBA = Bra('ab',a,b,reverse=True)
    assert is_equivalent(braAB.as_direct_product(tjs=1, use_dummies=0),
            (-1)**(j_a + j_b - J_ab)*braBA.as_direct_product(tjs=1, use_dummies=0))
    assert is_equivalent(Dagger(braAB).as_direct_product(tjs=1, use_dummies=0),
            (-1)**(j_a + j_b - J_ab)*Dagger(braBA).as_direct_product(tjs=1, use_dummies=0))

    c,d = Ket('c'), Ket('d')
    ketCD = Ket('cd',c,d)
    ketDC = Ket('cd',c,d,reverse=True)
    assert is_equivalent(ketCD.as_direct_product(tjs=1, use_dummies=0),
            (-1)**(j_c + j_d - J_cd)*ketDC.as_direct_product(tjs=1, use_dummies=0))
    assert is_equivalent(Dagger(ketCD).as_direct_product(tjs=1, use_dummies=0),
            (-1)**(j_c + j_d - J_cd)*Dagger(ketDC).as_direct_product(tjs=1, use_dummies=0))

    AB_op_CD = MatrixElement(braAB, Op, ketCD)
    BA_op_DC = MatrixElement(braBA, Op, ketDC)
    assert is_equivalent(
            AB_op_CD.as_direct_product(tjs=1, use_dummies=0),
            BA_op_DC.as_direct_product(tjs=1, use_dummies=0)*(-1)**(j_a+j_b-J_ab+j_c+j_d-J_cd),
            verbosity = 1)

    # test vacuum-shifted coupling and decoupling
    assert DirectMatrixElement((a,-b),Op,(c,-d)) == ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d)).get_related_direct_matrix()
    assert ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d)).get_direct_product_ito_self(use_dummies=False) == (-1)**(m_b-j_b)*(-1)**(m_d-j_d)*ASigma(J_ab, J_cd, M_ab, M_cd)*ClebschGordanCoefficient(j_a, m_a, j_b, -m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, -m_d,J_cd, M_cd)*ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d))
    # Kuo & al. eq (38)
    assert ThreeTensorMatrixElement(Bra('ad',a,-Dagger(d)),Op,Ket('cb',c,-Dagger(b))).as_direct_product(use_dummies=False, only_particle_states=True) == (-1)**(j_b-m_b)*(-1)**(j_d-m_d)*ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, m_a, j_d, -m_d,J_ad, M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b,J_cb, M_cb)*DirectMatrixElement((a,b),Op,(c,d))*(-1) #FIXME (-1) ???
    # FIXME: Kuo & al. do not have the (-1) above, but they choose to ignore
    # phases from "contractions of fermionic operators".  Is this consistent then?

    # test cross coupled matrix element
    crossmat = ThreeTensorMatrixElement(Bra('ca',a,c,reverse=True),Op,Ket('db',b,d,reverse=True))
    assert DirectMatrixElement((a,b),Op,(c,d)) == crossmat.get_related_direct_matrix()
    # Kuo & al. eq (20)
    assert crossmat.as_direct_product(use_dummies=False) == (-1)**(J_ca - M_ca)*(-1)**(j_b-m_b)*(-1)**(j_a-m_a)*ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_c,  m_c, j_a,-m_a,J_ca, -M_ca)*ClebschGordanCoefficient(j_d, m_d, j_b, -m_b,J_db, M_db)*DirectMatrixElement((a,b),Op,(c,d))

def test_MatrixElement_recoupling():
    global_assumptions.clear()
    j_a, j_b, j_c, j_d = symbols('j_a j_b j_c j_d', nonnegative=True)
    m_a, m_b, m_c, m_d = symbols('m_a m_b m_c m_d')
    J_ab, J_cd, J_ad, J_cb = symbols('J_ab J_cd J_ad J_cb', nonnegative=True)
    M_ab, M_cd, M_ad, M_cb = symbols('M_ab M_cd M_ad M_cb')
    k, q = symbols('k q')
    Op = SphericalTensorOperator('T',k,q)
    Bra = SphFermBra
    Ket = SphFermKet
    a,b = Bra('a'), Bra('b')
    c,d = Ket('c'), Ket('d')

    shifted = ThreeTensorMatrixElement(Bra('ad',a,-Dagger(d)),Op,Ket('cb',c,-Dagger(b)))
    straight = ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d))

    # test vacuum shifted coupling
    assert shifted.get_related_direct_matrix() == DirectMatrixElement((a,-Dagger(d)),Op,(c,-Dagger(b)))
    assert shifted.get_related_direct_matrix(only_particle_states=True) == -DirectMatrixElement((a,b),Op,(c,d))

    # test straight coupling
    assert straight.get_related_direct_matrix() == DirectMatrixElement((a,b),Op,(c,d))

    # verify equations
    assert shifted.as_other_coupling(straight, use_dummies=False) == (-1)**(J_ad + M_ab - J_ab - M_ad)*ASigma(J_ab, J_cd, M_ab, M_cd, m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_ab, -M_ab)*ClebschGordanCoefficient(j_a, -m_a, j_d, m_d, J_ad, -M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b, J_cb, M_cb)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d, J_cd, M_cd)*straight
    assert straight.as_other_coupling(shifted, use_dummies=False) == (-1)**(J_ab + M_ad - J_ad - M_ab)*ASigma(J_ad, J_cb, M_ad, M_cb, m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_ab, -M_ab)*ClebschGordanCoefficient(j_a, -m_a, j_d, m_d, J_ad, -M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b, J_cb, M_cb)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d, J_cd, M_cd)*shifted

    # relate reduced matrix elements
    assert shifted.as_other_coupling(straight, wigner_eckardt=True, use_dummies=False) == (-1)**(J_ad + M_ab - J_ab - M_ad)*ASigma(J_ab, J_cd, M_ab, M_cd, M_cb, m_a, m_b, m_c, m_d, q)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_ab, -M_ab)*ClebschGordanCoefficient(j_a, -m_a, j_d, m_d, J_ad, -M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b, J_cb, M_cb)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d, J_cd, M_cd)*ClebschGordanCoefficient(J_cb, M_cb, k, q, J_ad, M_ad)*ClebschGordanCoefficient(J_cd, M_cd, k, q, J_ab, M_ab)*straight.get_related_redmat()
    assert straight.as_other_coupling(shifted, wigner_eckardt=True, use_dummies=False) == (-1)**(-J_ad - M_ab + J_ab + M_ad)*ASigma(J_ad, J_cb, M_ad, M_cb, M_cd, m_a, m_b, m_c, m_d, q)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_ab, -M_ab)*ClebschGordanCoefficient(j_a, -m_a, j_d, m_d, J_ad, -M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b, J_cb, M_cb)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d, J_cd, M_cd)*ClebschGordanCoefficient(J_cb, M_cb, k, q, J_ad, M_ad)*ClebschGordanCoefficient(J_cd, M_cd, k, q, J_ab, M_ab)*shifted.get_related_redmat()

