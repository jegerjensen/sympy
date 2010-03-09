from sympy import symbols, C, S

from sympy.physics.braket import (
        SphericalTensorOperator, Dagger, BosonState, FermionState, Ket, Bra,
        QuantumState, SphFermKet, SphFermBra, FermKet, FermBra, ASigma,
        DirectQuantumState, SphericalTensor, ClebschGordanCoefficient,
        MatrixElement, SphericalTensorOperator, ThreeTensorMatrixElement,
        ReducedMatrixElement, DirectMatrixElement, SphericalQuantumState,
        BraKet, WignerEckardDoesNotApply, QuantumVacuumKet, QuantumVacuumBra
        )

from sympy.utilities import raises


def test_SphericalTensorOperator():
    k,q,T = symbols('k q T')
    assert Dagger(SphericalTensorOperator(T,k,q))==(-1)**(k+q)*SphericalTensorOperator(T, k, -q)

def test_Fermion_Boson():
    assert BosonState('a').spin_assume == 'integer'
    assert FermionState('a').spin_assume == 'half_integer'

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

    assert QuantumState('a').is_negative is None
    assert FermionState('a').is_negative is None
    assert BraKet('a').is_negative is None
    assert SphFermKet('a').is_negative is None

    assert QuantumState('a').is_positive is None
    assert FermionState('a').is_positive is None
    assert BraKet('a').is_positive is None
    assert SphFermKet('a').is_positive is None

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
    a,b,c,d = symbols('a b c d')
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

def test_as_coeff_tensor():
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
    full_bra_decoupling = SphFermBra('c', SphFermBra('a'), SphFermBra('b')).as_coeff_tensor(deep=True)
    assert full_bra_decoupling[0] == (-1)**(J_c-M_c)*(-1)**(j_a - m_a)*(-1)**(j_b - m_b)*ASigma(m_a, m_b)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_c, -M_c)
    assert full_bra_decoupling[1] == SphericalTensor(t,j_a, -m_a)*SphericalTensor(t,j_b, -m_b)

def test_as_coeff_sp_states():
    t, T, j_a, m_a = symbols('t T j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    J_c, M_c = symbols('J_c M_c')

    # sp states
    assert SphFermKet('a').as_coeff_sp_states() == (1, (SphFermKet('a'),))
    assert SphFermBra('a').as_coeff_sp_states() == (1, (SphFermBra('a'),))

    # hole states
    assert SphFermKet('a', hole=True).as_coeff_sp_states() == (1, (SphFermKet('a', hole=True),))
    assert SphFermBra('a', hole=True).as_coeff_sp_states() == (1, (SphFermBra('a', hole=True),))

    # coupled states
    a = SphFermKet('a')
    b = SphFermKet('b')
    assert SphFermKet('c', a, b).as_coeff_sp_states() == (ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_c,M_c), (a,b))
    a = SphFermBra('a')
    b = SphFermBra('b')
    assert SphFermBra('c', a, b).as_coeff_sp_states() == (ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,m_a,j_b,m_b,J_c,M_c), (a,b))

    # strict bra coupling
    assert SphFermBra('c', a, b).as_coeff_sp_states(strict_bra_coupling=True) == ((-1)**(J_c-M_c)*(-1)**(j_a - m_a)*(-1)**(j_b - m_b)*ASigma(m_a, m_b)*ClebschGordanCoefficient(j_a, -m_a, j_b, -m_b, J_c, -M_c), (a,b))

    # hole couplings
    a = SphFermKet('a', hole=True)
    b = SphFermKet('b')
    assert SphFermKet('c', a, b).as_coeff_sp_states() == ((-1)**(j_a-m_a)*ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,-m_a,j_b,m_b,J_c,M_c), (a,b))
    a = SphFermBra('a', hole=True)
    b = SphFermBra('b')
    assert SphFermBra('c', a, b).as_coeff_sp_states() == ((-1)**(j_a-m_a)*ASigma(m_a,m_b)*ClebschGordanCoefficient(j_a,-m_a,j_b,m_b,J_c,M_c), (a,b))

def test_MatrixElement_construction():
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

def test_ReducedMatrixElement():
    k, q, j_a, m_a = symbols('k q j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    J_ac, M_ac = symbols('J_ac M_ac')
    bra_a = SphFermBra('a')
    ket_b = SphFermKet('b')
    Op = SphericalTensorOperator('T','k','q')
    assert ReducedMatrixElement(bra_a, Op, ket_b, wigner_eckardt=True)==ClebschGordanCoefficient(j_b,m_b,k,q,j_a,m_a)*ReducedMatrixElement(bra_a, Op, ket_b)
    assert ReducedMatrixElement(bra_a, Op, ket_b)._get_reduction_factor()==ClebschGordanCoefficient(j_b,m_b,k,q,j_a,m_a)
    assert ReducedMatrixElement(bra_a, Op, ket_b)._get_ThreeTensorMatrixElement()==ThreeTensorMatrixElement(bra_a, Op, ket_b)

    bra_ac = SphFermBra('ac', bra_a, SphFermBra('c'))
    assert ReducedMatrixElement(bra_ac, Op, ket_b, wigner_eckardt=True)==ClebschGordanCoefficient(j_b,m_b,k,q,J_ac,M_ac)*ReducedMatrixElement(bra_ac, Op, ket_b)
    assert ReducedMatrixElement(bra_ac, Op, ket_b)._get_reduction_factor()==ClebschGordanCoefficient(j_b,m_b,k,q,J_ac,M_ac)
    assert ReducedMatrixElement(bra_ac, Op, ket_b)._get_ThreeTensorMatrixElement()==ThreeTensorMatrixElement(bra_ac, Op, ket_b)

def test_DirectMatrixElement():
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
    j_a, m_a = symbols('j_a m_a')
    j_b, m_b = symbols('j_b m_b')
    j_c, m_c = symbols('j_c m_c')
    j_d, m_d = symbols('j_d m_d')
    k, q = symbols('k q')
    J_ab, M_ab = symbols('J_ab M_ab')
    J_cd, M_cd = symbols('J_cd M_cd')
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
    assert ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).get_direct_product_ito_self() == ASigma(J_ab, J_cd, M_ab, M_cd)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d))
    assert ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d)).as_direct_product() == ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d,J_cd, M_cd)*DirectMatrixElement((a,b),Op,(c,d))

    # test vacuum-shifted coupling and decoupling
    assert DirectMatrixElement((a,-b),Op,(c,-d)) == ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d)).get_related_direct_matrix()
    assert ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d)).get_direct_product_ito_self() == (-1)**(j_b-m_b)*(-1)**(j_d-m_d)*ASigma(J_ab, J_cd, M_ab, M_cd)*ClebschGordanCoefficient(j_a, m_a, j_b, -m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, -m_d,J_cd, M_cd)*ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d))
    assert ThreeTensorMatrixElement(Bra('ab',a,-b),Op,Ket('cd',c,-d)).as_direct_product() == (-1)**(j_b-m_b)*(-1)**(j_d-m_d)*ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, m_a, j_b, -m_b,J_ab, M_ab)*ClebschGordanCoefficient(j_c, m_c, j_d, -m_d,J_cd, M_cd)*DirectMatrixElement((a,-b),Op,(c,-d))

def test_MatrixElement_recoupling():
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

    shifted = ThreeTensorMatrixElement(Bra('ad',a,-Dagger(d)),Op,Ket('cb',c,-Dagger(b)))
    straight = ThreeTensorMatrixElement(Bra('ab',a,b),Op,Ket('cd',c,d))

    # test vacuum shifted coupling
    assert shifted.get_related_direct_matrix() == DirectMatrixElement((a,-Dagger(d)),Op,(c,-Dagger(b)))
    assert shifted.get_related_direct_matrix(only_particle_states=True) == -DirectMatrixElement((a,b),Op,(c,d))

    # test straight coupling
    assert straight.get_related_direct_matrix() == DirectMatrixElement((a,b),Op,(c,d))

    # verify equations
    assert shifted.as_other_coupling(straight) == -(-1)**(j_b - m_b)*(-1)**(j_d - m_d)*ASigma(J_ab, J_cd, M_ab, M_cd)*ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b, J_ab, M_ab)*ClebschGordanCoefficient(j_a, m_a, j_d, -m_d, J_ad, M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b, J_cb, M_cb)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d, J_cd, M_cd)*straight

    assert straight.as_other_coupling(shifted) == -(-1)**(j_b - m_b)*(-1)**(j_d - m_d)*ASigma(J_ad, J_cb, M_ad, M_cb)*ASigma(m_a, m_b, m_c, m_d)*ClebschGordanCoefficient(j_a, m_a, j_b, m_b, J_ab, M_ab)*ClebschGordanCoefficient(j_a, m_a, j_d, -m_d, J_ad, M_ad)*ClebschGordanCoefficient(j_c, m_c, j_b, -m_b, J_cb, M_cb)*ClebschGordanCoefficient(j_c, m_c, j_d, m_d, J_cd, M_cd)*shifted


