from sympy.physics.racahalgebra import (
        refine_phases, refine_tjs2sjs, convert_cgc2tjs, is_equivalent,
        SixJSymbol, ASigma, combine_ASigmas, evaluate_sums, apply_deltas,
        apply_orthogonality, ClebschGordanCoefficient, extract_symbol2dummy_dict
        )
from sympy.physics.braket import (
        MatrixElement, ReducedMatrixElement, apply_wigner_eckardt,
        rewrite_as_direct_product, DirectMatrixElement,
        SphericalTensorOperator, Dagger, SphFermKet, SphFermBra,
        FermKet, FermBra, ThreeTensorMatrixElement,
        inject_every_symbol_globally, rewrite_coupling,
        braket_assumptions, DualSphericalTensorOperator
        )
from sympy import (
        Symbol, symbols, global_assumptions, Assume, ask, Q, Mul, S, sqrt,
        pprint, Eq, pprint_use_unicode, latex, preview, fcode, Function
        )
import sys

pprint_use_unicode(False)
def _report(expr):

    pprint(str(expr))

    # print fcode(expr)

    # print(expr)
    # pprint(expr)
    # print latex(expr)
    # preview(expr)

    print

def make_spherical_sp_states(str1):
    states = []
    for i in symbols(str1):
        states.append(SphFermKet(i))
    return states


Ket = FermKet
Bra = FermBra

i, j, k, l = make_spherical_sp_states('i j k l')
a, b, c, d = make_spherical_sp_states('a b c d')

J_A = Symbol('J_A', nonnegative=True)
M_A = Symbol('M_A')
J_Am1 = Symbol('J_Am1', nonnegative=True)
M_Am1 = Symbol('M_Am1')
braket_assumptions.add(Assume(J_A, Q.integer))
braket_assumptions.add(Assume(M_A, Q.integer))
braket_assumptions.add(Assume(J_Am1, 'half_integer'))
braket_assumptions.add(Assume(M_Am1, 'half_integer'))

LA = DualSphericalTensorOperator('L', J_A, M_A)
LAm1 = DualSphericalTensorOperator('L', J_Am1, M_Am1)
RA = SphericalTensorOperator('R', J_A, M_A)
RAm1 = SphericalTensorOperator('R', J_Am1, M_Am1)
V = SphericalTensorOperator('V', S.Zero, S.Zero)
T = SphericalTensorOperator('T', S.Zero, S.Zero)

print
print "Defined tensor operators:"
print LA
print RA
print LAm1
print RAm1
print V
print T


print
print "*************** <i| L | > *****************"
print

l_i = MatrixElement(Dagger(i), LA, "")
l_i_sph = ReducedMatrixElement(Dagger(i), LA, "")
print Eq(l_ai, l_ai_sph.get_direct_product_ito_self(tjs=0))

print
print fcode(l_i_sph.get_direct_product_ito_self(tjs=1))

print
print "*************** <ij| L | a> *****************"
print

l_aij = MatrixElement((Dagger(i), Dagger(j)), LA, a)
l_aij_sph = ReducedMatrixElement(SphFermBra('ij', Dagger(i), Dagger(j)), LA, a)
print Eq(l_aij, l_aij_sph.get_direct_product_ito_self(tjs=0))

print
print fcode(l_aij_sph.get_direct_product_ito_self(tjs=1))

print
print "*************** <a| T | i> *****************"
print

t_ai = MatrixElement(Dagger(a), T, i)
t_ai_sph = ReducedMatrixElement(Dagger(a), T, i)
print Eq(t_ai, t_ai_sph.get_direct_product_ito_self(tjs=0))

print
print fcode(t_ai_sph.get_direct_product_ito_self(tjs=1))

print
print "*************** <ab| T | ij> *****************"
print

t_abij = MatrixElement((Dagger(a), Dagger(b)), T, (i, j))
t_abij_sph = ReducedMatrixElement(SphFermBra('ab', Dagger(a), Dagger(b), reverse=0), T, SphFermKet('ij', i, j, reverse=0))
print Eq(t_abij, t_abij_sph.get_direct_product_ito_self(tjs=0))

print
print fcode(t_abij_sph.get_direct_product_ito_self(tjs=1))



print
print "*************** <| R | i> *****************"
print

r_ai = MatrixElement(a, RAm1, i)
r_ai_sph = ReducedMatrixElement(a, RAm1, i)
print Eq(r_ai, r_ai_sph.get_direct_product_ito_self(tjs=0))
print Eq(r_ai, r_ai_sph.get_direct_product_ito_self(tjs=1))

print
print fcode(r_ai_sph.get_direct_product_ito_self(tjs=1))
print
print "*************** <ab| R | ij> *****************"
print

r_abij = MatrixElement((Dagger(a), Dagger(b)), RAm1, (i, j))
r_abij_sph = ReducedMatrixElement(SphFermBra('ab', Dagger(a), Dagger(b)), RAm1, SphFermKet('ij', i, j))
print Eq(r_abij, r_abij_sph.get_direct_product_ito_self(tjs=0))

print
print fcode(r_abij_sph.get_direct_product_ito_self(tjs=1))

J_ij, J_Am1, j_b, j_i, j_j = symbols('J_ij J_Am1 j_b j_i j_j', nonnegative=True)
M_ij, M_Am1, m_b, m_i, m_j = symbols('M_ij M_Am1 m_b m_i m_j')

J_Am1 = RAm1.rank





SF = Symbol('SF')

"""
Ket = FermKet
Bra = FermBra

i, j, k, l = make_spherical_sp_states('i j k l')
a, b, c, d = make_spherical_sp_states('a b c d')

LA, RA = make_tensor_operators('J_A M_A J_A M_A',Q.integer)
LAm1,RAm1 = make_tensor_operators('J_Am1 M_Am1 J_Am1 M_Am1','half_integer')
# LA = LA.subs(Symbol('t'), Symbol('L'))
LA = Dagger(LA.subs(Symbol('t'), Symbol('L')))
# LA = SphericalTensorOperator('L', S.Zero, S.Zero)
RA = RA.subs(Symbol('t'), Symbol('R'))
LAm1 = Dagger(LAm1.subs(Symbol('t'), Symbol('L')))
RAm1 = RAm1.subs(Symbol('t'), Symbol('R'))
V = SphericalTensorOperator('V', S.Zero, S.Zero)
T = SphericalTensorOperator('T', S.Zero, S.Zero)
cre_p = SphericalTensorOperator('a','j','m')
ann_p = Dagger(SphericalTensorOperator('a','j','m'))
"""
print
print "Defined tensor operators:"
print LA
print RA
print LAm1
print RAm1
print V
print T

print
print "==== Recoupling L_{A} a' R_{A-1} ===="
print

# """
j_a = Symbol('j_a', nonnegative=True)
m_a = Symbol('m_a')
sf_reduction = (-1)**(j_a - m_a)*ClebschGordanCoefficient('J_A', 'M_A', j_a, -m_a, 'J_Am1', 'M_Am1')*ASigma('M_A', 'm_a')
# sf_reduction = (-1)**(j_a - m_a)*ClebschGordanCoefficient(0, 0, j_a, -m_a, 'J_Am1', 'M_Am1')*ASigma('M_A', 'm_a')

print "reduction factor"
print sf_reduction
print fcode(refine_phases(convert_cgc2tjs(sf_reduction)))

l0 = Symbol('l_0')
r0 = Symbol('r_0')

# r_i = MatrixElement("",RAm1, i)
# r_aij = MatrixElement(Dagger(a), RAm1, (i, j))
# r_i_sph = ReducedMatrixElement("",RAm1, i)
# r_aij_sph = ReducedMatrixElement(Dagger(a), RAm1, SphFermKet('ij',i, j))

# l_ai = MatrixElement(Dagger(i),LA,a)
# l_ai_sph = ReducedMatrixElement(Dagger(i),LA,a)
# l_abij = MatrixElement((Dagger(i),Dagger(j)),LA,(a,b))
# l_abij_sph = ReducedMatrixElement(SphFermBra('ij',Dagger(i),Dagger(j), reverse=0), LA, SphFermKet('ab', a, b,reverse=0))

# t_ai = MatrixElement(Dagger(a),T ,i)
# t_ai_sph = ReducedMatrixElement(Dagger(a),T ,i)
# t_abij = MatrixElement((Dagger(a), Dagger(b)), T,(i, j))
# t_abij_sph = ReducedMatrixElement(SphFermBra('ab',Dagger(a), Dagger(b), reverse=0), T, SphFermKet('ij', i, j,reverse=0))

_report("*** coupled elements: ***")
_report(Eq(r_i_sph,rewrite_coupling(r_i_sph, r_i)))
_report(Eq(r_aij_sph,rewrite_coupling(r_aij_sph, r_aij)))
_report(Eq(l_ai_sph,rewrite_coupling(l_ai_sph, l_ai)))
_report(Eq(l_abij_sph,rewrite_coupling(l_abij_sph, l_abij)))
_report(Eq(t_ai_sph,rewrite_coupling(t_ai_sph, t_ai)))
_report(Eq(t_abij_sph,rewrite_coupling(t_abij_sph, t_abij)))

"""
coupled_subs = {
        l_ai: rewrite_coupling(l_ai, l_ai_sph),
        r_i: rewrite_coupling(r_i, r_i_sph)
        }
print
_report("recoupling diagram 1")
expr_msc = combine_ASigmas(l_ai*r_i*ASigma('m_i') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
print fcode(expr_sph)


print
_report("recoupling diagram 2")
r_bij = r_aij.subs(Dagger(a), Dagger(b))
r_bij_sph = r_aij_sph.subs(Dagger(a), Dagger(b))

coupled_subs = {
        l_abij: rewrite_coupling(l_abij, l_abij_sph),
        r_bij: rewrite_coupling(r_bij, r_bij_sph)
        }

expr_msc = combine_ASigmas(S(1)/2*l_abij*r_bij*ASigma('m_b','m_i','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_b', 'm_i', 'm_j'])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases((expr_sph))
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1, let_pass=0)
_report(Eq(SF, expr_sph))
print fcode(expr_sph)


print
print "==== Recoupling L_{A} i' R_{A-1} ===="
print


j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = 1/((-1)**(j_i - m_i)*ClebschGordanCoefficient(0, 0, j_i, -m_i, 'J_Am1', 'M_Am1').get_as_ThreeJSymbol())


coupled_subs = {
        r_i: rewrite_coupling(r_i, r_i_sph)
        }
print
_report("recoupling diagram 3")
expr_msc = combine_ASigmas(l0*r_i*sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
print fcode(expr_sph)



print
_report("recoupling diagram 4")
l_aj = l_ai.subs(Dagger(i),Dagger(j))
l_aj_sph = l_ai_sph.subs(Dagger(i),Dagger(j))
coupled_subs = {
        l_aj: rewrite_coupling(l_aj, l_aj_sph),
        r_aij: rewrite_coupling(r_aij, r_aij_sph)
        }
j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient('J_A', 'M_A', j_i, -m_i, 'J_Am1', 'M_Am1')*ASigma('M_A', 'm_i')
print
expr_msc = combine_ASigmas(l_aj*r_aij*ASigma('m_a','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
print fcode(expr_sph)

print
_report("recoupling diagram 5")
l_aj = l_ai.subs(Dagger(i),Dagger(j))
l_aj_sph = l_ai_sph.subs(Dagger(i),Dagger(j))
r_j = r_i.subs(i,j)
r_j_sph = r_i_sph.subs(i,j)
coupled_subs = {
        t_ai: rewrite_coupling(t_ai, t_ai_sph),
        l_aj: rewrite_coupling(l_aj, l_aj_sph),
        r_j: rewrite_coupling(r_j, r_j_sph)
        }
j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient('J_A', 'M_A', j_i, -m_i, 'J_Am1', 'M_Am1')*ASigma('M_A', 'm_i')
print
expr_msc = combine_ASigmas(l_aj*r_j*t_ai*ASigma('m_a','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph, forbidden=[S('M_A'), S('M_Am1'), S('m_a')])
_report(Eq(SF, expr_sph))
print fcode(expr_sph)

"""

print
_report("recoupling diagram 6")
l_abjk = l_abij.subs(Dagger(j),Dagger(k))
l_abjk = l_abjk.subs(Dagger(i),Dagger(j))
l_abjk_sph = l_abij_sph.subs(SphFermBra('ij',Dagger(i),Dagger(j)), SphFermBra('jk', Dagger(j), Dagger(k)))
r_bjk = r_aij.subs(j,k)
r_bjk = r_bjk.subs(i,j)
r_bjk = r_bjk.subs(Dagger(a), Dagger(b))
r_bjk_sph = r_aij_sph.subs({Dagger(a): Dagger(b), SphFermKet('ij',i,j): SphFermKet('jk',j,k)})
coupled_subs = {
        t_ai: rewrite_coupling(t_ai, t_ai_sph),
        l_abjk: rewrite_coupling(l_abjk, l_abjk_sph),
        r_bjk: rewrite_coupling(r_bjk, r_bjk_sph)
        }
j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient('J_A', 'M_A', j_i, -m_i, 'J_Am1', 'M_Am1')*ASigma('M_A', 'm_i')
print
expr_msc = combine_ASigmas(l_abjk*r_bjk*t_ai*ASigma('m_a','m_j','m_k','m_b') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_j', 'm_k'])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
print fcode(expr_sph)




print
_report("recoupling diagram 7")
l_abjk = l_abij.subs(Dagger(j),Dagger(k))
l_abjk = l_abjk.subs(Dagger(i),Dagger(j))
l_abjk_sph = l_abij_sph.subs(SphFermBra('ij',Dagger(i),Dagger(j)), SphFermBra('jk', Dagger(j), Dagger(k)))
r_j = r_i.subs(i,j)
r_j_sph = r_i_sph.subs(i,j)
t_abik = t_abij.subs(j,k)
t_abik_sph = t_abij_sph.subs(SphFermKet('ij',i,j), SphFermKet('ik',i,k))
coupled_subs = {
        t_abik: rewrite_coupling(t_abik, t_abik_sph),
        l_abjk: rewrite_coupling(l_abjk, l_abjk_sph),
        r_j: rewrite_coupling(r_j, r_j_sph)
        }
j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient('J_A', 'M_A', j_i, -m_i, 'J_Am1', 'M_Am1')*ASigma('M_A', 'm_i')
print
expr_msc = combine_ASigmas(l_abjk*r_j*t_abik*ASigma('m_a','m_j','m_k','m_b') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_b', 'm_a'])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
print fcode(expr_sph)

