from sympy.physics.racahalgebra import (
        refine_phases, refine_tjs2sjs, convert_cgc2tjs, is_equivalent,
        SixJSymbol, ASigma, combine_ASigmas, evaluate_sums, apply_deltas,
        apply_orthogonality, ClebschGordanCoefficient, extract_symbol2dummy_dict,
        extract_dummy2symbol_dict
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

    if isinstance(expr, basestring):
        print r"\begin{verbatim}"
        print expr
        print r"\end{verbatim}"
        return

    # pprint(str(expr))

    # print fcode(expr)

    # print(expr)

    # pprint(expr)
    subsdict = extract_dummy2symbol_dict(expr)
    expr = expr.subs(subsdict)
    print latex(expr, mode='dmath', mul_symbol='dot')

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
_report( LA)
_report( RA)
_report( LAm1)
_report( RAm1)
_report( V)
_report( T)


print
print "*************** <i| L | > *****************"
print

l_i = MatrixElement(Dagger(i), LAm1, "")
l_i_sph = ReducedMatrixElement(Dagger(i), LAm1, "")
_report( Eq(l_i, l_i_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(l_i_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <ij| L | a> *****************"
print

l_aij = MatrixElement((Dagger(i), Dagger(j)), LAm1, a)
l_aij_sph = ReducedMatrixElement(SphFermBra('ij', Dagger(i), Dagger(j)), LAm1, a)
_report(Eq(l_aij, l_aij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(l_aij_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <a| T | i> *****************"
print

t_ai = MatrixElement(Dagger(a), T, i)
t_ai_sph = ReducedMatrixElement(Dagger(a), T, i)
_report(Eq(t_ai, t_ai_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(t_ai_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <ab| T | ij> *****************"
print

t_abij = MatrixElement((Dagger(a), Dagger(b)), T, (i, j))
t_abij_sph = ReducedMatrixElement(SphFermBra('ab', Dagger(a), Dagger(b), reverse=0), T, SphFermKet('ij', i, j, reverse=0))
_report(Eq(t_abij, t_abij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(t_abij_sph.get_direct_product_ito_self(tjs=1)))



print
print "*************** <a| R | i> *****************"
print

r_ai = MatrixElement(Dagger(a), RA, i)
r_ai_sph = ReducedMatrixElement(Dagger(a), RA, i)
_report(Eq(r_ai, r_ai_sph.get_direct_product_ito_self(tjs=0)))
_report(Eq(r_ai, r_ai_sph.get_direct_product_ito_self(tjs=1)))

print
_report(fcode(r_ai_sph.get_direct_product_ito_self(tjs=1)))
print
print "*************** <ab| R | ij> *****************"
print

r_abij = MatrixElement((Dagger(a), Dagger(b)), RA, (i, j))
r_abij_sph = ReducedMatrixElement(SphFermBra('ab', Dagger(a), Dagger(b)), RA, SphFermKet('ij', i, j))
_report(Eq(r_abij, r_abij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(r_abij_sph.get_direct_product_ito_self(tjs=1)))


print("*** coupled elements: ***")
_report(Eq(r_ai_sph,rewrite_coupling(r_ai_sph, r_ai)))
_report(Eq(r_abij_sph,rewrite_coupling(r_abij_sph, r_abij)))
_report(Eq(l_i_sph,rewrite_coupling(l_i_sph, l_i)))
_report(Eq(l_aij_sph,rewrite_coupling(l_aij_sph, l_aij)))
_report(Eq(t_ai_sph,rewrite_coupling(t_ai_sph, t_ai)))
_report(Eq(t_abij_sph,rewrite_coupling(t_abij_sph, t_abij)))


l0 = Symbol('l_0')
r0 = Symbol('r_0')
SF = Symbol('SF')




print
print "==== Recoupling <{A-1}|i|{A}>===="
print

j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
braket_assumptions.add(Assume(j_i, 'half_integer'))
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient(J_A, M_A, j_i, -m_i, J_Am1, M_Am1)*ASigma(M_A, m_i)

print "reduction factor"
_report( sf_reduction)
_report(fcode(refine_phases(convert_cgc2tjs(sf_reduction))))

coupled_subs = {
        l_i: rewrite_coupling(l_i, l_i_sph),
        J_A: 0,
        M_A: 0
        }
print
print("recoupling diagram 1")
expr_msc = combine_ASigmas(r0*l_i*sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))

print
_report("recoupling diagram 2")
r_aj = r_ai.subs(i,j)
r_aj_sph = r_ai_sph.subs(i,j)
coupled_subs = {
        r_aj: rewrite_coupling(r_aj, r_aj_sph),
        l_aij: rewrite_coupling(l_aij, l_aij_sph)
        }
j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient(J_A, M_A, j_i, -m_i, J_Am1, M_Am1)*ASigma('M_A', 'm_i')
print
expr_msc = combine_ASigmas(r_aj*l_aij*ASigma('m_a','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))


print
print "==== Recoupling <{A-1}|a|{A}> ==="
print

j_a = Symbol('j_a', nonnegative=True)
m_a = Symbol('m_a')
sf_reduction = (-1)**(j_a - m_a)*ClebschGordanCoefficient(J_A, M_A, j_a, -m_a, J_Am1, M_Am1)*ASigma(M_A, m_a)

print "reduction factor"
_report(sf_reduction)
_report(fcode(refine_phases(convert_cgc2tjs(sf_reduction))))
print
print("recoupling diagram 3")
coupled_subs = {
        t_ai: rewrite_coupling(t_ai, t_ai_sph),
        l_i: rewrite_coupling(l_i, l_i_sph),
        J_A: 0,
        M_A: 0
        }

print
expr_msc = combine_ASigmas(t_ai*l_i*r0*ASigma('m_i') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))

_report("recoupling diagram 4")
coupled_subs = {
        l_i: rewrite_coupling(l_i, l_i_sph),
        r_ai: rewrite_coupling(r_ai, r_ai_sph)
        }
print
expr_msc = combine_ASigmas(l_i*r_ai*ASigma('m_i') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [M_A, m_a])
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))

print
_report("recoupling diagram 5")
l_bij = l_aij.subs(a,b)
l_bij_sph = l_aij_sph.subs(a,b)
coupled_subs = {
        t_abij: rewrite_coupling(t_abij, t_abij_sph),
        l_bij: rewrite_coupling(l_bij, l_bij_sph),
        J_A: 0,
        M_A: 0
        }

print
expr_msc = combine_ASigmas(l_bij*r0*t_abij*ASigma('m_j','m_i','m_b') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_i', 'm_j'])
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
s2d = extract_symbol2dummy_dict(expr_sph)
expr_sph = apply_orthogonality(expr_sph, ['m_a', 'm_b', s2d[S('M_ab')]])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))


print
_report("recoupling diagram 6")
l_bij = l_aij.subs(a, b)
l_bij_sph = l_aij_sph.subs(a, b)
r_bj = r_ai.subs({Dagger(a):Dagger(b), i:j})
r_bj_sph = r_ai_sph.subs({Dagger(a):Dagger(b), i:j})

coupled_subs = {
        t_ai: rewrite_coupling(t_ai, t_ai_sph),
        r_bj: rewrite_coupling(r_bj, r_bj_sph),
        l_bij: rewrite_coupling(l_bij, l_bij_sph)
        }

expr_msc = combine_ASigmas(t_ai*r_bj*l_bij*ASigma('m_b','m_i','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases((expr_sph))
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1, let_pass=0)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))


print
_report("recoupling diagram 7")
l_bij = l_aij.subs(a, b)
l_bij_sph = l_aij_sph.subs(a, b)

coupled_subs = {
        r_abij: rewrite_coupling(r_abij, r_abij_sph),
        l_bij: rewrite_coupling(l_bij, l_bij_sph)
        }

expr_msc = combine_ASigmas(S(1)/2*r_abij*l_bij*ASigma('m_b','m_i','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_i', 'm_j'])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1, let_pass=0)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))
