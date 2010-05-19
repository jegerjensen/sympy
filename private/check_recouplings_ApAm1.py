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
print "*************** <i| L | a> *****************"
print

l_ai = MatrixElement(Dagger(i), LA, a)
l_ai_sph = ReducedMatrixElement(Dagger(i), LA, a)
_report( Eq(l_ai, l_ai_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(l_ai_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <ij| L | ab> *****************"
print

l_abij = MatrixElement((Dagger(i), Dagger(j)), LA, (a, b))
l_abij_sph = ReducedMatrixElement(SphFermBra('ij', Dagger(i), Dagger(j), reverse=0), LA, SphFermKet('ab', a, b, reverse=0))
_report( Eq(l_abij, l_abij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(l_abij_sph.get_direct_product_ito_self(tjs=1)))

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
print "*************** <| R | i> *****************"
print

r_i = MatrixElement(0, RAm1, i)
r_i_sph = ReducedMatrixElement(0, RAm1, i)
_report( Eq(r_i, r_i_sph.get_direct_product_ito_self(tjs=0)))
_report( Eq(r_i, r_i_sph.get_direct_product_ito_self(tjs=1)))

print
_report(fcode(r_i_sph.get_direct_product_ito_self(tjs=1)))
print
print "*************** <a| R | ij> *****************"
print

r_aij = MatrixElement(Dagger(a), RAm1, (i, j))
r_aij_sph = ReducedMatrixElement(Dagger(a), RAm1, SphFermKet('ij', i, j, reverse=0))
_report( Eq(r_aij, r_aij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(r_aij_sph.get_direct_product_ito_self(tjs=1)))

# As Gaute told me on chat:
#  <b|R|ij> = (ji,mi,jj,mj|Jij Mij) (jb,mb,J,M|Jij Mij) <b||R||ij>
J_ij, J_Am1, j_b, j_i, j_j = symbols('J_ij J_Am1 j_b j_i j_j', nonnegative=True)
M_ij, M_Am1, m_b, m_i, m_j = symbols('M_ij M_Am1 m_b m_i m_j')

J_Am1 = RAm1.rank


print
print
print
gautes_raij = (ASigma(m_b, M_Am1, m_i, m_j)
        *MatrixElement(Dagger(b), RAm1, (i, j))
        *ClebschGordanCoefficient(j_b, m_b, J_Am1, M_Am1, J_ij, M_ij)
        *ClebschGordanCoefficient(j_i, m_i, j_j, m_j, J_ij, M_ij)
        )
print "As Gaute told me on chat:"
print ' "<b|R|ij> = (ji,mi,jj,mj|Jij Mij) (jb,mb,J,M|Jij Mij) <b||R||ij>" '
print " so Gautes reduced matrix element can be written"
_report( gautes_raij)
print "The other way around, we can express the direct matrix as"
gautes = Function('Gautes')
direct_as_gautes = ( ASigma(J_ij, M_ij)
        *ClebschGordanCoefficient(j_b, m_b, J_Am1, M_Am1, J_ij, M_ij)
        *ClebschGordanCoefficient(j_i, m_i, j_j, m_j, J_ij, M_ij)
        *gautes(Dagger(b), SphFermKet('ij',i,j))
        )
_report( direct_as_gautes)
print """
Here, we have inserted the R operator directly, but in order to make Gautes
reduced matrix element independent of projection, it is necessary to treat the
R operator as a dual SphericalTensorOperator, transforming like (-1)**(j-m)T(j,-m).
In the wikipedian reduced matrix element, we expect R to transform as a
tensor T(j,m).  Since we prefer to insert the R operator directly in the
wikipedian element, the R operator in gautes must be substituted with
(-1)**(j+m)R(j,-m).  The expression is then """
direct_as_gautes = ( ASigma(J_ij, M_ij)*(-1)**(J_Am1 + M_Am1)
        *ClebschGordanCoefficient(j_b, m_b, J_Am1, -M_Am1, J_ij, M_ij)
        *ClebschGordanCoefficient(j_i, m_i, j_j, m_j, J_ij, M_ij)
        *gautes(Dagger(b), SphFermKet('ij',i,j))
        )
# We should use dummies for summation indices
direct_as_gautes = direct_as_gautes.subs([(J_ij, J_ij.as_dummy()), (M_ij, M_ij.as_dummy())])
_report( direct_as_gautes)
print "The wikipedian reduced matrix element can also be written in terms of the"
print( "direct product:")
mine_as_direct = r_aij_sph.as_direct_product()
_report( mine_as_direct)
subsdict = extract_symbol2dummy_dict(mine_as_direct)
direct_as_gautes = direct_as_gautes.subs(subsdict)
direct_w_dummies = r_aij_sph.get_related_direct_matrix().subs(subsdict)
relation_mine_ito_gautes = mine_as_direct.subs(direct_w_dummies, direct_as_gautes)
print "Inserting the expression for direct in terms of Gautes gives the relation"
_report( relation_mine_ito_gautes)
print "Which we can simplify to"
relation_mine_ito_gautes = apply_orthogonality(relation_mine_ito_gautes, [ subsdict[m_i], subsdict[m_j]])
_report( relation_mine_ito_gautes)
relation_mine_ito_gautes = evaluate_sums(relation_mine_ito_gautes)
_report( relation_mine_ito_gautes)
relation_mine_ito_gautes = convert_cgc2tjs(relation_mine_ito_gautes)
_report( relation_mine_ito_gautes)
subsdict = extract_symbol2dummy_dict(relation_mine_ito_gautes)
relation_mine_ito_gautes = apply_orthogonality(relation_mine_ito_gautes, [ subsdict[M_ij], subsdict[M_Am1]])
_report( relation_mine_ito_gautes)
print
_report(fcode(relation_mine_ito_gautes))




l0 = Symbol('l_0')
r0 = Symbol('r_0')
SF = Symbol('SF')




print
print "==== Recoupling <{A}|a'|{A-1}> ===="
print

j_a = Symbol('j_a', nonnegative=True)
m_a = Symbol('m_a')
braket_assumptions.add(Assume(j_a, 'half_integer'))
sf_reduction = (-1)**(j_a - m_a)*ClebschGordanCoefficient(J_A, M_A, j_a, -m_a, J_Am1, M_Am1)*ASigma(M_A, m_a)

print "reduction factor"
_report( sf_reduction)
_report(fcode(refine_phases(convert_cgc2tjs(sf_reduction))))


coupled_subs = {
        l_ai: rewrite_coupling(l_ai, l_ai_sph),
        r_i: rewrite_coupling(r_i, r_i_sph)
        }
print
print("recoupling diagram 1")
expr_msc = combine_ASigmas(l_ai*r_i*ASigma('m_i') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [M_A, m_a])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
_report( fcode(expr_sph))


print
print("recoupling diagram 2")
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
expr_sph = refine_phases(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1, let_pass=0)
_report(Eq(SF, expr_sph))
_report( fcode(expr_sph))


print
print "==== Recoupling <{A}|i'|{A-1}> ===="
print

j_i = Symbol('j_i', nonnegative=True)
m_i = Symbol('m_i')
sf_reduction = (-1)**(j_i - m_i)*ClebschGordanCoefficient(J_A, M_A, j_i, -m_i, J_Am1, M_Am1)*ASigma(M_A, m_i)


coupled_subs = {
        r_i: rewrite_coupling(r_i, r_i_sph)
        }
print
print("recoupling diagram 3")
expr_msc = combine_ASigmas(l0*r_i*sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = expr_sph.subs([(J_A, 0), (M_A,0)])
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
_report( fcode(expr_sph))



print
print("recoupling diagram 4")
l_aj = l_ai.subs(Dagger(i),Dagger(j))
l_aj_sph = l_ai_sph.subs(Dagger(i),Dagger(j))
coupled_subs = {
        l_aj: rewrite_coupling(l_aj, l_aj_sph),
        r_aij: rewrite_coupling(r_aij, r_aij_sph)
        }
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
_report( fcode(expr_sph))

print
print("recoupling diagram 5")
l_aj = l_ai.subs(Dagger(i),Dagger(j))
l_aj_sph = l_ai_sph.subs(Dagger(i),Dagger(j))
r_j = r_i.subs(i,j)
r_j_sph = r_i_sph.subs(i,j)
coupled_subs = {
        t_ai: rewrite_coupling(t_ai, t_ai_sph),
        l_aj: rewrite_coupling(l_aj, l_aj_sph),
        r_j: rewrite_coupling(r_j, r_j_sph)
        }
print
expr_msc = combine_ASigmas(l_aj*r_j*t_ai*ASigma('m_a','m_j') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [M_A, m_a])
_report(Eq(SF, expr_sph))
_report( fcode(expr_sph))


print
print("recoupling diagram 6")
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
print
expr_msc = combine_ASigmas(l_abjk*r_bjk*t_ai*ASigma('m_a','m_j','m_k','m_b') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_j', 'm_k'])
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
_report(fcode(expr_sph))




print
print("recoupling diagram 7")
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
print
expr_msc = combine_ASigmas(l_abjk*r_j*t_abik*ASigma('m_a','m_j','m_k','m_b') * sf_reduction)
_report(Eq(SF, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(SF, expr_sph))
expr_sph = apply_orthogonality(expr_sph, ['m_b', 'm_a'])
_report(Eq(SF, expr_sph))
expr_sph = convert_cgc2tjs(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = evaluate_sums(expr_sph, all_deltas=True)
_report(Eq(SF, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(SF, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(SF, expr_sph))
_report( fcode(expr_sph))

