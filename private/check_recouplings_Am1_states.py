from sympy.physics.racahalgebra import (
        refine_phases, refine_tjs2sjs, convert_cgc2tjs, is_equivalent,
        SixJSymbol, ASigma, combine_ASigmas, evaluate_sums, apply_deltas,
        apply_orthogonality, ClebschGordanCoefficient, extract_symbol2dummy_dict,
        extract_dummy2symbol_dict, AngularMomentumSymbol
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

    pprint(str(expr))


    # subsdict = extract_dummy2symbol_dict(expr)
    # expr = expr.subs(subsdict)
    # print latex(expr, mode='dmath', mul_symbol='dot')

    print



class ReducedMatrixElement_gaute(ReducedMatrixElement):
    _definition = 'gaute'

    def get_reduction_factor(self, **kw_args):
        left,op,right = self.args
        c_bra, t_bra = left.as_coeff_tensor()
        j1, m1 = t_bra.get_rank_proj()
        c_op, t_op = op.as_coeff_tensor()
        j2, m2 = t_op.get_rank_proj()
        J, M = right._j, right._m

        factor = c_op*c_bra*(-1)**(J - M)*ClebschGordanCoefficient(j1, m1, j2, m2, J, -M)

        if kw_args.get('tjs'):
            return refine_phases(convert_cgc2tjs(factor))
        else:
            return factor

    def get_inverse_reduction_factor(self, **kw_args):
        from sympy.physics.racahalgebra import convert_sumindex2dummy
        left,op,right = self.args
        c_bra, t_bra = left.as_coeff_tensor()
        j1, m1 = t_bra.get_rank_proj()
        c_op, t_op = op.as_coeff_tensor()
        j2, m2 = t_op.get_rank_proj()
        J, M = right._j, right._m

        factor = ASigma(m1, m2)*(-1)**(J-M)*ClebschGordanCoefficient(j1, m1, j2, m2, J, -M)/c_bra/c_op

        if kw_args.get('use_dummies', True):
            factor = convert_sumindex2dummy(factor)
        if kw_args.get('tjs'):
            factor = refine_phases(convert_cgc2tjs(factor))
        return factor

import sympy
sympy.physics.braket.ReducedMatrixElement_gaute = ReducedMatrixElement_gaute


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

LA = SphericalTensorOperator('L', J_A, M_A)
LAm1 = SphericalTensorOperator('L', J_Am1, M_Am1)
RA = DualSphericalTensorOperator('R', J_A, M_A)
RAm1 = DualSphericalTensorOperator('R', J_Am1, M_Am1)
X = SphericalTensorOperator('X', S.Zero, S.Zero)
T = SphericalTensorOperator('T', S.Zero, S.Zero)

print
print "Defined tensor operators:"
_report( LA)
_report( RA)
_report( LAm1)
_report( RAm1)
_report( X)
_report( T)


print
print "*************** <j| X | i> *****************"
print

x_kj = MatrixElement(Dagger(k), X, j)
x_kj_sph = ReducedMatrixElement(Dagger(k), X, j)
_report(Eq(x_kj, x_kj_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(x_kj_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <k| X | c> *****************"
print

x_ja = MatrixElement(Dagger(j), X, a)
x_ja_sph = ReducedMatrixElement(Dagger(j), X, a)
_report(Eq(x_ja, x_ja_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(x_ja_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <k -c| X | j -b> *****************"
print

x_kcjb = MatrixElement((Dagger(k), Dagger(b)), X, (c, j))
# x_kcjb_sph = ReducedMatrixElement(SphFermBra('ck', Dagger(k),-Dagger(c), reverse=1), X, SphFermKet('jb', j, -b, reverse=0))
x_kcjb_sph = ReducedMatrixElement(SphFermBra('kc', Dagger(k), c, reverse=0), X, SphFermKet('jb', Dagger(b), j, reverse=1))
_report(Eq(x_kcjb, x_kcjb_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(x_kcjb_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <j c| X | k b> *****************"
print

x_jckb = MatrixElement((Dagger(j), Dagger(c)), X, (k, b))
x_jckb_sph = ReducedMatrixElement(SphFermBra('jc', Dagger(j), Dagger(c), reverse=0), X, SphFermKet('kb', k,b, reverse=0))
_report(Eq(x_jckb, x_jckb_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(x_jckb_sph.get_direct_product_ito_self(tjs=1)))


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
print "*************** <i| L | > *****************"
print

l_i = MatrixElement(Dagger(i), LAm1, "")
l_i_sph = ReducedMatrixElement(Dagger(i), LAm1, "", definition='wikipedia')
_report( Eq(l_i, l_i_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(l_i_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <ij| L | a> *****************"
print

l_aij = MatrixElement((Dagger(i), Dagger(j)), LAm1, a)
l_aij_sph = ReducedMatrixElement(SphFermBra('ij', Dagger(i), Dagger(j)), LAm1, a, definition='wikipedia')
_report(Eq(l_aij, l_aij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(l_aij_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <| R | i> *****************"
print

r_i = MatrixElement(0, RAm1, i)
r_i_sph = ReducedMatrixElement(0, RAm1, i, definition='gaute')
_report( Eq(r_i, r_i_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(r_i_sph.get_direct_product_ito_self(tjs=1)))
print
print "*************** <a| R | ij> *****************"
print

r_aij = MatrixElement(Dagger(a), RAm1, (i, j))
r_aij_sph = ReducedMatrixElement(Dagger(a), RAm1, SphFermKet('ij', i, j, reverse=0), definition='gaute')
_report( Eq(r_aij, r_aij_sph.get_direct_product_ito_self(tjs=0)))
r_aij_cross = ReducedMatrixElement(SphFermBra('aj', Dagger(a), j, reverse=True), RAm1, i, definition='gaute')
_report( Eq(r_aij, r_aij_cross.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(r_aij_sph.get_direct_product_ito_self(tjs=1)))

J_ij, J_Am1, j_a, j_i, j_j, j_b, j_c, j_k = symbols('J_ij J_Am1 j_a j_i j_j j_b j_c j_k', nonnegative=True)
M_ij, M_Am1, m_a, m_i, m_j, m_b, m_c, m_k = symbols('M_ij M_Am1 m_a m_i m_j m_b m_c m_k')


print
print "****************** Left Am1, I5 to Lbij **************"
print

diagram = Symbol('I5_L2')

coupled_subs = {
        x_ja: rewrite_coupling(x_ja, x_ja_sph),
        l_i: rewrite_coupling(l_i, l_i_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
cgc_factor = l_aij_sph.as_direct_product(use_dummies=0).subs(l_aij, 1)
_report(cgc_factor)
expr_msc = l_i*x_ja*cgc_factor
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(expr_sph, [j_a])
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [m_a, M_Am1])
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))


print
print "****************** Right Am1, I4 only regular coupling **************"
print

diagram = Symbol('I4_R')
M_aj = Symbol('M_aj')

r_bij_sph = r_aij_sph.subs(Dagger(a), Dagger(b))
r_bij = r_aij.subs(Dagger(a), Dagger(b))
# r_cik = r_aij.subs([(Dagger(a), Dagger(c)), (j, k)])
# r_cik_sph = r_aij_sph.subs([(Dagger(a), Dagger(c)), (j, k)])
r_cik = MatrixElement(Dagger(c), RAm1, (i, k))
r_cik_sph = ReducedMatrixElement(Dagger(c), RAm1, SphFermKet('ik', i, k, reverse=0), definition='gaute')

coupled_subs = {
        x_jckb: rewrite_coupling(x_jckb, x_jckb_sph),
        r_cik: rewrite_coupling(r_cik, r_cik_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
print("Coefficient from left hand side is")
cgc_factor = r_bij_sph.as_direct_product(use_dummies=0).subs(r_bij, 1)
_report(cgc_factor)
expr_msc = r_cik*x_jckb*ASigma(m_k, m_c)*cgc_factor
expr_msc = combine_ASigmas(expr_msc)
_report(Eq(diagram, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(expr_sph, [M_ij, M_Am1, m_a, m_i, m_j, m_b, m_c, m_k])
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))


stop




print
print "****************** Right Am1, I7 to Rbij **************"
print

diagram = Symbol('I7_R2')

r_bij = r_aij.subs(Dagger(a), Dagger(b))
r_bik = r_aij.subs([(Dagger(a), Dagger(b)), (j, k)])

r_bij_sph = r_aij_sph.subs(Dagger(a), Dagger(b))
r_bik_sph = r_aij_sph.subs([(Dagger(a), Dagger(b)), (j, k)])

coupled_subs = {
        x_kj: rewrite_coupling(x_kj, x_kj_sph),
        r_bik: rewrite_coupling(r_bik, r_bik_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
cgc_factor = r_bij_sph.as_direct_product(use_dummies=0).subs(r_bij, 1)
_report(cgc_factor)
expr_msc = r_bik*x_kj*ASigma(m_k, j_k)*cgc_factor
expr_msc = combine_ASigmas(expr_msc)
_report(Eq(diagram, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [m_j, m_i, m_b, M_Am1])
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(expr_sph)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

print
print "****************** Right Am1, I4 **************"
print

diagram = Symbol('I4_R')
M_aj = Symbol('M_aj')

r_bij_cross = r_aij_cross.subs(Dagger(a), Dagger(b))
r_cik = r_aij.subs([(Dagger(a), Dagger(c)), (j, k)])
r_cik_sph = r_aij_cross.subs([(Dagger(a), Dagger(c)), (j, k)])
# r_cik_sph = r_aij_sph.subs([(Dagger(a), Dagger(c)), (j, k)])

coupled_subs = {
        x_kcjb: rewrite_coupling(x_kcjb, x_kcjb_sph),
        r_cik: rewrite_coupling(r_cik, r_cik_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
cgc_factor = r_bij_cross.as_direct_product(use_dummies=0).subs(r_bij_cross.get_related_direct_matrix(), 1)
_report(cgc_factor)
expr_msc = r_cik*x_kcjb*ASigma(m_k, m_c)*cgc_factor
expr_msc = combine_ASigmas(expr_msc)
_report(Eq(diagram, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
junk, angmom_symbs = expr_sph.as_coeff_terms(AngularMomentumSymbol)
expr_sph = apply_orthogonality(expr_sph, [m_c, m_k, m_b, m_j])
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [M_Am1, M_aj])
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

print
print "****************** Right Am1, I4 Direct to 6j-symbol **************"
print

diagram = Symbol('I4_R')

r_bij = r_aij.subs(Dagger(a), Dagger(b))
r_cik = r_aij.subs([(Dagger(a), Dagger(c)), (j, k)])

r_bij_sph = r_aij_sph.subs(Dagger(a), Dagger(b))
r_cik_sph = r_aij_cross.subs([(Dagger(a), Dagger(c)), (j, k)])

coupled_subs = {
        x_kcjb: rewrite_coupling(x_kcjb, x_kcjb_sph),
        r_cik: rewrite_coupling(r_cik, r_cik_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
cgc_factor = r_bij_sph.as_direct_product(use_dummies=0).subs(r_bij_sph.get_related_direct_matrix(), 1)
_report(cgc_factor)
expr_msc = r_cik*x_kcjb*ASigma(m_k, m_c)*cgc_factor
expr_msc = combine_ASigmas(expr_msc)
_report(Eq(diagram, expr_msc))
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
junk, angmom_symbs = expr_sph.as_coeff_terms(AngularMomentumSymbol)
expr_sph = apply_orthogonality(expr_sph, [m_c, m_k])
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(expr_sph, identity_sources=angmom_symbs, try_hard=True)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

print
print "****************** Check recoupling of right Am1 ************"
print

print "******* cross -> regular ******"
print

diagram = r_aij_sph

expr_sph = rewrite_coupling(r_aij_sph, r_aij_cross)
_report(Eq(diagram, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

print
print "******* regular -> cross ******"
print
diagram = r_aij_cross

expr_sph = rewrite_coupling(r_aij_cross, r_aij_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph, verbose=1)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))


print
print "****************** Norm of Left and right Am1 **************"
print

r_i = MatrixElement(0, RAm1, i)
r_i_sph = ReducedMatrixElement(0, RAm1, i, definition='wikipedia')
_report( Eq(r_i, r_i_sph.get_direct_product_ito_self(tjs=0)))

r_aij = MatrixElement(Dagger(a), RAm1, (i, j))
r_aij_sph = ReducedMatrixElement(Dagger(a), RAm1, SphFermKet('ij', i, j, reverse=0), definition='wikipedia')
_report( Eq(r_aij, r_aij_sph.get_direct_product_ito_self(tjs=0)))

diagram = Symbol('norm_1')

coupled_subs = {
        l_i: rewrite_coupling(l_i, l_i_sph, verbose=1, strict=1),
        r_i: rewrite_coupling(r_i, r_i_sph, verbose=1, strict=1)
        }
print
print("norm of r1, l1")
expr_msc = l_i*r_i*ASigma(m_i)
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(expr_sph, [j_a])
_report(Eq(diagram, expr_sph))
_report(fcode(expr_sph))


diagram = Symbol('norm_2')

coupled_subs = {
        l_aij: rewrite_coupling(l_aij, l_aij_sph, verbose=1, strict=1),
        r_aij: rewrite_coupling(r_aij, r_aij_sph, verbose=1, strict=1)
        }
print
print("norm of r2, l2")
expr_msc = l_aij*r_aij*ASigma(m_i,m_j,m_a)
expr_sph = expr_msc.subs(coupled_subs)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [m_j, m_i, m_b, M_Am1])
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [], all=True, mode='projections')
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(expr_sph, [j_a])
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))
