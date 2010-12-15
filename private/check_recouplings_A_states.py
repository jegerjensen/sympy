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

Sum=ASigma

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

RA = SphericalTensorOperator('R', J_A, M_A)
RAm1 = SphericalTensorOperator('R', J_Am1, M_Am1)
LA = DualSphericalTensorOperator('L', J_A, M_A)
LAm1 = DualSphericalTensorOperator('L', J_Am1, M_Am1)
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
x_kcjb_sph = ReducedMatrixElement(SphFermBra('kc', Dagger(k), c, reverse=0), X, SphFermKet('jb', Dagger(b), j, reverse=1))
_report(Eq(x_kcjb, x_kcjb_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(x_kcjb_sph.get_direct_product_ito_self(tjs=1)))

print
print "*************** <k b| X | i j> *****************"
print

x_kbij = MatrixElement((Dagger(k), Dagger(b)), X, (i, j))
x_kbij_sph = ReducedMatrixElement(SphFermBra('kb', Dagger(k), Dagger(b), reverse=0), X, SphFermKet('ij', i,j, reverse=0))
_report(Eq(x_kbij, x_kbij_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(x_kbij_sph.get_direct_product_ito_self(tjs=1)))


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
print "*************** <a| R | i> *****************"
print

r_ai = MatrixElement(Dagger(a), RAm1, i)
r_ai_sph = ReducedMatrixElement(Dagger(a), RA, i, definition='wikipedia')
_report( Eq(r_ai, r_ai_sph.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(r_ai_sph.get_direct_product_ito_self(tjs=1)))
print
print "*************** <a| R | ij> *****************"
print

r_abij = MatrixElement((Dagger(a),Dagger(b)), RA, (i, j))
r_abij_sph = ReducedMatrixElement(SphFermBra('ab', Dagger(a), Dagger(b)),
        RA, SphFermKet('ij', i, j, reverse=0), definition='wikipedia')
_report( Eq(r_abij, r_abij_sph.get_direct_product_ito_self(tjs=0)))
r_abij_cross = ReducedMatrixElement(SphFermBra('aj', Dagger(a), j, reverse=True), RAm1, SphFermKet('bi', b, i))
_report( Eq(r_abij, r_abij_cross.get_direct_product_ito_self(tjs=0)))

print
_report(fcode(r_abij_sph.get_direct_product_ito_self(tjs=1)))

J_ij, J_Am1, j_a, j_i, j_j, j_b, j_c, j_k = symbols('J_ij J_Am1 j_a j_i j_j j_b j_c j_k', nonnegative=True)
M_ij, M_Am1, m_a, m_i, m_j, m_b, m_c, m_k = symbols('M_ij M_Am1 m_a m_i m_j m_b m_c m_k')

print
print "****************** Right A, r_ai = X_kc*r_acik *************"
print

diagram = Symbol('XkcR2')

x_kc = MatrixElement(Dagger(k), X, c)
x_kc_sph = ReducedMatrixElement(Dagger(k), X, c)
r_acik = MatrixElement((Dagger(a),Dagger(c)), RA, (i, k))
r_acik_sph = ReducedMatrixElement(SphFermBra('ac', Dagger(a), Dagger(c)),
        RA, SphFermKet('ik', i, k, reverse=0), definition='wikipedia')
_report( Eq(x_kc, x_kc_sph.get_direct_product_ito_self(tjs=0)))
_report( Eq(r_acik, r_acik_sph.get_direct_product_ito_self(tjs=0)))

coupling_substitution = {
        r_acik: rewrite_coupling(r_acik, r_acik_sph),
        x_kc: rewrite_coupling(x_kc, x_kc_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
lhs_coupling_factor = r_ai_sph.as_direct_product(use_dummies=False).subs(r_ai, 1)
expr_msc = lhs_coupling_factor*r_acik*x_kc*Sum(m_c)
expr_sph = expr_msc.subs(coupling_substitution)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [m_a, m_b])
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

sys.exit()

print
print "****************** Right A, r_abij = X_kbij*r_ak **************"
print

diagram = Symbol('I4_R1')

r_ak = MatrixElement(Dagger(a), RA, k)
r_ak_sph = ReducedMatrixElement(Dagger(a), RA, k, definition='wikipedia')
_report( Eq(r_ak, r_ak_sph.get_direct_product_ito_self(tjs=0)))
_report( Eq(x_kbij, x_kbij_sph.get_direct_product_ito_self(tjs=0)))

coupling_substitution = {
        x_kbij: rewrite_coupling(x_kbij, x_kbij_sph),
        r_ak: rewrite_coupling(r_ak, r_ak_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
lhs_coupling_factor = r_abij_sph.as_direct_product(use_dummies=False).subs(r_abij, 1)
expr_msc = lhs_coupling_factor*x_kbij*r_ak*Sum(m_k)
expr_sph = expr_msc.subs(coupling_substitution)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [m_i, m_j])
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

print
print "****************** Right A, r_abij = X_abic*r_cj **************"
print

diagram = Symbol('I4_R1')

x_abic = MatrixElement((Dagger(a), Dagger(b)), X, (i, c))
x_abic_sph = ReducedMatrixElement(SphFermBra('ab', Dagger(a), Dagger(b), reverse=0), X, SphFermKet('ic', i,c, reverse=0))
r_cj = MatrixElement(Dagger(c), RA, j)
r_cj_sph = ReducedMatrixElement(Dagger(c), RA, j, definition='wikipedia')
_report( Eq(r_cj, r_cj_sph.get_direct_product_ito_self(tjs=0)))
_report( Eq(x_abic, x_abic_sph.get_direct_product_ito_self(tjs=0)))

coupling_substitution = {
        x_abic: rewrite_coupling(x_abic, x_abic_sph),
        r_cj: rewrite_coupling(r_cj, r_cj_sph, verbose=1, strict=1)
        }
print
print("recoupling diagram 1")
lhs_coupling_factor = r_abij_sph.as_direct_product(use_dummies=False).subs(r_abij, 1)
expr_msc = lhs_coupling_factor*x_abic*r_cj*Sum(m_c)
expr_sph = expr_msc.subs(coupling_substitution)
_report(Eq(diagram, expr_sph))
expr_sph = refine_phases(convert_cgc2tjs(expr_sph))
_report(Eq(diagram, expr_sph))
expr_sph = apply_orthogonality(expr_sph, [m_a, m_b])
_report(Eq(diagram, expr_sph))
expr_sph = evaluate_sums(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = apply_deltas(expr_sph)
_report(Eq(diagram, expr_sph))
expr_sph = refine_tjs2sjs(expr_sph)
_report(Eq(diagram, expr_sph))

_report(fcode(expr_sph))

