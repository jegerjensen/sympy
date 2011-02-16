from sympy import (
        Basic, Function, Mul, sympify, Integer, Add, sqrt, Pow, Symbol, latex,
        cache, powsimp, symbols, Rational, S
        )
from sympy.functions import Dij
from sympy.assumptions import (
        register_handler, remove_handler, Q, ask, Assume, refine, global_assumptions
        )
from sympy.physics.racahalgebra import (
        SixJSymbol, ThreeJSymbol, ClebschGordanCoefficient, refine_tjs2sjs,
        convert_cgc2tjs, convert_tjs2cgc, ASigma, SphericalTensor,
        CompositeSphericalTensor, AtomicSphericalTensor, is_equivalent,
        _standardize_coeff, evaluate_sums, NineJSymbol, _identify_NineJSymbol
        )
from sympy.utilities.pytest import raises


def test_half_integer_ask_handler():
    x = Symbol('x')
    y = Symbol('y')
    assert ask(x,'real', Assume(x,'half_integer')) == None
    assert ask(x,'half_integer', Assume(x,'half_integer')) == True
    assert ask(x,'half_integer', Assume(x,'half_integer', False)) == False
    assert ask(x,'half_integer', Assume(x,'integer')) == False
    assert ask(x,'integer', Assume(x,'half_integer')) == False
    assert ask(2*x,Q.even, Assume(x,'half_integer')) == False
    assert ask(2*x,Q.odd, Assume(x,'half_integer')) == True
    assert ask(y + x, Q.integer, Assume(x,'half_integer')
                               & Assume(y,'half_integer')) == True


def test_evaluate_sums():
    f = Function('f')
    i,j,k,l,m,n = symbols('i j k l m n')
    fn = f(i,j,k,l,m,n)

    assert evaluate_sums(ASigma(i,j,k)*fn*Dij(k,m)) == ASigma(i,j)*f(i,j,m,l,m,n)
    assert evaluate_sums(ASigma(i,j,k)*fn*Dij(i,l)*Dij(k,m)) == ASigma(j)*f(l,j,m,l,m,n)
    assert evaluate_sums(ASigma(i,j,k)*fn*Dij(i,k)*Dij(k,m)) == ASigma(j)*f(m,j,m,l,m,n)
    assert evaluate_sums(ASigma(i,j,k)*fn*Dij(i,m)*Dij(k,i)) == ASigma(j)*f(m,j,m,l,m,n)


def test_tjs_methods():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

    assert (0 < A)
    assert (S.Zero < A)

    # canonical ordering
    assert (ThreeJSymbol(A, C, B, a, c, b) == (-1)**(A + B + C)*ThreeJSymbol(A, B, C, a, b, c))
    assert ( ThreeJSymbol(A, C, B, -a, c, b) == ThreeJSymbol(A, B, C, a, -b, -c))
    assert ( ThreeJSymbol(A, C, B, -2*a, c, b) == ThreeJSymbol(A, B, C, 2*a, -b, -c))

    # methods
    tjs = ThreeJSymbol(A,B,C,a,-b,3*c)
    assert tjs.magnitudes == (A, B, C)
    assert tjs.projections == (a, -b, 3*c)
    assert tjs.get_projection(A) == a
    assert tjs.get_projection(B) == -b
    assert tjs.get_projection(C) == 3*c
    assert tjs.get_projection_symbol(A) == a
    assert tjs.get_projection_symbol(B) == b
    assert tjs.get_projection_symbol(C) == c
    assert tjs.get_magnitude_projection_dict() == { A:a, B:-b, C:3*c }

    assert tjs.get_as_ClebschGordanCoefficient() == (-1)**(A-B-3*c)*ClebschGordanCoefficient(A, a, B, -b, C, -3*c)/sqrt(2*C+1)

def test_coupling_direction():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')
    assert ThreeJSymbol(A,B,C,a,b,c) == powsimp((-1)**(-A-B-C)*ThreeJSymbol(B,A,C,b,a,c))


def test_determine_best_phase():
    from sympy.physics.racahalgebra import _determine_best_phase, __all_phases_seen_in_search
    a,b,c,d,e,f = symbols('a b c d e f')
    __all_phases_seen_in_search.clear()
    __all_phases_seen_in_search |= set([a+b, a+b+c, c])
    assert _determine_best_phase(set(),set()) == c
    __all_phases_seen_in_search |= set([a+b, a+b+d, c])
    assert _determine_best_phase(set([c]),set()) == a+b
    __all_phases_seen_in_search |= set([a+b, a+b+d, c])
    assert _determine_best_phase(set(),set([d])) == a+b +d
    assert len(__all_phases_seen_in_search) == 0


def test_sjs_methods():
    a,b,d,e,f,z = symbols('abdefz')
    A,B,D,E,F,Z = symbols('ABDEFZ')

    # canonical ordering
    assert SixJSymbol(B, A, Z, E, D, F) == SixJSymbol(A, F, E, D, Z, B)

    global_assumptions.add( Assume(a, 'half_integer') )
    global_assumptions.add( Assume(b, 'half_integer') )
    global_assumptions.add( Assume(d, 'half_integer') )
    global_assumptions.add( Assume(A, 'half_integer') )
    global_assumptions.add( Assume(B, 'half_integer') )
    global_assumptions.add( Assume(D, 'half_integer') )
    global_assumptions.add( Assume(e, Q.integer) )
    global_assumptions.add( Assume(f, Q.integer) )
    global_assumptions.add( Assume(E, Q.integer) )
    global_assumptions.add( Assume(F, Q.integer) )
    global_assumptions.add( Assume(Z, 'half_integer') )
    global_assumptions.add( Assume(z, 'half_integer') )

    sjs = SixJSymbol(A, B, E, D, Z, F);
    assert sjs.get_ito_ThreeJSymbols((a,b,e,d,z,f), definition='brink_satchler') == (-1)**(A + E + Z - a - e - z)*ASigma(a, b, z, d, e, f)*ThreeJSymbol(A, F, Z, a,  f, -z)*ThreeJSymbol(Z, D, E, z, d, -e)*ThreeJSymbol(E, B, A, e,  b, -a)*ThreeJSymbol(B, D, F, b, d, f)

    global_assumptions.clear()

def test_cgc_methods():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

    assert ClebschGordanCoefficient(A, a, B, b, C, c).get_as_ThreeJSymbol() == (-1)**(A + c - B)*ThreeJSymbol(A, B, C, a, b, -c)*sqrt(1 + 2*C)

def test_9jsymbol():
    a,b,c,d,e,f,g,h,i = symbols('abcdefghi')
    njs = NineJSymbol(a, b, c, d, e, f, g, h, i)
    assert njs.reflect_major_diagonal() == NineJSymbol(a, d, g, b, e, h, c, f, i)
    assert njs.reflect_minor_diagonal() == NineJSymbol(i, f, c, h, e, b, g, d, a)
    assert njs.permute_rows(0, 1) == NineJSymbol(d, e, f, a, b, c, g, h, i)*(-1)**(Add(*njs.args))
    assert njs.permute_rows(0, 2) == NineJSymbol(g, h, i, d, e, f, a, b, c)*(-1)**(Add(*njs.args))
    assert njs.permute_rows(1, 2) == NineJSymbol(a, b, c, g, h, i, d, e, f)*(-1)**(Add(*njs.args))
    assert njs.permute_columns(0, 1) == NineJSymbol(b, a, c, e, d, f, h, g, i)*(-1)**(Add(*njs.args))
    assert njs.permute_columns(0, 2) == NineJSymbol(c, b, a, f, e, d, i, h, g)*(-1)**(Add(*njs.args))
    assert njs.permute_columns(1, 2) == NineJSymbol(a, c, b, d, f, e, g, i, h)*(-1)**(Add(*njs.args))
    assert njs == njs.permute_rows(1, 1)
    assert njs == njs.permute_columns(1, 1)

def test_9jsymbol_canonicalization():
    a,b,c,d,e,f,g,h,i = sorted(symbols('abcdefghi'))
    njs = NineJSymbol(a, b, c, d, e, f, g, h, i)
    assert njs.canonical_form() == NineJSymbol(a, b, c, d, e, f, g, h, i)
    njs = NineJSymbol(d, e, f, a, b, c, g, h, i)
    assert njs.canonical_form() == NineJSymbol(a, b, c, d, e, f, g, h, i)*(-1)**(Add(*njs.args))
    njs = NineJSymbol(i, b, g, d, e, f, c, h, a)
    assert njs.canonical_form() == NineJSymbol(a, h, c, f, e, d, g, b, i)
    njs = NineJSymbol(a, b, c, d, e, f, g, i, h)
    assert njs.canonical_form() == NineJSymbol(a, c, b, d, f, e, g, h, i)*(-1)**(Add(*njs.args))
    njs = NineJSymbol(a, b, c, d, e, i, g, h, f)
    assert njs.canonical_form() == NineJSymbol(a, b, c, g, h, f, d, e, i)*(-1)**(Add(*njs.args))
    njs = NineJSymbol(a, b, c, d, i, f, g, h, e)
    assert njs.canonical_form() == NineJSymbol(a, c, b, g, e, h, d, f, i)

def test_9jsymbol_identification():
    a,b,c,d,e,f,g,h,i = sorted(symbols('abcdefghi'))
    A,B,C,D,E,F,G,H,I = sorted(symbols('ABCDEFGHI'))
    njs = NineJSymbol(A, B, C, D, E, F, G, H, I)
    expr = njs.get_ito_ThreeJSymbols([a, b, c, d, e, f, g, h, i])
    assert _identify_NineJSymbol(expr.atoms(ThreeJSymbol)) == njs

def test_ASigma():
    a,b,c = symbols('abc')
    assert ASigma(b,c,a).args == (a,b,c)
    assert str(ASigma(a,b,c)) == 'Sum(a, b, c)'
    assert ASigma(a,b) == ASigma(a,-b)
    raises(ValueError, "ASigma(a,2*b*a)")

def test_SphericalTensor_creation():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

    assert isinstance(SphericalTensor('T', A, a), AtomicSphericalTensor)
    B = SphericalTensor('t', B, b)
    C = SphericalTensor('t', C, c)
    assert isinstance(SphericalTensor('T', A, a, B, C), CompositeSphericalTensor)

def test_SphericalTensor_methods():
    a = symbols('a')
    A = symbols('A')
    t = SphericalTensor('T', A, a)
    assert t.rank == A
    assert t.projection == a
    assert t.symbol == Symbol('T')

def test_AtomicSphericalTensor_methods():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

    t = SphericalTensor('T', A, a)
    assert t.as_direct_product() == t
    assert t.get_direct_product_ito_self() == t
    assert str(t) == 'T(A, a)'

def test_CompositeSphericalTensor_methods():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

    t1 = SphericalTensor('t1', A, a)
    t2 = SphericalTensor('t2', B, b)
    T = SphericalTensor('T',C, c, t1, t2)
    assert T.tensor1 == t1
    assert T.tensor2== t2
    assert str(T) == 'T[t1(A)*t2(B)](C, c)'
    assert T.as_direct_product(use_dummies=False) == ASigma(a, b)*ClebschGordanCoefficient(A,a,B,b,C,c)*t1*t2
    assert T.get_direct_product_ito_self(use_dummies=False) == ASigma(C, c)*ClebschGordanCoefficient(A,a,B,b,C,c)*T

def test_CompositeSphericalTensor_nestings():
    a,b,c,d,e,f = symbols('abcdef')
    A,B,C,D,E,F = symbols('ABCDEF')

    t1 = SphericalTensor('t1', A, a)
    t2 = SphericalTensor('t2', B, b)
    t3 = SphericalTensor('t3',C, c)
    T = SphericalTensor('T',D, d, t1, t2)
    U = SphericalTensor('U',E, e, T, t3)
    assert U.as_direct_product(use_dummies=False) == ASigma(a, b, c, d)*t1*t2*t3*ClebschGordanCoefficient(A, a, B, b, D, d)*ClebschGordanCoefficient(D, d, C, c, E, e)
    assert U.as_direct_product(deep=False, use_dummies=False) == ASigma(c, d)*T*t3*ClebschGordanCoefficient(D, d, C, c, E, e)

def test_3b_coupling_schemes():
    a,b,c,d,e,f,g = symbols('abcdefg')
    A,B,C,D,E,F,G = symbols('ABCDEFG')

    t1 = SphericalTensor('t1', A, a)
    t2 = SphericalTensor('t2', B, b)
    t3 = SphericalTensor('t3', C, c)
    T12 = SphericalTensor('T12', D, d, t1, t2)
    T23 = SphericalTensor('T23', F, f, t2, t3)

    U = SphericalTensor('U',E, e, T12, t3)
    V = SphericalTensor('V',G, g, t1, T23)

    assert V.as_other_coupling(U, use_dummies=False) == (
            ASigma(D, a, b, c, d, f)
            *ClebschGordanCoefficient(A, a, B, b, D, d)
            *ClebschGordanCoefficient(A, a, F, f, G, g)
            *ClebschGordanCoefficient(B, b, C, c, F, f)
            *ClebschGordanCoefficient(D, d, C, c, E, e)
            *U).subs({E: G, e: g})

def test_4b_coupling_schemes():
    a,b,c,d,e,f,g,h,y,z = symbols('abcdefghyz')
    A,B,C,D,E,F,G,H,Y,Z = symbols('ABCDEFGHYZ')

    l1 = SphericalTensor('l1', A, a)
    l2 = SphericalTensor('l2', B, b)
    s1 = SphericalTensor('s1', C, c)
    s2 = SphericalTensor('s2', D, d)

    L  = SphericalTensor( 'L', E, e, l1, l2)
    U  = SphericalTensor( 'U', F, f, s1, s2)
    j1 = SphericalTensor('j1', G, g, l1, s1)
    j2 = SphericalTensor('j2', H, h, l2, s2)

    Tls = SphericalTensor('Tls', Y, y,  L,  U)
    Tjj = SphericalTensor('Tjj', Z, z, j1, j2)

    assert Tls.as_other_coupling(Tjj, use_dummies=False) == (
            ASigma(a, b, c, d, e, f, g, h, G, H)
            *ClebschGordanCoefficient(A, a, C, c, G, g) # j1
            *ClebschGordanCoefficient(B, b, D, d, H, h) # j2
            *ClebschGordanCoefficient(A, a, B, b, E, e) # L
            *ClebschGordanCoefficient(C, c, D, d, F, f) # U
            *ClebschGordanCoefficient(E, e, F, f, Y, y) # Tls
            *ClebschGordanCoefficient(G, g, H, h, Z, z) # Tjj
            *Tjj).subs({Z: Y, z: y})

def test_standardize_coeff():
    a,b,c = symbols('a b c')
    global_assumptions.clear()
    global_assumptions.add( Assume( a, 'half_integer') )
    global_assumptions.add( Assume( b, 'integer') )
    assert _standardize_coeff(3*a) == -a
    assert _standardize_coeff(-2*a) == 2*a
    assert _standardize_coeff(-4*a) == S.Zero
    assert _standardize_coeff(-4*a, True) == 4*a

    assert _standardize_coeff( 3*b) == b
    assert _standardize_coeff(-3*b) == b
    assert _standardize_coeff(-2*b) == S.Zero
    assert _standardize_coeff(-2*b, True) == 2*b

    assert _standardize_coeff( 3*c) == 3*c
    assert _standardize_coeff(-3*c) == -3*c
    assert _standardize_coeff(-2*c) == -2*c
    assert _standardize_coeff(-4*c) == -4*c

def test_refine_tjs2sjs():
    m1, m2, M, M12, M23, m3 = symbols('m1 m2 M M12 M23 m3')
    j1, j2, J, J12, J23, j3 = symbols('j1 j2 J J12 J23 j3')

    global_assumptions.clear()
    global_assumptions.add( Assume( j1, 'half_integer') )
    global_assumptions.add( Assume( m1, 'half_integer') )
    global_assumptions.add( Assume( j2, 'half_integer') )
    global_assumptions.add( Assume( m2, 'half_integer') )
    global_assumptions.add( Assume( j3, 'half_integer') )
    global_assumptions.add( Assume( m3, 'half_integer') )
    global_assumptions.add( Assume(J12, Q.integer) )
    global_assumptions.add( Assume(M12, Q.integer) )
    global_assumptions.add( Assume(J23, Q.integer) )
    global_assumptions.add( Assume(M23, Q.integer) )
    global_assumptions.add( Assume(  J, 'half_integer') )
    global_assumptions.add( Assume(  M, 'half_integer') )

    t1 =    SphericalTensor('t1', j1, m1)
    t2 =    SphericalTensor('t2', j2, m2)
    t3 =    SphericalTensor('t3', j3, m3)
    T12 =   SphericalTensor('T12', J12, M12, t1, t2)
    T23 =   SphericalTensor('T23', J23, M23, t2, t3)
    T12_3 = SphericalTensor('T12_3',J, M, T12, t3)
    T1_23 = SphericalTensor('T1_23',J, M, t1, T23)

    expr_tjs = convert_cgc2tjs(T1_23.as_other_coupling(T12_3, use_dummies=False))
    expr_heyde = (ASigma(J12)*(-1)**(j1+j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T12_3)

    expr = refine_tjs2sjs(expr_tjs, definition='brink_satchler')
    assert is_equivalent(expr_heyde, expr)

    expr = refine_tjs2sjs(expr_tjs, definition='edmonds')
    assert is_equivalent(expr_heyde, expr)


def test_is_equivalent():
    m1, m2, M, M12, M23, m3 = symbols('m1 m2 M M12 M23 m3')
    j1, j2, J, J12, J23, j3 = symbols('j1 j2 J J12 J23 j3')

    global_assumptions.clear()
    global_assumptions.add( Assume( j1, 'half_integer') )
    global_assumptions.add( Assume( m1, 'half_integer') )
    global_assumptions.add( Assume( j2, 'half_integer') )
    global_assumptions.add( Assume( m2, 'half_integer') )
    global_assumptions.add( Assume( j3, 'half_integer') )
    global_assumptions.add( Assume( m3, 'half_integer') )
    global_assumptions.add( Assume(J12, Q.integer) )
    global_assumptions.add( Assume(M12, Q.integer) )
    global_assumptions.add( Assume(J23, Q.integer) )
    global_assumptions.add( Assume(M23, Q.integer) )
    global_assumptions.add( Assume(  J, 'half_integer') )
    global_assumptions.add( Assume(  M, 'half_integer') )

    t1 =    SphericalTensor('t1', j1, m1)
    t2 =    SphericalTensor('t2', j2, m2)
    t3 =    SphericalTensor('t3', j3, m3)
    T12 =   SphericalTensor('T12', J12, M12, t1, t2)
    T23 =   SphericalTensor('T23', J23, M23, t2, t3)
    T12_3 = SphericalTensor('T12_3',J, M, T12, t3)
    T1_23 = SphericalTensor('T1_23',J, M, t1, T23)

    expr1 = (ASigma(J12)*(-1)**(j1+j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T12_3)
    expr2 = (ASigma(J12)*(-1)**(j1+j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T12_3)
    assert is_equivalent(expr1, expr2)  # identical
    expr2 = (ASigma(J12)*(-1)**(j1-j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T12_3)
    assert is_equivalent(expr1, expr2) == False # different phase
    expr2 = (ASigma(J23)*(-1)**(j1+j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T12_3)
    assert is_equivalent(expr1, expr2) == False # different sum
    expr2 = (ASigma(J12)*(-1)**(j1+j2+j3+J)*((2*J12+2)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T12_3)
    assert is_equivalent(expr1, expr2) == False # different factor
    expr2 = (ASigma(J12)*(-1)**(j1+j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j3, J12, j2, J, J23)*T12_3)
    assert is_equivalent(expr1, expr2) == False # different 6j symbol
    expr2 = (ASigma(J12)*(-1)**(j1+j2+j3+J)*((2*J12+1)*(2*J23+1))**(Rational(1,2))
            * SixJSymbol(j1, j2, J12, j3, J, J23)*T1_23)
    assert is_equivalent(expr1, expr2) == False # different tensor


