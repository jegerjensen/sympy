from sympy import (
        Basic, Function, Mul, sympify, Integer, Add, sqrt, Pow, Symbol, latex,
        cache, powsimp, symbols
        )
from sympy.functions import Dij
from sympy.assumptions import (
        register_handler, remove_handler, Q, ask, Assume, refine, global_assumptions
        )
from sympy.physics.racahalgebra import (
        SixJSymbol, ThreeJSymbol, ClebschGordanCoefficient, refine_tjs2sjs,
        convert_cgc2tjs, convert_tjs2cgc, ASigma, SphericalTensor, CompositeSphericalTensor, AtomicSphericalTensor
        )
from sympy.utilities import raises


def test_half_integer_ask_handler():
    x = Symbol('x')
    assert ask(x,'half_integer', Assume(x,'half_integer')) == True
    assert ask(x,'half_integer', Assume(x,'half_integer', False)) == False
    assert ask(2*x,Q.even, Assume(x,'half_integer')) == False
    assert ask(2*x,Q.odd, Assume(x,'half_integer')) == True

def test_tjs_methods():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

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


def test_sjs_methods():
    a,b,c,d,e,f,z = symbols('abcdefz')
    A,B,C,D,E,F,Z = symbols('ABCDEFZ')

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
    global_assumptions.add( Assume(C, 'half_integer') )
    global_assumptions.add( Assume(c, 'half_integer') )

    sjs = SixJSymbol(A, B, E, D, C, F);
#    FIXME: check this expression!!!
    assert sjs.get_ito_ThreeJSymbols((a,b,e,d,c,f), definition='brink_satchler') == (-1)**(C + D + F - a - c - e)*ASigma(a, b, c, d, e, f)*ThreeJSymbol(A, B, E, a, -e, -b)*ThreeJSymbol(A, C, F, a, f, -c)*ThreeJSymbol(B, D, F, e, -d, -c)*ThreeJSymbol(C, D, E, f, d, b)


    global_assumptions.clear()

def test_cgc_methods():
    a,b,c = symbols('abc')
    A,B,C = symbols('ABC')

    assert ClebschGordanCoefficient(A, a, B, b, C, c).get_as_ThreeJSymbol() == (-1)**(A + c - B)*ThreeJSymbol(A, B, C, a, b, -c)*sqrt(1 + 2*C)


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
    assert t.get_direct_product_ito_self(drop_self=True) == 1
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
    assert T.as_direct_product() == ASigma(a, b)*ClebschGordanCoefficient(A,a,B,b,C,c)*t1*t2
    assert T.get_direct_product_ito_self() == ASigma(C, c)*ClebschGordanCoefficient(A,a,B,b,C,c)*T

def test_CompositeSphericalTensor_nestings():
    a,b,c,d,e,f = symbols('abcdef')
    A,B,C,D,E,F = symbols('ABCDEF')

    t1 = SphericalTensor('t1', A, a)
    t2 = SphericalTensor('t2', B, b)
    t3 = SphericalTensor('t3',C, c)
    T = SphericalTensor('T',D, d, t1, t2)
    S = SphericalTensor('S',E, e, T, t3)
    assert S.as_direct_product() == ASigma(a, b, c, d)*t1*t2*t3*ClebschGordanCoefficient(A, a, B, b, D, d)*ClebschGordanCoefficient(D, d, C, c, E, e)
    assert S.as_direct_product(deep=False) == ASigma(c, d)*T*t3*ClebschGordanCoefficient(D, d, C, c, E, e)

def test_3b_coupling_schemes():
    a,b,c,d,e,f,g = symbols('abcdefg')
    A,B,C,D,E,F,G = symbols('ABCDEFG')

    t1 = SphericalTensor('t1', A, a)
    t2 = SphericalTensor('t2', B, b)
    t3 = SphericalTensor('t3',C, c)
    T12 = SphericalTensor('T12',D, d, t1, t2)
    T23 = SphericalTensor('T23',F, f, t2, t3)

    S = SphericalTensor('S',E, e, T12, t3)
    V = SphericalTensor('V',G, g, t1, T23)

    assert V.get_ito_other_coupling_order(S) == (
            ASigma(D, a, b, c, d, f)
            *ClebschGordanCoefficient(A, a, B, b, D, d)
            *ClebschGordanCoefficient(A, a, F, f, G, g)
            *ClebschGordanCoefficient(B, b, C, c, F, f)
            *ClebschGordanCoefficient(D, d, C, c, E, e)
            *S*Dij(G,E)*Dij(g,e)
            )

def test_4b_coupling_schemes():
    a,b,c,d,e,f,g,h,y,z = symbols('abcdefghyz')
    A,B,C,D,E,F,G,H,Y,Z = symbols('ABCDEFGHYZ')

    l1 = SphericalTensor('l1', A, a)
    l2 = SphericalTensor('l2', B, b)
    s1 = SphericalTensor('s1', C, c)
    s2 = SphericalTensor('s2', D, d)

    L  = SphericalTensor( 'L', E, e, l1, l2)
    S  = SphericalTensor( 'S', F, f, s1, s2)
    j1 = SphericalTensor('j1', G, g, l1, s1)
    j2 = SphericalTensor('j2', H, h, l2, s2)

    Tls = SphericalTensor('Tls', Y, y,  L,  S)
    Tjj = SphericalTensor('Tjj', Z, z, j1, j2)

    assert Tls.get_ito_other_coupling_order(Tjj) == (
            ASigma(a, b, c, d, e, f, g, h, G, H)
            *ClebschGordanCoefficient(A, a, C, c, G, g) # j1
            *ClebschGordanCoefficient(B, b, D, d, H, h) # j2
            *ClebschGordanCoefficient(A, a, B, b, E, e) # L
            *ClebschGordanCoefficient(C, c, D, d, F, f) # S
            *ClebschGordanCoefficient(E, e, F, f, Y, y) # Tls
            *ClebschGordanCoefficient(G, g, H, h, Z, z) # Tjj
            *Tjj*Dij(Y,Z)*Dij(y,z)
            )

def test_refine_tjs2sjs():
    a,b,c,d,e,f,g = symbols('abcdefg')
    A,B,C,D,E,F,G = symbols('ABCDEFG')

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
    global_assumptions.add( Assume(C, 'half_integer') )
    global_assumptions.add( Assume(c, 'half_integer') )

    t1 = SphericalTensor('t1', A, a)
    t2 = SphericalTensor('t2', B, b)
    t3 = SphericalTensor('t3',D, d)
    T12 = SphericalTensor('T12',E, e, t1, t2)
    T23 = SphericalTensor('T23',F, f, t2, t3)
    S = SphericalTensor('S',C, c, T12, t3)
    V = SphericalTensor('V',C, c, t1, T23)

    expr_tjs = convert_cgc2tjs(V.get_ito_other_coupling_order(S))
    expr_tjs = refine(powsimp(expr_tjs))
    print
    print expr_tjs
    print refine_tjs2sjs(expr_tjs)




