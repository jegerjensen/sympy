from sympy import sqrt, exp, S, laguerre_l, pi, ratsimp, Rational, powsimp, Integer
from sympy.mpmath import fac, fac2, gamma
from sympy import gamma, factorial, factorial2

def R_nl(n, l, nu, r):
    """
    Returns the radial wavefunction R_{nl} for a 3d isotropic harmonic oscillator.

    ``n``
        the "nodal" quantum number.  Corresponds to the number of nodes in the
        wavefunction.  n >= 0
    ``l``
        the quantum number for orbital angular momentum
    ``nu``
        mass-scaled frequency: nu = m*omega/(2*hbar) where `m' is the mass and
        `omega' the frequency of the oscillator.  (in atomic units nu == omega)
    ``r``
        Radial coordinate


    :Examples:

    >>> from sympy.physics.sho import R_nl
    >>> from sympy import var
    >>> var("r nu l")
    (r, nu, l)
    >>> R_nl(1, 0, 1, r)
    2*2**(3/4)*exp(-r**2)/pi**(1/4)
    >>> R_nl(2, 0, 1, r)
    4*2**(1/4)*3**(1/2)*(3/2 - 2*r**2)*exp(-r**2)/(3*pi**(1/4))

    l, nu and r may be symbolic:

    >>> R_nl(1, 0, nu, r)
    2*2**(3/4)*(nu**(3/2))**(1/2)*exp(-nu*r**2)/pi**(1/4)
    >>> R_nl(1, l, 1, r)
    r**l*(2**(2 + l)*2**(3/2 + l)/(1 + 2*l)!!)**(1/2)*exp(-r**2)/pi**(1/4)

    The normalization of the radial wavefunction is::

    >>> from sympy import Integral, oo
    >>> Integral(R_nl(1, 0, 1, r)**2 * r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(2, 0, 1, r)**2 * r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(2, 1, 1, r)**2 * r**2, (r, 0, oo)).n()
    1.00000000000000

    """
    n, l, nu, r = map(S, [n, l, nu, r])

    # formula uses n >= 1 (instead of nodal n >= 0)
    n = n + 1
    C = sqrt(
            ((2*nu)**(l + Rational(3, 2))*2**(n+l+1)*factorial(n-1))/
            (sqrt(pi)*(factorial2(2*n + 2*l - 1)))
            )
    return  C*r**(l)*exp(-nu*r**2)*laguerre_l(n-1, l + S(1)/2, 2*nu*r**2)
