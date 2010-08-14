#!/usr/bin/env python
"""
Setup ufuncs for the legendre polynomials
-----------------------------------------

This example demonstrates how you can use the autowrap module in Sympy to
create fast, customized universal functions for use with numpy arrays.
An autowrapped sympy expression can be significantly faster than what you would
get by applying a sequence of the ufuncs shipped with numpy. [0]

You need to have numpy installed to run this example, as well as a working
fortran compiler.

[0]: http://ojensen.wordpress.com/2010/08/10/fast-ufunc-ish-hydrogen-solutions/
"""

import sys

try:
    import numpy as np
except ImportError:
    sys.exit("Cannot import numpy. Exiting.")

import sympy.mpmath as mpmath
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.autowrap import autowrap
from sympy import symbols, Idx, IndexedBase, Eq, Lambda, legendre, Plot, pprint


def main():

    print __doc__

    # arrays are represented with IndexedBase, indices with Idx
    m = symbols('m', integer=True)
    i = Idx('i', m)
    A = IndexedBase('A')
    B = IndexedBase('B')
    x = symbols('x')

    # a numpy array we can apply the ufuncs to
    grid = np.linspace(-1, 1, 1000)

    # set mpmath precision to 20 significant numbers for verification
    mpmath.mp.dps = 20

    print "Compiling legendre ufuncs and checking results:"

    # Let's also plot the ufunc's we generate
    plot1 = Plot(visible=False)
    for n in range(6):

        # Setup the Sympy expression to ufuncify
        expr = legendre(n, x)

        print "The polynomial of degree %i is" % n
        pprint(expr)

        # Create a symbolic scalar lambda function
        scalar_lambda = Lambda(x, expr)

        # attach it to a Sympy function
        f = implemented_function('f', scalar_lambda)

        # Define the outgoing array element in terms of the incoming
        instruction = Eq(A[i], f(B[i]))

        # implement, compile and wrap the element-wise function
        bin_poly = autowrap(instruction)

        # It's now ready for use with numpy arrays
        polyvector = bin_poly(grid)

        # let's check the values against mpmath's legendre function
        maxdiff = 0
        for j in range(len(grid)):
            precise_val = mpmath.legendre(n, grid[j])
            diff = abs(polyvector[j] - precise_val)
            if diff > maxdiff:
                maxdiff = diff
        print "The largest error in applied ufunc was %e" % maxdiff
        assert maxdiff < 1e-14

        # We can also attach the autowrapped legendre polynomial to a sympy
        # function and plot values as they are calculated by the binary function
        g = implemented_function('g', bin_poly)
        plot1[n] = g(x), [200]

    print "Here's a plot with values calculated by the wrapped binary function"
    plot1.show()

if __name__ == '__main__':
    main()
