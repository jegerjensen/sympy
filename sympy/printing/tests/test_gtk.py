from sympy import print_gtk, sin
from sympy.utilities.pytest import XFAIL

# this test fails if python-libxml2 isn't installed. We don't want to depend on
# anything with SymPy
@XFAIL
def test1():
    from sympy.abc import x
    print_gtk(x**2, start_viewer=False)
    print_gtk(x**2+sin(x)/4, start_viewer=False)
