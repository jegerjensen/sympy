from sympy.core.function import Function
from sympy.core import sympify, S

###############################################################################
###################### Kronecker Delta, Levi-Civita etc. ######################
###############################################################################

class Dij(Function):
    """
    Represents the Kronecker Delta Function

    >>> from sympy import Dij, symbols
    >>> i,j = symbols('ij')
    >>> Dij(-j, -2*i)
    Dij(j, 2*i)
    >>> Dij(j, i)
    Dij(i, j)
    >>> Dij(i, j).subs(j,i)
    1
    >>> Dij(1, 2)
    0
    """
    nargs = (1, 2)

    @classmethod
    def eval(cls, i, j=0):
        i, j = map(sympify, (i, j))
        if i == j:
            return S.One
        elif i.is_number and j.is_number:
            return S.Zero
        elif i.is_Mul and j.is_Mul:
            ci, ti = i.as_coeff_terms()
            cj, tj = i.as_coeff_terms()
            if ci.is_negative and cj.is_negative:
                return cls(-i, -j)
        if i > j: return cls(j, i)

    def _latex(self, p, exp=None):
        args = ", ".join([ p._print(a) for a in self.args ])
        return r"\delta_{%s}" % args

class Eijk(Function):
    """
    Represents the Levi-Civita symbol (antisymmetric symbol)
    """
    nargs = 3

    @classmethod
    def eval(cls, i, j, k):
        i, j, k = map(sympify, (i, j, k))
        if (i,j,k) in [(1,2,3), (2,3,1), (3,1,2)]:
            return S.One
        elif (i,j,k) in [(1,3,2), (3,2,1), (2,1,3)]:
            return S.NegativeOne
        elif i==j or j==k or k==i:
            return S.Zero
