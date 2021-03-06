"""Implementation of RootOf class and related tools. """

from sympy.core import (
    S, Basic, Expr, Integer, Float, I, Add, Lambda, symbols,
)

from sympy.polys.polytools import Poly, factor

from sympy.polys.rootisolation import (
    dup_isolate_complex_roots_sqf,
    dup_isolate_real_roots_sqf,
)

from sympy.polys.polyroots import (
    roots_linear, roots_quadratic,
    roots_binomial, preprocess_roots,
)

from sympy.polys.rationaltools import (
    together,
)

from sympy.polys.polyfuncs import (
    symmetrize, viete,
)

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
)

from sympy.polys.domains import QQ

from sympy.mpmath import (
    mp, mpf, mpc, mpi, findroot,
)

from sympy.simplify import collect
from sympy.utilities import any, lambdify

import operator

def dup_minpoly_add(f, g, K):
    F = dmp_raise(f, 1, 0, K)
    G = dmp_raise(g, 1, 0, K)

    H = [[-K.one], [K.one, K.zero]]
    F = dmp_compose(F, H, 1, K)

    return dmp_resultant(F, G, 1, K)

def dup_minpoly_sub(f, g, K):
    F = dmp_raise(f, 1, 0, K)
    G = dmp_raise(g, 1, 0, K)

    H = [[K.one], [K.one, K.zero]]
    F = dmp_compose(F, H, 1, K)

    return dmp_resultant(F, G, 1, K)

def dup_minpoly_mul(f, g, K):
    f, F = reversed(f), []

    for i, c in enumerate(f):
        if not c:
            F.append([])
        else:
            F.append(dup_lshift([c], i, K))

    F = dmp_strip(F)
    G = dmp_raise(g, 1, 0, K)

    return dmp_resultant(F, G, 1, K)

def dup_minpoly_div(f, g, K):
    F = dmp_raise(f, 1, 0, K)
    G = dmp_raise(g, 1, 0, K)

    H = [[K.one, K.zero], []]
    F = dmp_compose(F, H, 1, K)

    return dmp_resultant(F, G, 1, K)

def dup_minpoly_pow(f, p, q, K):
    d = {(p, 0): -K.one, (0, q): K.one}

    F = dmp_raise(f, 1, 0, K)
    G = dmp_from_dict(d, 1, K)

    return dmp_resultant(F, G, 1, K)

_rootof_trivial_cache = {}

def roots_trivial(poly, radicals=True):
    """Compute roots in linear, quadratic and binomial cases. """
    if poly.degree() == 1:
        return roots_linear(poly)
    else:
        if not radicals:
            return None

        if poly in _rootof_trivial_cache:
            roots = _rootof_trivial_cache[poly]
        else:
            if radicals and poly.degree() == 2:
                roots = roots_quadratic(poly)
            elif radicals and poly.length() == 2 and poly.TC():
                roots = roots_binomial(poly)
            else:
                return None

            _rootof_trivial_cache[poly] = roots

        return roots

_rootof_reals_cache = {}
_rootof_complexes_cache = {}

def _rootof_get_reals_sqf(factor):
    """Compute real isolating intervals for a square-free polynomial. """
    if factor in _rootof_reals_cache:
        real_part = _rootof_reals_cache[factor]
    else:
        _rootof_reals_cache[factor] = real_part = \
            dup_isolate_real_roots_sqf(factor.rep.rep, factor.rep.dom, blackbox=True)

    return real_part

def _rootof_get_complexes_sqf(factor):
    """Compute complex isolating intervals for a square-free polynomial. """
    if factor in _rootof_complexes_cache:
        complex_part = _rootof_complexes_cache[factor]
    else:
        _rootof_complexes_cache[factor] = complex_part = \
            dup_isolate_complex_roots_sqf(factor.rep.rep, factor.rep.dom, blackbox=True)

    return complex_part

def _rootof_get_reals(factors):
    """Compute real isolating intervals for a list of factors. """
    reals = []

    for factor, k in factors:
        real_part = _rootof_get_reals_sqf(factor)
        reals.extend([ (root, factor, k) for root in real_part ])

    return reals

def _rootof_get_complexes(factors):
    """Compute complex isolating intervals for a list of factors. """
    complexes = []

    for factor, k in factors:
        complex_part = _rootof_get_complexes_sqf(factor)
        complexes.extend([ (root, factor, k) for root in complex_part ])

    return complexes

def _rootof_reals_sorted(reals):
    """Make real isolating intervals disjoint and sort roots. """
    cache = {}

    for i, (u, f, k) in enumerate(reals):
        for j, (v, g, m) in enumerate(reals[i+1:]):
            u, v = u.refine_disjoint(v)
            reals[i+j+1] = (v, g, m)

        reals[i] = (u, f, k)

    reals = sorted(reals, key=lambda r: (r[0].a, r[0].b))

    for root, factor, _ in reals:
        if factor in cache:
            cache[factor].append(root)
        else:
            cache[factor] = [root]

    for factor, roots in cache.iteritems():
        _rootof_reals_cache[factor] = roots

    return reals

def _rootof_complexes_sorted(complexes):
    """Make complex isolating intervals disjoint and sort roots. """
    cache = {}

    for i, (u, f, k) in enumerate(complexes):
        for j, (v, g, m) in enumerate(complexes[i+1:]):
            u, v = u.refine_disjoint(v)
            complexes[i+j+1] = (v, g, m)

        complexes[i] = (u, f, k)

    complexes = sorted(complexes, key=lambda r: (r[0].ax, r[0].ay))

    for root, factor, _ in complexes:
        if factor in cache:
            cache[factor].append(root)
        else:
            cache[factor] = [root]

    for factor, roots in cache.iteritems():
        _rootof_complexes_cache[factor] = roots

    return complexes

def _rootof_reals_index(reals, index):
    """Transform ``RootOf`` index concerning real roots. """
    i = 0

    for j, (_, factor, k) in enumerate(reals):
        if index < i + k:
            poly, index = factor, 0

            for _, factor, _ in reals[:j]:
                if factor == poly:
                    index += 1

            return poly, index, None, None
        else:
            i += k

def _rootof_complexes_index(complexes, index):
    """Transform ``RootOf`` index concerning complex roots. """
    index, conjugate, i = index, False, 0

    for j, (_, factor, k) in enumerate(complexes):
        if index < i + 2*k:
            if index >= i + k:
                conjugate = True

            poly, pointer = factor, 0

            for _, factor, _ in complexes[:j]:
                if factor == poly:
                    pointer += 1

            index = len(_rootof_reals_cache[poly])

            if not conjugate:
                index += 2*pointer
            else:
                index += 2*pointer + 1

            return poly, index, pointer, conjugate
        else:
            i += 2*k

def _rootof_data(poly, indices):
    """Construct ``RootOf`` data from a polynomial and indices. """
    (_, factors) = poly.factor_list()

    reals = _rootof_get_reals(factors)
    real_count = sum([ k for _, _, k in reals ])

    if indices is None:
        reals = _rootof_reals_sorted(reals)

        for index in xrange(0, real_count):
            yield _rootof_reals_index(reals, index)
    else:
        if any(index < real_count for index in indices):
            reals = _rootof_reals_sorted(reals)

            for index in indices:
                if index < real_count:
                    yield _rootof_reals_index(reals, index)

        if any(index >= real_count for index in indices):
            complexes = _rootof_get_complexes(factors)
            complexes = _rootof_complexes_sorted(complexes)

            for index in indices:
                if index >= real_count:
                    yield _rootof_complexes_index(complexes, index-real_count)

class RootOf(Expr):
    """Represents ``k``-th root of a univariate polynomial. """

    __slots__ = ['poly', 'index', 'pointer', 'conjugate']

    def __new__(cls, f, x=None, indices=None, radicals=True, expand=True):
        """Construct a new ``RootOf`` object for ``k``-th root of ``f``. """
        if indices is None and (not isinstance(x, Basic) or x.is_Integer):
            x, indices = None, x

        poly = Poly(f, x, greedy=False, expand=expand)

        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        degree = poly.degree()

        if degree <= 0:
            raise PolynomialError("can't construct RootOf object for %s" % f)

        if indices is not None and indices is not True:
            if hasattr(indices, '__iter__'):
                indices, iterable = list(indices), True
            else:
                indices, iterable = [indices], False

            indices = map(int, indices)

            for i, index in enumerate(indices):
                if index < -degree or index >= degree:
                    raise IndexError("root index out of [%d, %d] range, got %d" % (-degree, degree-1, index))
                elif index < 0:
                    indices[i] += degree
        else:
            iterable = True

            if indices is True:
                indices = range(degree)

        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        roots = roots_trivial(poly, radicals)

        if roots is not None:
            if indices is not None:
                result = [ roots[index] for index in indices ]
            else:
                result = [ root for root in roots if root.is_real ]
        else:
            coeff, poly = preprocess_roots(poly)
            dom = poly.get_domain()

            if not dom.is_ZZ:
                raise NotImplementedError("RootOf is not supported over %s" % dom)

            result = []

            for data in _rootof_data(poly, indices):
                poly, index, pointer, conjugate = data

                roots = roots_trivial(poly, radicals)

                if roots is not None:
                    result.append(coeff*roots[index])
                else:
                    result.append(coeff*cls._new(poly, index, pointer, conjugate))

        if not iterable:
            return result[0]
        else:
            return result

    @classmethod
    def _new(cls, poly, index, pointer=None, conjugate=None):
        """Construct new ``RootOf`` instance from valid ``RootOf`` data. """
        obj = Expr.__new__(cls)

        obj.poly = poly
        obj.index = index

        if pointer is None:
            obj.pointer = index
        else:
            obj.pointer = pointer

        obj.conjugate = conjugate

        return obj

    def _hashable_content(self):
        return (self.expr, self.index)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, Integer(self.index))

    @property
    def is_commutative(self):
        return True

    @property
    def is_real(self):
        """Return ``True`` if the root in consideration is real. """
        return self.conjugate is None

    @property
    def is_complex(self):
        """Return ``True`` if the root in consideration is complex. """
        return self.conjugate is not None

    @property
    def is_conjugate(self):
        """Return ``True`` if the root is located in the lower half-plane. """
        return self.is_complex and self.conjugate

    def _get_interval(self):
        """Internal function for retrieving isolation interval from cache. """
        if self.is_real:
            return _rootof_reals_cache[self.poly][self.pointer]
        else:
            return _rootof_complexes_cache[self.poly][self.pointer]

    def _set_interval(self, interval):
        """Internal function for updating isolation interval in cache. """
        if self.is_real:
            _rootof_reals_cache[self.poly][self.pointer] = interval
        else:
            _rootof_complexes_cache[self.poly][self.pointer] = interval

    def _eval_evalf(self, prec):
        """Evaluate this complex root to the given precision. """
        _prec, mp.prec = mp.prec, prec

        try:
            func = lambdify(self.poly.gen, self.expr)
            interval, refined = self._get_interval(), False

            while True:
                if self.is_real:
                    x0 = mpf(str(interval.center))
                else:
                    re, im = interval.center

                    re = mpf(str(re))
                    im = mpf(str(im))

                    x0 = mpc(re, im)

                try:
                    root = findroot(func, x0)
                except ValueError:
                    interval = interval.refine()
                    refined = True
                    continue
                else:
                    if refined:
                        self._set_interval(interval)

                    if self.is_conjugate:
                        root = root.conjugate()

                    break
        finally:
            mp.prec = _prec

        return Float._new(root.real._mpf_, prec) + I*Float._new(root.imag._mpf_, prec)

class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """

    __slots__ = ['poly', 'fun', 'auto']

    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        """Construct a new ``RootSum`` instance carrying all roots of a polynomial. """
        coeff, poly = cls._transform(expr, x)

        if not poly.is_univariate:
            raise MultivariatePolynomialError("only univariate polynomials are allowed")

        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            try:
                is_func = func.is_Function
            except AttributeError:
                is_func = False

            if is_func and (func.nargs == 1 or 1 in func.nargs):
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                raise ValueError("expected a univariate function, got %s" % func)

        var, expr = func.variables[0], func.expr

        if coeff is not S.One:
            expr = expr.subs(var, coeff*var)

        deg = poly.degree()

        if not expr.has(var):
            return deg*expr

        if expr.is_Add:
            add_const, expr = expr.as_independent(var)
        else:
            add_const = S.Zero

        if expr.is_Mul:
            mul_const, expr = expr.as_independent(var)
        else:
            mul_const = S.One

        func = Lambda(var, expr)

        rational = cls._is_func_rational(poly, func)
        (_, factors), terms = poly.factor_list(), []

        for poly, k in factors:
            if poly.is_linear:
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                term = sum(map(func, roots_quadratic(poly)))
            else:
                if not rational or not auto:
                    term = cls._new(poly, func, auto)
                else:
                    term = cls._rational_case(poly, func)

            terms.append(k*term)

        return mul_const*Add(*terms) + deg*add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        """Construct new raw ``RootSum`` instance. """
        obj = Expr.__new__(cls)

        obj.poly = poly
        obj.fun  = func
        obj.auto = auto

        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        """Construct new ``RootSum`` instance. """
        if not func.expr.has(*func.variables):
            return func.expr

        rational = cls._is_func_rational(poly, func)

        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        """Transform an expression to a polynomial. """
        poly = Poly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        """Check if a lambda is areational function. """
        var, expr = func.variables[0], func.expr
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        """Handle the rational function case. """
        roots = symbols('r:%d' % poly.degree())
        var, expr = func.variables[0], func.expr

        f = sum(expr.subs(var, r) for r in roots)
        p, q = together(f).as_numer_denom()

        domain = QQ[roots]

        p = p.expand()
        q = q.expand()

        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            p, p_coeff = None, (p,)
        else:
            p_monom, p_coeff = zip(*p.terms())

        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            q, q_coeff = None, (q,)
        else:
            q_monom, q_coeff = zip(*q.terms())

        coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
        formulas, values = viete(poly, roots), []

        for (sym, _), (_, val) in zip(mapping, formulas):
            values.append((sym, val))

        for i, (coeff, _) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)

        n = len(p_coeff)

        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]

        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            (p,) = p_coeff

        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            (q,) = q_coeff

        return factor(p/q)

    def _hashable_content(self):
        return (self.expr, self.fun)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, self.fun, self.poly.gen)

    @property
    def is_commutative(self):
        return True

    def doit(self, **hints):
        if hints.get('roots', True):
            return Add(*map(self.fun, RootOf(self.poly, True)))
        else:
            return self

    def _eval_derivative(self, x):
        var, expr = self.fun.args
        func = Lambda(var, expr.diff(x))
        return self.new(self.poly, func, self.auto)
