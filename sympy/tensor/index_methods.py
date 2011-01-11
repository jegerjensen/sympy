"""Module with functions operating on IndexedBase, Indexed and Idx objects

    - Check shape conformance
    - Determine indices in resulting expression

    etc.

    Methods in this module could be implemented by calling methods on Expr
    objects instead.  When things stabilize this could be a useful refactoring.
"""

from sympy.tensor.indexed import Idx, IndexedBase, Indexed
from sympy.utilities import all
from sympy.functions import exp
from sympy.core import C


class IndexConformanceException(Exception):
    pass

def _remove_repeated(inds):
    """Removes repeated Idx from a sequence

    Returns a set of the unique objects and a tuple of all that have been
    removed.

    >>> from sympy.tensor.index_methods import _remove_repeated
    >>> from sympy.tensor import Idx
    >>> l1 = map(Idx, 'abb')
    >>> _remove_repeated(l1)
    (set([a]), (b,))

    """
    sum_index = {}
    for i in inds:
        if i in sum_index and isinstance(i, Idx):
            sum_index[i] += 1
        else:
            sum_index[i] = 0
    inds = filter(lambda x: not sum_index[x], inds)
    dummies = tuple([ i for i in sum_index if sum_index[i] ])
    return set(inds), dummies

def _get_indices_Mul(expr, return_dummies=False):
    """Determine the outer indices of a Mul object.

    >>> from sympy.tensor.index_methods import _get_indices_Mul
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Mul(x[i, k]*y[j, k])
    (set([i, j]), {})
    >>> _get_indices_Mul(x[i, k]*y[j, k], return_dummies=True)
    (set([i, j]), {}, (k,))

    """

    junk, factors = expr.as_coeff_mul()
    inds = map(get_indices, factors)
    inds, syms = zip(*inds)

    inds = map(list, inds)
    inds = reduce(lambda x, y: x + y, inds)
    inds, dummies = _remove_repeated(inds)

    symmetry = {}
    for s in syms:
        for pair in s:
            if pair in symmetry:
                symmetry[pair] *= s[pair]
            else:
                symmetry[pair] = s[pair]

    if return_dummies:
        return inds, symmetry, dummies
    else:
        return inds, symmetry

def _get_indices_Pow(expr):
    """Determine outer indices of a power or an exponential.

    A power is considered a universal function, so that the indices of a Pow is
    just the collection of indices present in the expression.  This may be
    viewed as a bit inconsistent in the special case:

        x[i]**2 = x[i]*x[i]                                                      (1)

    The above expression could have been interpreted as the contraction of x[i]
    with itself, but we choose instead to interpret it as a function

        lambda y: y**2

    applied to each element of x (a universal function in numpy terms).  In
    order to allow an interpretation of (1) as a contraction, we need
    contravariant and covariant Idx subclasses.  (FIXME: this is not yet
    implemented)

    Expressions in the base or exponent are subject to contraction as usual,
    but an index that is present in the exponent, will not be considered
    contractable with its own base.  Note however, that indices in the same
    exponent can be contracted with each other.

    >>> from sympy.tensor.index_methods import _get_indices_Pow
    >>> from sympy import Pow, exp, IndexedBase, Idx
    >>> A = IndexedBase('A')
    >>> x = IndexedBase('x')
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> _get_indices_Pow(exp(A[i, j]*x[j]))
    (set([i]), {})
    >>> _get_indices_Pow(Pow(x[i], x[i]))
    (set([i]), {})
    >>> _get_indices_Pow(Pow(A[i, j]*x[j], x[i]))
    (set([i]), {})

    """
    base, exp = expr.as_base_exp()
    binds, bsyms = get_indices(base)
    einds, esyms = get_indices(exp)

    inds = binds | einds

    # FIXME: symmetries from power needs to check special cases, else nothing
    symmetries = {}

    return inds, symmetries

def _get_indices_Add(expr):
    """Determine outer indices of an Add object.

    In a sum, each term must have the same set of outer indices.  A valid
    expression could be

        x(i)*y(j) - x(j)*y(i)

    But we do not allow expressions like:

        x(i)*y(j) - z(j)*z(j)

    FIXME: Add support for Numpy broadcasting

    >>> from sympy.tensor.index_methods import _get_indices_Add
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Add(x[i] + x[k]*y[i, k])
    (set([i]), {})

    """

    inds = map(get_indices, expr.args)
    inds, syms = zip(*inds)

    # allow broadcast of scalars
    non_scalars = filter(lambda x: x != set(), inds)
    if not non_scalars:
        return set(), {}

    if not all(map(lambda x: x == non_scalars[0], non_scalars[1:])):
        raise IndexConformanceException("Indices are not consistent: %s"%expr)
    if not reduce(lambda x, y: x!=y or y, syms):
        symmetries = syms[0]
    else:
        # FIXME: search for symmetries
        symmetries = {}

    return non_scalars[0], symmetries

def get_indices(expr):
    """Determine the outer indices of expression ``expr``

    By *outer* we mean indices that are not summation indices.  Returns a set
    and a dict.  The set contains outer indices and the dict contains
    information about index symmetries.  Only indices of class Idx are subject
    to implicit summation.

    :Examples:

    >>> from sympy.tensor.index_methods import get_indices
    >>> from sympy.tensor import IndexedBase, Idx
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j = map(Idx, ['i', 'j'])

    The indices of the total expression is determined, Repeated indices imply a
    summation, for instance the trace of a matrix A:

    >>> get_indices(A[i, i])
    (set(), {})

    In the case of many terms, the terms are required to have identical
    outer indices.  Else an IndexConformanceException is raised.

    >>> get_indices(x[i] + A[i, j]*y[j])
    (set([i]), {})

    :Exceptions:

    An IndexConformanceException means that the terms ar not compatible, e.g.

    >>> get_indices(x[i] + y[j])                #doctest: +SKIP
            (...)
    IndexConformanceException: Indices are not consistent: x(i) + y(j)

    .. warning::
       The concept of *outer* indices applies recursively, starting on the deepest
       level.  This implies that dummies inside parenthesis are assumed to be
       summed first, so that the following expression is handled gracefully:

       >>> get_indices((x[i] + A[i, j]*y[j])*x[j])
       (set([i, j]), {})

       This is correct and may appear convenient, but you need to be careful
       with this as Sympy wil happily .expand() the product, if requested.  The
       resulting expression would mix the outer ``j`` with the dummies inside
       the parenthesis, which makes it a different expression.  To be on the
       safe side, it is best to avoid such ambiguities by using unique indices
       for all contractions that should be held separate.

    """
    # We call ourself recursively to determine indices of sub expressions.

    # break recursion
    if isinstance(expr, Indexed):
        inds, dummies = _remove_repeated(expr.indices)
        return inds, {}
    elif expr is None:
        return set(), {}
    elif expr.is_Atom:
        return set(), {}
    elif isinstance(expr, Idx):
        return set([expr]), {}

    # recurse via specialized functions
    else:
        if expr.is_Mul:
            return _get_indices_Mul(expr)
        elif expr.is_Add:
            return _get_indices_Add(expr)
        elif expr.is_Pow or isinstance(expr, exp):
            return _get_indices_Pow(expr)

        elif isinstance(expr, C.Piecewise):
            # FIXME:  No support for Piecewise yet
            return set(), {}
        elif isinstance(expr, C.Function):
            # Support ufunc like behaviour by returning indices from arguments.
            # Functions do not interpret repeated indices across argumnts
            # as summation
            ind0 = set()
            for arg in expr.args:
                ind, sym = get_indices(arg)
                ind0 |= ind
            return ind0, sym

        # this test is expensive, so it should be at the end
        elif not expr.has(Indexed):
            return set(), {}
        raise NotImplementedError(
                "FIXME: No specialized handling of type %s"%type(expr))

def get_contraction_structure(expr):
    """Determine dummy indices of ``expr`` and describe it's structure

    By *dummy* we mean indices that are summation indices.

    The stucture of the expression is determined and described as follows:

    1) A conforming summation of Indexed objects is described with a dict where
       the keys are summation indices and the corresponding values are sets
       containing all terms for which the summation applies.  All Add objects
       in the Sympy expression tree are described like this.

    2) For all nodes in the Sympy expression tree that are *not* of type Add, the
       following applies:

       If a node discovers contractions in one of it's arguments, the node
       itself will be stored as a key in the dict.  For that key, the
       corresponding value is a list of dicts, each of which is the result of a
       recursive call to get_contraction_structure().  The list contains only
       dicts for the non-trivial deeper contractions, ommitting dicts with None
       as the one and only key.

    .. Note:: The presence of expressions among the dictinary keys indicates
       multiple levels of index contractions.  A nested dict displays nested
       contractions and may itself contain dicts from a deeper level.  In
       practical calculations the summation in the deepest nested level must be
       calculated first so that the outer expression can access the resulting
       indexed object.

    :Examples:

    >>> from sympy.tensor.index_methods import get_contraction_structure
    >>> from sympy.tensor import IndexedBase, Idx
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j, k, l = map(Idx, ['i', 'j', 'k', 'l'])
    >>> get_contraction_structure(x[i]*y[i] + A[j, j])
    {(i,): set([x[i]*y[i]]), (j,): set([A[j, j]])}
    >>> get_contraction_structure(x[i]*y[j])
    {None: set([x[i]*y[j]])}

    A multiplication of contracted factors results in nested dicts representing
    the internal contractions.

    >>> d = get_contraction_structure(x[i, i]*y[j, j])
    >>> sorted(d.keys())
    [None, x[i, i]*y[j, j]]
    >>> d[None]  # Note that the product has no contractions
    set([x[i, i]*y[j, j]])
    >>> sorted(d[x[i, i]*y[j, j]])  # factors are contracted ``first''
    [{(i,): set([x[i, i]])}, {(j,): set([y[j, j]])}]

    A parenthesized Add object is also returned as a nested dictionary.  The
    term containing the parenthesis is a Mul with a contraction among the
    arguments, so it will be found as a key in the result.  It stores the
    dictionary resulting from a recursive call on the Add expression.

    >>> d = get_contraction_structure(x[i]*(y[i] + A[i, j]*x[j]))
    >>> sorted(d.keys())
    [(i,), x[i]*(A[i, j]*x[j] + y[i])]
    >>> d[(i,)]
    set([x[i]*(A[i, j]*x[j] + y[i])])
    >>> d[x[i]*(A[i, j]*x[j] + y[i])]
    [{None: set([y[i]]), (j,): set([A[i, j]*x[j]])}]

    Powers with contractions in either base or exponent will also be found as
    keys in the dictionary, mapping to a list of results from recursive calls:

    >>> d = get_contraction_structure(A[j, j]**A[i, i])
    >>> d[None]
    set([A[j, j]**A[i, i]])
    >>> nested_contractions = d[A[j, j]**A[i, i]]
    >>> nested_contractions[0]
    {(j,): set([A[j, j]])}
    >>> nested_contractions[1]
    {(i,): set([A[i, i]])}

    The description of the contraction structure may appear complicated when
    represented with a string in the above examples, but it is easy to iterate
    over:

    >>> from sympy import Expr
    >>> for key in d:
    ...     if isinstance(key, Expr):
    ...         continue
    ...     for term in d[key]:
    ...         if term in d:
    ...             # treat deepest contraction first
    ...             pass
    ...     # treat outermost contactions here

    """

    # We call ourself recursively to inspect sub expressions.

    if isinstance(expr, Indexed):
        junk, key = _remove_repeated(expr.indices)
        return {key or None: set([expr])}
    elif expr.is_Atom:
        return {None: set([expr])}
    elif expr.is_Mul:
        junk, junk, key = _get_indices_Mul(expr, return_dummies=True)
        result = {key or None: set([expr])}
        # recurse on every factor
        nested = []
        for fac in expr.args:
            facd = get_contraction_structure(fac)
            if not (None in facd and len(facd) == 1):
                nested.append(facd)
        if nested:
            result[expr] = nested
        return result
    elif expr.is_Pow or isinstance(expr, exp):
        # recurse in base and exp separately.  If either has internal
        # contractions we must include ourselves as a key in the returned dict
        b, e = expr.as_base_exp()
        dbase = get_contraction_structure(b)
        dexp = get_contraction_structure(e)

        dicts = []
        for d in dbase, dexp:
            if not (None in d and len(d) == 1):
                dicts.append(d)
        result = {None: set([expr])}
        if dicts:
            result[expr] = dicts
        return result
    elif expr.is_Add:
        # Note: we just collect all terms with identical summation indices, We
        # do nothing to identify equivalent terms here, as this would require
        # substitutions or pattern matching in expressions of unknown
        # complexity.
        result = {}
        for term in expr.args:
            # recurse on every term
            d = get_contraction_structure(term)
            for key in d:
                if key in result:
                    result[key] |= d[key]
                else:
                    result[key] = d[key]
        return result

    elif isinstance(expr, C.Piecewise):
        # FIXME:  No support for Piecewise yet
        return {None: expr}
    elif isinstance(expr, C.Function):
        # Collect non-trivial contraction structures in each argument
        # We do not report repeated indices in separate arguments as a
        # contraction
        deeplist = []
        for arg in expr.args:
            deep = get_contraction_structure(arg)
            if not (None in deep and len(deep) == 1):
                deeplist.append(deep)
        d = {None: set([expr])}
        if deeplist:
            d[expr] = deeplist
        return d

    # this test is expensive, so it should be at the end
    elif not expr.has(Indexed):
        return {None: set([expr])}
    raise NotImplementedError(
            "FIXME: No specialized handling of type %s"%type(expr))
