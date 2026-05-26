#Code isn't used by pipeline
from numpy import linalg, zeros, ones, hstack, asarray
import itertools

def basis_vector(n, i):
    """ Return an array like [0, 0, ..., 1, ..., 0, 0]

    >>> from multipolyfit.core import basis_vector
    >>> basis_vector(3, 1)
    array([0, 1, 0])
    >>> basis_vector(5, 4)
    array([0, 0, 0, 0, 1])
    """
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

def as_tall(x):
    return x.reshape(x.shape + (1,))

def multipolyfit(xs, y, deg, full=False, model_out=False, powers_out=False):
   
    y = asarray(y).squeeze()
    rows = y.shape[0]
    xs = asarray(xs)
    num_covariates = xs.shape[1]
    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))

    generators = [basis_vector(num_covariates+1, i)
                            for i in range(num_covariates+1)]

    # All combinations of degrees
    powers = [sum(x) for x in itertools.combinations_with_replacement(generators, deg)]

    # Raise data to specified degree pattern, stack in order
    A = hstack(asarray([as_tall((xs**p).prod(1)) for p in powers]))

    beta = linalg.lstsq(A, y)[0]

    if model_out:
        return mk_model(beta, powers)

    if powers_out:
        return beta, powers
    return beta

def mk_model(beta, powers):

    # Create a function that takes in many x values
    # and returns an approximate y value
    def model(*args):
        num_covariates = len(powers[0]) - 1
        if len(args)!=(num_covariates):
            raise ValueError("Expected %d inputs"%num_covariates)
        xs = asarray((1,) + args)
        return sum([coeff * (xs**p).prod()
                             for p, coeff in zip(powers, beta)])
    return model

def mk_sympy_function(beta, powers):
    from sympy import symbols, Add, Mul, S
    num_covariates = len(powers[0]) - 1
    xs = (S.One,) + symbols('x0:%d'%num_covariates)
    return Add(*[coeff * Mul(*[x**deg for x, deg in zip(xs, power)])
                        for power, coeff in zip(powers, beta)])