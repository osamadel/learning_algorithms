import numpy as np
from numpy.linalg import norm

__n = 2

def generate_population(n, lower_bound, upper_bound):
    """
    generate a population for the PSO. The search space is N-dimensional hence,
    the lower_bound and upper_bound are N-dimensional vectors and each
    particle is also a N-dimensional vector. Consequently, if the size of the
    population is n, then this functions returns nxN population matrix.

    Parameters
    ----------
    n : int
        Size of population.
    lower_bound : np.ndarray
        A vector with each entry representing the lower bound at that dimension.
    upper_bound : np.ndarray
        A vector with each entry representing the upper bound at that dimension.

    Returns
    -------
    nxN np.ndarray where n is the size of the population and N is the dimension
    of the search space (particle vector dimension).

    """
    return np.random.normal(loc=0.0, scale=upper_bound-lower_bound, size=(n, __n))



def FER(pop, i, j, ub, lb, f):
    fs = f(pop)
    p_g = pop[np.argmax(fs, axis=0)]
    p_w = pop[np.argmin(fs, axis=0)]
    S = np.sum((ub - lb)**2, axis=0)
    alpha =  S / (f(p_g) - f(p_w))
    return alpha * (fs[j] - fs[i]) / norm(pop[None,j]-pop[None,i])



def fitness(x):
    """
    calculates the value of the function to optimize at input x

    Parameters
    ----------
    x : np.ndarray
        MxN matrix where M is any integer number of particles, N is the
        dimension of the particle vector.

    Returns
    -------
    np.ndarray
        Mx1 fitness value of the input particles x.

    """
    return (4 - 2.1*x[:,0]**2+x[:,0]**4/3) * x[:,0]**2 + np.prod(x, axis=1) + \
                    (-4+4*x[:,1]**2)*x[:,1]**2