import numpy as np
from numpy.linalg import norm

__OMEGA = 1
__C1 = 0.1
__C2 = 0.1
__R1 = np.random.uniform(0,1)
__R2 = np.random.randint(0,1)
__N = 50
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


def generate_velocity(n):
    return np.random.uniform(low=0.0, high=1.0, size=(n, __n))



def FER(pop, i, j, ub, lb, f):
    fs = f(pop) # fitness evaluation for all population
    p_g = pop[None, np.argmax(fs, axis=0)] # global best
    p_w = pop[None, np.argmin(fs, axis=0)] # global worst
    S = np.sum((ub - lb)**2, axis=0) # size of search space
    alpha =  S / (f(p_g) - f(p_w)) # normalization factor
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


def update_pbest(population, pbest, fitness_function):
    """
    updates the best personal historical location of each particle in the 
    population.

    Parameters
    ----------
    population : np.ndarray
        nxN population matrix where n is the size of the population and N is 
        the dimension of each particle in the population.
    pbest : np.ndarray
        nxN matrix of the best personal location of all n particles.
    fitness_function : function
        the fitness function, returns a scalar.

    Returns
    -------
    None.

    """
    for index, particle in enumerate(population):
        if fitness(particle[None,:]) > fitness(pbest[None, index]):
            pbest[index] = particle


def update_velocity(v, population, pbest, gbest):
    v = __OMEGA * v + __C1*__R1*(pbest - population) + __C2*__R2*(gbest-population)
    return v

def update_position(v, population):
    return population + v

if __name__ == '__main__':
    lb = np.array([-2, 2])
    ub = np.array([-2, 2])
    pop = generate_population(__N, lb, ub)
    v = generate_velocity(__N)
    pbest = np.zeros(pop.shape)
    gbest = np.zeros(pop.shape)
    k = 0
    gbests = []
    while k < 10:
        update_pbest(pop, pbest, fitness)
        
        for i in range(pop.shape[0]):
            fer = np.zeros((pop.shape[0]))
            for j in range(pop.shape[0]):
                fer[j] = FER(pop, i, j, ub, lb, fitness)
            gbest[i] = pbest[np.argmax(fer)]
            
        v = update_velocity(v, pop, pbest, gbest)
        pop = update_position(v, pop)
        # Plot moving gbest
        # gbests.append(gbest)
        gbests.append(fitness(gbest).mean())
        k += 1
    
    import matplotlib.pyplot as plt
    plt.figure()
    # Plot moving gbest
    # plt.scatter([x[0,0] for x in gbests], [x[0,1] for x in gbests])
    plt.plot(gbests)
    plt.show()
    









