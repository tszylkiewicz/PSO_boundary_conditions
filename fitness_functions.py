import numpy as np


class Sphere:

    def __init__(self):
        self.bounds = [-5.12, 5.12]

    def calculate_fitness(self, position):
        fitness = 0
        for i in range(len(position)):
            fitness += position[i]**2

        return fitness


class Rosenbrock:

    def __init__(self):
        self.bounds = [-5, 10]

    def calculate_fitness(self, position):
        fitness = 0
        for i in range(len(position)-1):
            fitness += 100*((position[i]**2) -
                            position[i+1])**2 + (1-position[i])**2
        return fitness


class Rastrigin:

    def __init__(self):
        self.bounds = [-5.12, 5.12]

    def calculate_fitness(self, position):
        fitness = 10.0*len(position)
        for i in range(len(position)):
            fitness += position[i]**2.0 - (10.0*np.cos(2.0*np.pi*position[i]))
        return fitness


class Griewank:

    def __init__(self):
        self.bounds = [-600, 600]

    def calculate_fitness(self, position):
        firstSum = 0
        secondSum = 1
        for i in range(len(position)):
            firstSum += position[i]**2
            secondSum *= np.cos(float(position[i]) / np.sqrt(i+1))
        fitness = 1 + (float(firstSum)/4000.0) - float(secondSum)
        return fitness


def ackley(position):
    firstSum = 0.0
    secondSum = 0.0
    for c in position:
        firstSum += c**2.0
        secondSum += np.cos(2.0*np.pi*c)
    n = float(len(position))
    return -20.0*np.exp(-0.2*np.sqrt(firstSum/n)) - np.exp(secondSum/n) + 20 + np.e
