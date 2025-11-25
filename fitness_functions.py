import numpy as np

from abc import (ABC, abstractmethod)


class FitnessFunction(ABC):

    @property
    @abstractmethod
    def label(self):
        pass

    @property
    @abstractmethod
    def bounds(self):
        pass

    @abstractmethod
    def calculate_fitness(self, x):
        pass
    
    def __str__(self) -> str:
        return self.label

    def bounds_str(self):
        return "[{0}, {1}]".format(self.bounds[0], self.bounds[1])


class Sphere(FitnessFunction):
    label = 'Sphere'
    bounds = [-5.12, 5.12]

    def calculate_fitness(self, x):
        fitness = 0
        for i in range(len(x)):
            fitness += x[i]**2

        return fitness


class Rosenbrock(FitnessFunction):
    label = 'Rosenbrock'
    bounds = [-2.048, 2.048]

    def calculate_fitness(self, x):
        fitness = 0
        for i in range(len(x)-1):
            fitness += 100*((x[i]**2) -
                            x[i+1])**2 + (1-x[i])**2
        return fitness


class Rastrigin(FitnessFunction):
    label = 'Rastrigin'
    bounds = [-5.12, 5.12]

    def calculate_fitness(self, x):
        fitness = 10.0*len(x)
        for i in range(len(x)):
            fitness += x[i]**2.0 - (10.0*np.cos(2.0*np.pi*x[i]))
        return fitness


class Griewank(FitnessFunction):
    label = 'Griewank'
    bounds = [-600.0, 600.0]

    def calculate_fitness(self, x):
        firstSum = 0
        secondSum = 1
        for i in range(len(x)):
            firstSum += x[i]**2
            secondSum *= np.cos(float(x[i]) / np.sqrt(i+1))
        fitness = 1 + (float(firstSum)/4000.0) - float(secondSum)
        return fitness


class Ackley(FitnessFunction):
    label = 'Ackley'
    bounds = [-32.768, 32.768]

    def calculate_fitness(self, x):
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c**2.0
            secondSum += np.cos(2.0*np.pi*c)
        n = float(len(x))
        return -20.0*np.exp(-0.2*np.sqrt(firstSum/n)) - np.exp(secondSum/n) + 20 + np.e
