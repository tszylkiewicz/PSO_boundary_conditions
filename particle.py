import random
import copy
import numpy as np

from fitness_functions import FitnessFunction


class Particle:

    def __init__(self, dim, function: FitnessFunction):
        self.position = np.array(
            [self.init_position(function.bounds[0], function.bounds[1]) for _ in range(dim)])
        self.velocity = np.zeros(dim)
        self.fitness = function.calculate_fitness(self.position)

        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_fitness = function.calculate_fitness(self.pbest_position)

    def init_position(self, min, max):
        return min + (np.random.uniform() * (max-min))

    def update_fitness(self, new_fitness):
        self.fitness = new_fitness

        if self.fitness < self.pbest_fitness:
            self.pbest_position = self.position
            self.pbest_fitness = self.fitness
