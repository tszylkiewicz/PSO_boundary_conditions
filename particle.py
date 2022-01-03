import random
import copy
import numpy as np

from fitness_functions import FitnessFunction


class Particle:

    def __init__(self, dim, fitness_function: FitnessFunction):
        self.bounds = fitness_function.bounds
        self.valid_fitness = True
        
        self.position = np.array(
            [self.init_position(self.bounds[0], self.bounds[1]) for _ in range(dim)])
        self.velocity = np.zeros(dim)
        self.fitness = fitness_function.calculate_fitness(self.position)

        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_fitness = fitness_function.calculate_fitness(self.pbest_position)

    def init_position(self, x_min, x_max):
        return x_min + (np.random.uniform() * (x_max - x_min))

    def update_fitness(self, new_fitness):
        self.fitness = new_fitness

        if self.fitness < self.pbest_fitness:
            self.pbest_position = self.position
            self.pbest_fitness = self.fitness
