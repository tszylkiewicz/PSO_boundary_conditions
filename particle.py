import random
import copy
import numpy as np


def init_position(min, max):
    return min + (np.random.uniform() * (max-min))


class Particle:

    def __init__(self, dim, function, boundary_condition, c1, c2, w):
        self.position = np.array([init_position(function.bounds[0], function.bounds[1]) for _ in range(dim)])
        self.velocity = np.zeros(dim)
        self.fitness = function.calculate_fitness(self.position)

        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_fitness = function.calculate_fitness(self.pbest_position)

        self.boundary_condition = boundary_condition
        self.objective_function = function.calculate_fitness
        self.bounds = function.bounds
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def update_velocity(self, gbest_position):
        r1 = random.random()
        r2 = random.random()

        cognitive = self.c1 * r1 * (self.pbest_position - self.position)
        social = self.c2 * r2 * (gbest_position - self.position)
        self.velocity = (self.w * self.velocity) + cognitive + social

    def update_position(self):
        self.position = self.position + self.velocity

        self.position, self.velocity, skip_evaluation = self.boundary_condition(
            self.position, self.velocity, self.bounds)
        return skip_evaluation

    def evaluate(self):
        self.fitness = self.objective_function(self.position)

        if self.fitness < self.pbest_fitness:
            self.pbest_position = self.position
            self.pbest_fitness = self.fitness
