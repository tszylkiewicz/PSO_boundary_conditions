import random
import numpy as np
import math
from boundary_conditions import BoundaryCondition
from dimensionality import Dimensionality
from fitness_functions import FitnessFunction

from particle import Particle
from pso import PSO


class CfbPso(PSO):

    def __init__(self, dimensionality: Dimensionality, fitness_function: FitnessFunction, boundary_condition: BoundaryCondition):
        super().__init__(dimensionality, fitness_function, boundary_condition)

        self.c1 = 2.1
        self.c2 = 2.1
        self.fi = self.c1 + self.c2
        self.K = self.calculate_K()
    
    def update_velocity(self, particle: Particle):
        r1 = random.random()
        r2 = random.random()

        cognitive = self.c1 * r1 * \
            (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self.gbest_position - particle.position)
        particle.velocity = particle.velocity + cognitive + social
        np.multiply(particle.velocity, self.K)

    def calculate_K(self):
        square_root = math.sqrt(self.fi ** 2 - (4.0 * self.fi))
        denominator = 2.0 - self.fi - square_root
        result = 2.0 / denominator
        return result
