import random
import numpy as np
from boundary_conditions import BoundaryCondition
from pso import PSO
from dimensionality import Dimensionality
from fitness_functions import FitnessFunction
from particle import Particle


class InteriaWeightPSO(PSO):

    def __init__(self, dimensionality: Dimensionality, fitness_function: FitnessFunction, boundary_condition: BoundaryCondition):
        super().__init__(dimensionality, fitness_function, boundary_condition)
        
        self.w = 0.0
        self.w_max = 0.9
        self.w_min = 0.4

    def optimize(self):
        for _ in range(self.dimensionality.max_iterations):
            self.w = self.w_max - (((self.w_max - self.w_min) /
                                    self.dimensionality.max_iterations) * (self.iteration))
            self.iteration += 1
            for particle in self.swarm_particle:
                self.update_particle(particle)
                if particle.valid_fitness:
                    self.evaluate_particle(particle)

            self.gbest.append(self.gbest_fitness)
            if self.gbest_fitness < 0.1:
                break

    def update_velocity(self,  particle: Particle):
        r1 = random.random()
        r2 = random.random()

        cognitive = self.c1 * r1 * \
            (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self.gbest_position - particle.position)
        np.multiply(particle.velocity, self.w)
        particle.velocity = particle.velocity + cognitive + social
