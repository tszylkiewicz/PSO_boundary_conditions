import random
import numpy as np
from numpy.linalg import cond
from boundary_conditions import BoundaryCondition
from dimensionality import Dimensionality
from fitness_functions import FitnessFunction

from particle import Particle


class InteriaWeightPso:

    def __init__(self, dimensionality: Dimensionality, function: FitnessFunction, boundary_condition: BoundaryCondition, c1, c2):
        self.bounds = function.bounds
        self.dimensionality = dimensionality
        self.gbest = []
        self.gbest_position = []
        self.gbest_fitness = float('inf')
        self.r_norm = 0
        self.c1 = c1
        self.c2 = c2
        self.w = 0.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.boundary_condition = boundary_condition.calculate
        self.max_velocity = 0.5*(function.bounds[1]-function.bounds[0])

        self.function = function.calculate_fitness

        self.swarm_particle = [Particle(dimensionality.dimensions, function)
                               for _ in range(self.dimensionality.swarm_size)]

        self.diameter = self.calculate_diameter()
        self.iteration = 0

        for j in range(self.dimensionality.swarm_size):
            if self.swarm_particle[j].fitness < self.gbest_fitness:
                self.gbest_position = list(self.swarm_particle[j].position)
                self.gbest_fitness = float(self.swarm_particle[j].fitness)

    def optimize(self):
        for _ in range(self.dimensionality.max_iterations):
            self.w = self.w_max - (((self.w_max - self.w_min) /
                               self.dimensionality.max_iterations) * (self.iteration))
            self.iteration += 1            
            for particle in self.swarm_particle:
                skip_evaluation = self.update_particle(particle)
                if not skip_evaluation:
                    self.evaluate_particle(particle)

            self.gbest.append(self.gbest_fitness)
            self.r_norm = self.calculate_r_norm()
            if self.r_norm < 0.35:
                break

    def update_particle(self,  particle: Particle):
        r1 = random.random()
        r2 = random.random()

        cognitive = self.c1 * r1 * \
            (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self.gbest_position - particle.position)
        test = np.multiply(particle.velocity, self.w)
        particle.velocity = test + cognitive + social

        # Velocity clamping
        for i in range(self.dimensionality.dimensions):
            if particle.velocity[i] >= self.max_velocity:
                particle.velocity[i] = self.max_velocity

        particle.position = particle.position + particle.velocity

        particle.position, particle.velocity, skip_evaluation = self.boundary_condition(
            particle.position, particle.velocity, self.bounds)
        return skip_evaluation

    def evaluate_particle(self, particle: Particle):
        particle.update_fitness(self.function(particle.position))

        if particle.fitness < self.gbest_fitness:
            self.gbest_position = list(
                particle.position)
            self.gbest_fitness = float(
                particle.fitness)

    def calculate_diameter(self):
        result = 0.0
        for i in range(self.dimensionality.swarm_size):
            for j in range(self.dimensionality.swarm_size):
                if i != j:
                    dist = np.linalg.norm(
                        self.swarm_particle[i].position - self.swarm_particle[j].position)
                    if dist > result:
                        result = dist
        return result

    def calculate_r_norm(self):
        result = 0.0
        for particle1 in self.swarm_particle:
            dist = np.linalg.norm(
                particle1.position - self.gbest_position)
            if dist > result:
                result = dist

        result = result / self.diameter
        return result
