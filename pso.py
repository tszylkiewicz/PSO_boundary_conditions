import random
import numpy as np
from boundary_conditions import BoundaryCondition
from dimensionality import Dimensionality
from fitness_functions import FitnessFunction
from particle import Particle


class PSO:

    def __init__(self, dimensionality: Dimensionality, fitness_function: FitnessFunction, boundary_condition: BoundaryCondition):
        self.dimensionality = dimensionality
        self.fitness_function = fitness_function

        self.gbest = []
        self.gbest_position = []
        self.gbest_fitness = float('inf')

        self.r_norm = 0
        self.c1 = 2.0
        self.c2 = 2.0
        self.boundary_condition = boundary_condition.calculate
        self.max_velocity = 0.1 * \
            (self.fitness_function.bounds[1] - self.fitness_function.bounds[0])

        self.swarm_particle = [Particle(dimensionality.dimensions, self.fitness_function)
                               for _ in range(self.dimensionality.swarm_size)]

        self.diameter = self.calculate_diameter()
        self.iteration = 0

        for j in range(self.dimensionality.swarm_size):
            if self.swarm_particle[j].fitness < self.gbest_fitness:
                self.gbest_position = list(self.swarm_particle[j].position)
                self.gbest_fitness = float(self.swarm_particle[j].fitness)

    def optimize(self):
        for _ in range(self.dimensionality.max_iterations):
            self.iteration += 1
            for particle in self.swarm_particle:
                self.update_particle(particle)
                if particle.valid_fitness:
                    self.evaluate_particle(particle)

            self.gbest.append(self.gbest_fitness)
            self.r_norm = self.calculate_r_norm()
            if self.r_norm < 0.15:
                break

    def update_particle(self,  particle: Particle):
        self.update_velocity(particle)

        # Velocity clamping
        for i in range(self.dimensionality.dimensions):
            if particle.velocity[i] >= self.max_velocity:
                particle.velocity[i] = self.max_velocity

        particle.position = particle.position + particle.velocity

        self.boundary_condition(particle)

    def update_velocity(self, particle: Particle):
        r1 = random.random()
        r2 = random.random()

        cognitive = self.c1 * r1 * \
            (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self.gbest_position - particle.position)

        particle.velocity = particle.velocity + cognitive + social

    def evaluate_particle(self, particle: Particle):
        particle.update_fitness(self.fitness_function.calculate_fitness(particle.position))

        if particle.fitness < self.gbest_fitness:
            self.gbest_position = list(particle.position)
            self.gbest_fitness = float(particle.fitness)

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
