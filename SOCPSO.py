import random
import numpy as np
from numpy.linalg import cond
from SOCParticle import SocParticle
from boundary_conditions import BoundaryCondition
from dimensionality import Dimensionality
from fitness_functions import FitnessFunction

from scipy.spatial import KDTree

from pso import PSO


class SocPso(PSO):

    def __init__(self, dimensionality: Dimensionality, fitness_function: FitnessFunction, boundary_condition: BoundaryCondition):
        super().__init__(dimensionality, fitness_function, boundary_condition)
        
        self.w = 0.0
        self.w_max = 0.9
        self.w_min = 0.4

        self.td_max = abs(self.fitness_function.bounds[0] - self.fitness_function.bounds[1]) * 0.01
        self.td_min = 0.0
        self.td = 1.0
        self.cr = 2. / self.dimensionality.swarm_size
        self.cl = self.dimensionality.swarm_size / 5

        self.swarm_particle = [SocParticle(dimensionality.dimensions, self.fitness_function)
                               for _ in range(self.dimensionality.swarm_size)]

    def optimize(self):
        for _ in range(self.dimensionality.max_iterations):
            self.w = self.w_max - (((self.w_max - self.w_min) /
                                    self.dimensionality.max_iterations) * (self.iteration))
            self.td = self.td_max - (((self.td_max - self.td_min) /
                                      self.dimensionality.max_iterations) * (self.iteration))
            self.iteration += 1
            for particle in self.swarm_particle:
                self.update_particle(particle)
                if particle.valid_fitness:
                    self.evaluate_particle(particle)

            self.calculate_criticality()

            self.gbest.append(self.gbest_fitness)
            if self.gbest_fitness < 0.1:
                break
    
    def update_velocity(self, particle: SocParticle):
        r1 = random.random()
        r2 = random.random()

        cognitive = self.c1 * r1 * \
            (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self.gbest_position - particle.position)
        np.multiply(particle.velocity, self.w)
        particle.velocity = particle.velocity + cognitive + social

    def calculate_criticality(self):
        # calculate criticality
        for i in range(self.dimensionality.swarm_size):
            for j in range(self.dimensionality.swarm_size):
                if i != j:
                    dist = np.linalg.norm(
                        self.swarm_particle[i].position - self.swarm_particle[j].position)
                    if dist < self.td:
                        self.swarm_particle[i].c += 1.

        # reduce criticality
        for particle in self.swarm_particle:
            particle.c *= self.cr

        positions = [o.position for o in self.swarm_particle]
        tree = KDTree(positions)
        while True:
            for i in range(self.dimensionality.swarm_size):
                if self.swarm_particle[i].c > self.cl:
                    _, ii = tree.query(
                        self.swarm_particle[i].position, k=self.cl+1)
                    popped_element = ii.pop(0)
                    for index in ii:
                        self.swarm_particle[index].c += 1.
                    self.swarm_particle[popped_element].c -= self.cl
                    self.swarm_particle[popped_element].position = np.array(
                        [self.swarm_particle[popped_element].init_position(self.fitness_function.bounds[0], self.fitness_function.bounds[1]) for _ in range(self.dimensionality.dimensions)])
            if not any(p.c > self.cl for p in self.swarm_particle):
                break
