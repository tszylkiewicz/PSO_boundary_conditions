import numpy as np
from numpy.linalg import cond

from particle import Particle


class BasicPso:

    def __init__(self, dim, particles, epochs, function, boundary_condition, c1, c2, w):
        self.particles = particles
        self.bounds = function.bounds
        self.epochs = epochs
        self.gbest = []
        self.gbest_position = []
        self.gbest_fitness = float('inf')
        self.r_norm = 0

        self.function = function.calculate_fitness

        self.swarm_particle = np.array([Particle(dim, function, boundary_condition, c1, c2, w)
                                        for _ in range(particles)])

        self.diameter = self.calculate_diameter()
        self.iteration = 0

        for j in range(self.particles):
            if self.swarm_particle[j].fitness < self.gbest_fitness:
                self.gbest_position = list(self.swarm_particle[j].position)
                self.gbest_fitness = float(self.swarm_particle[j].fitness)

    def optimize(self):
        for i in range(self.epochs):
            self.iteration += 1
            for j in range(self.particles):
                self.swarm_particle[j].update_velocity(self.gbest_position)
                skip_evaluation = self.swarm_particle[j].update_position()
                if not skip_evaluation:
                    self.swarm_particle[j].evaluate()

                    if self.swarm_particle[j].fitness < self.gbest_fitness:
                        self.gbest_position = list(
                            self.swarm_particle[j].position)
                        self.gbest_fitness = float(
                            self.swarm_particle[j].fitness)

            self.gbest.append(self.gbest_fitness)
            self.r_norm =self.calculate_r_norm()
            if self.r_norm < 0.35:
                break

    def calculate_diameter(self):
        result = 0.0
        for i in range(self.particles):
            for j in range(self.particles):
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