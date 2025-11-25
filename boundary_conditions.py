import random

from abc import (ABC, abstractmethod)

from particle import Particle


class BoundaryCondition(ABC):

    @property
    @abstractmethod
    def label(self):
        pass

    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def marker(self):
        pass

    @abstractmethod
    def calculate(self, particle: Particle):
        pass

    def __str__(self):
        return self.label


class Absorbing(BoundaryCondition):

    label = 'Absorbing'
    marker = '.'
    color = 'r'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0]:
                particle.position[i] = particle.bounds[0]
                particle.velocity[i] = 0
            if particle.position[i] > particle.bounds[1]:
                particle.position[i] = particle.bounds[1]
                particle.velocity[i] = 0


class Reflecting(BoundaryCondition):

    label = 'Reflecting'
    marker = 'v'
    color = 'g'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0]:
                particle.position[i] = particle.bounds[0]
                particle.velocity[i] *= -1.0
            if particle.position[i] > particle.bounds[1]:
                particle.position[i] = particle.bounds[1]
                particle.velocity[i] *= -1.0


class Dumping(BoundaryCondition):

    label = 'Damping'
    marker = '^'
    color = 'b'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0]:
                particle.position[i] = particle.bounds[0]
                particle.velocity[i] *= -random.random()
            if particle.position[i] > particle.bounds[1]:
                particle.position[i] = particle.bounds[1]
                particle.velocity[i] *= -random.random()


class Invisible(BoundaryCondition):

    label = 'Invisible'
    marker = '<'
    color = 'c'

    def calculate(self, particle: Particle):
        particle.valid_fitness = True
        for i in range(len(particle.position)):
            if particle.position[i] < particle.bounds[0] or particle.position[i] > particle.bounds[1]:
                particle.valid_fitness = False


class InvisibleReflecting(BoundaryCondition):

    label = 'Invisible/Reflecting'
    marker = '>'
    color = 'm'

    def calculate(self, particle: Particle):
        particle.valid_fitness = True
        for i in range(len(particle.position)):
            if particle.position[i] < particle.bounds[0] or particle.position[i] > particle.bounds[1]:
                particle.velocity[i] *= -1.0
                particle.valid_fitness = False


class InvisibleDamping(BoundaryCondition):

    label = 'Invisible/Damping'
    marker = 's'
    color = 'k'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0] or particle.position[i] > particle.bounds[1]:
                particle.velocity[i] *= -random.random()
                particle.valid_fitness = False


class Teleporting(BoundaryCondition):

    label = 'Teleporting'
    marker = 'p'
    color = 'y'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0]:
                diff = particle.bounds[0] - particle.position[i]
                particle.position[i] = particle.bounds[1] - diff
                particle.velocity[i] = 0
            if particle.position[i] > particle.bounds[1]:
                diff = particle.position[i] - particle.bounds[1]
                particle.position[i] = particle.bounds[0] + diff
                particle.velocity[i] = 0


class Mirroring(BoundaryCondition):

    label = 'Mirroring'
    marker = '*'
    color = '#8a42f5'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0]:
                diff = particle.bounds[0] - particle.position[i]
                particle.position[i] = particle.bounds[0] + diff
                particle.velocity[i] = 0
            if particle.position[i] > particle.bounds[1]:
                diff = particle.position[i] - particle.bounds[1]
                particle.position[i] = particle.bounds[1] - diff
                particle.velocity[i] = 0


class Reseting(BoundaryCondition):

    label = 'Restarting'
    marker = '+'
    color = '#00ff08'

    def calculate(self, particle: Particle):
        for i in range(len(particle.position)):
            particle.valid_fitness = True
            if particle.position[i] < particle.bounds[0] or particle.position[i] > particle.bounds[1]:
                particle.position[i] = particle.pbest_position[i]
                particle.velocity[i] = 0
