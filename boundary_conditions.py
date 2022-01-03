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
    def calculate(self, particle: Particle, bounds):
        pass

    def __str__(self):
        return self.label


class Absorbing(BoundaryCondition):

    label = 'Ściana pochłaniająca'
    marker = '.'
    color = 'r'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0]:
                particle.position[i] = bounds[0]
                particle.velocity[i] = 0
            if particle.position[i] > bounds[1]:
                particle.position[i] = bounds[1]
                particle.velocity[i] = 0

        return self.skip


class Reflecting(BoundaryCondition):

    label = 'Ściana odbijająca'
    marker = 'v'
    color = 'g'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0]:
                particle.position[i] = bounds[0]
                particle.velocity[i] *= -1.0
            if particle.position[i] > bounds[1]:
                particle.position[i] = bounds[1]
                particle.velocity[i] *= -1.0

        return self.skip


class Dumping(BoundaryCondition):

    label = 'Ściana tłumiąca'
    marker = '^'
    color = 'b'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0]:
                particle.position[i] = bounds[0]
                particle.velocity[i] *= -random.random()
            if particle.position[i] > bounds[1]:
                particle.position[i] = bounds[1]
                particle.velocity[i] *= -random.random()

        return self.skip


class Invisible(BoundaryCondition):

    label = 'Brak ścian'
    marker = '<'
    color = 'c'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        self.skip = False
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0] or particle.position[i] > bounds[1]:
                self.skip = True

        return self.skip


class InvisibleReflecting(BoundaryCondition):

    label = ' invisible/reflecting'
    marker = '>'
    color = 'm'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        self.skip = False
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0] or particle.position[i] > bounds[1]:
                particle.velocity[i] *= -1.0
                self.skip = True

        return self.skip


class InvisibleDamping(BoundaryCondition):

    label = 'invisible/damping'
    marker = 's'
    color = 'k'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        self.skip = False
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0] or particle.position[i] > bounds[1]:
                particle.velocity[i] *= -random.random()
                self.skip = True

        return self.skip


class Teleport(BoundaryCondition):

    label = 'Ściana teleportująca'
    marker = 'p'
    color = 'y'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0]:
                diff = bounds[0] - particle.position[i]
                particle.position[i] = bounds[1] - diff
                particle.velocity[i] = 0
            if particle.position[i] > bounds[1]:
                diff = particle.position[i] - bounds[1]
                particle.position[i] = bounds[0] + diff
                particle.velocity[i] = 0

        return self.skip


class Swap(BoundaryCondition):

    label = 'Ściana testowa'
    marker = '*'
    color = 'k'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0]:
                diff = bounds[0] - particle.position[i]
                particle.position[i] = bounds[0] + diff
                particle.velocity[i] = 0
            if particle.position[i] > bounds[1]:
                diff = particle.position[i] - bounds[1]
                particle.position[i] = bounds[1] - diff
                particle.velocity[i] = 0

        return self.skip


class Testing(BoundaryCondition):

    label = 'Ściana testowa 2'
    marker = '+'
    color = '#0f3d11'

    def __init__(self):
        self.skip = False

    def calculate(self, particle: Particle, bounds):
        for i in range(len(particle.position)):
            if particle.position[i] < bounds[0] or particle.position[i] > bounds[1]:
                particle.position[i] = particle.pbest_position[i]
                particle.velocity[i] = 0

        return self.skip
