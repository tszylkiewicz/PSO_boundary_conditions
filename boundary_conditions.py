import random

from abc import (ABC, abstractmethod)


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
    def calculate(self, position, velocity, bounds):
        pass


class Absorbing(BoundaryCondition):

    label = 'Ściana pochłaniająca'
    marker = '.'
    color = 'r'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[0]:
                position[i] = bounds[0]
                velocity[i] = 0
            if position[i] > bounds[1]:
                position[i] = bounds[1]
                velocity[i] = 0

        return position, velocity, self.skip


class Reflecting(BoundaryCondition):

    label = 'Ściana odbijająca'
    marker = 'v'
    color = 'g'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[0]:
                position[i] = bounds[0]
                velocity[i] *= -1.0
            if position[i] > bounds[1]:
                position[i] = bounds[1]
                velocity[i] *= -1.0

        return position, velocity, self.skip


class Dumping(BoundaryCondition):

    label = 'Ściana tłumiąca'
    marker = '^'
    color = 'b'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[0]:
                position[i] = bounds[0]
                velocity[i] *= -random.random()
            if position[i] > bounds[1]:
                position[i] = bounds[1]
                velocity[i] *= -random.random()

        return position, velocity, self.skip


class Invisible(BoundaryCondition):

    label = 'Brak ścian'
    marker = '<'
    color = 'c'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        self.skip = False
        for i in range(len(position)):
            if position[i] < bounds[0] or position[i] > bounds[1]:
                self.skip = True

        return position, velocity, self.skip


class InvisibleReflecting(BoundaryCondition):

    label = ' invisible/reflecting'
    marker = '>'
    color = 'm'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        self.skip = False
        for i in range(len(position)):
            if position[i] < bounds[0] or position[i] > bounds[1]:
                velocity[i] *= -1.0
                self.skip = True

        return position, velocity, self.skip


class InvisibleDamping(BoundaryCondition):

    label = 'invisible/damping'
    marker = 's'
    color = 'k'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        self.skip = False
        for i in range(len(position)):
            if position[i] < bounds[0] or position[i] > bounds[1]:
                velocity[i] *= -random.random()
                self.skip = True

        return position, velocity, self.skip


class Teleport(BoundaryCondition):

    label = 'Ściana teleportująca'
    marker = 'p'
    color = 'y'

    def __init__(self):
        self.skip = False

    def calculate(self, position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[0]:
                diff = bounds[0] - position[i]
                position[i] = bounds[1] - diff
            if position[i] > bounds[1]:
                diff = position[i] - bounds[1]
                position[i] = bounds[0] + diff

        return position, velocity, self.skip


# def invisible_accelerating(position, velocity, bounds):
#     skip = False
#     for i in range(len(position)):
#         if position[i] < bounds[0] or position[i] > bounds[1]:
#             velocity[i] *= -random.random()
#             skip = True

#     return position, velocity, skip
