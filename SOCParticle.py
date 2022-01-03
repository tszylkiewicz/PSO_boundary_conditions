import random
import copy
import numpy as np

from fitness_functions import FitnessFunction
from particle import Particle


class SocParticle(Particle):

    def __init__(self, dim, function: FitnessFunction):
        super().__init__(dim, function)
        self.c = 0
