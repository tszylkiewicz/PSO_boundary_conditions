class Dimensionality:

    def __init__(self, dimensions, swarm_size, max_iterations):
        self.dimensions = dimensions
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations

    def __str__(self):
        return 'Wymiary: {0}; Rozmiar roju: {1}; Max iteracji: {2}'.format(self.dimensions, self.swarm_size, self.max_iterations)
