import os
from statistics import stdev, mean
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from classic_pso import BasicPso
from boundary_conditions import absorbing, dumping, invisible, invisible_damping, invisible_reflecting, reflecting, teleport
from fitness_functions import Griewank, Rastrigin, Rosenbrock, Sphere

functions = [Sphere(), Rosenbrock()]

boundary_conditions = [absorbing, reflecting, dumping, teleport]
particle_size = 30
iterations = 200
dim = 3
cognitive_param = 2.0
social_param = 2.0
w = 0.9

runs = 50

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def main():

    labels = ["absorbing", "reflecting", "dumping","invisible", "invisible_damping", "invisible_reflecting"]
    markers = [".", "v", "^", "<", ">", "s"]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i in range(len(functions)):
        print(functions[i].__class__.__name__)
        fig = plt.figure()
        ax = fig.add_subplot()   
        ax.set_xlabel('Liczba iteracji')
        ax.set_ylabel('Global best')
        ax.set_yscale('log')
        ax.set_title(functions[i].__class__.__name__)
        for j in range(len(boundary_conditions)):
            print("\t"+boundary_conditions[j].__name__)
            gbest_runs = []
            for _ in tqdm(range(runs)):
                swarm = BasicPso(dim, particle_size, iterations, functions[i],
                                boundary_conditions[j], cognitive_param, social_param, w)
                swarm.optimize()
                gbest_runs.append(swarm.gbest)

            y, error = tolerant_mean(gbest_runs)
            ax.plot(np.arange(swarm.iteration), y, color=colors[j], marker=markers[j], label=labels[j])
            print('Max: {0} | Min: {1} | Mean: {2} | Stdev: {3} | Iteration: {4}'.format(
                max(swarm.gbest), min(swarm.gbest), mean(swarm.gbest), stdev(swarm.gbest), swarm.iteration))
        ax.legend(labels, loc="upper right")
        plt.savefig('test_'+functions[i].__class__.__name__+'.png')


# for i in range(len(boundary_conditions)):
#     swarm = Swarm(dim, particle_size, iterations, function,
#                   bounds, boundary_conditions[i], cognitive_param, social_param, w)
#     swarm.optimize()

#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot(swarm.gbest, color='r')
#     plt.savefig('test_'+boundary_conditions[i].__name__+'.png')

#     print('Max: {0} | Min: {1} | Mean: {2} | Stdev: {3}'.format(
#         max(swarm.gbest), min(swarm.gbest), mean(swarm.gbest), stdev(swarm.gbest)))


if __name__ == "__main__":
    main()
