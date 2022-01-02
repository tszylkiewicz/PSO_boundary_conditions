import os
from statistics import stdev, mean
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from CFBPSO import CfbPso
from InteriaWeightPSO import InteriaWeightPso

from dimensionality import Dimensionality
from classic_pso import BasicPso
from boundary_conditions import Absorbing, Dumping, Reflecting, Teleport
from fitness_functions import Griewank, Rastrigin, Rosenbrock, Sphere

functions = [Sphere(), Rosenbrock()]
boundary_conditions = [Absorbing(), Reflecting(), Dumping(), Teleport()]
dimensions = [Dimensionality(3, 30, 200)]

cognitive_param = 2.0
social_param = 2.1

runs = 50

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)


def main():

    labels = [o.label for o in boundary_conditions]
   
    for eval_function in functions:
        print(eval_function.__class__.__name__)

        for dimension in dimensions:
            print('Number of dimensions: {0}'.format(dimension.dimensions))

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_xlabel('Liczba iteracji')
            ax.set_ylabel('Global best')
            ax.set_yscale('log')
            ax.set_title('Funkcja: {0}, N={1}'.format(
                eval_function.__class__.__name__, dimension.dimensions))

            for boundary_condition in boundary_conditions:
                print("\t"+boundary_condition.__class__.__name__)

                gbest_runs = []
                for _ in tqdm(range(runs)):
                    # swarm = BasicPso(dimension, eval_function,
                                    #  boundary_condition, cognitive_param, social_param)
                    # swarm = InteriaWeightPso(dimension, eval_function,
                    #                  boundary_condition, cognitive_param, social_param)
                    swarm = CfbPso(dimension, eval_function,
                                     boundary_condition, cognitive_param, social_param)
                    swarm.optimize()
                    gbest_runs.append(swarm.gbest)

                y, error = tolerant_mean(gbest_runs)
                ax.plot(np.arange(len(y)), y,
                        color=boundary_condition.color, marker=boundary_condition.marker, label=boundary_condition.label)
                print('Max: {0} | Min: {1} | Mean: {2} | Stdev: {3} | Iteration: {4}'.format(
                    max(swarm.gbest), min(swarm.gbest), mean(swarm.gbest), stdev(swarm.gbest), swarm.iteration))
            ax.legend(labels, loc="upper right")
            plt.savefig('{0}_{1}_{2}.png'.format(swarm.__class__.__name__,
                eval_function.__class__.__name__, dimension.dimensions))


if __name__ == "__main__":
    main()
