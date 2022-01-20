import os
import csv
from statistics import median, stdev, mean
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pso import PSO
from CFBPSO import CfbPso
from FCPSO import FcPso
from InteriaWeightPSO import InteriaWeightPSO
from SOCPSO import SocPso

from dimensionality import Dimensionality
from boundary_conditions import Absorbing, Dumping, Reflecting, Swap, Teleport, Testing, Invisible, InvisibleDamping, InvisibleReflecting
from fitness_functions import Ackley, Griewank, Rastrigin, Rosenbrock, Sphere
from scipy.spatial import KDTree

functions = [Sphere(), Griewank(), Rastrigin(), Rosenbrock(), Ackley()]
boundary_conditions = [Absorbing(), Reflecting(), Dumping(), Invisible(
), InvisibleDamping(), InvisibleReflecting(), Teleport(), Swap(), Testing()]
dimensions = [Dimensionality(3, 100, 1000), Dimensionality(
    30, 150, 1500), Dimensionality(50, 250, 2500)]

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

    csv_headers = ['Funkcja', 'Parametry', 'Warunek brzegowy', 'Najgorsze rozwiązanie',
                   'Najlepsze rozwiązanie', 'Średnia', 'Odchylenie standardowe', 'Median']
    f = open('results/results.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(csv_headers)

    for eval_function in functions:
        print(eval_function)

        for dimension in dimensions:
            print('Number of dimensions: {0}'.format(dimension.dimensions))

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_xlabel('Liczba iteracji')
            ax.set_ylabel('Średnie rozwiązanie po 50 uruchomieniach')
            ax.set_yscale('log')
            ax.set_title('{0}, N={1}'.format(
                eval_function, dimension.dimensions))

            for boundary_condition in boundary_conditions:
                print('\t {0}'.format(boundary_condition))

                gbest_runs = []
                for _ in tqdm(range(runs)):
                    swarm = PSO(dimension, eval_function, boundary_condition)
                    # swarm = InteriaWeightPSO(dimension, eval_function,
                    #                          boundary_condition)
                    # swarm = CfbPso(dimension, eval_function,
                    #                boundary_condition)
                    # swarm = FcPso(dimension, eval_function,
                    #   boundary_condition)
                    # swarm = SocPso(dimension, eval_function,
                    #                boundary_condition)
                    swarm.optimize()
                    gbest_runs.append(swarm.gbest)

                y, error = tolerant_mean(gbest_runs)
                ax.plot(np.arange(len(y)), y,
                        color=boundary_condition.color, marker=boundary_condition.marker, label=boundary_condition.label)
                print('Results:')
                print('Max: {0} | Min: {1} | Mean: {2} | Stdev: {3} | Median: {4}'.format(
                    max(y), min(y), mean(y), stdev(y), median(y)))

                writer.writerow([eval_function, dimension, boundary_condition, max(
                    y), min(y), mean(y),  stdev(y), median(y)])

            ax.legend(labels, loc="upper right", fontsize=9)
            plt.savefig('results/{0}_{1}_{2}.png'.format(swarm.__class__.__name__,
                                                         eval_function.__class__.__name__, dimension.dimensions))

    f.close()


if __name__ == "__main__":
    main()
