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
from boundary_conditions import Absorbing, Dumping, Reflecting, Mirroring, Teleporting, Reseting, Invisible, InvisibleDamping, InvisibleReflecting
from fitness_functions import Ackley, Griewank, Rastrigin, Rosenbrock, Sphere

runs = 20

boundary_conditions = [Absorbing(), Reflecting(), Dumping(),
                       Invisible(), InvisibleDamping(), InvisibleReflecting(),
                       Mirroring(), Reseting(), Teleporting()]
dimensions = [Dimensionality(20, 50, 200), Dimensionality(30, 100, 300), Dimensionality(50, 200, 500)]
# dimensions = [Dimensionality(2, 10, 20)]
functions = [Sphere(), Griewank(), Rastrigin(), Rosenbrock(), Ackley()]
algorithms = ['pso','iwpso','cfbpso','fcpso','socpso']

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1)


def main():

    labels = [o.label for o in boundary_conditions]

    csv_headers = ['Funkcja', 'Parametry', 'Warunek brzegowy', 'Najgorsze rozwiązanie',
                   'Najlepsze rozwiązanie', 'Średnia', 'Odchylenie standardowe', 'Median']

    for algorithm in algorithms:               
        f = open('results/{0}_results.csv'.format(algorithm), 'w', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        for eval_function in functions:
            print('Function name: {0}'.format(eval_function))

            for dimension in dimensions:
                print('Number of dimensions: {0}'.format(dimension.dimensions))

                fig = plt.figure()
                ax = fig.add_subplot()
                ax.set_xlabel('Liczba iteracji')
                ax.set_ylabel(
                    'Średnie rozwiązanie po {0} uruchomieniach'.format(runs))
                ax.set_yscale('log')
                ax.set_title('{0}, N={1}'.format(
                    eval_function, dimension.dimensions))

                for boundary_condition in boundary_conditions:
                    print('\t {0}'.format(boundary_condition))

                    gbest_runs = []
                    for _ in tqdm(range(runs)):
                        if algorithm == 'pso':
                            swarm = PSO(dimension, eval_function,
                                        boundary_condition)
                        elif algorithm == 'iwpso':
                            swarm = InteriaWeightPSO(
                                dimension, eval_function, boundary_condition)
                        elif algorithm == 'cfbpso':
                            swarm = CfbPso(dimension, eval_function,
                                        boundary_condition)
                        elif algorithm == 'fcpso':
                            swarm = FcPso(dimension, eval_function,
                                        boundary_condition)
                        else:
                            swarm = SocPso(dimension, eval_function,
                                        boundary_condition)
                        swarm.optimize()
                        gbest_runs.append(swarm.gbest)

                    y = tolerant_mean(gbest_runs)
                    ax.plot(np.arange(len(y)), y, marker=boundary_condition.marker, lw=0.5, ms=5, mew=1, markevery=10,
                            color=boundary_condition.color, label=boundary_condition.label)
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
