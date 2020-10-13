# coding:utf:8
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import operator
import time


def cal_distmat(city_matrix):
    length = city_matrix.shape[0]
    matrix = np.zeros((length, length))

    for i in range(length):
        for j in range(i + 1, length):
            matrix[i][j] = matrix[j][i] = np.linalg.norm(city_matrix[i] - city_matrix[j])
    return matrix


def get_distance(sequence):
    global distance_matrix

    cost = 0
    for i in range(len(sequence)):
        cost += distance_matrix[sequence[i - 1]][sequence[i]]
    return cost


def cal_fitness(population):
    fitness = []

    for i in range(len(population)):
        fitness.append(1 / get_distance(population[i]))

    return fitness


def init_population():
    global individual_size, population_size
    population_init = []
    for i in range(population_size):
        tmp_list = list(range(individual_size))
        population_init.append(random.sample(tmp_list, individual_size))
    return population_init


def select_chromosome(fitness, population, population_size_next):
    # global population_size # not use ?
    sort_result = {}
    for i in range(len(population)):
        sort_result[(fitness[i], 1 / fitness[i])] = i
    sort_key = sorted(sort_result.keys(), key=operator.itemgetter(0), reverse=True)
    sort_index = [sort_result[i] for i in sort_key]
    sort_population = [population[i] for i in sort_index]
    return sort_population[:population_size_next]


def selection(fitness, num):
    def select_one(cumulative_probability):
        rand_num = random.random()
        for index in range(len(cumulative_probability)):
            if rand_num < cumulative_probability[index]:
                return index

    fitness_sum = sum(fitness)
    fitness_q = []
    accumulation = 0
    for i in range(len(fitness)):
        accumulation = accumulation + fitness[i] / fitness_sum
        if accumulation > 1:
            accumulation = 1
        fitness_q.append(accumulation)

    res = set()
    while len(res) < num:
        t = select_one(fitness_q)
        res.add(t)
    return res


def crossover(parent1, parent2):
    global individual_size

    a = random.randint(1, individual_size - 1)
    child1, child2 = parent1[:a], parent2[:a]

    for i in range(individual_size):
        if parent2[i] not in child1:
            child1.append(parent2[i])

        if parent1[i] not in child2:
            child2.append(parent1[i])

    return child1, child2


def mutation(gene):
    gene_length = len(gene)
    index1 = random.randint(0, gene_length - 1)
    index2 = random.randint(0, gene_length - 1)
    new_gene = gene[:]
    new_gene[index1], new_gene[index2] = new_gene[index2], new_gene[index1]
    return new_gene


def find_best_individual(fitness, population):
    max_index = fitness.index(max(fitness))
    max_fitness = fitness[max_index]
    max_individual = population[max_index]

    return max_fitness, max_individual


def record(f):
    global record_distance
    record_distance.append(1 / f)


def genetic_algorithm():
    global mutation_possibility, max_generation
    generation = 1

    population_current = init_population()
    fitness = cal_fitness(population_current)

    time_start = time.time()

    # generation iteration
    while generation < max_generation:

        population_next = select_chromosome(fitness, population_current, population_size // 4)
        # crossover
        for i in range(population_size):
            p1, p2 = selection(fitness, 2)
            child1, child2 = crossover(population_current[p1], population_current[p2])

            # mutation
            if random.random() < mutation_possibility:
                child1 = mutation(child1)
            if random.random() < mutation_possibility:
                child2 = mutation(child2)

            population_next.append(child1)
            population_next.append(child2)
        population_next = select_chromosome(cal_fitness(population_next), population_next, population_size)
        tmp_max_fitness, tmp_max_individual = find_best_individual(fitness, population_current)
        record(tmp_max_fitness)

        population_current = population_next
        generation += 1

        fitness = cal_fitness(population_current)

    # record the best result
    best_fitness, best_individual = find_best_individual(fitness, population_current)
    record(best_fitness)

    time_end = time.time()
    print('Time Cost：{:.2f}'.format(time_end - time_start), '(s)')
    print("Final Costs：{:.2f} (km)".format(get_distance(best_individual)))

    plot(best_individual)


def plot(sequence):
    global record_distance, city_coordinates

    plt.figure(figsize=(15, 6))
    plt.subplot(121)

    plt.plot(record_distance)
    plt.ylabel('travel distance')
    plt.xlabel('iteration times')

    plt.subplot(122)

    x_list = []
    y_list = []
    for i in range(len(sequence)):
        x_list.append(city_coordinates[sequence[i]][1])
        y_list.append(city_coordinates[sequence[i]][0])
    x_list.append(city_coordinates[sequence[0]][1])
    y_list.append(city_coordinates[sequence[0]][0])

    plt.plot(x_list, y_list, 'c-', label='Route')
    plt.plot(x_list, y_list, 'ro', label='Location')

    # 防止科学计数法
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Tsp Route")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # read data from file
    city_file = 'info.csv'
    city_coordinates = np.loadtxt(city_file, delimiter=',', skiprows=1)
    distance_matrix = cal_distmat(city_coordinates)
    # Parameter initialization
    individual_size = city_coordinates.shape[0]
    max_generation = 70
    population_size = 10
    mutation_possibility = 0.2
    record_distance = []
    # run the algortim
    genetic_algorithm()
