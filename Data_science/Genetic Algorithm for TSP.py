# Genetic algorithm for TSP problem
# version 1.0
# based on Fundamentals of Artificial Intelligence seminar
# Machine Learning and Data Science programme on
# University of Economics in Katowice (2022/2023)

'''DESCRIPTION'''
# The point is to keep the algorithm as basic as possible, using only the
# fundamentals of Python programming (even if less elegant).

# Problem
# TSP problem is all about finding an optimal route between points
# http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsplib.html
# Input: matrix showing distances between points

# Solution: genetic algorithm
# Representation: the order of points, counted from 0
# Fitness function: sum of distance between points
# Initial population: n random individuals
# Selecion: tournament
# Crossover: PMX
# Mutation: inversion

'''PSEUDOCODE'''
# INITIALISATION
# 1. Load data from the file and prepare matrices.
# 2. Create an initial population.
# 3. Evaluation - sum distances between points and apply to the whole population.
# GENETIC ALGORITHM LOOP
# 4. Selection: from 'k' random individuals, choose the best (lowest sum).
# 5. Crossover: PMX.
# 6. Mutation: inversion.
# 7. Succession.
# RESULTS
# 8. Print the best solution, together with its representation.

'''MODULES'''
import random

'''INPUT FILE'''
my_file = <<<file name>>>  # Here we can change an input file easily
my_dir = <<<directory>>> + "\\" + my_file + ".txt"

'''ALGORITHM PARAMETERS'''
num_of_ind = 100  # individuals
num_of_gen = 10000  # generations
k = int(round(0.3 * num_of_ind, 0))  # for selection
pc = 0.85  # crossover probability
pm = 0.05  # mutation probability

'''FUNCTIONS'''
# n - number of individuals
# m - number of points
# ind - individual
# pop - population

# For initialisation
def new_population(n, m):
    population = []
    for _ in range(n):
        population.append(new_individual(m))
    return population

def new_individual(m):
    ind = [i for i in range(m)]  # individual has all numbers of points
    random.shuffle(ind)
    return ind

# For evaluation
def calculate_fitness(ind, distance_matrix):
    sum_distance = 0
    for i in range(len(ind)-1):  # all indicies except the last
        point_1 = ind[i]
        point_2 = ind[i+1]
        sum_distance += distance_matrix[point_1][point_2]
    sum_distance += distance_matrix[ind[0]][ind[-1]]
    return sum_distance

def evaluate_population(pop, distance_matrix):
    fitness = []
    for ind in pop:
        fitness.append(calculate_fitness(ind, distance_matrix))
    return fitness    
    
# For selection
def selection(pop, fitness, k = None):
    new_population = []  # for selected individuals
    for _ in range(len(pop)):
        selected = tournament(pop, fitness, k)
        new_population.append(selected)
    return new_population

def tournament(pop, fitness, k):
    num_of_ind = len(pop)
    min_index = random.randint(0, num_of_ind-1)  # first individual for comparison
    for _ in range(k-1):
        random_index = random.randint(0, num_of_ind-1)
        if fitness[min_index] > fitness[random_index]:
            min_index = random_index
    return pop[min_index][:]  # the best individual in the tournament

# For crossover
def crossover(pop, pc):
    new_population = []
    for i in range(0, len(pop), 2):  # every other because we're crossing pairs
        p1 = pop[i]  # parent1
        p2 = pop[i+1]  # parent2
        if random.random() < pc:  # check if crossing should occur
            c1,c2 = crossover_PMX(p1,p2)
            new_population.append(c1)  # pass new individuals to pop
            new_population.append(c2)
        else:
            new_population.append(p1[:])  # pass (copies) of parents
            new_population.append(p2[:])
    return new_population

def crossover_PMX(p1, p2):
    cut1 = random.randint(1,len(p1)-2)
    cut2 = random.randint(cut1+1,len(p1)-1) 

    c1 = p1[cut1:cut2]
    c2 = p2[cut1:cut2]

    prefix1 = pmx_fix(p2[:cut1], c1, c2)
    prefix2 = pmx_fix(p1[:cut1], c2, c1)

    postfix1 = pmx_fix(p2[cut2:], c1, c2)
    postfix2 = pmx_fix(p1[cut2:], c2, c1)

    c1 = prefix1 + c1 + postfix1
    c2 = prefix2 + c2 + postfix2

    return c1, c2

def pmx_fix(parent, self_mid, mid):
    fix = []  # part of individual to be passed
    for gene in parent:
        while gene in self_mid:  # if the gene already exist, seek a new one
            gene = mid[self_mid.index(gene)]
        fix.append(gene)
    return fix

# For mutation
def mutation(pop, pm):
    # note that we're CHANGING individuals, so there is no new_population like
    # in selection or crossover
    for i in range(len(pop)):
        if random.random() < pm:  # check if mutation should occur
            mutation_inv(pop[i])
            
def mutation_inv(ind):
    cut1 = random.randint(1,len(ind))
    cut2 = random.randint(1,len(ind))
    while cut2==cut1:
        cut2 = random.randint(1,len(ind))
    if cut1>cut2:
        cut1, cut2 = cut2, cut1

    ind[cut1:cut2] = ind[cut2-1:cut1-1:-1]

# For succession
def succession(pop, fitness, pop_0, fitness_0):
    pop += pop_0
    fitness += fitness_0
    fit = [[f, i] for f,i in zip(fitness,range(len(fitness)))]
    fit.sort()
    return [pop[el[1]] for el in fit[:len(fit)//2]]

# For results
def print_individual(ind, fitness):
    print("-".join(map(str,ind)),fitness)

'''INITIALISATION'''
lines = open(my_dir).readlines()

# Create empty matrices (filled with 0s)
num_of_points = int(lines[0])  # the first line in a file is a number of points
distance_matrix = [[0 for _ in range(num_of_points)] for _ in range(num_of_points)]

# Fill the matrices
row = 1
for line in lines[2:]: 
    columns = list(map(int,line.strip().split()))  # we need a list without spaces
    for column in range(len(columns)):
        # filling the matrices with distances based on row/col
        distance_matrix[row][column] = columns[column]
        # creating the other half of each matrix
        distance_matrix[column][row] = columns[column]
    row +=1

# Initial population
pop = new_population(num_of_ind, num_of_points)

# Evaluate population
fitness = evaluate_population(pop, distance_matrix)

min_ind = (pop[0][:], fitness[0])  # a starting individual for comparison

for i in range(1, len(pop)):
    if fitness[i] < min_ind[1]:  # min_ind[1] - indicates the sum in min_ind
        min_ind = (pop[i][:], fitness[i])  # overwrite min_ind

print("The best individual after selection:", min_ind[1])

'''GENETIC ALGORITHM LOOP'''
for gen in range(num_of_gen):
    pop = selection(pop, fitness, k)
    pop_0 = crossover(pop, pc)
    mutation(pop_0, pm)
    fitness_0 = evaluate_population(pop_0, distance_matrix)
    pop = succession(pop, fitness, pop_0, fitness_0)
    fitness = evaluate_population(pop, distance_matrix)
    
    for i in range(num_of_ind):
        if fitness[i] < min_ind[1]:
            min_ind = (pop[i][:],fitness[i]) 
    
    if gen % 100 == 0:  # print every 100th result for monitoring purposes
        print(min_ind[1])

'''RESULTS'''
print("Best solution found:")
print_individual(min_ind[0],min_ind[1])
