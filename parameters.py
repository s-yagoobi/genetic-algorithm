################################
# Simulation parameters
################################
# Population size

population_size = 4

# Number of edges of a fully-connected graph
b = int(population_size * (population_size - 1) / 2)

# Number of parents producing next generation
top_parents = 5

# Number of realization until a mutant gets fixed or goes extinct
number_of_realization = 100

fitness = 2

# size of the pool
pool_size = 10

# Number of times the process for finding the optimized graphs repeats
number_of_GA_iterations = 10

# rate of error in recombination
error_rate = 0.001
