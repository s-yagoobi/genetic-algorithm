################################
# Simulation parameters
################################
# Population size
population_size = 10

# Number of edges of a fully-connected graph
b=int(population_size*(population_size-1)/2)

# Number of parents producing next generation
top_parents = 5

# Number of realization until a mutant gets fixed or goes extinct
number_of_realization = 100
fitness = 2

# The number of parents
number_of_parents = 10

# Number of times the process for finding the optimized graphs repeats
number_of_GA_iterations = 10

#probability of recombination
recom_prob=0.1