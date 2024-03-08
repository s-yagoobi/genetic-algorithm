import networkx as nx
import numpy as np
import random

from parameters import pool_size, population_size, b,\
    number_of_realization,\
    number_of_GA_iterations,\
    fitness, top_parents, error_rate


def random_graphs(pool_size):
    """
    Generate a list of random connected graphs.
    
    Parameters:
        - pool_size: Size of the pool or the number of graphs in the pool.

    Returns:
        - RG: List of pool_size initial random connected graphs.

    Generates a list of connected graphs by repeatedly calling the `get_RG` function.
    The number of graphs generated is determined by the variable `pool_size`.
    """
    RG = []
    for i in range(pool_size):
        RG.append(get_RG(population_size, b, i))
    return RG

def get_RG(population_size, b, seed=0):
    """
    Generate a random connected graph
    
    Parameters:
        - population_size: Number of nodes of a network or a graph.
        - b: Number of possible links or edges of a graph or network.
        - seed: The random seed for reproducibility. (Default: 0)

    Returns:
        - G: A random connected graph as a NetworkX graph.

    Randomly generates a binary matrix of size `b` and constructs an upper triangular
    adjacency matrix with the same values. The matrix is used to create a graph, and the
    process is repeated until a connected graph is obtained.

    Note: The generated graph is guaranteed to be connected.
    """
    random.seed(seed)
    
    if population_size <= 0:
        raise ValueError("Population size must be greater than 0")
    if b <= 0:
        raise ValueError("Parameter 'b' must be greater than 0")
    
    # Loop until we find a connected graph.
    while True:
        # generate a binary random matrix of size b 
        matrix1 = np.array([np.random.randint(2) for _ in range(b)]) 
        
        # retrieve the indices of the upper triangle of a matrix with the size population_size*population_size
        upper_indices = np.triu_indices(population_size, k=1)

        G = np.zeros([population_size, population_size])
        G[upper_indices] = matrix1
        G += G.T

        # Convert array G to a graph
        G1 = nx.from_numpy_array(G)
        
        # Check if the generated random graph is connected
        if nx.is_connected(G1):
            return nx.to_numpy_array(G1)




def fixation_probability(graph, number_of_realization):
    """
    Calculate the average fixation probability of a mutant in a graph

    Parameters:
        - number_of_realization: number of iterations of the fixation or extinction process
          type: int

    Returns:
        - fixation_probability: Probability of fixation for a mutant in the given graph.
    """
    fixation_times = []

    for i in range(number_of_realization):
        # Calculate fixation time
        time = fixation_time(graph, population_size, fitness)
        fixation_times.append(time)
    
    # Remove the entries of the list that equal to 1 which correspond to extinction
    fixation_times = list(filter(lambda x: x != 1, fixation_times))

    return len(fixation_times) / number_of_realization


def fixation_time(graph, population_size, fitness):
    """
    Calculate the fixation time of a mutant in a graph
    Returns 1 if instead of fixation there is extinction

    Parameters:
        - graph: The adjacency matrix representing the population structure.
        - population_size: number of nodes in the population
        - fitness: fitness of mutant

    Returns:
        - time: The number of steps until the mutant takes over the entire population.
    """
    # Configuration determines whether a node in graph is occupied by a mutant or a wild-type
    # initially the whole network is occupied by wild-types
    Configuration = np.zeros(population_size)  
    
    # The fitness of a mutant is 1 if it is a wild-type otherwise it is equal to fitness
    Fitness = np.ones(population_size)  

    # One of the individuals goes through mutation randomly
    first_mutant = random.randrange(population_size)  
    Configuration[first_mutant] = 1
    Fitness[first_mutant] = fitness
    number_of_mutants = 1
    time = 0

    while number_of_mutants != population_size:
        time += 1
        
        # The probability of selecting an individual for birth
        probability_of_birth = Fitness / (population_size - number_of_mutants + fitness * number_of_mutants)  
        
        # Cumulative of probability
        cumulative_of_probability_of_birth = np.cumsum(probability_of_birth) 

        # Selecting an individual for birth randomly but proportional to its fitness
        rand = random.random()  
        birth_node = np.where(rand <= cumulative_of_probability_of_birth)[0][0] 

        # Find all neighbors of the individual that is selected for birth
        neighbor_of_birth_node = np.where(graph[birth_node, 0:population_size] == 1)[0]  

        # Select one of the neighbors randomly  
        death_node = random.choice(neighbor_of_birth_node)

        # Replace the node chosen for death by the offspring of the node chosen for birth 
        Configuration[death_node] = Configuration[birth_node]
        Fitness[death_node] = Fitness[birth_node]

        # Calculate number of mutants in the population
        number_of_mutants = sum(Configuration)  
        
        # If the number of mutants is zero it means that we reached extinction 
        # and we need to exclude this trial as we are interested in fixation
        if number_of_mutants == 0:
            time = 1
            break

    return time




def cross_over(G1, G2, population_size,  error_rate ):
    """
    Combine two graphs G1 and G2 and reproduce a new graph

    Parameters:
        - G1: The adjacency matrix of the first parent.
          type: array
        - G2: The adjacency matrix of the second parent.
          type: array
        - error_rate: The probability that the offspring does not inherit a specific characteristic of the parents.
          type: float
        - population_size: Number of nodes in the population.
          type: int

    Returns:
        - offspring: The offspring of G1 and G2, plus mutation.
          type: array

    parents1:
        - The upper triangle elements of the adjacency matrix of G1.
    parents2:
        - The upper triangle elements of the adjacency matrix of G2.
    birth:
        - Recombination of parents1 and parents2 plus mutation.
    """
    
    # Pick up the upper triangle elements of the matrices G1 and G2
    # Because the matrices are symmetric and having only the upper triangle is sufficient
    parents1 = G1[np.triu_indices(population_size, k=1)] 
    parents2 = G2[np.triu_indices(population_size, k=1)]
    
    # The process of recombination    
    # If both or none of the parents have a link, this will be inherited to the offspring
    # If one of the parents has a link and the other doesn't, the offspring either has the link with probability 0.5
    # or it does not have with probability 0.5
    # np.where(condition, x, y): This function selects elements from x or y based on the boolean array condition.
    # Wherever condition is True, the corresponding element from x is selected, and where it is False, the corresponding element from y is selected.
    birth = np.where(parents1 == parents2, parents1, np.random.randint(2, size=len(parents1)))
    
    # During recombination, an error might occur with probability error_rate
    # Mutation
    if random.random() < error_rate:
        rand_index = random.randrange(len(birth))
        birth[rand_index] = abs(birth[rand_index] - 1)
        
    # Build the adjacency matrix for the offspring
    offspring = np.zeros([population_size, population_size])
    upper_indices = np.triu_indices(population_size, k=1)
    offspring[upper_indices] = birth
    offspring += offspring.T
    
    return offspring




def new_parent(G1, G2):
    """
    Check the offspring produced by the crossover function if it is connected or not 
    and if its fixation probability is higher or lower than its parents.
    If it is lower, then we keep the parents with the highest fixation probability in the new pool, 
    otherwise, we keep the offspring as the new parent in the next generation.

    Parameters:
        - G1: The adjacency matrix of the first parent.
          type: array
        - G2: The adjacency matrix of the second parent.
          type: array

    Returns:
        - offspring: The adjacency matrix of the new parent.
          type: array
        - fix_prob_offspring: The fixation probability of the new parent.
          type: float
    """
    
    # Crossover 
    offspring = cross_over(G1, G2, population_size, error_rate)
    
    # Convert offspring to a format that is readable by networkx
    offspring_matrix = np.asmatrix(offspring)
    offspring_graph = nx.from_numpy_matrix(offspring_matrix)

    # Check if the offspring is connected
    if nx.is_connected(offspring_graph):
        
        # Compare the fixation probability of the newborn with the parents
        # Save the graph with the maximum fixation probability as the new offspring
        new_family = [G1, G2, offspring]
        fix_prob_1 = fixation_probability(G1, number_of_realization)
        fix_prob_2 = fixation_probability(G2, number_of_realization)
        fix_prob_offspring = fixation_probability(offspring, number_of_realization)

        # Find the index of the graph with maximum fixation probability
        max_ind = np.argmax([fix_prob_1, fix_prob_2, fix_prob_offspring])

        # Save the graph with the maximum fixation probability as a new generation graph
        offspring = new_family[max_ind]
        fix_prob_offspring = max([fix_prob_1, fix_prob_2, fix_prob_offspring])
        
    else:
        new_family = [G1, G2]
        fix_prob_1 = fixation_probability(G1, number_of_realization)
        fix_prob_2 = fixation_probability(G2, number_of_realization)

        # Find the index of the graph with maximum fixation probability
        max_ind = np.argmax([fix_prob_1, fix_prob_2])

        # Save the graph with the maximum fixation probability as a new generation graph
        offspring = new_family[max_ind]
        fix_prob_offspring = max([fix_prob_1, fix_prob_2])
                
    return offspring, fix_prob_offspring

  


def new_generation(pool, fix_prob_pool, top_parents):
    """
    Produce the next generation using the top parents.

    Parameters:
        - pool: A list of graphs in the current generation.
          type: list of arrays
        - fix_prob_pool: The fixation probability of the graphs in the current pool.
          type: list of floats
        - top_parents: Number of parents with the highest fixation probability.
          type: int

    Returns:
        - new_pool: A list of graphs in the new pool.
          type: list of arrays
        - fix_prob_new_pool: The fixation probability of the graphs in the new pool.
          type: list of floats
    """
    
    # Sort the pool based on the fixation probability in descending order
    sorted_indices = sorted(range(len(fix_prob_pool)), key=lambda k: fix_prob_pool[k], reverse=True)
    
    # Get the indices of the top parents
    top_indices = sorted_indices[:top_parents]

    # Get the graphs with the highest fixation probabilities
    top_pool = [pool[i] for i in top_indices]
    
    # Build the new generation
    new_pool = []
    fix_prob_new_pool = []
    for _ in range(len(pool)):
        # Choose two parents randomly from the top pool for crossover
        parents = random.sample(top_pool, 2)
        G1, G2 = parents[0], parents[1]
        offspring, fix_prob_offspring = new_parent(G1, G2)
        new_pool.append(offspring)
        fix_prob_new_pool.append(fix_prob_offspring)
    
    return new_pool, fix_prob_new_pool




def find_graph_with_highest_fix_prob(number_of_GA_iterations):
    """
    For a population of size `population_size`, find out how individuals in the population should be connected
    in order for the population to have the highest fixation probability.

    Parameters:
        - number_of_GA_iterations: the number of times that we repeat genetic algorithm
        

    Returns:    
        - max_fix_prob: a list of maximum fixation probabilities along the genetic algorithm process
        - graph_with_max_fix_prob : a list of graphs with the maximum fixation probability along the genetic algorithm process
    """

    max_fix_prob = []
    graph_with_max_fix_prob = []

    # Initialize the pool of graphs
    pool = random_graphs(pool_size)
    
    # Calculate the fixation probability of the initial pool
    fix_prob_pool = [fixation_probability(graph, number_of_realization) for graph in pool]

    graph_with_max_fix_prob.append(pool[np.argmax(fix_prob_pool)])    
    max_fix_prob.append(max(fix_prob_pool))
    

    # Start the process of updating the pool over and over again
    for i in range(number_of_GA_iterations):

        new_pool, fix_prob_new_pool = new_generation(pool, fix_prob_pool, top_parents)
        
        if max(fix_prob_new_pool) >= max_fix_prob[-1]:

            # Graph with the maximum fixation probability in the pool
            graph_with_max_fix_prob.append(new_pool[np.argmax(fix_prob_new_pool)])

            # The maximum fixation probability in the pool
            max_fix_prob.append(max(fix_prob_new_pool))
            pool = new_pool
            fix_prob_pool = fix_prob_new_pool

    

    return max_fix_prob, graph_with_max_fix_prob

         

print(find_graph_with_highest_fix_prob(number_of_GA_iterations))




        
    
    
