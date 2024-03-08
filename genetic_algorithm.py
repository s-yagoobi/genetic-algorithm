import networkx as nx
import numpy as np
import random

def random_graphs(pool_size, population_size, b):
     """
    Generate a list of random connected graphs.
    
    Parameters:
        -pool_size: size of the pool or the number of graphs in the pool
        -population_size: number of nodes of a network or a graph
        -b: number of possible links or edges of a graph or network

    Returns:
        - RG: List of pool_size initial random connected graphs.

    Generates a list of connected graphs by repeatedly calling the `get_RG` function.
    The number of graphs generated is determined by the variable `pool_size`.

    :return: List of connected graphs.
    """
    i = 0
    RG = []
    
    while len(RG) != pool_size:
        RG.append(get_RG(population_size, b, i))
        i += 1
        
    return RG

def get_RG(population_size, b, seed=0):
    """
    Generate a random connected graph
    
    Parameters:
        -population_size: number of nodes of a network or a graph
        -b: number of possible links or edges of a graph or network
        - seed: The random seed for reproducibility.
          (Default: 0)

    Returns:
        - G: A random connected graph as an adjacency matrix.

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
        # generate a a binary random matrix of size b 
        matrix1 = np.array([np.random.randint(2) for _ in range(b)]) 
        
        # retrieve the indices of the upper triangle of a matrix with the size population_size*population_size
        upper_indices = np.triu_indices(population_size, k = 1)

        G = np.zeros([population_size, population_size])
        G[upper_indices] = matrix1
        G += G.T
        #convert array G to a format that is readable to library networkx
        G1 = nx.from_numpy_array(G)
        
        #check if the generated random graph is connected
        if nx.is_connected(G1):
            return G1





def fixation_probability(graph, number_of_realization, fitness, population_size):
    """
    Calculate the average fixation probability of a mutant in a graph

    Parameters:
        - graph: The adjacency matrix representing the population structure.
        - number_of_realization: number 

    Returns:
        - fixation_probability: Probability of fixation for a mutant in the given graph.

    Fixation_Time: 
        - List storing the fixation times for multiple realizations.
    """
    Fixation_Time = []

    for i in range(number_of_realization):
        # Calculate fixation time
        time = fixation_time(graph, population_size, fitness)
        Fixation_Time.append(time)
    
    # Remove the entries of the list that equal to 1 which correspond to extinction
    Fixation_Time = list(filter((1).__ne__, Fixation_Time))

    return len(Fixation_Time) / number_of_realization

def fixation_time(graph, population_size, fitness):
    """
    This function calculates the fixation time of a mutant in a graph
    it reaturn 1 if instead of fixation there is an extinction
    """
    # Configuration determines whether a node in graph is occupied by a mutant or a wild-type
    # initially the whole network is occupied by wild-types
    Configuration = np.zeros(population_size)  
    
    # the fitness of a mutant is 1 if it is a wild-type otherwise it is equal to fitness
    Fitness = np.ones(population_size)  

    # one of the individuals goes through mutation randomly
    first_mutant = random.randrange(population_size)  
    Configuration[first_mutant] = 1
    Fitness[first_mutant] = fitness
    number_of_mutants = 1
    time = 0

    while number_of_mutants != population_size:
        time += 1
        
        # the probability of selecting an individual for birth
        probability_of_Birth = Fitness / (population_size - number_of_mutants + fitness * number_of_mutants)  
        
        # cumulative of probability
        cumulative_of_probability_of_birth = np.cumsum(probability_of_Birth) 

        #selecting an individual for birth randomly but proportional to its fitness
        rand = random.random()  
        birth_node = np.where(rand <= cumulative_of_probability_of_birth)[0][0] 

        # find all neighbors of the individual that is selected for birth
        neighbor_of_birth_node = np.where(graph[birth_node, 0:population_size] == 1)[0]  
        
        # converting array to list.
        neighbor_of_birth_node.tolist()

        # select one  of the neighbors randomly  
        death_node = random.choice(neighbor_of_birth_node)

        # replace the node chosen for death by the offspring of the node chosen for birth 
        Configuration[death_node] = Configuration[birth_node]
        Fitness[death_node] = Fitness[birth_node]

        # calculate number of mutants in the population
        number_of_mutants = sum(Configuration)  

        #if the number of mutants is zero it means that we reached extinction 
        # and we need to exclude this trial as we are interested in fixation
        if number_of_mutants == 0:
            time = 1
            break

    return time





def cross_over(G1, G2, error_rate, population_size):
    """
    Combine two graphs G1 and G2 and reproduce a new graph
    """
    
    # Pick up the upper triangle elements of the matrices G1 and G2
    # Because the matrices are symmetric and having only the upper triangle is sufficient
    parents1 = G1[np.triu_indices(population_size, k=1)] 
    parents2 = G2[np.triu_indices(population_size, k=1)]
    
    # The process of recombination    
    # If both or none of the parents have a link this will be inherited to the offspring
    # If one of the parents has a link and the other doesn't, the offspring either has the link with probability 0.5
    # or it does not have with probability 0.5     
    birth = np.where(parents1 == parents2, parents1, np.random.randint(2, size=len(parents1)))
    
    # During recombination, an error might occur with probability error_rate
    # Mutation
    if random.random() >= error_rate:
        rand_index = random.randrange(len(birth))
        birth[rand_index] = abs(birth[rand_index] - 1)
        
    # Build the adjacency matrix for the offspring
    offspring = np.zeros([population_size, population_size])
    upper_indices = np.triu_indices(population_size, k=1)
    offspring[upper_indices] = birth
    offspring += offspring.T
   
    
    return offspring




def new_offspring(G1, G2, number_of_realization, fitness, population_size, error_rate):
    """
    Check the offspring if it is connected or not and if its fixation probability is higher or lower than its parents.
    If it is lower then we keep the parents with the highest fixation probability in the new pool, otherwise we keep the offspring.
    """
    
    # Crossover 
    offspring = cross_over(G1, G2, error_rate, population_size)
    
    # Convert offspring to a format that is readable by networkx
    offspring_matrix = np.asmatrix(offspring)
    offspring_graph = nx.from_numpy_matrix(offspring_matrix)

    # Check if the offspring is connected
    if nx.is_connected(offspring_graph):
        
        # Compare the fixation probability of the newborn with the parents
        # Save the graph with the maximum fixation probability as the new offspring
        new_family = [G1, G2, offspring]
        fix_prob_1 = fixation_probability(G1, number_of_realization, fitness, population_size)
        fix_prob_2 = fixation_probability(G2, number_of_realization, fitness, population_size)
        fix_prob_offspring = fixation_probability(offspring, number_of_realization, fitness, population_size)

        # Find the index of the graph with maximum fixation probability
        max_ind = np.argmax([fix_prob_1, fix_prob_2, fix_prob_offspring])

        # Save the graph with the maximum fixation probability as a new generation graph
        offspring = new_family[max_ind]
        fix_prob_offspring = max([fix_prob_1, fix_prob_2, fix_prob_offspring])
        
    else:
        new_family = [G1, G2]
        fix_prob_1 = fixation_probability(G1, number_of_realization, fitness, population_size)
        fix_prob_2 = fixation_probability(G2, number_of_realization, fitness, population_size)

        # Find the index of the graph with maximum fixation probability
        max_ind = np.argmax([fix_prob_1, fix_prob_2])

        # Save the graph with the maximum fixation probability as a new generation graph
        offspring = new_family[max_ind]
        fix_prob_offspring = max([fix_prob_1, fix_prob_2])
                
    return offspring, fix_prob_offspring
  


def new_generation(pool, fix_prob_pool, top_fix_prob):

    # produce the next generation
    # sort the graphs based on their fixation probability
    # Sort the list in descending order
    sorted_fixation_probabilities = sorted(fix_prob_pool, reverse=True)    

    # Get the highest fixation probabilities in the pool
    top_fixation_probabilities = sorted_fixation_probabilities[:top_fix_prob]

    # Get the indices of the graphs with the highest fixation probabilities
    top_indices_fixation_probabilities = [fix_prob_pool.index(val) for val in top_fixation_probabilities]

    # from the graphs with highest fixation porbability build the new generation
    new_pool = []
    fix_prob_new_pool = []
    for i in range(pool_size):
        # from the top graphs choose two graphs to recombine
        parents = random.sample(top_indices_fixation_probabilities, 2)
        G1 = pool[parents[0]]
        G2 = pool[parents[1]]
        offspring, fix_prob_offspring = new_offspring(G1, G2, number_of_realization, fitness, population_size, error_rate)
        new_pool.append(offspring)
        fix_prob_new_pool.append(fix_prob_offspring)

    return new_pool, fix_prob_new_pool



def find_graph_with_highest_fix_prob(number_of_iteration):

    max_fix_prob = []
    #initialize the pool of graphs
    pool = random_graphs(pool_size, population_size)
    
    # calculate the fixation probability of the initial pool
    fix_prob_pool = []
    for graph in pool:
        fix_prob_pool.append(fixation_probability(graph, number_of_realization, fitness, population_size))
    max_fix_prob.append(max(fix_prob_pool))
    for i in range(number_of_iteration):

        pool, fix_prob_pool = new_generation(pool, fix_prob_pool, top_fix_prob)
        max_fix_prob.append(max(fix_prob_pool))

    return max_fix_prob
         






        
    
    
