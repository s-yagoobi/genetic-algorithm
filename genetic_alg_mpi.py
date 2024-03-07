"""
:module : Script to find the graph with a certain size "b" with the highest fixation probability! 
This code is an MPI (Message Passing Interface) parallelization setup for a genetic algorithm (GA) to evolve and optimize graphs.
"""

from mpi4py import MPI
import networkx as nx
import numpy as np
import random
import time

from parameters import population_size,\
     number_of_GA_iterations,\
    number_of_parents,\
    number_of_realization,\
    fitness,\
    b,\
    top_parents


def random_graphs():
    """
    Generate a list of random connected graphs.

    Returns:
        - RG: List of number_of_parents initial random connected graphs.

    Generates a list of connected graphs by repeatedly calling the `get_RG` function.
    The number of graphs generated is determined by the variable `number_of_parents`.

    :return: List of connected graphs.
    """
    i = 0
    RG = []

    while len(RG) != number_of_parents:
        RG.append(get_RG(i))
        i += 1

    return RG


def get_RG(seed=0):
    """
    Generate a random connected graph.

    Parameters:
        - seed: The random seed for reproducibility.
          (Default: 0)

    Returns:
        - G: A random connected graph as an adjacency matrix.

    Randomly generates a binary matrix of size `b` and constructs an upper triangular
    adjacency matrix with the same values. The matrix is used to create a graph, and the
    process is repeated until a connected graph is obtained.

    :param seed: The random seed for reproducibility.
    :type seed: int

    Note: The generated graph is guaranteed to be connected.
    """
    random.seed(seed)

    # Loop until we find a connected graph.
    while True:
        # Generate a random binary matrix of size b
        matrix1 = np.array([np.random.randint(2) for _ in range(b)])

        # Create upper triangular indices and corresponding lower indices
        upper_indices = np.triu_indices(population_size, k=1)
        lower_indices = (upper_indices[1], upper_indices[0])

        # Build the upper triangular adjacency matrix
        G = np.zeros([population_size, population_size])
        G[upper_indices] = matrix1
        G[lower_indices] = matrix1

        # Convert the adjacency matrix to a graph
        G1 = nx.from_numpy_array(G)

        # Check if the graph is connected
        if nx.is_connected(G1):
            return G



def fix_prob(graph):
    """
    Calculate the fixation probability of a mutant in a graph.

    Parameters:
        - graph: The adjacency matrix representing the population structure.

    Returns:
        - fixation_probability: Probability of fixation for a mutant in the given graph.

    Fixation_Time: 
        - List storing the fixation times for multiple realizations.
    """
    Fixation_Time = []

    for i in range(number_of_realization):
        time = calc_time(graph)
        Fixation_Time.append(time)

    # Filter out realizations with fixation time equal to 1 (extinction)
    Fixation_Time = list(filter((1).__ne__, Fixation_Time))

    # Calculate fixation probability
    fixation_probability = len(Fixation_Time) / number_of_realization
    return fixation_probability



def calc_time(graph):
    """
    Calculate the fixation time of a mutant in a population.

    Parameters:
        - graph: The adjacency matrix representing the population structure.

    Returns:
        - fixation_time: The number of steps until the mutant takes over the entire population.

    Configuration: 
        - The configuration of the population. If a site is occupied by a mutant, its value is one; otherwise, it is 0.
    Fitness: 
        - The fitness array of individuals. Its value is 1 if occupied by a wild-type and 'fitness' if occupied by a mutant.
    first_mutant: 
        - The index of the first individual turned into a mutant.
    number_of_mutants: 
        - The number of mutants in the entire population.
    fixation_time: 
        - The number of steps it takes until the mutant takes over the whole population.
    """
    Configuration = np.zeros(population_size)  
    Fitness = np.ones(population_size)  
    first_mutant = random.randrange(population_size)  
    Configuration[first_mutant] = 1
    Fitness[first_mutant] = fitness
    number_of_mutants = 1
    fixation_time = 0

    while number_of_mutants != population_size:
        fixation_time = fixation_time + 1
        probability_of_Birth = Fitness / (population_size - number_of_mutants + fitness * number_of_mutants)  # probability matrix for selection.
        cumulative_of_probability_of_birth = np.cumsum(probability_of_Birth)  # cumulative of probability
        rand = random.random()  # generating a random number
        birth_node = np.where(rand <= cumulative_of_probability_of_birth)[0][0] 

        Nighbor_of_birth_node = np.where(graph[birth_node, 0:population_size] == 1)[0]  # find all neighbors of birth_node
        
        Nighbor_of_birth_node.tolist()  # converting array to list.
        death_node = random.choice(Nighbor_of_birth_node)  # select a neighbor.
        # replace the death_node with the offspring of birth_node
        Configuration[death_node] = Configuration[birth_node]
        Fitness[death_node] = Fitness[birth_node] 
        number_of_mutants = sum(Configuration)  # calculate the number of mutants in each time

        # if the mutant gets extinct, break.
        if number_of_mutants == 0:
            fixation_time = 1
            break

    return fixation_time



def cross_over(G1, G2):
    """
    Combine two graphs to obtain a new parent.

    Parameters:
        - G1: The adjacency matrix of the first parent.
        - G2: The adjacency matrix of the second parent.

    Returns:
        - offspring: The offspring of G1 and G2, plus mutation.

    parents1: 
        - The upper triangle elements of the adjacency matrix of the first parent.
    parents2: 
        - The upper triangle elements of the adjacency matrix of the second parent.
    birth1: 
        - Recombination of parents1 and parents2.
    birth: 
        - Mutated offspring obtained from birth1.
    """
    parents1 = G1[np.triu_indices(population_size, k=1)]  # pick up the upper triangle elements of the matrix
    parents2 = G2[np.triu_indices(population_size, k=1)]

    # the process of recombination
    birth1 = np.concatenate((parents1, parents2), axis=0)

    random.shuffle(birth1)

    birth = np.zeros([len(parents1)])
    birth = birth1[0:len(parents1)]

    # mutation
    rand = random.randrange(len(birth))
    birth[rand] = abs(birth[rand] - 1)

    # build the adjacency matrix for the offspring
    offspring = np.zeros([population_size, population_size])
    upper_indices = np.triu_indices(population_size, k=1)
    lower_indices = (upper_indices[1], upper_indices[0])
    offspring[upper_indices] = birth
    offspring[lower_indices] = birth
    return offspring



def new_offspring(random_graphs, fixation_probabilities):
    """
    Generate a new offspring through crossover and mutation.

    Parameters:
        - random_graphs: List of random graphs representing the current population.
        - fixation_probabilities: Array of fixation probabilities corresponding to each graph in the population.

    Returns:
        - offspring: The new offspring graph.
        - fix_prob_offspring: Fixation probability of the new offspring.
        - duration: Time taken for the crossover operation.

    Crossover Process:
        - Choose two individuals with the highest fixation probabilities to mate.
        - Generate offspring through crossover using the `cross_over` function.
        - Check if the offspring is a connected graph among parents and offspring.
        
    If Connected:
        - Choose the individual with the highest fixation probability among parents and offspring.

    If Not Connected:
        - Choose the individual with the highest fixation probability among parents.

    """

    # Crossover
    start = time.time()
    # Choose two individuals among the highest fix-prob to mate
    top_performers = np.argsort(fixation_probabilities)[number_of_parents - top_parents:number_of_parents]

    parents = np.random.choice(
        top_performers,
        size=2,
        replace=False,
    )
    try:
        G1 = random_graphs[parents[0]]
        G2 = random_graphs[parents[1]]
    except:
        print("rank = {}, parents = ".format(rank, parents), flush=True)
        raise

    offspring = cross_over(G1, G2)

    # Check if the offspring is connected
    offspring_matrix = np.asmatrix(offspring)
    offspring_graph = nx.from_numpy_matrix(offspring_matrix)

    if nx.is_connected(offspring_graph):
        # If offspring is a connected graph among parents and offspring, choose the one with the highest fixation probability
        new_family = [G1, G2, offspring]
        Fixation_new_family = [fixation_probabilities[parents[0]], fixation_probabilities[parents[1]], fix_prob(offspring)]
        max_ind = Fixation_new_family.index(max(Fixation_new_family))
        offspring = new_family[max_ind]
        fix_prob_offspring = max(Fixation_new_family)
    else:
        # If the offspring is not a connected graph among the parents, pick the one with the highest fixation probability as the new offspring
        new_family = [G1, G2]
        Fixation_new_family = [fixation_probabilities[parents[0]], fixation_probabilities[parents[1]]]
        max_ind = Fixation_new_family.index(max(Fixation_new_family))
        offspring = new_family[max_ind]
        fix_prob_offspring = max(Fixation_new_family)

    return offspring, fix_prob_offspring, time.time() - start


def new_generation(random_graphs, fixation_probabilities):
    """
    Generate a new generation of graphs using the `new_offspring` function.

    Parameters:
        - random_graphs: List of random graphs representing the current population.
        - fixation_probabilities: Array of fixation probabilities corresponding to each graph in the population.

    Returns:
        - offspring: The new generation of graphs.
        - fix_prob_offspring: Fixation probabilities of the new generation.
    """

    offspring, fix_prob_offspring, duration = new_offspring(random_graphs, fixation_probabilities)

    return offspring, fix_prob_offspring


def mpi_parallel_genetic_algorithm():
    """
    Execute a parallel genetic algorithm using MPI.

    This function sets up MPI communication, initializes random graphs,
    calculates fixation probabilities, and performs the genetic algorithm in parallel.

    MPI Communication Setup:
        - Initializes MPI communication, retrieves the size and rank of the MPI process.

    Initial Population Generation:
        - In the master process (rank 0), generates an initial pool of random graphs and calculates fixation probabilities.

    Genetic Algorithm Iterations:
        - For each generation, broadcasts random graphs and fixation probabilities to all processes.
        - In each process, generates new offspring for a subset of parents.
        - Gathers the new offspring and fixation probabilities back to the master process.
        - In the master process, prints information about the top-performing graphs in each generation.

    """
    # Setup the MPI communication
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("*****************************")
        print("* Welcome to MPI            *")
        print("*                           *")
        print("* MPI size = {}              *".format(size))
    else:
        pass

    comm.barrier()
    print("* MPI rank = {}              *".format(rank))

    # Calculate the fixation probability of an initial pool of random graphs
    RG = None
    fixation_probabilities = None
    if rank == 0:
        RG = random_graphs()
        fixation_probabilities = []
        # Calculate fixation probabilities for each graph in the initial pool
        for g in RG:
            fixation_probabilities.append(fix_prob(g))
        fixation_probabilities = np.asarray(fixation_probabilities)

    # Master process iterates through generations
    for j in range(number_of_GA_iterations):
        # Broadcast the random graphs and fixation probabilities to all processes
        RG = comm.bcast(RG, root=0)
        fixation_probabilities = comm.bcast(fixation_probabilities, root=0)

        if rank == 0:
            # Print information about the current generation for the master process
            print("************************************")
            print(" Generation {}".format(j + 1))
            print(" Top performers:")
            # Find and print the indices of the top-performing graphs
            top_performers = np.argsort(fixation_probabilities)[number_of_parents - top_parents:number_of_parents]
            print(top_performers)
            print("************************************")

        my_RGs = []
        my_FPs = []

        # Each process generates new offspring for a subset of parents
        for i in range(number_of_parents):
            if i % size == rank:
                # Generate new offspring for the current parent
                rg, fp = new_generation(RG, fixation_probabilities)
                my_RGs.append(rg)
                my_FPs.append(fp)

        # Gather the new offspring and fixation probabilities to the master process
        RG = comm.gather(my_RGs, root=0)
        fixation_probabilities = comm.gather(my_FPs, root=0)

        # Unpack the gathered data on the master process
        if rank == 0:
            RG = [rg for y in RG for rg in y]
            fixation_probabilities = [fp for y in fixation_probabilities for fp in y]

# Execute the MPI parallel genetic algorithm
mpi_parallel_genetic_algorithm()
