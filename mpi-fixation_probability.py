"""
:module : Script to calculate fixation probabilities in parallel using MPI.
"""

from mpi4py import MPI
import mpi4py
import networkx as nx
import numpy as np
import random
import time

################################
# Simulation parameters
################################
# Population size
population_size=10          

# Number of edges of a fully-connected graph
b=int(population_size*(population_size-1)/2) 

# Number of parents producing next generation
top_parents=10          

# Number of realization untill a mutant gets fixed or goes extict
number_of_realization=1000          
fitness=2

# The number of parents
number_of_parents=100          

# Number of times the process for finding the optimized graphs repeats
time_genetic_algorithm=10             

#Initial random graphs
def random_graphs():
    '''generating number_of_parents initial random graphs'''
    i=0
    RG=[]
    
    while len(RG) !=number_of_parents:
        RG.append(get_RG(i))
        i += 1
        
    return RG

def get_RG(seed=0):
    """
    Generate a random graph
    
    :param seed: The random seet
    :type  seed: int
    
    """
    
    random.seed(seed)
    
    # Loop until we find a connected graph.
    while True:
        matrix1 = np.array([np.random.randint(2) for _ in range(b)]) # generate a random matrix of size b

        upper_indices=np.triu_indices(population_size, k = 1)
        lower_indices=(upper_indices[1],upper_indices[0])

        G=np.zeros([population_size,population_size])
        G[upper_indices]=matrix1
        G[lower_indices]=matrix1
        G1 = nx.from_numpy_array(G)

        if (nx.is_connected(G1)==True):
            return G

#fixation probability of a graph
def fix_prob(graph):    
    
    Fixation_Time=[]
    
    for i in range(number_of_realization):
        time = calc_time(graph)
        Fixation_Time.append(time)
    Fixation_Time =list(filter((1).__ne__, Fixation_Time))
    return len(Fixation_Time)/number_of_realization

def calc_time(graph):
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

        Nighbor_of_birth_node = np.where(graph[birth_node, 0:population_size] == 1)[0]  # find all neighbors of m
        
        Nighbor_of_birth_node.tolist()  # converting array to list.
        death_node = random.choice(Nighbor_of_birth_node)  # select a neighbor.
        Configuration[death_node] = Configuration[birth_node]
        Fitness[death_node] = Fitness[birth_node]
        number_of_mutants = sum(Configuration)  # calculate number of B in each time
        if number_of_mutants == 0:
            fixation_time = 1
            break

    return fixation_time


# In[5]:


def cross_over(G1,G2):
    parents1=G1[np.triu_indices(population_size, k = 1)] #pick up the upper triangle elements of the matrix
    #print(parents1)
    parents2=G2[np.triu_indices(population_size, k = 1)]
    
    #the process of recombination         
    rand=random.random()
    birth1=np.concatenate((parents1,parents2),axis=0)
    
    random.shuffle(birth1)
    
    birth=np.zeros([len(parents1)])
    birth=birth1[0:len(parents1)]
       
    #mutation
    rand=random.randrange(len(birth))
    birth[rand]=abs(birth[rand]-1)
        
    #build the adjacency matrix for the offspring
    offspring=np.zeros([population_size,population_size])
    upper_indices=np.triu_indices(population_size, k = 1)
    lower_indices=(upper_indices[1],upper_indices[0])
    offspring[upper_indices]=birth
    offspring[lower_indices]=birth
    return offspring


# In[6]:


def new_offspring(i): 
            start=time.time()
            
            
            
            #crossover 
            parents = random.sample(FIX_Prob_sorted[number_of_parents - top_parents:number_of_parents], 2)  # choose two individual among the highest fix-prob to mate
            G1 = RG[parents[0]]
            G2 = RG[parents[1]]
            
            offspring = cross_over(G1, G2)
            
            # check if the offspring is connected
            offspring_matrix = np.asmatrix(offspring)
            offspring_graph = nx.from_numpy_matrix(offspring_matrix)
             
            if nx.is_connected(offspring_graph):
                

                new_family = [G1, G2, offspring]
                Fixation_new_family = [FIX_Prob[parents[0]], FIX_Prob[parents[1]], fix_prob(offspring)]
                max_ind = Fixation_new_family.index(max(Fixation_new_family))
                offspring=new_family[max_ind]
                fix_prob_offspring=max(Fixation_new_family)
                #RG_new.append(new_family[max_ind])
                #FIX_prob_new.append(max(Fixation_new_family))
            else:
                new_family = [G1, G2]
                Fixation_new_family = [FIX_Prob[parents[0]], FIX_Prob[parents[1]]]
                max_ind = Fixation_new_family.index(max(Fixation_new_family))
                offspring=new_family[max_ind]
                fix_prob_offspring=max(Fixation_new_family)
                
            return offspring, fix_prob_offspring, time.time()-start    
         
# Setup the MPI communication
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Empty array to store results from each process.
fxp=np.zeros(number_of_parents, dtype=np.float64)

# Distribute in round-robin fashion.
fxp = None
for i in range(number_of_parents):
    if i % size == rank:
        graph = get_RG(i)
        print("Running random graph {} on rank {}.".format(i,rank))
        fxp = fix_prob(graph)

print("Rank {} is alive.".format(rank))

fxp = comm.gather(fxp, root=0)

if rank == 0:
    print(fxp)
