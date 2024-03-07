"""
:module : Script to calculate fixation probabilities in parallel using MPI.
"""

#from mpi4py import MPI
import networkx as nx
import numpy as np
import random
import time
import math

from parameters import population_size,\
     number_of_GA_iterations,\
    number_of_parents,\
    number_of_realization,\
    fitness,\
    b,\
    top_parents,\
    recom_prob

def star():
    S=np.zeros([population_size, population_size])
    S[0,:]=1
    S[:,0]=1
    S[0,0]=0
    return S

def complete():    
    C=np.ones([population_size, population_size])
    for i in range(population_size):
        C[i,i]=0
    return C    

def random_graphs(n):
    '''generating number_of_parents initial random graphs
    n: population size
    '''
    RG=[]
    RG.append(star())
    l=prob_distribution(n)
    L=np.arange(n-1, n*(n-1)//2+1)
    for _ in range(number_of_parents-1):
        
        k=np.random.choice(L,size=1,replace=False, p=l)
        G=get_RG(n, k)
        
        RG.append(np.asarray(nx.adjacency_matrix(G).todense()))
        
    return RG



def get_RG(num_nodes,num_edges):
    """
    Generate a random graph
    """
    # Loop until we find a connected graph.
    while True:
       
        G=nx.gnm_random_graph(num_nodes, num_edges, seed=None, directed=False)
        if (nx.is_connected(G)==True):
            return G

def prob_distribution(n):
    """
    probability distribution of graphs with different number of edges 
    """
    prob=[]
    for k in range(n-1, n*(n-1)//2+1):
        prob.append(math.factorial(n*(n-1)//2)/(math.factorial(k)*math.factorial(n*(n-1)//2-k)))
    prob1=np.array(prob)
    return prob1/np.sum(prob1)#*number_of_parents/np.sum(prob1)





def fix_prob(graph):
    """ fixation probability of a graph """

    Fixation_Time=[]
    
    for i in range(number_of_realization):
        time = calc_time(graph , i)
        #print(time)
        Fixation_Time.append(time)
    Fixation_Time =list(filter((1).__ne__, Fixation_Time))
    return len(Fixation_Time)/number_of_realization, sum(Fixation_Time)/len(Fixation_Time)


def calc_time(graph, i):
    Configuration = np.zeros(population_size)  
    Fitness = np.ones(population_size)  
    first_mutant =1 #i % population_size 
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
        #print(death_node)
        Configuration[death_node] = Configuration[birth_node]
        Fitness[death_node] = Fitness[birth_node]
        number_of_mutants = sum(Configuration)  # calculate number of B in each time
        #print(number_of_mutants)
        if number_of_mutants == 0:
            fixation_time = 1
            break

    return fixation_time

t0=time.time()
FP, FT = fix_prob(star())
print(FT)    

print(time.time()-t0)

"""
def cross_over(G1,G2):
    '''combine two graphs to obtain a new parents'''
    parents1=G1[np.triu_indices(population_size, k = 1)] #pick up the upper triangle elements of the matrix
    #print(parents1)
    parents2=G2[np.triu_indices(population_size, k = 1)]
    
    #the process of recombination         
    birth1=np.concatenate((parents1,parents2),axis=0)
    
    random.shuffle(birth1)
    
    birth=np.zeros([len(parents1)])
    birth=birth1[0:len(parents1)]
       
    
        
    #build the adjacency matrix for the offspring
    offspring=np.zeros([population_size,population_size])
    upper_indices=np.triu_indices(population_size, k = 1)
    lower_indices=(upper_indices[1],upper_indices[0])
    offspring[upper_indices]=birth
    offspring[lower_indices]=birth
    return offspring

def mutation(G):
    rand=np.random.choice(population_size,size=2, replace=False)
    G[rand[0],rand[1]]=abs(G[rand[0],rand[1]]-1)
    G[rand[1],rand[0]]=abs(G[rand[1],rand[0]]-1)
    return G

  

def new_offspring(random_graphs, fixation_probabilities):
    p=random.random()
    top_performers = np.argsort(fixation_probabilities)[len(fixation_probabilities) - top_parents:len(fixation_probabilities)]
    if p<recom_prob:

       #crossover
       start = time.time()
       # choose two individual among the highest fix-prob to mate
       

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

       # check if the offspring is connected
       offspring_matrix = np.asmatrix(offspring)
       offspring_graph = nx.from_numpy_matrix(offspring_matrix)

       if nx.is_connected(offspring_graph):
          new_family = [G1, G2, offspring]
          Fixation_new_family = [fixation_probabilities[parents[0]], fixation_probabilities[parents[1]], fix_prob(offspring)]
          max_ind = Fixation_new_family.index(max(Fixation_new_family))
          offspring=new_family[max_ind]
          fix_prob_offspring=max(Fixation_new_family)
       else:
          new_family = [G1, G2]
          Fixation_new_family = [fixation_probabilities[parents[0]], fixation_probabilities[parents[1]]]
          max_ind = Fixation_new_family.index(max(Fixation_new_family))
          offspring=new_family[max_ind]
          fix_prob_offspring=max(Fixation_new_family)

    else:
        #mutation
        parent=np.random.choice(top_performers,size=1)
        G = random_graphs[parent]
        offspring=mutation(G)
        # check if the offspring is connected
        offspring_matrix = np.asmatrix(offspring)
        offspring_graph = nx.from_numpy_matrix(offspring_matrix)

       if nx.is_connected(offspring_graph):
          new_family = [G, offspring]
          Fixation_new_family = [fixation_probabilities[parent], fix_prob(offspring)]
          max_ind = Fixation_new_family.index(max(Fixation_new_family))
          offspring=new_family[max_ind]
          fix_prob_offspring=max(Fixation_new_family)
       else:
          offspring=G
          fix_prob_offspring=fixation_probabilities[parent]

    return offspring, fix_prob_offspring, time.time()-start




def new_generation(random_graphs, fixation_probabilities):

    offspring, fix_prob_offspring, duration = new_offspring(random_graphs, fixation_probabilities)

    return offspring, fix_prob_offspring


   



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
    fixation_probabilities=[]
    for g in RG:
        fixation_probabilities.append(fix_prob(g))
    fixation_probabilities = np.asarray(fixation_probabilities)

# Master process

for j in range(number_of_GA_iterations):

    RG = comm.bcast(RG, root=0)
    fixation_probabilities = comm.bcast(fixation_probabilities, root=0)

    if rank == 0:
        print("************************************")
        print(" Generation {}".format(j+1))
        print(" Top performers:")
        top_performers = np.argsort(fixation_probabilities)[len(fixation_probabilities) - top_parents:len(fixation_probabilities)]
        print(top_performers)
        print("************************************")

    my_RGs = []
    my_FPs = []
    for i in range(len(fixation_probabilities)):
        if i % size == rank:
            rg, fp = new_generation(RG, fixation_probabilities)
            my_RGs.append(rg)
            my_FPs.append(fp)

    RG = comm.gather(my_RGs, root=0)
    fixation_probabilities = comm.gather(my_FPs, root=0)

    # Unpack
    if rank == 0:
        RG = [rg for y in RG for rg in y]
        fixation_probabilities = [fp for y in fixation_probabilities for fp in y]


"""
  
