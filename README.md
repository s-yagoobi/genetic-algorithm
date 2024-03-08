# Find the network with the highest fixation probability using the genetic algorithm

In this project, we aim to find out how individuals in a population with a certain size should be connected for the population to have the highest fixation probability.
We only investigate the undirected symmetric networks. 

## Brute Force method
One way to proceed is to examine the fixation probability of all the possible networks with a certain size and compare them. However, this is not a practical approach as by increasing the population size the number of possible networks increases insanely. For instance, with only 10 nodes in a graph, there are 11,716,571 unique connected networks.

## Genetic algorithm
In this approach, we initiate a pool of random graphs. From this pool, we pick those who perform well in terms of fixation probability and build a new generation by crossing over these well-performed graphs. In order not to stuck in the local minima we add an error. We repeat this process over and over again until it converges.

## Requirements
* python3
* networkx





