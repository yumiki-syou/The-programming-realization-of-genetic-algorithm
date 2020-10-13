# The-programming-realization-of-genetic-algorithm
Experiment 1 The programming realization of genetic algorithm
1.	Experiment Goal：

a)	Learn to write genetic algorithm program.

b)	To be familiar with the overall structure of GA algorithm.

c)	To master the programming of selection, crossover, mutation and other operators

d)	To master the programming of probability parameters such as roulette.

2.	Introduction to GA algorithm application scenarios: 

  The Traveling Salesman Problem (TSP) is a classic operational research optimization problem. It has been studied in discrete combinatorial optimization. It has a wide range of engineering applications, such as the arrangement of aircraft routes, the construction of road networks, etc.
  The Traveling Salesman Problem (TSP) refers to a single traveling salesman who needs to go to N cities to sell goods. The salesman starting from a certain city and passing through n-1 cities. The traveling salesman can pass through n-1 cities and can only pass once, and then return to the departure city. This problem asks to find the shortest possible route that visits each city exactly once and returns to the origin city?"

3.	Algorithm Architecture

a)	Encoding Gene
The arrangement of city numbers is used as the decoding method, and the first city in a set of arrangements is defaulted as the departure city.  For example, the city scale is 8, according to the coding rules, a set of chromosomes is 2-4-1-6-8-3-5-7, which means that the traveling salesman starts from city 2 and passes through cities 4, 1, 6, 8, 3, 5, 7, and finally back to city 2.

b)	Calculate Fitness
The target value is sum of the distance between adjacent cities on the traveling salesman’s path and the distance between the last arrival city and the departure city. To enlarge the gap, we take the fitness as 1 / the sum of distance

c)	Selection
Roulette Wheel Selection

d)	Cross Over
One-point Crossover: refers to only one crossover point is randomly set in the individual code string, and then part of the chromosomes of two paired individuals are exchanged at this point

e)	Mutation
Basic bit mutation (Simple Mutation): Perform mutation operation on a certain bit or a few randomly designated positions in the individual code string with the probability of mutation.

