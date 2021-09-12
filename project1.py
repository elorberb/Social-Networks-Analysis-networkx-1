from builtins import print

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import tqdm as tm
import powerlaw as pl
from scipy import stats
import seaborn as sb
import statsmodels as sm
import math
import statistics as st



def get_name(): # name function
    name = "itay lorebrboym"
    print(name)
    return name


def get_id(): # id function
    id = "314977596"
    print(id)
    return id


# --------------1------------------- Generating and analysing random networks

# 1.i ----------

def random_networks_generator(n, p, num_networks=1, directed=False,
                              seed=np.random.seed(314977596)):  # generating a list containing random networks
    networks = []
    for i in range(num_networks):
        network = nx.gnp_random_graph(n, p, seed, directed) # creating network
        networks.append(network) # inserting to list
    return networks


# 1.ii

def std_calculator(network):
    lst1 = get_degrees(network)
    mean = sum(lst1) / len(lst1)
    variance = sum([((x - mean) ** 2) for x in lst1]) / len(lst1)
    std = variance ** 0.5
    return std

def get_min_degrees(network):
    return min(get_degrees(network))

def get_max_degrees(network):
    return max(get_degrees(network))

def get_degrees(network):
    dict1 = dict(nx.degree(network))
    lst1 = dict1.values()
    return lst1

def get_avg_degrees(network):
    return 2 * network.number_of_edges() / network.number_of_nodes()

def network_stats(network):
    try:
        spl = (nx.average_shortest_path_length(network))
    except:
        pass
    try:
        diameter = (nx.diameter(network))
    except:
        pass

    stats = {'degrees_avg': get_avg_degrees(network)
        , 'degrees_std': std_calculator(network)
        , 'degrees_min': get_min_degrees(network)
        , 'degrees_max': get_max_degrees(network)
        , 'spl': spl
        , 'diameter': diameter
             }
    return stats


# 1.iii

def networks_avg_stats(networks):
    stats = {'degrees_avg': 0
        , 'degrees_std': 0
        , 'degrees_min': 0
        , 'degrees_max': 0
        , 'spl': 0
        , 'diameter': 0
             }
    for network in networks:
        net_stats = network_stats(network)
        for key in stats:
            if key in net_stats:
                stats[key] = stats[key] + net_stats[key]  # adding values with same keys
            else:
                pass

    for key in stats:
        stats[key] = stats.get(key) / len(networks)  # sum of values divide by the size of networks to get average
    return stats

# tests 1 ------------ open questions data

type_a = random_networks_generator(100, 0.1, 20)
print(len(type_a))
for i in range(len(type_a)):
    print("network", i + 1, network_stats(type_a[i]))
print("average type a networks stats: ", networks_avg_stats(type_a))
type_b = random_networks_generator(100, 0.6, 20)
print(len(type_b))
for i in range(len(type_b)):
    print("network", i + 1, network_stats(type_b[i]))
print("average type b networks stats: ", networks_avg_stats(type_b))
type_c = random_networks_generator(1000, 0.1, 10)
print(len(type_c))
for i in range(len(type_c)):
    print("network", i + 1, network_stats(type_c[i]))
print("average type c networks stats: ", networks_avg_stats(type_c))
type_d = random_networks_generator(1000, 0.6, 10)
print(len(type_d))
for i in range(len(type_d)):
    print("network", i + 1, network_stats(type_d[i]))
print("average type d networks stats: ", networks_avg_stats(type_d))


#  --------------2------------------- Random networks - hypothesis testing

# 2.i ----- loading networks

rand_networks = pd.read_pickle("rand_nets.p") # reading pickle file to list

# 2.ii -----

def rand_net_hypothesis_testing(network, therotical_p, alpha=0.05): # hypothesis test
    edges = nx.number_of_edges(network)
    nodes = nx.number_of_nodes(network)
    total_edges_options = (nodes * (nodes - 1)) / 2 # total possible edges
    test_p_value = stats.binom_test(edges, total_edges_options, therotical_p)  # binom test for the network
    if (test_p_value > (alpha / 2) and test_p_value < 1 - (
            alpha / 2)):  # if the p value is between alpha/2 and 1-alpha/2 so we accept
        return (test_p_value, "accept")
    else:  # else we reject
        return (test_p_value, "reject")


# 2.iii -----

def most_probable_p(network):

    possible_values = [0.01, 0.1, 0.3, 0.6]
    for value in possible_values:
        p = rand_net_hypothesis_testing(network, value) #check if the value is accepted or denay by the hypothesis test
        if (p[1] == "accept"):
            return value
    return -1 #  none of the options are accepted



# tests 2 ------------------------ open question data

print(" optimal p:" , most_probable_p(rand_networks[0]))
print("hypothesis test with 10% bigger therotical_p (p =0.33):",rand_net_hypothesis_testing(rand_networks[0], 0.33))
print("hypothesis test with 100% bigger therotical_p (p =0.6):", rand_net_hypothesis_testing(rand_networks[0], 0.6))

# --------------3------------------- Find an optimal 𝛾 parameter to a scale-free network

# 3.i ------------- loading networks

scale_free_networks = pd.read_pickle("scalefree_nets.p") # reading pickle file to list

# 3.ii ------------ opt gamma

def find_opt_gamma(network, treat_as_social_network=True):
    data = [d for n, d in nx.degree(network)] # list of all nodes
    data = list(filter(lambda a: a != 0, data))  # filtering 0 values
    fit = pl.Fit(data, treat_as_social_network, verbose=False)
    return fit.alpha


# tests 3 --------------------------- open question data

for i in range(len(scale_free_networks)):
    print("network",i+1,"opt gamma: ",find_opt_gamma(scale_free_networks[i]))

print("scale stats: ",network_stats(scale_free_networks[0]))
rand_test = random_networks_generator(331,0.022,1)
print("rand stats: ",network_stats(rand_test[0]))




# --------------4------------------- Distinguish between random networks and scale free networks

# 4.i ------------- loading networks

mixed_networks = pd.read_pickle("mixed_nets.p") # reading mixed networks

# 4.ii ------------ network classifier

def network_classifier(network):
    gamma = find_opt_gamma(network)
    if (gamma > 3): #  if gamma is above 3 the network is random
        return 1
    elif (gamma < 3) and (gamma > 2):  # if between 2 and 3 network is scale free
        return 2
    return -1  # neither one of them - network is unknown

# test 4 ---------------------------- open question data

for i in range(len(mixed_networks)):
    print("Network",i+1,"classified: ", network_classifier(mixed_networks[i]))
