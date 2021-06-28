import numpy as np
import networkx as nx
from numpy import random as rand

import time
import tempfile
import subprocess

def get_path_length(graph, path_list, directed = False):

    '''
        Returns lengths of a list of paths

        Inputs : 
            graph     : nx Graph object - Input graph
            path_list : List - List of paths 
            directed  : Bool - Directed / Undirected
    '''

    length_list = []

    # Iterate over all paths
    for path in path_list:
        length = 0
        # Add weights of each edge along the path 
        for edge in get_edges(path, directed):
            length = length + graph.edges[edge]['weight']
        length_list.append(length)

    return length_list


def get_shortest_paths(graph, source, target,  num_paths = None, desired_path = None):
    
    '''

        Gets shortest simple paths upto a fixed number of paths or till a particular path is reached. 
        Uses networkx's nx.shortest_simple_paths

        Inputs : 
            graph : nx Graph object - Input graph
            source : Integer - Source node 
            target : Integer - Target node
            num_paths : Integer - Number of paths to iterate (optional)
            desired_path : List - The desired path to iterate till (optional)
    '''

    paths = []
    X = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
    k = num_paths
    
    if desired_path == None: 
        # Iterate over num_paths
        for counter, path in enumerate(X):
            paths.append(path)
            if counter == k - 1:
                break

        return paths
    
    else: 
        # Keep iterating till desired_path is found
        for counter, path in enumerate(X):
            paths.append(path)
            if path == desired_path:
                break 
    
        return paths


def get_edges(path, directed = False):

    '''
        Returns a list of edges from a path 
        
        Inputs : 
            path     : List - List of nodes forming a path 
            directed : Bool - Directed / undirected
    '''

    edge_list = []
    for i in range(len(path)):

        if i + 1 < len(path):
            if directed == False:
                lower_node = min(path[i : i + 2])
                upper_node = max(path[i : i + 2])
                edge_list.append((lower_node, upper_node))
            else:
                edge_list.append(tuple(path[i : i + 2]))


    return edge_list

def sort_dict(in_dict, reverse = False):

    '''
        Returns a dictionary sorted by keys 
        Inputs : 
            in_dict : dict - Dictionary to sort
            reverse : Bool - Ascending / Descending
    '''

    return sorted(in_dict.items(), key = lambda kv : kv[1], reverse = reverse)

def randomized_rounding(G, s, t, E, delta, nPaths, pStar, min_length):
    uniqueVals = np.unique(delta)
    entropy = -np.sum(np.nan_to_num(delta*np.log2(delta)))
    m = len(E)
    #if list(uniqueVals) == [0, 1]:
    #    dPrime = delta
    #    cutEdges = [E[i] for i in np.where(dPrime)[0]]
    #    assert(entropy==0.0)
    #else:
    nSamples = np.ceil(np.log(4*nPaths))
    sample = np.zeros(delta.shape)
    numTries = 1

    for ii in range(int(nSamples)):
        r = rand.rand(m)
        sample = np.logical_or(sample, [(r[ii] < delta[ii]) for ii in range(m)])

    cutEdges = [E[i] for i in np.where(sample)[0]]
    cutGraph = G.copy()
    for e in cutEdges:
        cutGraph.remove_edge(*e)

    sps = nx.shortest_simple_paths(cutGraph, source=s, target=t, weight = 'weight')
    shortest_path = next(sps)
    if shortest_path == pStar:
        try:
            shortest_path = next(sps)
            spl = get_path_length(cutGraph, [shortest_path])[0]
        except:
            spl = np.inf
    else:
        spl = get_path_length(cutGraph, [shortest_path])[0]
    while spl <= min_length:
        numTries += 1
        if numTries > 100:
            print('failed after 100 tries, just use everything with a nonzero probability . . . ')
            sample = int(delta > 0)
        else:
            print('need to try again')
            sample = np.zeros(delta.shape)

            for ii in range(int(nSamples)):
                r = rand.rand(m)
                sample = np.logical_or(sample, [(r[ii] < delta[ii]) for ii in range(m)])

        cutEdges = [E[i] for i in np.where(sample)[0]]
        cutGraph = G.copy()
        for e in cutEdges:
            cutGraph.remove_edge(*e)

        #spl = nx.shortest_path_length(cutGraph, source=s, target=t, weight='weight')
        sps = nx.shortest_simple_paths(cutGraph, source=s, target=t, weight = 'weight')
        shortest_path = next(sps)
        if shortest_path == pStar:
            try:
                shortest_path = next(sps)
                spl = get_path_length(cutGraph, [shortest_path])[0]
            except:
                spl = np.inf
        else:
            spl = get_path_length(cutGraph, [shortest_path])[0]

    return cutEdges, entropy, numTries

def randomized_rounding_noGraph(P, s, t, E, delta, nPaths, pStar, min_length):
    uniqueVals = np.unique(delta)
    entropy = -np.sum(np.nan_to_num(delta*np.log2(delta)))
    m = len(E)
    #if list(uniqueVals) == [0, 1]:
    #    dPrime = delta
    #    cutEdges = [E[i] for i in np.where(dPrime)[0]]
    #    assert(entropy==0.0)
    #else:
    nSamples = np.ceil(np.log(4*nPaths))
    sample = np.zeros(delta.shape)
    numTries = 1

    for ii in range(int(nSamples)):
        r = rand.rand(m)
        sample = np.logical_or(sample, [(r[ii] < delta[ii]) for ii in range(m)])

    while np.sum(P@sample < .5) > 0:
        numTries += 1
        for ii in range(int(nSamples)):
             r = rand.rand(m)
             sample = np.logical_or(sample, [(r[ii] < delta[ii]) for ii in range(m)])
    
    cutEdges = [E[i] for i in np.where(sample)[0]]

    return cutEdges, entropy, numTries



def kronecker_graph(size, density, initiator=None):
    '''
        Generate a stochastic Kronecker graph using SNAP.
    '''

    command = './Snap-6.0-Ubuntu18.04/krongen'
    tmp = tempfile.NamedTemporaryFile()
    output_flag = '-o:{}'.format(tmp.name)
    iters = int(round(np.log2(size)))
    iter_flag = '-i:{}'.format(iters)
    seed_flag = '-s:{}'.format(int(round(rand.rand()*10000000)))
    #    time.time())

    # The expected number of edges is (\sum_ij a_ij)^k, where a_ij is the
    # matrix passed in mat_flag, and k is the number of iterations.
    mat = initiator if initiator is not None else initiator_matrix(size, density)
    mat = '; '.join([', '.join([str(el) for el in row]) for row in mat])
    mat_flag = '-m:{}'.format(mat)
    subprocess.run([command, output_flag, iter_flag, mat_flag, seed_flag],
                   stdout=subprocess.DEVNULL)
    graph = nx.read_edgelist(tmp.name)
    graph.graph['name'] = 'Kronecker(N={}, m={})'.format(2**iters, mat)

    # assert graph.order() == size
    return nx.convert_node_labels_to_integers(graph)


def initiator_matrix(size, density):
    
    '''
        Return an initiator matrix for the stochastic Kronecker model.

        The stochastic Kronecker random graph needs an initiator matrix. This
        function returns a 2x2 initiator matrix from which the stochastic
        Kronecker model will generate a network with expected number of nodes
        and edges equal to size and density, respectively.

        With an initiator matrix [a, b; b, c], the expected number of nodes and
        edges is 2**k, and (a+2b+c)**k, respectively, where k is the number of
        iterations of the Kronecker product. This allows us to draw a, b, and c
        randomly in such a way that fixes the expected density.

    '''

    iters = int(round(np.log2(size)))
    # We have to make sure that the resulting initiator matrix generates a
    # densification power law.  Thus, we draw a, and c randomly until the
    # condition for densification is satisfied. (See
    # https://arxiv.org/pdf/0812.4905.pdf, section 3.7 'Densification'.)
    # We also fix a around 0.95 and c near 0.48, as inspired by the values
    # of Table 4 from the reference.
    mina, maxa = 0.90, 0.99
    minc, maxc = 0.46, 0.51
    a, b, c = 0, 0, 0
    while a + 2*b + c < 2:
        a = (maxa - mina) * np.random.random() + mina
        c = (maxc - minc) * np.random.random() + minc
        b = (np.power(density * size * (size-1) / 2, 1/iters) - a - c) / 2
    # np.array([[0.9, 0.46], [0.43, 0.48]])
    return [[a, b], [b, c]]

