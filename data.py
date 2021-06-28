"""
Script to create input data for experiments outlined in “PATHATTACK: Attacking 
shortest paths in complex networks” by Benjamin A. Miller, Zohair Shafi, 
Wheeler Ruml, Yevgeniy Vorobeychik, Tina Eliassi-Rad, and Scott Alfeld at 
ECML/PKDD 2021.

This material is based upon work supported by the United States Air Force under
Air  Force  Contract  No.  FA8702-15-D-0001  and  the  Combat  Capabilities  
Development Command Army Research Laboratory (under Cooperative Agreement Number
W911NF-13-2-0045).  Any  opinions,  findings,  conclusions  or  recommendations
expressed in this material are those of the authors and do not necessarily 
reflect theviews of the United States Air Force or Army Research Laboratory.

Copyright (C) 2021
Benjamin A. Miller [1], Zohair Shafi [1], Wheeler Ruml [2],
Yevgeniy Vorobeychik [3],Tina Eliassi-Rad [1], and Scott Alfeld [4]

[1] Northeastern Univeristy
[2] University of New Hampshire
[3] Washington University in St. Louis
[4] Amherst College
"""
import networkx as nx
import random
import pickle as pkl
from numpy import random as rand

from numpy.random import MT19937

from numpy.random import RandomState, SeedSequence


import sys

from algorithms import *

# add_weights: add weights to an unweighted networkx graph via a specified method
#    Inputs:    G - an unweighted, undirected networkx graph
#               weights - a string denoting the distribution from which to
#                   draw weights (Poisson, Uniform, or Equal)
#   Outputs:    None (G is modified directly)
def add_weights(G, weights):

<<<<<<< HEAD
    # get one weight for each edge
    nWeights = G.number_of_edges()

    #create a vector of weights
=======
    '''
        Add weights to a graph
        Input : 
            G       : nx Graph object - Input graph
            weights : String - Poisson | Uniform
    '''

    nWeights = G.number_of_edges()
    
>>>>>>> ee82767d5aa5dddf6c742eb449261ff688c38bd9
    if weights == 'Poisson':
        # draw weights from a Poisson distribution
        w = 1+rand.poisson(20, (nWeights))
    elif weights == 'Uniform':
        # draw weights from a uniform distribution
        w = 1+rand.randint(41, size=(nWeights))
    else:
        # make all weights equal (1)
        w = np.ones((nWeights))

    # assign each weight to an edge
    ctr = 0
    for e in G.edges:
        G.edges[e]['weight'] = w[ctr]
        ctr += 1
        

if __name__ == "__main__":

    # process inputs
    graphName = sys.argv[1]     # string with the graph name (from the list below)
    weights = sys.argv[2]       # distribution of weights ('Poisson', 'Uniform', or 'Equal')
    seedPlus = int(sys.argv[3]) # integer (experiments used 0 to 9)
    nTrials = int(sys.argv[4])  # number of trials (random weights and topologies)
    inputDir = sys.argv[5]      # directory to store input files

    # sets seeds for reproducibility
    random.seed(81238.2345+9235.893456*seedPlus)
    rand.seed(892358293+27493463*seedPlus)

    assert(weights in ['Poisson', 'Uniform', 'Equal'])
    print(weights, ' weights')
    if graphName == 'er':    # Erdos-Renyi graph
        G = []
        for ii in range(nTrials):
            Gtemp = nx.erdos_renyi_graph(16000, .00125)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    elif graphName == 'ba':   # Barabasi-Albert graph
        G = []
        for ii in range(nTrials):
            Gtemp = nx.barabasi_albert_graph(16000, 10)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    elif graphName == 'ws':   # Watts-Strogatz graph
        G = []
        for ii in range(nTrials):
            Gtemp = nx.watts_strogatz_graph(n=16000, k=20, p=0.02)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    elif graphName == 'kron':   # Kronecker graph
        G = []
        for ii in range(nTrials):
            Gtemp = kronecker_graph(16000, 0.00125)
            for n in Gtemp.nodes():
                if Gtemp.has_edge(n, n):
                    Gtemp.remove_edge(n, n)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    elif graphName == 'lattice':   # lattice graph
        G = []
        Gtemp = nx.Graph(nx.adjacency_matrix(nx.grid_2d_graph(285, 285)))
        for ii in range(nTrials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graphName == 'complete':   # Complete graph
        G = []
        Gtemp = nx.complete_graph(565)
        for ii in range(nTrials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graphName == 'wiki':   # Wikispeedia Graph
        G = []
        with open('./wikispeediaGraph.pkl', 'rb') as f:
            Gtemp = pkl.load(f)
        for ii in range(nTrials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graphName == 'as':   # Autonomous System Graph
        G = []
        with open('./autonomousSystemGraph.pkl', 'rb') as f:
            Gtemp = pkl.load(f)
        for ii in range(nTrials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graphName == 'PARoad':   # Pennsylvania Road Network
        G = []
        with open('./roadNet-PA.pkl', 'rb') as f:
            Gtemp = pkl.load(f)
        for ii in range(nTrials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    else:
        print('invalid graph name')
        raise
    print('done making graphs', flush=True)


    # file name to save input data
    filename = inputDir+'/'+graphName+'_'+weights+'Weights_part_'+str(seedPlus)+'_'+str(nTrials)+'trials.pkl'

    input_list = []
    if graphName not in ['er', 'ba', 'kron', 'ws']:
        lcc = list(max(nx.connected_components(G[0]), key=len))
    for ii in range(nTrials):
        print('   graph ', ii)
        if graphName in ['er', 'ba', 'kron', 'ws']:
            lcc = list(max(nx.connected_components(G[ii]), key=len))

        # for lattice-like networks
        if graphName in ['lattice', 'PARoad']:
            # random node from largest connected component
            s = rand.choice(lcc)
            n = len(G[ii])
            A = nx.adjacency_matrix(G[ii])
            A += sp.diags(np.ones(n)) #add diagonal to adjacency matrix

            # initialize vector as an indicator for selected node
            v = np.zeros((n))
            v[s] = 1
            # spread over 49 hops
            for _ in range(49):
                v = A @ v
                
            # find nodes exactly 50 hops away
            neighbors = np.nonzero(v)[0]
            v = A @ v
            nHopNeighbors = np.setdiff1d(np.nonzero(v)[0], neighbors)
            # target is one of these nodes (randomly selected)
            t = rand.choice(nHopNeighbors)

            # only consider paths within the induced subgraph of nodes within
            # 60 hops of the source
            for _ in range(10):
                v = A @ v
            sg = np.nonzero(v)[0]
            Gpath = nx.subgraph(G[ii], sg)
        else:  # other (non-lattice-like) networks)
            # choose two nodes at random from the largest connected component
            terminals = rand.choice(lcc, size=2, replace=False)
            s = terminals[0]
            t = terminals[1]
            Gpath = G[ii] # consider paths over the entire graph

        # dictionary to store path information
        path_dict = dict()
        ctr = 0
        # iterate over paths from s to t
        for p in nx.shortest_simple_paths(Gpath, s, t, weight='weight'):
            ctr += 1
            # save the 100th, 200th, and 400th paths
            if ctr in [100, 200, 400, 800]:
                print('      path %d'%ctr, flush=True)
                path_dict[ctr] = p
            if ctr == 800: #stop after path 800
                break

        # save input information: Graph (with weights), source and 
        # target, and paths
        input_dict = dict()
        input_dict['graph'] = G[ii]
        input_dict['source'] = s
        input_dict['destination'] = t
        input_dict['paths'] = path_dict
        input_list.append(input_dict)

    # write file
    with open(filename, 'wb') as f:
        pkl.dump(input_list, f)


