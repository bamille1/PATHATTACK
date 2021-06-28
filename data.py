import networkx as nx
import random
import pickle as pkl
from numpy import random as rand

from numpy.random import MT19937

from numpy.random import RandomState, SeedSequence


import sys

from algorithms import *

def add_weights(G, weights):

    '''
        Add weights to a graph
        Input : 
            G       : nx Graph object - Input graph
            weights : String - Poisson | Uniform
    '''

    nWeights = G.number_of_edges()
    
    if weights == 'Poisson':
        w = 1+rand.poisson(20, (nWeights))
    elif weights == 'Uniform':
        w = 1+rand.randint(41, size=(nWeights))
    else:
        w = np.ones((nWeights))

    ctr = 0
    for e in G.edges:
        G.edges[e]['weight'] = w[ctr]
        ctr += 1
        

if __name__ == "__main__":

    # process inputs
    graphName = sys.argv[1]     # string with the graph name (from the list below)
    weights = sys.argv[2]       # distribution of weights ('Poisson', 'Uniform', or 'Equal')
    seedPlus = int(sys.argv[3]) # integer from 0 to 10
    nTrials = int(sys.argv[4])  # number of trials (random weights and topologies)

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
    filename = 'inputs/'+graphName+'_'+weights+'Weights_part_'+str(seedPlus)+'_'+str(nTrials)+'trials.pkl'

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


