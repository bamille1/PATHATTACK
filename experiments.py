import networkx as nx
import random
import pickle as pkl
from numpy import random as rand

from numpy.random import MT19937

from numpy.random import RandomState, SeedSequence


import sys

from algorithms import *

def add_costs(G, costs):
    for e in G.edges:
        if costs=='weight':
            G.edges[e]['cost'] = G.edges[e]['weight']
        elif costs == 'invWeight':
            w = G.edges[e]['weight']
            if w > 0:
                G.edges[e]['cost'] = 1.0/w
            else:
                print('bad edge weight')
        else:
            G.edges[e]['cost'] = 1.0


if __name__ == "__main__":

    graphName = sys.argv[1]
    weights = sys.argv[2]
    pathRank = int(sys.argv[3])
    nTrials = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    costs = sys.argv[6]

    if weights not in ['Poisson', 'Uniform', 'Equal']:
        print('invalid weights')
        raise

    if costs not in ['weight', 'invWeight', 'equal']:
        print('invalid costs')
        raise

    filename = 'results/'+graphName+'_rank'+str(pathRank)+'_'+weights+'Weights_'\
            +costs+'Cost'+'_batch'+str(batch_size)+'_'+str(nTrials)+'trials_noEig.pkl'


    results = {}
    results['LP'] = []
    results['GreedySC'] = []
    results['GreedyEdge'] = []
    results['GreedyEig'] = []
    if graphName=='NortheastRoad':
        with open('NortheastUSRoad.pkl', 'rb') as f:
            G = pkl.load(f)
    elif graphName=='PARoad':
        with open('roadNet-PA.pkl', 'rb') as f:
            G = pkl.load(f)
    elif graphName=='LJ':
        with open('livejournal.pkl', 'rb') as f:
            G = pkl.load(f)
    elif graphName=='DBLP':
        with open('DBLP.pkl', 'rb') as f:
            G = pkl.load(f)
        #make distance inverse of number of connections
        for e in G.edges:
            G.edges[e]['weight'] = 1.0/G.edges[e]['weight']
    elif graphName=='LBL':
        with open('LBL.pkl', 'rb') as f:
            G = pkl.load(f)
        #make distance inverse of number of connections
        for e in G.edges:
            G.edges[e]['weight'] = 1.0/G.edges[e]['weight']
    elif graphName=='grid':
        with open('ChileanGrid.pkl', 'rb') as f:
            G = pkl.load(f)

    for ii in range(nTrials):
        if (ii%10) == 0:
            inputFile = 'inputs/'+graphName+'_'+weights+'Weights_part_'+str(int(np.round(ii/10)))+'_10trials.pkl'
            with open(inputFile, 'rb') as f:
                inputData = pkl.load(f)
        if graphName in ['PARoad', 'LJ']:
            ctr = 0
            for e in G.edges:
                G.edges[e]['weight'] = inputData[ii%10]['weight'][ctr]
                ctr += 1
        elif graphName not in ['NortheastRoad', 'DBLP', 'LBL', 'grid']:
            G = inputData[ii%10]['graph']
        s = inputData[ii%10]['source']
        t = inputData[ii%10]['destination']
        p = inputData[ii%10]['paths'][pathRank]

        if not (graphName=='grid' and costs=='equal'):
            add_costs(G, costs)
            for e in G.edges:
                G.edges[e]['cost'] += 0.000000001
        
        results['LP'].append(force_path_cut_approx(G, s, t, p, batch_size=batch_size))
        print('done with linear programming based algorithm', flush=True)
        print(results['LP'][-1])
        results['GreedySC'].append(greedy_batch_cut(G, s, t, p, batch_size=batch_size))
        print('done with greedy set cover based algorithm', flush=True)
        print(results['GreedySC'][-1])
        results['GreedyEdge'].append(naive_edge_cut(G, p, False, True))
        print('done with greedy edge cut', flush=True)
        print(results['GreedyEdge'][-1])
        results['GreedyEig'].append(eigen_score_cut(G, p, False, False, False))
        print('done with greedy eigenscore cut', flush=True)
        print(results['GreedyEig'][-1])

    with open(filename, 'wb') as f:
        pkl.dump(results, f)


