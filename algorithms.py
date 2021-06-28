"""
Code for algorithms outlined in “PATHATTACK: Attacking shortest paths
in complex networks” by Benjamin A. Miller, Zohair Shafi, Wheeler Ruml,
Yevgeniy Vorobeychik, Tina Eliassi-Rad, and Scott Alfeld at ECML/PKDD 2021.

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
import numpy as np
import scipy.sparse as sp
import numpy.random as rand
import time
import pandas as pd

import gurobipy as gp
from gurobipy import GRB


from utils import *

# PATHATTACK_LP: Run the linear-programming-based PATHATTACK algorithm
#   to identify edges to cut to make p* the shortest path
# Inputs:   G - an undirected networkx graph with weights and costs on
#               each edge
#           s - source vertex of the path
#           t - destination vertex of the path
#           pStar - a list of vertices denoting the order of the path
#           batch_size - the number of shortest paths to add before 
#               retrying the optimization
# Outputs:  return_dict - a dictionary containing statistics of the 
#               of the optimization, including running time and
#               costs of edges cut
def PATHATTACK_LP(G, s, t, pStar, batch_size=1):
    
    start_time = time.time()
    ######################set up######################
    E = list(G.edges()) # list of edges
    M = len(E)          # number of edges  
    
    # create a map from edges to indices in E
    eInd = dict()
    for i in range(len(E)):
        # edges are undirected, so map pairs in both orders
        eInd[E[i]] = i
        eInd[E[i][1], E[i][0]] = i
    
    #sanity check: make sure p* is a path from s to t
    assert((pStar[0]==s) and (pStar[-1]==t)) #source and destination are correct
    assert(len(pStar)==len(np.unique(pStar))) #no cycles
    #all edges are in the graph
    assert(np.prod([(((pStar[i], pStar[i+1]) in E) or ((pStar[i+1], pStar[i]) in E)) for i in range(len(pStar)-1)]))
    
    #make the p* vector
    xp = np.zeros((len(E)))
    for i in range(len(pStar)-1):
        xp[eInd[(pStar[i], pStar[i+1])]] = 1
    
    #get weights 
    weights = nx.get_edge_attributes(G, "weight")
    w = np.array([weights[e] for e in E])
    
    #get costs 
    costs = nx.get_edge_attributes(G, "cost")
    c = np.array([costs[e] for e in E])
    ##################################################
    
    
    zeros = np.zeros((M))       # vector of zeros
    ones = np.ones((M))         # vector of ones
    xp_ind = np.nonzero(xp)[0]  # indices of edges in p*
    max_length = np.dot(w, xp)  # length of p* (maximum length of paths to cut)
    
    # create a Gurobi model
    mod = gp.Model('minCut')
    # delta: indicator vector of edges to cut (continuous variable, apply
    # randomized rounding
    delta = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="delta")

    # constraints on delta (at most 1, at least 0, don't cut p*)
    mod.addConstr(delta <= ones, "deltaUpper")
    mod.addConstr(delta >= zeros, "deltaLower")
    mod.addConstr(xp@delta == 0)
    # objective: minimize cost
    mod.setObjective(delta@c, GRB.MINIMIZE)
    
    
    done = False    # loop until done
    nConst = 0      # count constraints
    while not done:
        # perform minimization
        print('Minimize delta')
        mod.optimize()
        deltaHat = delta.X #get current value of delta
        
        
        print('find next constraint')
        # find new shortest path probably not cut
        Gtemp = G.copy()
        #cut edges where delta=1
        for i in np.where(deltaHat == 1)[0]:
            Gtemp.remove_edge(E[i][0], E[i][1])

        # iterate over paths from s to t
        P = nx.shortest_simple_paths(Gtemp, s, t, weight='weight')
        ctr = 0
        pConst = [] # list to hold new constraints
        while ctr < batch_size: # find batch_size new shortest paths
            try:
                p = next(P)
            except:
                print('no paths') # break if there are no more such paths
                break
            cut = 0     # accumulated cut vector values along path
            length = 0  # path length
            # loop over edges on the path
            for i in range(len(p)-1):
                ind = eInd[(p[i], p[i+1])]  # index of the current edge
                cut += deltaHat[ind]    # add cut vector value at current edge
                length += w[ind]        # add weight (length) at current edge
            
            # if the path is too long, don't consider anything further
            if length > max_length:
                break
            elif p == pStar: # if the path is p*, keep iterating
                print('we see p*')
                continue
            elif cut <= 0.999999: # allowing for some rounding error  #cut < 1:
                # if the accumulated cut value is not large enough,
                # add a new constraint
                pConst.append(p)
                ctr += 1
            print(p, ': ', cut)

        if len(pConst) == 0: # done if no new constraints were added
            done = True
        else: # otherwise, add the new constraints to the linear program
            for p in pConst: # for each constraint
                # get the indicator vector for edges in the path
                xHat = np.zeros((M))
                for i in range(len(p)-1):
                    ind = eInd[(p[i], p[i+1])]
                    xHat[ind] = 1

                # add the new constraint to the model
                nConst += 1
                mod.addConstr(xHat@delta >= 1, "path constraint "+str(nConst))
    
    cutEdges, entropy, nTries = randomized_rounding(G, s, t, E, deltaHat, nConst, pStar, max_length)
    totalCost = 0
    costRemoved = 0
    for e in G.edges:
        totalCost += G.edges[e]['cost']
        if (e in cutEdges) or ((e[1], e[0]) in cutEdges):
            costRemoved += G.edges[e]['cost']
    cut_graph = G.copy()
    for e in cutEdges:
        cut_graph.remove_edge(e[0], e[1])
    return_dict = dict()
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['cut_sp'] = nx.shortest_path(cut_graph, source=s, target=t, weight='weight')
    return_dict['edges_removed'] = cutEdges
    return_dict['num_edges_removed'] = len(cutEdges)
    return_dict['total_edges'] = G.number_of_edges()
    return_dict['cost_removed'] = costRemoved
    return_dict['total_cost'] = totalCost
    return_dict['num_constraints'] = nConst
    return_dict['entropy'] = entropy
    return_dict['num_randomized_rounding_tries'] = nTries
    
    return return_dict


def PATHATTACK_Greedy(graph, source, dest, pStar, batch_size=1):
    start_time = time.time()
    #setup
    
    ourEdges = []
    for ii in range(len(pStar)-1):
        if pStar[ii] < pStar[ii+1]:
            ourEdges.append((pStar[ii], pStar[ii+1]))
        else:
            ourEdges.append((pStar[ii+1], pStar[ii]))
        
    
    #make everything longer than p
    done = False
    edgeToPath = dict()
    pathToEdge = dict()
    cutEdges = []
    min_length = np.sum([graph.edges[(pStar[v], pStar[v+1])]['weight'] for v in range(len(pStar)-1)])
    while not done:
        print('cut edges: ', cutEdges)
        cut_graph = graph.copy()
        for e in cutEdges:
            cut_graph.remove_edge(e[0], e[1])
        

        #mod_check.setObjective(newWeights @ x, GRB.MINIMIZE)
        ######this part should really be its own function######
        # get new shortest paths
        ctr = 0
        sps = nx.shortest_simple_paths(cut_graph, source, dest, weight="weight")
        while ctr < batch_size:
            try:
                path = next(sps)
            except:
                print("in the exception")
                #no more paths
                if ctr == 0:
                    done = True
                break
            path_length = np.sum([cut_graph.edges[(path[v], path[v+1])]['weight'] for v in range(len(path)-1)])
            if path_length > min_length:
                print("we're done for now")
                if ctr ==  0:
                    done = True
                break
            if path == pStar: #don't add p*
                print('we see p*')
                continue
            
            #convert the path to a (hashable) string
            pathStr = str(path)
            if pathStr in pathToEdge:
                print("this shouldn't happen; this path should be cut")
                print(path)
                print(pathStr)
                print(pathToEdge[pathStr])
                for e in pathToEdge[pathStr]:
                    print(edgeToPath[e])
                return
            
            #add edges to the path, add the path to the edges
            pathToEdge[pathStr] = []
            for jj in range(len(path)-1):
                if path[jj] < path[jj+1]:
                    e = (path[jj], path[jj+1])
                else:
                    e = (path[jj+1], path[jj])
                
                if e not in edgeToPath:
                    edgeToPath[e] = []
                edgeToPath[e].append(path)
                pathToEdge[pathStr].append(e)
            ctr += 1
                

        #######################################################
        nPaths = dict()
        pathsPerCost = dict()
        edgeToPathCopy = dict()
        for e in edgeToPath:
            nPaths[e] = len(edgeToPath[e])
            pathsPerCost[e] = nPaths[e]/graph.edges[e]['cost']
            edgeToPathCopy[e] = edgeToPath[e].copy()
        
        
        #maxEdge = max(nPaths, key=nPaths.get)
        for e, _ in sorted(pathsPerCost.items(), key=lambda item: item[1], reverse=True):
            if e in ourEdges:
                continue
            maxEdge = e
            break
        
        cutEdges = []
        while nPaths[maxEdge] > 0:
            cutEdges.append(maxEdge)
            pathsBeingRemoved = edgeToPathCopy[maxEdge].copy()
            for p in pathsBeingRemoved:
                for e in pathToEdge[str(p)]:
                    nPaths[e] -= 1
                    pathsPerCost[e] = nPaths[e]/graph.edges[e]['cost']
                    edgeToPathCopy[e].remove(p)
            for e, _ in sorted(pathsPerCost.items(), key=lambda item: item[1], reverse=True):
                if e in ourEdges:
                    continue
                maxEdge = e
                break
        
    totalCost = 0
    costRemoved = 0
    for e in graph.edges:
        totalCost += graph.edges[e]['cost']
        if (e in cutEdges) or ((e[1], e[0]) in cutEdges):
            costRemoved += graph.edges[e]['cost']
    return_dict = dict()
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['cut_sp'] = nx.shortest_path(cut_graph, source=source, target=dest, weight='weight')
    return_dict['edges_removed'] = cutEdges
    return_dict['num_edges_removed'] = len(cutEdges)
    return_dict['total_edges'] = graph.number_of_edges()
    return_dict['cost_removed'] = costRemoved
    return_dict['total_cost'] = totalCost
    return_dict['num_constraints'] = len(pathToEdge)
    
    return return_dict


#Baselines

def naive_edge_cut(G, desired_path, return_graph, debug):

    '''
        Runs a naive algorithm that iteratively cuts the edge with the lowest cost that does not align with the desired path
        Inputs : 
            G            : nx Graph object - Input graph
            desired_path : List - List of nodes that represents the path
            return_graph : Bool - Return perturbed graph
            debug        : Bool - Print debug statements
        
    ''' 


    start_time = time.time()
    graph = G.copy()
    source = desired_path[0]
    target = desired_path[-1]
    edges_removed = []

    # Create a dictionary mapping edges to cost 
    costs = {edge : graph.edges[edge]['cost'] for edge in graph.edges}
    total_cost = np.sum(list(costs.values()))

    # Get edges along desired path directed from source to target
    desired_edges = get_edges(desired_path, directed = False)
    desired_path_length = get_path_length(graph, [desired_path])[0]

    # Set up a shortest path iterator 
    path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
    # Get current shortest path
    next_path = next(path_iterator)
    # Skip over the desired patth
    if next_path == desired_path: 
        try:
            next_path = next(path_iterator)
            path_length = get_path_length(graph, [next_path])[0]
        except:
            path_length = np.inf
    # If not the desired path, get the next one
    else:
        path_length = get_path_length(graph, [next_path])[0]

    # Iterate till conditions are satisfied
    while path_length <= desired_path_length:

        # Get edges along desired path directed from source to target
        current_edges = get_edges(next_path, directed = False)

        min_cost_edge = None
        min_cost = np.inf
        for edge in current_edges:
            # Find edge with minimum cost
            if edge not in desired_edges:
                if graph.edges[edge]['cost'] < min_cost:
                    min_cost = graph.edges[edge]['cost']
                    min_cost_edge = edge

        if debug :
            print ("Edge Removed : ", min_cost_edge, end = '\r')

        # Remove edge
        graph.remove_edge(*min_cost_edge)
        edges_removed.append(min_cost_edge)

        # Reinitialise Path Iterator
        path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
        next_path = next(path_iterator)
        # Skip over the desired patth
        if next_path == desired_path: 
            try:
                next_path = next(path_iterator)
                path_length = get_path_length(graph, [next_path])[0]
            except:
                path_length = np.inf
        else:
            path_length = get_path_length(graph, [next_path])[0]

    # Compute total costs
    cost_removed = 0
    for edge in edges_removed:
        try:
            cost_removed = cost_removed + costs[edge]
        except:
            cost_removed = cost_removed + costs[(edge[1], edge[0])]

    if debug :
        print ("Final Shortest Path : ", next_path)
        print ("Edges Removed : ", edges_removed)
        print ("Number Of Edges Removed : ", len(edges_removed), "/", len(costs))
        print ("Weight Removed : ", cost_removed, "/", total_cost)

    return_dict = {}

    if return_graph:
        return_dict['perturbed_graph'] = graph

    return_dict['perturbed_sp'] = nx.shortest_path(graph, source=source, target=target, weight='weight')
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['edges_removed'] = edges_removed
    return_dict['num_edges_removed'] = len(edges_removed)
    return_dict['total_edges'] = len(costs)
    return_dict['weight_removed'] = cost_removed
    return_dict['total_weight'] = total_cost

    return return_dict



def eigen_score_cut(G, desired_path, directed, return_graph, debug, use_abs = True):

    '''
    Runs a naive algorithm that iteratively cuts the edge with the highest eigen score that does not align with the desired path
        Inputs : 
            G            : nx Graph object - Input graph
            desired_path : List - List of nodes that represents the path
            directed     : Bool - Directed or undirected graph
            return_graph : Bool - Return perturbed graph
            debug        : Bool - Print debug statements
            use_abs      : Bool - Flag to determine using abolute values of eigen vectors

    '''

    start_time = time.time()
    graph = G.copy()
    if debug :
        print ("Running SVD")

    source = desired_path[0]
    target = desired_path[-1]

    desired_path_length = get_path_length(graph, [desired_path])[0]

    # Consider path to be directed from source to target
    desired_edges = get_edges(desired_path, directed = False)

    #u, s, v = sp.linalg.svds(sp.csc_matrix(nx.to_numpy_matrix(graph).astype(np.float)), k = 1)
    u, s, v = sp.linalg.svds(nx.adjacency_matrix(graph).astype(np.float), k = 1)

    edges_removed = []

    # Create dictionary mapping from edges to cost
    costs = {edge : graph.edges[edge]['cost'] for edge in graph.edges}
    total_cost = np.sum(list(costs.values()))

    # Make sure shapes are all right and make sense
    left_vector = u
    right_vector = v.T

    if use_abs:
        left_vector = np.abs(left_vector)
        right_vector = np.abs(right_vector)

    # Set up a shortest path iterator 
    path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
    # Get current shortest path
    next_path = next(path_iterator)
    # Skip if next path is the desired path
    if next_path == desired_path: 
        try:
            next_path = next(path_iterator)
            path_length = get_path_length(graph, [next_path])[0]
        except:
            path_length = np.inf
    else:
        path_length = get_path_length(graph, [next_path])[0]



    svd_end = time.time() - start_time
    edge_time = time.time()

    if debug :
        print ("SVD Complete, Removing Edges")

    # Iterate 
    while path_length <= desired_path_length:     

        # Consider path to be directed from source to target
        current_edges = get_edges(next_path, directed = False)
        allowed_edges = list(set(current_edges).difference(set(desired_edges)))

        # Compute eigen scores for each edge scaled by cost of the edge
        eigen_score = {}
        for edge in current_edges:

            eigen_score[edge] = left_vector[edge[0]] * right_vector[edge[1]] / graph.edges[edge]['cost']

        # Add reversed edges in case graph is undirected
        if not directed :
            allowed_edges = allowed_edges + [(x[1], x[0]) for x in allowed_edges]

        # Create dictionary mapping edges to eigen scores
        current_eigen_score = {edge : score for edge, score in eigen_score.items() if edge in allowed_edges}
        current_eigen_score = sort_dict(current_eigen_score, reverse = True)

        # Remove the top edge
        graph.remove_edge(*current_eigen_score[0][0])
        edges_removed.append(current_eigen_score[0][0])
        if debug :
            print ("Edge Removed : ", current_eigen_score[0][0], end = '\r')


        # Reinitialise Path Iterator
        path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
        next_path = next(path_iterator)
        if next_path == desired_path: #
            try:
                next_path = next(path_iterator)
                path_length = get_path_length(graph, [next_path])[0]
            except:
                path_length = np.inf
        else:
            path_length = get_path_length(graph, [next_path])[0]

    cost_removed = 0
    for edge in edges_removed:
        try:
            cost_removed = cost_removed + costs[edge]
        except:
            cost_removed = cost_removed + costs[(edge[1], edge[0])]

    if debug :
        print ("Final Shortest Path : ", next_path)
        print ("Edges Removed : ", edges_removed)
        print ("Number Of Edges Removed : ", len(edges_removed), "/", len(costs))
        print ("Weight Removed : ", cost_removed, "/", total_cost)

    return_dict = {}

    if return_graph:
        return_dict['perturbed_graph'] = graph

    return_dict['perturbed_sp'] = nx.shortest_path(graph, source=source, target=target, weight='weight')
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['edges_removed'] = edges_removed
    return_dict['num_edges_removed'] = len(edges_removed)
    return_dict['total_edges'] = len(costs)
    return_dict['cost_removed'] = cost_removed
    return_dict['total_cost'] = total_cost
    return_dict['svd_time'] = svd_end
    return_dict['edge_time'] = time.time() - edge_time

    return return_dict



