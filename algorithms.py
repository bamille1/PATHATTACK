import networkx as nx
import numpy as np
import scipy.sparse as sp
import numpy.random as rand
import time
import pandas as pd

import gurobipy as gp
from gurobipy import GRB


from utils import *

def force_path_cut_approx(G, s, t, pStar, batch_size=1):
    
    start_time = time.time()
    ######################set up######################
    #Gd = nx.DiGraph(G)
    ##note: columns of C are in the same order as E
    #C = nx.incidence_matrix(Gd, oriented=True)
    #C = sp.csr_matrix(C)
    #E = list(Gd.edges())
    #d = np.zeros((len(G)))
    #d[s] = -1
    #d[t] = 1
    E = list(G.edges())
    M = len(E)
    
    eInd = dict()
    for i in range(len(E)):
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
    
    
    
    #(N, M) = C.shape
    zeros = np.zeros((M))
    ones = np.ones((M))
    xp_ind = np.nonzero(xp)[0]
    max_length = np.dot(w, xp)
    
    mod = gp.Model('minCut')
    delta = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="delta")
    mod.addConstr(delta <= ones, "deltaUpper")
    mod.addConstr(delta >= zeros, "deltaLower")
    mod.addConstr(xp@delta == 0)
    mod.setObjective(delta@c, GRB.MINIMIZE)
    
    
    #define oracle to find most violated constraint
    #oracleMod = gp.Model('oracle')
    #Np = np.sum(xp)
    #x = oracleMod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="x")
    #oracleMod.addConstr(C @ x == d, name="x is a path")
    #oracleMod.addConstr(x <= ones, "xUpper")
    #oracleMod.addConstr(x >= zeros, "xLower")
    #oracleMod.addConstr(w@x <= max_length, "short enough")
    #oracleMod.addConstr(x@xp <= Np-1, "not p*")
    
    done = False
    nConst = 0
    while not done:
        print('Minimize delta')
        mod.optimize()
        deltaHat = delta.X #get current value of delta
        
        
        print('find next constraint')
        Gtemp = G.copy()
        #cut edges where delta=1
        for i in np.where(deltaHat == 1)[0]:
            Gtemp.remove_edge(E[i][0], E[i][1])
        P = nx.shortest_simple_paths(Gtemp, s, t, weight='weight')
        ctr = 0
        pConst = []
        while ctr < batch_size:
            try:
                p = next(P)
            except:
                print('no paths')
                break
            cut = 0
            length = 0
            for i in range(len(p)-1):
                ind = eInd[(p[i], p[i+1])]
                cut += deltaHat[ind]
                length += w[ind]
            
            if length > max_length:
                break
            elif p == pStar:
                print('we see p*')
                continue
            elif cut <= 0.999999: #allowing for some rounding error  #cut < 1:
                pConst.append(p)
                ctr += 1
            print(p, ': ', cut)
        if len(pConst) == 0:
            done = True
        else:
            for p in pConst:
                xHat = np.zeros((M))
                for i in range(len(p)-1):
                    ind = eInd[(p[i], p[i+1])]
                    xHat[ind] = 1
                nConst += 1
                mod.addConstr(xHat@delta >= 1, "path constraint "+str(nConst))
        ##update oracle objective
        #oracleMod.setObjective(deltaHat@x, GRB.MINIMIZE)
        #print('find the right constraint')
        #oracleMod.optimize() #run optimization
        #
        #if oracleMod.status != GRB.OPTIMAL: #optimization failed: no violated constraints
        #    done = True
        #else:
        #    xHat = x.X #get result
        #    xHat = _extract_path(C, E, s, t, xHat, xp, xp_ind)
        #    if np.dot(xHat, deltaHat) < 1:# if a constraint is violated
        #        #add constraint
        #        nConst += 1
        #        mod.addConstr(xHat@delta >= 1, "path constraint "+str(nConst))
        #    else: #shortest uncut path is longer
        #        done = True
    
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

def force_path_cut_approx_cutFirst(G, s, t, pStar, batch_size=1):
    
    start_time = time.time()
    ######################set up######################
    Gd = nx.DiGraph(G)
    #note: columns of C are in the same order as E
    C = nx.incidence_matrix(Gd, oriented=True)
    C = sp.csr_matrix(C)
    E = list(Gd.edges())
    d = np.zeros((len(G)))
    d[s] = -1
    d[t] = 1
    E = list(G.edges())
    M = len(E)
    
    eInd = dict()
    for i in range(len(E)):
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
    
    
    
    #(N, M) = C.shape
    zeros = np.zeros((M))
    ones = np.ones((M))
    xp_ind = np.nonzero(xp)[0]
    max_length = np.dot(w, xp)
    
    mod = gp.Model('minCut')
    delta = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="delta")
    mod.addConstr(delta <= ones, "deltaUpper")
    mod.addConstr(delta >= zeros, "deltaLower")
    mod.addConstr(xp@delta == 0)
    mod.setObjective(delta@c, GRB.MINIMIZE)

    P = sp.csr_matrix((0, M), dtype=float)
    
    
    done = False
    nConst = 0
    while not done:
        print('Minimize delta')
        mod.optimize()
        deltaHat = delta.X #get current value of delta
        if nConst == 0:
            cutEdges = []
            entropy = 0
            nTries = 0
        else:
            cutEdges, entropy, nTries = randomized_rounding_noGraph(P, s, t, E, deltaHat, nConst+1, pStar, max_length)
        cut_graph = G.copy()
        for e in cutEdges:
            cut_graph.remove_edge(e[0], e[1])

        
        print('find next constraint')
        #update oracle objective
        sps = nx.shortest_simple_paths(cut_graph, s, t, weight='weight')

        p = next(sps)
        if p==pStar:
            try:
                p = next(sps)
            except:
                print("no more paths, we're done")
                done=True
                break
        length = 0
        for i in range(len(p)-1):
            ind = eInd[(p[i], p[i+1])]
            length += w[ind] 
        if length > max_length:
            done=True
            break
        else:
            xHat = sp.csr_matrix((1, M))
            for i in range(len(p)-1):
                ind = eInd[(p[i], p[i+1])]
                xHat[0, ind] = 1
            nConst += 1
            mod.addConstr(xHat@delta >= 1, "path constraint "+str(nConst))
            P = sp.vstack([P, xHat])

    
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

def greedy_batch_cut(graph, source, dest, pStar, batch_size=1):
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

    start_time = time.time()
    graph = G.copy()
    source = desired_path[0]
    target = desired_path[-1]
    edges_removed = []

    costs = {edge : graph.edges[edge]['cost'] for edge in graph.edges}
    total_cost = np.sum(list(costs.values()))

    # Get edges along desired path directed from source to target
    desired_edges = get_edges(desired_path, directed = False)
    desired_path_length = get_path_length(graph, [desired_path])[0]

    path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
    next_path = next(path_iterator)
    if next_path == desired_path: #skip over the desired patth
        try:
            next_path = next(path_iterator)
            path_length = get_path_length(graph, [next_path])[0]
        except:
            path_length = np.inf
    else:
        path_length = get_path_length(graph, [next_path])[0]

    while path_length <= desired_path_length:

        # Get edges along desired path directed from source to target
        current_edges = get_edges(next_path, directed = False)

        minCostEdge = None
        minCost = np.inf
        for edge in current_edges:
            # Remove the first edge not part of our desired path
            if edge not in desired_edges:
                if graph.edges[edge]['cost'] < minCost:
                    minCost = graph.edges[edge]['cost']
                    minCostEdge = edge

        if debug :
            print ("Edge Removed : ", minCostEdge, end = '\r')

        graph.remove_edge(*minCostEdge)
        edges_removed.append(minCostEdge)

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
    return_dict['weight_removed'] = cost_removed
    return_dict['total_weight'] = total_cost

    return return_dict



def eigen_score_cut(G, desired_path, directed, return_graph, debug, use_abs = True):

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

    costs = {edge : graph.edges[edge]['cost'] for edge in graph.edges}
    total_cost = np.sum(list(costs.values()))

    # Make sure shapes are all right and make sense
    left_vector = u
    right_vector = v.T

    if use_abs:
        left_vector = np.abs(left_vector)
        right_vector = np.abs(right_vector)

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



    svd_end = time.time() - start_time
    edge_time = time.time()

    if debug :
        print ("SVD Complete, Removing Edges")

    while path_length <= desired_path_length:     # next_path != desired_path:

        # Consider path to be directed from source to target
        current_edges = get_edges(next_path, directed = False)
        allowed_edges = list(set(current_edges).difference(set(desired_edges)))

        eigen_score = {}
        for edge in current_edges:

            eigen_score[edge] = left_vector[edge[0]] * right_vector[edge[1]] / graph.edges[edge]['cost']

        if not directed :
            allowed_edges = allowed_edges + [(x[1], x[0]) for x in allowed_edges]

        current_eigen_score = {edge : score for edge, score in eigen_score.items() if edge in allowed_edges}
        current_eigen_score = sort_dict(current_eigen_score, reverse = True)


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

def _force_path_core(G, E, eInd, w, buffer, pStar, xp, batch_size=1, init_constraints=[]):
    M = len(E)
    zeros = np.zeros((M))
    ones = np.ones((M))
    xp_ind = np.nonzero(xp)[0]
    s = pStar[0]
    t = pStar[-1]

    min_length = get_path_length(G, [pStar])[0] + buffer

    mod = gp.Model('minPerturbation')
    delta = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="delta")
    mod.addConstr(delta >= zeros, "deltaLower")
    mod.addConstr(xp@delta == 0)
    mod.setObjective(delta@ones, GRB.MINIMIZE)


    #define oracle to find most violated constraint
    #oracleMod = gp.Model('oracle')
    #Np = np.sum(xp)
    #x = oracleMod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="x")
    #oracleMod.addConstr(C @ x == d, name="x is a path")
    #oracleMod.addConstr(x <= ones, "xUpper")
    #oracleMod.addConstr(x >= zeros, "xLower")
    #oracleMod.addConstr(w@x <= max_length, "short enough")
    #oracleMod.addConstr(x@xp <= Np-1, "not p*")

    constraints_used = []
    for p in init_constraints:
        xHat = np.zeros((M))
        for i in range(len(p)-1):
            ind = eInd[(p[i], p[i+1])]
            xHat[ind] = 1
        constraints_used.append(p)
        mod.addConstr(xHat@delta >= (min_length-xHat@w),\
                      "path constraint "+str(len(constraints_used)))


    done = False
    while not done:
        print('Minimize delta')
        mod.optimize()
        deltaHat = delta.X #get current value of delta


        print('find next constraint')
        Gtemp = G.copy()
        #cut edges where delta=1
        for i in np.where(deltaHat > 0)[0]:
            Gtemp.edges[(E[i][0], E[i][1])]['weight'] += deltaHat[i]
        P = nx.shortest_simple_paths(Gtemp, s, t, weight='weight')
        ctr = 0
        pConst = []
        while ctr < batch_size:
            try:
                p = next(P)
            except:
                print('no paths')
                break


            if p == pStar:
                print('we see p*')
                continue
            else:
                length = get_path_length(Gtemp, [p])[0]

            if length >= min_length-.000001:
                break
            else:
                pConst.append(p)
                ctr += 1
        print('pConst: ', pConst)
        if len(pConst) == 0:
            done = True
        else:
            for p in pConst:
                xHat = np.zeros((M))
                for i in range(len(p)-1):
                    ind = eInd[(p[i], p[i+1])]
                    xHat[ind] = 1
                constraints_used.append(p)
                mod.addConstr(xHat@delta >= (min_length-xHat@w),\
                              "path constraint "+str(len(constraints_used)))
    return Gtemp, deltaHat, constraints_used


def force_path(G, s, t, pStar, batch_size=1, buffer=0.1, init_constraints=[]):
    
    start_time = time.time()
    ######################set up######################
    #Gd = nx.DiGraph(G)
    ##note: columns of C are in the same order as E
    #C = nx.incidence_matrix(Gd, oriented=True)
    #C = sp.csr_matrix(C)
    #E = list(Gd.edges())
    #d = np.zeros((len(G)))
    #d[s] = -1
    #d[t] = 1
    E = list(G.edges())
    M = len(E)
    
    eInd = dict()
    for i in range(len(E)):
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
    ##################################################
    
    
    

    
    perturbed_graph, deltaHat, constraints_used = _force_path_core(G, E, eInd, w, buffer,\
                                                                   pStar, xp,\
                                                                   batch_size=batch_size,\
                                                                   init_constraints=init_constraints)
    
    perturbed_edges = []
    perturbation_values = []
    for i in np.where(deltaHat)[0]:
        perturbed_edges.append(i)
        perturbation_values.append(deltaHat[i])
    
    return_dict = dict()
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['perturbed_sp'] = nx.shortest_path(perturbed_graph, source=s, target=t, weight='weight')
    return_dict['edges_perturbed'] = perturbed_edges
    return_dict['perturbation_values'] = perturbation_values
    return_dict['total_edges'] = G.number_of_edges()
    return_dict['total_weight'] = np.sum(w)
    return_dict['constraints_used'] = constraints_used
    
    return return_dict


def naive_edge_perturb(G, desired_path, return_graph, buffer=0.1, debug = True):

    start_time = time.time()

    graph = G.copy()
    source = desired_path[0]
    target = desired_path[-1]
    edges_perturbed = []
    perturbation_mag = []

    weights = {edge : graph.edges[edge]['weight'] for edge in graph.edges}
    total_weight = np.sum(list(weights.values()))

    ## Epsilon - The gap required between desired shortest path and the next
    #epsilon = 1 / len(graph.edges)

    # Get edges along desired path directed from source to target
    desired_edges = get_edges(desired_path, directed = True)
    desired_path_length = get_path_length(graph, [desired_path])[0]

    path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
    next_path = next(path_iterator)
    if next_path == desired_path:
        try:
            next_path = next(path_iterator)
            next_path_length = get_path_length(graph, [next_path])[0]
        except:
            next_path_length = np.inf
    else:
        next_path_length = get_path_length(graph, [next_path])[0]



    while next_path_length < desired_path_length + buffer:

        perturbation = desired_path_length + buffer - next_path_length

        # Get edges along desired path directed from source to target
        current_edges = get_edges(next_path, directed = True)

        for edge in current_edges:

            # Add all perturbation to the first edge not part of our desired path
            if edge not in desired_edges:

                if debug :
                    print ("Edge Perturbed : ", edge, end = '\r')

                graph.edges[edge]['weight'] = graph.edges[edge]['weight'] + perturbation
                edges_perturbed.append(edge)
                perturbation_mag.append(perturbation)
                break

        # Reinitialise Path Iterator
        path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
        next_path = next(path_iterator)

        if next_path == desired_path:
            if debug :
                print ("Ties Found, Breaking", end = '\r')
            try:
                next_path = next(path_iterator)
                next_path_length = get_path_length(graph, [next_path])[0]
            except:
                next_path_length = np.inf
        else:
            next_path_length = get_path_length(graph, [next_path])[0]



    delta = {edges_perturbed[i] : perturbation_mag[i] for i in range(len(perturbation_mag))}
    perturbed_path = next(nx.shortest_simple_paths(graph, source, target, weight = 'weight'))
    if debug :
        print ("Final Shortest Path : ", perturbed_path)
        print ("Number Of Edges Perturbed : ", len(edges_perturbed), "/", len(weights))
        print ("Total Perturbation : ", np.sum(perturbation_mag), "/", total_weight)
        print ("Mean Perturbation (Non-Zero): ", np.mean(perturbation_mag), "+/-", np.std(perturbation_mag))
        if perturbed_path == desired_path:
            print ("Successful Completion")
        else:
            print ("Failed")

        perturb_sp = get_shortest_paths(graph = graph,
                                        source = source,
                                        target = target,
                                        num_paths = 5)
        sp_length = get_path_length(graph = graph,
                                   path_list = perturb_sp,
                                   directed = False)

        perturb_df = pd.DataFrame(list(zip(perturb_sp, sp_length)))
        perturb_df.columns = ['Path', 'Length']
        print(perturb_df) #display(perturb_df)

    return_dict = {}


    if return_graph:
        return_dict['perturbed_graph'] = graph

    return_dict['perturbed_sp'] = perturbed_path
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['edges_perturbed'] = edges_perturbed
    return_dict['total_edges'] = len(weights)
    return_dict['perturbation_values'] = perturbation_mag
    return_dict['total_weight'] = total_weight
    return_dict['success'] = perturbed_path == desired_path

    return return_dict


def naive_edge_perturb_minWeight(G, desired_path, return_graph, buffer=0.1, debug = True):

    start_time = time.time()

    graph = G.copy()
    source = desired_path[0]
    target = desired_path[-1]
    edges_perturbed = []
    perturbation_mag = []

    weights = {edge : graph.edges[edge]['weight'] for edge in graph.edges}
    total_weight = np.sum(list(weights.values()))

    ## Epsilon - The gap required between desired shortest path and the next
    #epsilon = 1 / len(graph.edges)

    # Get edges along desired path directed from source to target
    desired_edges = get_edges(desired_path, directed = True)
    desired_path_length = get_path_length(graph, [desired_path])[0]

    path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
    next_path = next(path_iterator)
    if next_path == desired_path:
        try:
            next_path = next(path_iterator)
            next_path_length = get_path_length(graph, [next_path])[0]
        except:
            next_path_length = np.inf
    else:
        next_path_length = get_path_length(graph, [next_path])[0]



    while next_path_length < desired_path_length + buffer:

        perturbation = desired_path_length + buffer - next_path_length

        # Get edges along desired path directed from source to target
        current_edges = get_edges(next_path, directed = True)

        minEdge = None
        minWeight = np.inf
        for edge in current_edges:
            # Remove the first edge not part of our desired path
            if edge not in desired_edges:
                if graph.edges[edge]['weight'] < minWeight:
                    minWeight = graph.edges[edge]['weight']
                    minEdge = edge

        if debug :
            print ("Edge Removed : ", minEdge, end = '\r')

        graph.edges[minEdge]['weight'] += perturbation
        edges_perturbed.append(minEdge)
        perturbation_mag.append(perturbation)

        # Reinitialise Path Iterator
        path_iterator = nx.shortest_simple_paths(graph, source, target, weight = 'weight')
        next_path = next(path_iterator)

        if next_path == desired_path:
            if debug :
                print ("Ties Found, Breaking", end = '\r')
            try:
                next_path = next(path_iterator)
                next_path_length = get_path_length(graph, [next_path])[0]
            except:
                next_path_length = np.inf
        else:
            next_path_length = get_path_length(graph, [next_path])[0]



    delta = {edges_perturbed[i] : perturbation_mag[i] for i in range(len(perturbation_mag))}
    perturbed_path = next(nx.shortest_simple_paths(graph, source, target, weight = 'weight'))
    if debug :
        print ("Final Shortest Path : ", perturbed_path)
        print ("Number Of Edges Perturbed : ", len(edges_perturbed), "/", len(weights))
        print ("Total Perturbation : ", np.sum(perturbation_mag), "/", total_weight)
        print ("Mean Perturbation (Non-Zero): ", np.mean(perturbation_mag), "+/-", np.std(perturbation_mag))
        if perturbed_path == desired_path:
            print ("Successful Completion")
        else:
            print ("Failed")

        perturb_sp = get_shortest_paths(graph = graph,
                                        source = source,
                                        target = target,
                                        num_paths = 5)
        sp_length = get_path_length(graph = graph,
                                   path_list = perturb_sp,
                                   directed = False)

        perturb_df = pd.DataFrame(list(zip(perturb_sp, sp_length)))
        perturb_df.columns = ['Path', 'Length']
        print(perturb_df) #display(perturb_df)

    return_dict = {}


    if return_graph:
        return_dict['perturbed_graph'] = graph

    return_dict['perturbed_sp'] = perturbed_path
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['edges_perturbed'] = edges_perturbed
    return_dict['total_edges'] = len(weights)
    return_dict['perturbation_values'] = perturbation_mag
    return_dict['total_weight'] = total_weight
    return_dict['success'] = perturbed_path == desired_path

    return return_dict
