import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp
import scipy.stats as st
import pandas as pd
import copy
import random
import itertools
import operator
from stackedBarGraph import StackedBarGrapher
sbg=StackedBarGrapher()

import timeit
from bisect import bisect


class TSP(object):
    def __init__(self, start=0, dissipation=.2, tolerance=.2, alpha=1.0, beta=1.0, explore=1.0, n_cities=4, force=None, damage_step=30, rug=1.0):
        # inputs
        self.n_cities = n_cities
        self.dissipation = dissipation
        self.explore = explore
        self.tolerance = tolerance
        self.force = force
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.rug=rug #ruggedness of landscape (exponentially scales distances)
        self.damage_step=damage_step
        self.step = 0

        # setup
        self.history_d = {}  # holds decisions about node order
        self.history_k = {}
        self.history_s = {}
        self.tsp = self.init_tsp_big(self.n_cities)
        # self.tsp=tsp_single()

        # Edge Tracker
        self.history_e = {}
        self.e_change = {}

        # Tracking - last[type][element][node history]
        ##type - p, dist, score, route, dec
        ##element - edge, edge, step, obj, none
        # self.last = {'p':{}, 'dist': {}, 'score': {}, 'route': {}, 'dec': {}}
        self.last = {'dist': {}, 'score': {}, 'route': {}, 'dec': {}}
        for e in self.tsp.edges():
            self.last['dist'][e] = []
            # self.last['p'][e] = []

        for i in xrange(self.tsp.number_of_nodes() - 1):
            self.last['route'][i] = []
            # self.last['dec'][i+1] = []

        self.last['score'][0] = []
        self.last['dec'][0] = []  # zero is placeholder

        # create last_action
        self.last_action = copy.deepcopy(self.last)

        self.d = [int(self.start)]

        # self.d=[]
        self.d_change = []
        self.history_d[self.step] = list(self.d)
        self.state = self.init_graph(self.tsp)
        self.history_s[self.step] = self.state.copy()
        self.k = self.init_k(self.state)

        # Track parameters
        self.tsp_sol_e = self.state.number_of_nodes()-2
        self.tsp_nonsol_e = self.state.number_of_edges()-(2*self.state.number_of_nodes()-3)
        self.start_sol_e = 1
        self.start_nonsol_e = n_cities

        # self.last_solution={}

    def init_tsp_big(self, num_cities=5):
        """
        Nine cities, with a shortest path of 8
        Adjacent cities have dist=1
        Otherwise cities dist are the average of their node labels
        """
        tsp = nx.DiGraph()
        n_l = range(1, num_cities + 1)
        tsp.add_nodes_from(n_l)
        # Edges:
        # Non-adjacent edges: Distance=average of node labels
        for i in tsp.nodes():
            for j in tsp.nodes():
                if i != j:
                    # tsp.add_edge(i,j,dist=(i+j)/2.0)
                    tsp.add_edge(i, j, dist=np.power(float(abs(i - j)),self.rug))

        # Distance=1 for nodes to the right and left
        for i in xrange(len(n_l)):
            n = n_l[i]
            i_l = (i - 1) % len(n_l)
            n_left = n_l[i_l]
            i_r = (i + 1) % len(n_l)
            n_right = n_l[i_r]
            # print i, i_l,i_r, (n_l[i],n_l[i_l],n_l[i_r])
            # tsp.add_edges_from([(n,n_left,{'dist':1.0}),(n,n_right,{'dist':1.0})])

        # infinite distance between first and last
        n = n_l[0]
        n_left = n_l[-1]
        # tsp.add_edges_from([(n,n_left,{'dist':100.0}),(n_left,n,{'dist':100.0})])

        # print tsp.edges(data=True)
        # start node
        tsp.add_node(0)
        for n in tsp.nodes():
            if n != 0:
                tsp.add_edge(0, n, dist=1.0)

        # print tsp.edges(data=True)

        return tsp

    def init_tsp(self):
        tsp = nx.DiGraph()
        tsp.add_edge(1, 2, dist=1.0)
        tsp.add_edge(2, 1, dist=1.0)
        tsp.add_edge(1, 3, dist=1.0)
        tsp.add_edge(3, 1, dist=1.0)
        tsp.add_edge(1, 4, dist=2.0)
        tsp.add_edge(4, 1, dist=2.0)
        tsp.add_edge(2, 3, dist=3.0)
        tsp.add_edge(3, 2, dist=3.0)
        tsp.add_edge(2, 4, dist=4.0)
        tsp.add_edge(4, 2, dist=4.0)
        tsp.add_edge(3, 4, dist=2.0)
        tsp.add_edge(4, 3, dist=2.0)

        # start node
        tsp.add_node(0)
        for n in tsp.nodes():
            if n != 0:
                tsp.add_edge(0, n, dist=0.0)

        return tsp

    def init_graph(self, tsp):
        g = tsp.copy()
        for u, v in g.edges():
            g[u][v]['p'] = 0.5
            g[u][v]['dist'] = 0.0

        # initialize nodes

        # print g.edges(data=True)

        return g

    def init_k(self, g):
        k = nx.DiGraph()
        last_visited = {}

        # k for edge data
        for u, v, d in g.edges(data=True):
            last_visited[(u, v)] = {}
            # Add pheromone k
            # n_label = k.number_of_nodes() + 1
            # k.add_node(n_label, label=(u, v), layer='k', p=float(d['p']), step=self.step)
            # self.last['p'][(u, v)].append(n_label)

            # Add distance k
            n_label = k.number_of_nodes() + 1
            k.add_node(n_label, label=(u, v), layer='k', dist=float(d['dist']), step=self.step)
            self.last['dist'][(u, v)].append(n_label)

        # k for decision
        # n_label = k.number_of_nodes() + 1
        # k.add_node(n_label, label='decision', layer='d', d=list(self.d), step=self.step)
        # self.last['dec'][0].append(n_label)

        self.history_e[self.step] = []
        self.e_change[self.step] = []
        self.history_k[self.step] = k.copy()
        self.step += 1

        return k  # ,last_visited

    def walk(self):
        # initialize
        g = self.state.copy()
        tsp = self.tsp.copy()
        # tabu=[int(self.start)]
        if not self.d:
            tabu = [random.choice(g.nodes())]
        else:
            tabu = [self.d[0]]

        d_index = []
        for i in xrange(g.number_of_nodes() - 1):
            # get pheromone list
            n_l = []
            p_l = []
            h_l = []
            dec_point = False  # checks if this step is affected by a decision
            for n in g.nodes():
                if n not in tabu:
                    n_l.append(n)
                    p_l.append(g[tabu[-1]][n]['p'])

                    if g[tabu[-1]][n]['p'] == 0.0:  # Part of a decision process
                        dec_point = True

                    if g[tabu[-1]][n]['dist'] == 0.0:
                        h_l.append(self.explore)  # 10.0
                    else:
                        h_l.append(1.0 / g[tabu[-1]][n]['dist'])

            # Combine pheromones and heuristic
            c_l = np.power(p_l, self.alpha) * np.power(h_l, self.beta)

            # Select next step
            n_index = self.make_decision(c_l)
            new_n = n_l[n_index]

            # Add action node for route
            a_label = self.k.number_of_nodes() + 1
            self.k.add_node(a_label, layer='a', label='selection', i=i, e=(tabu[-1], new_n), step=self.step)
            self.last_action['route'][i].append(a_label)

            # Add edges from knowledge to action
            # for e_i in xrange(len(tabu) - 1):  # iterate through route selections
            #     # add constraining, e_i is "edge selection i"
            #     last = self.last['route'][e_i][-1]
            #     self.k.add_edge(last, a_label, step=self.step, t=1)

            # if influenced by decision, edge from decision not other edge info

            if (dec_point or (len(n_l) == 1 and len(self.d) == self.tsp.number_of_nodes())):
                # only edge from decision
                d_index.append(i)
                # self.k.add_edge(self.last['dec'][0][i+1], a_label, step=self.step, t=1)

            else:  # add edges from distances and pheromones
                # for e_i in xrange(len(tabu) - 1):  # iterate through route selections
                #     # add constraining, e_i is "edge selection i"
                #     if e_i in d_index:
                #         # print self.d, d_index,e_i
                #         last=self.last['dec'][0][e_i+1]
                #         self.k.add_edge(last, a_label, step=self.step, t=1)
                #     else:
                #         last = self.last['route'][e_i][-1]
                #         self.k.add_edge(last, a_label, step=self.step, t=1)

                for n in n_l:  # n_l is the list of possible nodes to be selected
                    # if n not in tabu: #only non tabu
                    if n != tabu[-1]:  # no self-edge
                        e = (tabu[-1], n)
                        if (e[0], e[1]) not in self.tsp.edges():  # self.last_visited.keys():
                            e = (e[1], e[0])

                        # self.k.add_edge(self.last['p'][e][-1], n_label, step=self.step, t=2) #edges from pheromones
                        # if e[0] != self.start:
                        self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)

            for e_i in xrange(len(tabu) - 1):  # iterate through route selections
                # add constraining, e_i is "edge selection i"
                # last = self.last['route'][e_i][-1]
                # self.k.add_edge(last, a_label, step=self.step, t=1)

                if e_i in d_index:
                    # print self.d, d_index,e_i
                    last_index = len(self.last['dec'][0])
                    last = self.last['dec'][0][last_index - 1]
                    # last = self.last['dec'][0][e_i + 1]
                    self.k.add_edge(last, a_label, step=self.step, t=1)
                else:
                    last = self.last['route'][e_i][-1]
                    self.k.add_edge(last, a_label, step=self.step, t=1)

            # Action addition complete

            # Add new knowledge and edge from actions
            n_label = self.k.number_of_nodes() + 1

            # Node label is edge number in solution, e is edge added
            self.k.add_node(n_label, layer='k', label='selection', i=i, e=(tabu[-1], new_n), step=self.step)
            self.last['route'][i].append(n_label)
            self.k.add_edge(a_label, n_label, step=self.step)  # edge from action to new k

            # append new node to walk solution
            tabu.append(new_n)

        score = 0.0
        e_r = [tuple(tabu[i:i + 2]) for i in xrange(0, len(tabu), 1)]
        del e_r[-1]

        for e in e_r:
            score += float(self.tsp[e[0]][e[1]]['dist'])

        return tuple(tabu), score

    def update_edges(self, route, score):
        g_t = self.state.copy()
        g = self.state.copy()
        p = float(self.dissipation)
        tsp = self.tsp.copy()

        # print 'before', g_t.edges(data=True)
        # print 'reduction', p

        # update k with rout
        # n_label=self.k.number_of_nodes()+1
        # self.k.add_node(n_label,label=route,step=self.step)
        # self.last_solution['route']=n_label

        # k edges from tsp edge info to route - now handled in walk
        # for e in self.last_visited:
        # self.k.add_edge(self.last_visited[e]['p'],n_label,step=self.step)
        # self.k.add_edge(self.last_visited[e]['dist'],n_label,step=self.step)

        e_r = [tuple(route[i:i + 2]) for i in xrange(0, len(route), 1)]
        del e_r[-1]

        # update distances
        new_edges = []
        for e in e_r:
            sel_index = e_r.index(e)
            # check ordering
            if (e[0], e[1]) not in self.tsp.edges():
                e = (e[1], e[0])

            # update dist in k
            if g_t[e[0]][e[1]]['dist'] == 0.0:  # and e[0]!=self.start: #start is always 0.0
                dist = float(tsp[e[0]][e[1]]['dist'])
                g_t[e[0]][e[1]]['dist'] = dist

                # Add action node for changed distance
                a_label = self.k.number_of_nodes() + 1 #new action
                last_n = self.last['dist'][(e[0], e[1])][-1] #last dist
                self.k.add_node(a_label, layer='a', label='dist', e=(e[0], e[1]), dist=float(dist), step=self.step)
                self.k.add_edge(self.last['route'][sel_index][-1], a_label, step=self.step, t=1) #from route
                self.k.add_edge(last_n, a_label, step=self.step, t=1)  # from initial distance
                # self.k.add_edge(self.last['score'][0][-1],a_label, step=self.step, t=1) #from score
                self.last_action['dist'][(e[0], e[1])].append(a_label)

                # update k for changed distance from action
                n_label = self.k.number_of_nodes() + 1 #knowledge node
                # last_n = self.last['dist'][(e[0], e[1])][-1]
                self.k.add_node(n_label, layer='k', label='dist', e=(e[0], e[1]), dist=float(dist), step=self.step)
                self.k.add_edge(last_n, n_label, step=self.step, t=2)  # edge from refinement
                self.k.add_edge(a_label, n_label, step=self.step)  # edge from action

                self.last['dist'][(e[0], e[1])].append(n_label)

                # Update edge history
                new_edges.append(e)

        self.e_change[self.step] = new_edges
        old_edges = list(self.history_e[self.step - 1])
        all_edges = old_edges + new_edges
        self.history_e[self.step] = all_edges

        # Add action for score
        a_label = self.k.number_of_nodes() + 1
        self.k.add_node(a_label, layer='a', label='score', score=score, step=self.step)
        self.last_action['score'][0].append(a_label)

        # add k from route selections to score action
        for sel, n_list in self.last['route'].iteritems():
            self.k.add_edge(n_list[-1], a_label, step=self.step, t=1)

        # add edges from distances to actions
        for e in e_r:
            sel_index = e_r.index(e)
            # check ordering
            if (e[0], e[1]) not in self.tsp.edges():
                e = (e[1], e[0])

            # update k for route score for distance
            last_n = self.last['dist'][(e[0], e[1])][-1]
            self.k.add_edge(last_n, a_label, step=self.step, t=1)

            #update prototype for route score
            e_i=e_r.index(e)
            last_n = self.last['route'][e_i][-1]
            self.k.add_edge(last_n, a_label, step=self.step, t=1)

        # Action added

        # add k for score
        n_label = self.k.number_of_nodes() + 1
        self.k.add_node(n_label, layer='k', label='score', score=score, step=self.step)
        self.k.add_edge(a_label, n_label, step=self.step)  # add edge from action to knowledge

        self.last['score'][0].append(n_label)

        # Dissipate pheromone
        for u, v, d in g_t.edges(data=True):
            # update pheromone
            g_t[u][v]['p'] = float(g[u][v]['p']) * (1.0 - p)

            # update k for pheromone reduction
            # last_n = self.last['p'][(u, v)][-1]
            # n_label = self.k.number_of_nodes() + 1
            # self.k.add_node(n_label, label=(u, v), p=float(d['p']), step=self.step)
            # self.k.add_edge(last_n, n_label, step=self.step, t=2)
            # self.last['p'][(u, v)].append(n_label)

        # Update pheromone
        t_update = 1.0 / score
        for e in e_r:
            sel_index = e_r.index(e)
            # check ordering
            if (e[0], e[1]) not in self.tsp.edges():
                e = (e[1], e[0])

            # update pheromones on included edges
            g_t[e[0]][e[1]]['p'] = float(g_t[e[0]][e[1]]['p']) + t_update * p

            # update k for pheromone addition - add edge for walk update
            # last_n = self.last['p'][(e[0], e[1])][-1]
            # self.k.node[last_n]['p'] = float(g_t[e[0]][e[1]]['p'])
            # # n_label=self.k.number_of_nodes()+1
            # # self.k.add_node(n_label,label=(e[0],e[1]),p=float(g_t[e[0]][e[1]]['p']),step=self.step)
            # # self.k.add_edge(last_n,n_label,step=self.step)
            # self.k.add_edge(self.last['route'][sel_index][-1], last_n, step=self.step, t=2)
            # self.k.add_edge(self.last['score'][0][-1], last_n, step=self.step, t=2)
            # # self.last_visited[(e[0],e[1])]['p']=n_label

        if self.step == self.force:
            g_t = self.force_design(g_t)

        g_t = self.design(g_t, tolerance=self.tolerance)

        self.history_d[self.step] = list(self.d)
        self.history_k[self.step] = self.k.copy()
        self.history_s[self.step] = g_t.copy()
        self.step += 1
        # print 'after', g_t.edges(data=True)
        return g_t

    def force_design(self, g):
        # locks-in best next step
        g_t = g.copy()

        if not self.d:
            dec_node = 0
        else:
            dec_node = self.d[-1]

        # Get probabilities
        # get pheromone list
        n_l = []
        p_l = []
        h_l = []
        for n in g.nodes():
            if (n not in self.d) and (n != dec_node):
                n_l.append(n)
                p_l.append(g[dec_node][n]['p'])
                # print dec_node,n, n!=dec_node

                if g[dec_node][n]['dist'] == 0.0:  # only look at explored nodes
                    # h_l.append(0.0)
                    h_l.append(self.explore)
                    # print dec_node,n
                    # n_l.append(n)
                    # p_l.append(g[dec_node][n]['p'])

                else:
                    # h_l.append(1.0)
                    h_l.append(1.0 / g[dec_node][n]['dist'])

        ph_l = np.power(p_l, self.alpha) * np.power(h_l, 1.0)  # only use pheromone preference
        if sum(ph_l) == 0.0:
            perc_l = [1.0 / len(ph_l)] * len(ph_l)
        else:
            perc_l = [float(i) / sum(ph_l) for i in ph_l]

        dec_index = perc_l.index(max(perc_l))  # decision index
        # print n_l,dec_index
        node = n_l[dec_index]
        self.d.append(node)  # add node to decisions
        # print dec_node,self.d
        # print 'decision',self.step,dec_node,node,self.d_change

        # Eliminate other edge option
        for n in g.nodes():
            # print n, dec_node, n!=dec_node
            # print n, skip,n not in skip
            if n not in self.d:
                # print n,dec_node
                if (dec_node, n) in g_t.edges():
                    g_t[dec_node][n]['p'] = 0.0  # now prob of taking that edges is 0

        # Add action for decision
        a_label = self.k.number_of_nodes() + 1
        self.k.add_node(a_label, layer='a', label='dec', d=list(self.d), step=self.step)
        self.last_action['dec'][0].append(a_label)

        # add edges from pheromone knowledge to action
        for n in n_l:
            e = (dec_node, n)
            # self.k.add_edge(self.last['p'][e][-1], n_label, step=self.step, t=2)
            self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)

        # add edges from decision knowledge to action
        if self.last['dec'][0]:
            self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)

        # update k for decision from action
        n_label = self.k.number_of_nodes() + 1
        self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
        self.k.add_edge(self.last['dec'][0][-1], n_label, step=self.step, t=3)
        if self.last['dec'][0]:
            self.k.add_edge(a_label, n_label, step=self.step, t=3)
        self.last['dec'][0].append(n_label)
        self.d_change.append(self.step)

        return g_t

    def design(self, g, tolerance):
        # makes locks-in step in walk if p(1st node)-p(2nd node)
        # g=self.state.copy()
        g_t = g.copy()

        while True:
            if len(self.d) == (g.number_of_nodes()):
                break

            if not self.d:
                dec_node=0
            else:
                dec_node = self.d[-1]

            # Get probabilities
            # get pheromone list
            n_l = []
            p_l = []
            h_l = []
            for n in g.nodes():
                if (n not in self.d) and (n != dec_node):
                    n_l.append(n)
                    p_l.append(g[dec_node][n]['p'])
                    # print dec_node,n, n!=dec_node

                    if g[dec_node][n]['dist'] == 0.0:  # only look at explored nodes
                        # h_l.append(0.0)
                        h_l.append(self.explore)
                        # print dec_node,n
                        # n_l.append(n)
                        # p_l.append(g[dec_node][n]['p'])

                    else:
                        # h_l.append(1.0)
                        h_l.append(1.0 / g[dec_node][n]['dist'])

            ph_l = np.power(p_l, self.alpha) * np.power(h_l, 1.0)  # only use pheromone preference
            if sum(ph_l) == 0.0:
                perc_l = [1.0 / len(ph_l)] * len(ph_l)
            else:
                perc_l = [float(i) / sum(ph_l) for i in ph_l]
            l = list(perc_l)
            # print perc_l
            m_1 = l.pop(l.index(max(l)))
            # print l,m_1
            # print m_1-max(l)

            if not l:  # only one option
                # print 'decision'
                dec_index = perc_l.index(max(perc_l))  # decision index
                node = n_l[dec_index]
                self.d.append(node)  # add node to decisions
                # print 'decision',self.step,dec_node,node,self.d_change

                # Eliminate other edge options
                # skip=[int(self.start),dec_node]
                for n in g.nodes():
                    if n not in self.d:
                        if (dec_node, n) in g_t.edges():
                            g_t[dec_node][n]['p'] = 0.0  # now prob of taking that edges is 0

                # Add action for decision
                a_label = self.k.number_of_nodes() + 1
                self.k.add_node(a_label, layer='a', label='dec', d=list(self.d), step=self.step)
                self.last_action['dec'][0].append(a_label)

                # add edges from pheromone knowledge to action
                for n in n_l:
                    e = (dec_node, n)
                    self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)

                # add edges from decision knowledge to action
                self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)

                # update k for decision from action
                n_label = self.k.number_of_nodes() + 1
                self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
                self.k.add_edge(self.last['dec'][0][-1], n_label, step=self.step, t=3)
                self.k.add_edge(a_label, n_label, step=self.step, t=3)
                self.last['dec'][0].append(n_label)
                self.d_change.append(self.step)

            elif (m_1 - max(l)) >= tolerance:  # prob gap is larger than tolerance
                # print 'decision',self.step,dec_node

                dec_index = perc_l.index(max(perc_l))  # decision index
                # print n_l,dec_index
                node = n_l[dec_index]
                self.d.append(node)  # add node to decisions
                # print dec_node,self.d
                # print 'decision',self.step,dec_node,node,self.d_change

                # Eliminate other edge option
                for n in g.nodes():
                    # print n, dec_node, n!=dec_node
                    # print n, skip,n not in skip
                    if n not in self.d:
                        # print n,dec_node
                        if (dec_node, n) in g_t.edges():
                            g_t[dec_node][n]['p'] = 0.0  # now prob of taking that edges is 0

                # Add action for decision
                a_label = self.k.number_of_nodes() + 1
                self.k.add_node(a_label, layer='a', label='dec', d=list(self.d), step=self.step)
                self.last_action['dec'][0].append(a_label)

                # add edges from pheromone knowledge to action
                for n in n_l:
                    e = (dec_node, n)
                    self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)

                # add edges from decision knowledge to action
                if self.last['dec'][0]:
                    # print self.last['dec'][0]
                    # print self.last['dec'][0][-1]
                    # print a_label
                    self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)

                # update k for decision from action
                n_label = self.k.number_of_nodes() + 1
                self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
                if self.last['dec'][0]:
                    self.k.add_edge(self.last['dec'][0][-1], n_label, step=self.step, t=3)
                self.k.add_edge(a_label, n_label, step=self.step, t=3)
                self.last['dec'][0].append(n_label)
                self.d_change.append(self.step)

                # break
            else:
                break

        # self.history_d[self.step]=self.d

        return g_t


    def make_decision(self, ph_l):
        """
        Return decision index, based on pheromone list.
        """
        # convert pheromones to percentage
        if sum(ph_l) == 0.0:
            percent_list = [1.0 / len(ph_l)] * len(ph_l)
        else:
            percent_list = [float(i) / sum(ph_l) for i in ph_l]
        cumulative_percent = np.cumsum(percent_list)
        # print cumulative_percent

        # Choose decision index
        select_index = bisect(cumulative_percent, np.random.uniform(0, 1, 1))

        return select_index

    def problem_difficulty(self, decision):
        scores = []
        for route in itertools.permutations(self.tsp.nodes()):
            r = list(route)

            if r[:len(decision)] == decision:
                # valid given decision, get score
                # print r
                score = 0.0
                e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
                del e_r[-1]

                for e in e_r:
                    score += float(self.tsp[e[0]][e[1]]['dist'])

                scores.append(score)
            else:
                continue

        # print scores
        hist, bin_edges = np.histogram(scores, density=True)
        p = hist / hist.sum()
        a_p = np.array(p)
        a_p = a_p[np.nonzero(a_p)]
        h = -sum(a_p * np.log(a_p))

        m = np.mean(scores)
        v = np.var(scores)

        print decision, ' mean={}, entropy={}, variance={}'.format(np.mean(scores), h, np.var(scores))

        return m, h, v


def run_ACO(steps=100, to_step_lim=False, cities=5, tolerance=0.7, alpha=1.0, beta=1.0, dissipation=0.2, explore=1.0, force=None, rug=1.0):
    t = TSP(start=0, tolerance=tolerance, alpha=alpha, beta=beta, dissipation=dissipation, explore=explore,
            n_cities=cities, force=force,rug=rug)

    routes = []
    converge = 0
    decided = 0
    for i in xrange(steps):
        r, s = t.walk()
        routes.append((r, s))
        t.state = t.update_edges(r, s)

        # check convergence

        if len(routes) >= 2:
            if routes[-1] == routes[-2]:
                # print converge
                converge += 1
            else:
                converge = 0
        if to_step_lim==False:
            if len(t.d) == (cities + 1) and decided == 0:
                converge = 10
                decided = 1

            if converge >= 20:
                break

            # if len(t.d)==t.state.number_of_nodes():
            # break
            # print t.edges(data=True)
    #Create t.d if not decided
    if len(t.d)<t.tsp.number_of_nodes():
        t.d=list(routes[-1][0])
    # print routes
    # print t.state.edges(data=True)
    print t.d
    # print t.d_change
    return t, routes


def KAD_analysis(t,interval=1):
    #Removing d=[0] messed up data analysis of decision nodes?
    results = {}
    norm=True
    # analysis_type = ['out', 'close', 'between', 'katz', 'hits-a', 'hits-h']
    # analysis_type = ['out', 'in', 'close', 'between', 'hits-a', 'hits-h']
    # analysis_type = ['out', 'in', 'between', 'hits-a', 'hits-h']
    analysis_type = ['out', 'in', 'hits-a', 'hits-h']
    for a_type in analysis_type:
        results[a_type] = {}

    for s, kad in t.history_k.iteritems():
        if (s % interval != 0) and (s != 0):
            continue

        if s == 0:
            continue
            results['hits-h'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['hits-a'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['out'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['in'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            # results['close'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            # results['between'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            # results['katz'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            # results['eigen'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)

        if s != 0:
            (h, a) = nx.hits(kad, max_iter=10000, normalized=norm)  # (hubs,authorities)
            results['hits-h'][s] = h
            results['hits-a'][s] = a
            results['out'][s] = kad.out_degree() # nx.out_degree_centrality(kad)
            results['in'][s] = kad.in_degree() # nx.in_degree_centrality(kad)
            # results['close'][s] = nx.closeness_centrality(kad, normalized=norm)
            # results['between'][s] = nx.betweenness_centrality(kad, normalized=norm)
            # results['katz'][s] = nx.katz_centrality(kad, normalized=norm)
            # results['eigen'][s] = nx.eigenvector_centrality(kad.reverse(),max_iter=100000)

    return results

def analysis_suite(t, interval=1,offset_range=15):
    # Get results
    kad_results = KAD_analysis(t,interval=interval)
    ak_last={}
    ak_last['a']=t.last_action
    ak_last['k']=t.last

    # Get the final solution answers
    solution = t.d
    sol_edges = [tuple(solution[i:i + 2]) for i in xrange(0, len(solution), 1)]
    del sol_edges[-1]
    # columns_names
    param_columns=['N','p','exp','tol','force','alpha','beta',
                   'tsp_sol','tsp_nsol','start_sol','start_nsol','centrality','s']

    # columns_names
    cent_columns = param_columns+['layer']
    cent_param_columns=list(cent_columns)

    #parameters for Df
    params=[t.n_cities,t.dissipation,t.explore,t.tolerance,t.force,t.alpha,t.beta,
            t.tsp_sol_e,t.tsp_nonsol_e,t.start_sol_e,t.start_nonsol_e]
    cent_output = []

    #start path dictionary
    s_paths = {}

    # process
    for s,tsp in t.history_s.iteritems():
        if (s % interval!=0) and (s!=0):
            continue

        if s==0:
            continue
        #centrality
        #Df: params;centrality type;s;a/k;macro;edge;prototype;
        #Macro -> average by type -> average by nodes
        #M-D,A-D,M-Proto,Sep-Proto,A-Proto, M-Start,A-Sol Start,A-NSol Start, M-TSP, A-TSPSol, A-TSP NSol
        s_paths[s]={}



        for ana,results_series in kad_results.iteritems():
            results_dict=results_series[s]

            for layer,last in ak_last.iteritems():
                #analysis ID for Df
                a_ID=[ana,s,layer]

                # Get an macro, edge and prototype

                cent_state = params+a_ID
                #Decision
                tot_dec = []
                #n_nodes = 0
                for sel, nodes in last['dec'].iteritems():
                    for n in nodes:
                        if n in results_dict:
                            #n_nodes += 1
                            tot_dec.append(results_dict[n])
                column_add=['Decision Sum','Decision','Decision Mean']
                if not (set(column_add) < set(cent_columns)):
                    cent_columns.extend(column_add)
                if not tot_dec:
                    tot_dec=[0.0]
                cent_state.extend([np.sum(tot_dec),np.sum(tot_dec),np.mean(tot_dec)])

                # K Edge Selection
                tot_route = []
                for sel, nodes in last['route'].iteritems():
                    sel_tot = []
                    # n_nodes = 0
                    for n in nodes:
                        if n in results_dict:
                            # n_nodes += 1
                            tot_route.append(results_dict[n])
                            sel_tot.append(results_dict[n])
                    column_add=['Step-{} Sum'.format((sel+1)),'Step-{} Mean'.format(sel+1)]
                    if not (set(column_add) < set(cent_columns)):
                        cent_columns.extend(column_add)
                    cent_state.extend([np.sum(sel_tot),np.mean(sel_tot)])
                column_add=['Step Sum','Step Mean']
                if not (set(column_add) < set(cent_columns)):
                    cent_columns.extend(column_add)
                cent_state.extend([np.sum(tot_route),np.mean(tot_route)])

                # K solution Score
                tot_score = []

                for sel, nodes in last['score'].iteritems():
                    for n in nodes:
                        if n in results_dict:
                            tot_score.append(results_dict[n])
                column_add=['Score Sum', 'Score', 'Score Mean']
                if not (set(column_add) < set(cent_columns)):
                    cent_columns.extend(column_add)
                cent_state.extend([np.sum(tot_score), np.sum(tot_score), np.mean(tot_score)])

                # K edges - sort by start-non solution, start-solution, tsp-solution, tsp-non solution
                tot_start = []
                tot_tsp = []

                s_ns = []  # start non solution
                s_s = []  # start solution
                tsp_s = []  # tsp solution
                tsp_ns = []  # tsp non solution

                for sel, nodes in last['dist'].iteritems():
                    for n in nodes:
                        if n in results_dict:
                            if sel[0] == 0:
                                tot_start.append(results_dict[n])
                                if sel in sol_edges:  # in the solution
                                    s_s.append(results_dict[n])
                                else:  # not in solution
                                    s_ns.append(results_dict[n])

                            else:
                                tot_tsp.append(results_dict[n])
                                if sel in sol_edges:  # in the solution
                                    tsp_s.append(results_dict[n])
                                else:  # not in solution
                                    tsp_ns.append(results_dict[n])
                column_add=['Start Edges Sum','Start Edges Non-Solution Sum','Start Edges Solution Sum',
                                     'Start Edges Mean','Start Edges Non-Solution Mean', 'Start Edges Solution Mean',
                                     'TSP Edges Sum', 'TSP Edges Non-Solution Sum', 'TSP Edges Solution Sum',
                                     'TSP Edges Mean','TSP Edges Non-Solution Mean', 'TSP Edges Solution Mean',]
                if not (set(column_add) < set(cent_columns)):
                    cent_columns.extend(column_add)

                cent_state.extend([np.sum(tot_start),np.sum(s_ns),np.sum(s_s),
                                   np.mean(tot_start),np.mean(s_ns),np.mean(s_s),
                                   np.sum(tot_tsp), np.sum(tsp_ns), np.sum(tsp_s),
                                   np.mean(tot_tsp), np.mean(tsp_ns), np.mean(tsp_s)])

                cent_output.append(cent_state)

            #Strongest Path analysis

            # Get states edge information
            state_edges = {}
            state_edges['c'] = {}
            state_edges['ph'] = {}
            for u, v, d in tsp.edges(data=True):
                if d['dist'] == 0.0:
                    e_h = t.explore
                else:
                    e_h = 1.0 / tsp[u][v]['dist']
                e_ph = np.power(tsp[u][v]['p'], t.alpha) * np.power(e_h, 0.0)  # tsp_object.beta)
                state_edges['ph'][(u, v)] = e_ph

                # for last_node in tsp_object.last['dist'][(u,v)][::-1]:  # iterate through nodes to get centrality results
                #     if last_node in results_series[s]:
                #         e_cent = results_series[s][last_node]
                #         break  # Only use the most recent and exit

                e_cent = 0.0
                for last_node in t.last['dist'][(u, v)]:  # iterate through nodes to get centrality results
                    if last_node in results_dict:
                        e_cent += results_dict[last_node]
                state_edges['c'][(u, v)] = e_cent

            #get strongest paths
            # Strongest Paths
            tabu_p = [0]
            tabu_c = [0]
            unique = True
            for i in xrange(t.tsp.number_of_nodes() - 1):  # iterate through nodes
                # print unique
                # if unique == False:
                #     break
                step_e_p = {}
                step_e_c = {}
                for u, v in t.tsp.edges():
                    # cent
                    if u == tabu_c[-1] and (v not in tabu_c):
                        step_e_c[(u, v)] = state_edges['c'][(u, v)]
                    # ph
                    if u == tabu_p[-1] and (v not in tabu_p):
                        step_e_p[(u, v)] = state_edges['ph'][(u, v)]
                        # print (u,v),tabu_c, step_e_c, tabu_p, step_e_p

                # get maxes
                max_e_c = max(step_e_c.iteritems(), key=operator.itemgetter(1))[0] #max edge
                #print max_e_c
                c_keys = []
                for e_c, val in step_e_c.iteritems():
                    if val == step_e_c[max_e_c]:
                        c_keys.append(e_c)
                if len(c_keys) > 1:
                    unique = False
                    # print c_keys
                    tabu_c.append(random.choice(c_keys)[1])
                    #tabu_c.append(max_e_c[1]) #head of max edge

                else:
                    tabu_c.append(max_e_c[1]) #head of max edge

                max_e_p = max(step_e_p.iteritems(), key=operator.itemgetter(1))[0] #max edge
                p_keys = []
                for e_p, val in step_e_p.iteritems():
                    if val == step_e_p[max_e_p]:
                        p_keys.append(e_p)
                if len(p_keys) > 1:
                    unique = False
                    # print p_keys
                    tabu_p.append(random.choice(p_keys)[1])
                    #tabu_p.append(max_e_p[1])

                else:
                    tabu_p.append(max_e_p[1])

            # compare agreeance of paths
            e_path_ph = [tuple(tabu_p[i:i + 2]) for i in xrange(0, len(tabu_p), 1)]
            del e_path_ph[-1]

            if 'ph' not in s_paths[s]:
                s_paths[s]['ph']=e_path_ph

            e_path_cent = [tuple(tabu_c[i:i + 2]) for i in xrange(0, len(tabu_c), 1)]
            del e_path_cent[-1]
            s_paths[s][ana]=e_path_cent
    # print len(cent_output), cent_output[-1]
    # print len(cent_columns), cent_columns
    # for col_i in xrange(len(cent_columns)):
    #     print cent_columns[col_i],cent_output[-1][col_i]
    cent_df=pd.DataFrame(cent_output,columns=cent_columns)

    #Path error analysis
    # check prediction at different offsets
    # Df: params;centrality type;s;offset;E(path)
    path_columns=param_columns+['offset','Strongest Path Error']
    path_param_columns=list(param_columns+['offset'])
    path_output = []

    for offset in xrange(-offset_range, offset_range+1):  # amount to offset centrality results
        e_off_s = []
        for s in s_paths:  # go over each state
            s_off = s + offset
            # Track errors at this state
            if s_off not in s_paths:  # offset doesn't exist
                continue

            # sorted array of all centrality ml's
            path_ph=s_paths[s]['ph']

            for ana,path_c in s_paths[s_off].iteritems():
                if ana=='ph':
                    continue
                shared = list(set(path_ph).intersection(path_c))
                error_path = 1.0 - float(len(shared)) / len(path_c)

                #Add to output
                off_output=params+[ana,s,offset,error_path]
                path_output.append(off_output)
    # print path_output

    path_df = pd.DataFrame(path_output, columns=path_columns)

    return cent_df,path_df,cent_param_columns,path_param_columns


def influence_analysis(t,results):
    # Create array of centrality results
    # One array for knowledge
    # One array for action
    # Within array: list of lists - each list is a time step

    # Get the final solution answers
    solution = t.d
    sol_edges = [tuple(solution[i:i + 2]) for i in xrange(0, len(solution), 1)]
    del sol_edges[-1]

    colors = ['#2166ac', '#fee090', '#fdbb84', '#fc8d59', '#e34a33']

    # Array out: macro;edge;proto

    macro_influence = {}
    edge_influence = {}
    proto_influence = {}

    for ana, results_series in results.iteritems():
        # get analysis and corresponding dictionary steps
        macro_influence[ana] = {}
        k_results = []
        a_results = []

        edge_influence[ana] = {}
        ek_results = []
        ea_results = []

        proto_influence[ana] = {}
        pk_results = []
        pa_results = []

        labels = []
        for s, results_dict in results_series.iteritems():
            # sort results based on type in knowledge and action
            # types: Decision, Edges Selection, Solution Score, Starting Edges, TSP Edges,
            # self.last = {'dist': {}, 'score': {}, 'route': {}, 'dec': {}}
            labels.append(s)
            k_s = []
            a_s = []

            proto_k_s = []
            proto_a_s = []

            # K decision Score
            tot_dec = 0.0
            for sel, nodes in t.last['dec'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_dec += results_dict[n]

            k_s.append(tot_dec)

            # K Edge Selection
            tot_route = 0.0

            for sel, nodes in t.last['route'].iteritems():
                sel_tot = 0.0
                for n in nodes:
                    if n in results_dict:
                        tot_route += results_dict[n]
                        sel_tot += results_dict[n]
                proto_k_s.append(sel_tot)

            k_s.append(tot_route)

            # K solution Score
            tot_score = 0.0
            for sel, nodes in t.last['score'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_score += results_dict[n]

            k_s.append(tot_score)

            # K edges - sort by start-non solution, start-solution, tsp-solution, tsp-non solution
            tot_start = 0.0
            tot_tsp = 0.0

            s_ns = 0.0  # start non solution
            s_s = 0.0  # start solution
            tsp_s = 0.0  # tsp solution
            tsp_ns = 0.0  # tsp non solution

            for sel, nodes in t.last['dist'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        if sel[0] == 0:
                            tot_start += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                s_s += results_dict[n]
                            else:  # not in solution
                                s_ns += results_dict[n]

                        else:
                            tot_tsp += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                tsp_s += results_dict[n]
                            else:  # not in solution
                                tsp_ns += results_dict[n]
            k_s.append(tot_start)
            k_s.append(tot_tsp)
            edges_k_s = [s_ns, s_s, tsp_s, tsp_ns]

            # A decision Score
            tot_dec = 0.0
            for sel, nodes in t.last_action['dec'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_dec += results_dict[n]

            a_s.append(tot_dec)

            # A Edge Selection
            tot_route = 0.0

            for sel, nodes in t.last_action['route'].iteritems():
                sel_tot = 0.0
                for n in nodes:
                    if n in results_dict:
                        tot_route += results_dict[n]
                        sel_tot += results_dict[n]
                proto_a_s.append(sel_tot)

            a_s.append(tot_route)

            # A solution Score
            tot_score = 0.0
            for sel, nodes in t.last_action['score'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_score += results_dict[n]

            a_s.append(tot_score)

            # A edges
            tot_start = 0.0
            tot_tsp = 0.0

            s_ns = 0.0  # start non solution
            s_s = 0.0  # start solution
            tsp_s = 0.0  # tsp solution
            tsp_ns = 0.0  # tsp non solution

            for sel, nodes in t.last_action['dist'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        if sel[0] == 0:
                            tot_start += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                s_s += results_dict[n]
                            else:  # not in solution
                                s_ns += results_dict[n]
                        else:
                            tot_tsp += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                tsp_s += results_dict[n]
                            else:  # not in solution
                                tsp_ns += results_dict[n]
            a_s.append(tot_start)
            a_s.append(tot_tsp)
            edges_a_s = [s_ns, s_s, tsp_s, tsp_ns]

            # Append all
            k_results.append(k_s)
            a_results.append(a_s)
            ek_results.append(edges_k_s)
            ea_results.append(edges_a_s)
            pk_results.append(proto_k_s)
            pa_results.append(proto_a_s)

        # Save macro influence
        k = np.array(k_results)
        a = np.array(a_results)
        macro_influence[ana]['k'] = k
        macro_influence[ana]['a'] = a
        # Plot macro influence
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        sbg.stackedBarPlot(ax1, k, colors, xLabels=labels)
        plt.title('{} Knowledge Influence'.format(ana))

        ax2 = fig.add_subplot(212)
        sbg.stackedBarPlot(ax2, a, colors, xLabels=labels)
        plt.title('{} Action Influence'.format(ana))

        # Save edge influence
        ek = np.array(ek_results)
        ea = np.array(ea_results)
        edge_influence[ana]['k'] = ek
        edge_influence[ana]['a'] = ea
        # Plot edge influence
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        sbg.stackedBarPlot(ax1, ek, colors[1::], xLabels=labels)
        plt.title('{} Knowledge Edge Influence'.format(ana))

        ax2 = fig.add_subplot(212)
        sbg.stackedBarPlot(ax2, ea, colors[1::], xLabels=labels)
        plt.title('{} Action Edge Influence'.format(ana))

        # Save prototype influence
        pk = np.array(pk_results)
        pa = np.array(pa_results)
        proto_influence[ana]['k'] = pk
        proto_influence[ana]['a'] = pa
        # Plot edge influence
        cmap = cm.get_cmap('jet')
        c = [cmap(float(i) / t.tsp.number_of_nodes()) for i in xrange(t.tsp.number_of_nodes())]
        # print c
        # print pk

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        sbg.stackedBarPlot(ax1, pk, c, xLabels=labels)
        plt.title('{} Knowledge Prototype Influence'.format(ana))

        ax2 = fig.add_subplot(212)
        sbg.stackedBarPlot(ax2, pa, c, xLabels=labels)
        plt.title('{} Action Prototype Influence'.format(ana))

    return macro_influence, edge_influence, proto_influence


def influence_chart(t,results):
    # Create array of centrality results
    # One array for knowledge
    # One array for action
    # Within array: list of lists - each list is a time step

    # Get the final solution answers
    solution=t.d
    sol_edges=[tuple(solution[i:i + 2]) for i in xrange(0, len(solution), 1)]
    del sol_edges[-1]

    colors=['#2166ac','#fee090','#fdbb84','#fc8d59','#e34a33']

    macro_influence={}
    edge_influence={}
    proto_influence={}

    for ana, results_series in results.iteritems():
        #get analysis and corresponding dictionary steps
        macro_influence[ana]={}
        k_results=[]
        a_results=[]

        edge_influence[ana] = {}
        ek_results=[]
        ea_results=[]

        proto_influence[ana] = {}
        pk_results=[]
        pa_results=[]

        labels=[]
        for s, results_dict in results_series.iteritems():
            #sort results based on type in knowledge and action
            #types: Decision, Edges Selection, Solution Score, Starting Edges, TSP Edges,
            # self.last = {'dist': {}, 'score': {}, 'route': {}, 'dec': {}}
            labels.append(s)
            k_s=[]
            a_s=[]

            proto_k_s=[]
            proto_a_s = []

            # K decision Score
            tot_dec = 0.0
            for sel, nodes in t.last['dec'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_dec += results_dict[n]

            k_s.append(tot_dec)

            #K Edge Selection
            tot_route=0.0

            for sel,nodes in t.last['route'].iteritems():
                sel_tot=0.0
                for n in nodes:
                    if n in results_dict:
                        tot_route+=results_dict[n]
                        sel_tot+=results_dict[n]
                proto_k_s.append(sel_tot)

            k_s.append(tot_route)

            #K solution Score
            tot_score=0.0
            for sel,nodes in t.last['score'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_score += results_dict[n]

            k_s.append(tot_score)

            # K edges - sort by start-non solution, start-solution, tsp-solution, tsp-non solution
            tot_start = 0.0
            tot_tsp=0.0

            s_ns=0.0 #start non solution
            s_s=0.0 #start solution
            tsp_s=0.0 #tsp solution
            tsp_ns=0.0 #tsp non solution

            for sel, nodes in t.last['dist'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        if sel[0] == 0:
                            tot_start += results_dict[n]
                            if sel in sol_edges: #in the solution
                                s_s+=results_dict[n]
                            else: #not in solution
                                s_ns+=results_dict[n]

                        else:
                            tot_tsp += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                tsp_s += results_dict[n]
                            else:  # not in solution
                                tsp_ns += results_dict[n]
            k_s.append(tot_start)
            k_s.append(tot_tsp)
            edges_k_s=[s_ns,s_s,tsp_s,tsp_ns]


            # A decision Score
            tot_dec = 0.0
            for sel, nodes in t.last_action['dec'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_dec += results_dict[n]

            a_s.append(tot_dec)

            # A Edge Selection
            tot_route = 0.0

            for sel, nodes in t.last_action['route'].iteritems():
                sel_tot=0.0
                for n in nodes:
                    if n in results_dict:
                        tot_route += results_dict[n]
                        sel_tot += results_dict[n]
                proto_a_s.append(sel_tot)

            a_s.append(tot_route)

            # A solution Score
            tot_score = 0.0
            for sel, nodes in t.last_action['score'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        tot_score += results_dict[n]

            a_s.append(tot_score)

            # A edges
            tot_start = 0.0
            tot_tsp = 0.0

            s_ns = 0.0  # start non solution
            s_s = 0.0  # start solution
            tsp_s = 0.0  # tsp solution
            tsp_ns = 0.0  # tsp non solution

            for sel, nodes in t.last_action['dist'].iteritems():
                for n in nodes:
                    if n in results_dict:
                        if sel[0] == 0:
                            tot_start += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                s_s += results_dict[n]
                            else:  # not in solution
                                s_ns += results_dict[n]
                        else:
                            tot_tsp += results_dict[n]
                            if sel in sol_edges:  # in the solution
                                tsp_s += results_dict[n]
                            else:  # not in solution
                                tsp_ns += results_dict[n]
            a_s.append(tot_start)
            a_s.append(tot_tsp)
            edges_a_s = [s_ns, s_s, tsp_s, tsp_ns]

            #Append all
            k_results.append(k_s)
            a_results.append(a_s)
            ek_results.append(edges_k_s)
            ea_results.append(edges_a_s)
            pk_results.append(proto_k_s)
            pa_results.append(proto_a_s)

        # Save macro influence
        k = np.array(k_results)
        a = np.array(a_results)
        macro_influence[ana]['k']=k
        macro_influence[ana]['a']=a
        # Plot macro influence
        fig=plt.figure()
        ax1=fig.add_subplot(211)
        sbg.stackedBarPlot(ax1,k,colors,xLabels=labels)
        plt.title('{} Knowledge Influence'.format(ana))

        ax2 = fig.add_subplot(212)
        sbg.stackedBarPlot(ax2, a, colors, xLabels=labels)
        plt.title('{} Action Influence'.format(ana))

        # Save edge influence
        ek = np.array(ek_results)
        ea = np.array(ea_results)
        edge_influence[ana]['k'] = ek
        edge_influence[ana]['a'] = ea
        # Plot edge influence
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        sbg.stackedBarPlot(ax1, ek, colors[1::], xLabels=labels)
        plt.title('{} Knowledge Edge Influence'.format(ana))

        ax2 = fig.add_subplot(212)
        sbg.stackedBarPlot(ax2, ea, colors[1::], xLabels=labels)
        plt.title('{} Action Edge Influence'.format(ana))

        # Save prototype influence
        pk = np.array(pk_results)
        pa = np.array(pa_results)
        proto_influence[ana]['k'] = pk
        proto_influence[ana]['a'] = pa
        # Plot edge influence
        cmap = cm.get_cmap('jet')
        c = [cmap(float(i) /t.tsp.number_of_nodes()) for i in xrange(t.tsp.number_of_nodes())]
        # print c
        # print pk

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        sbg.stackedBarPlot(ax1, pk, c, xLabels=labels)
        plt.title('{} Knowledge Prototype Influence'.format(ana))

        ax2 = fig.add_subplot(212)
        sbg.stackedBarPlot(ax2, pa, c, xLabels=labels)
        plt.title('{} Action Prototype Influence'.format(ana))

    return macro_influence,edge_influence,proto_influence

def KD_analysis(t):
    ### Test Results ###
    results = {}
    results['out'] = {}

    for s, kad in t.history_k.iteritems():
        kd, action = project_KD_A(kad)

        c = nx.out_degree_centrality(kd)
        # c=nx.closeness_centrality(kd,normalized=True)
        # c=nx.betweenness_centrality(kd,normalized=True)
        for k in t.last:
            results['out'].setdefault(k, {})
            for e in t.last[k]:
                n_list = t.last[k][e]
                tc = 0.0
                for n in n_list:
                    if n in c:
                        tc += c[n]
                        # else:
                        # continue

                add = results['out'][k].setdefault(e, [])
                add.append(tc)

    results['hits-a'] = {}
    results['hits-h'] = {}
    for s, kad in t.history_k.iteritems():
        kd, action = project_KD_A(kad)

        if s != 0:
            (h, a) = nx.hits(kd, max_iter=10000, normalized=True)  # (hubs,authorities)
        else:
            h = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            a = dict.fromkeys(t.history_k[s].nodes(), 0.0)
        # c=nx.betweenness_centrality(kd,normalized=True)
        for k in t.last:
            results['hits-a'].setdefault(k, {})
            results['hits-h'].setdefault(k, {})
            for e in t.last[k]:
                n_list = t.last[k][e]
                ta = 0.0
                th = 0.0

                for n in n_list:
                    if n in a:
                        inc_a = a[n]
                        ta += a[n]
                    if n in h:
                        inc_h = h[n]
                        th += h[n]
                        # else:
                        # continue

                add = results['hits-a'][k].setdefault(e, [])
                add.append(ta)  # ta #use ta for accumulated, inc_a for timestep

                add = results['hits-h'][k].setdefault(e, [])
                add.append(th)  # th #use th for accumulated, inc_h for timestep

    return results


def A_analysis(t):
    ### Test Results ###
    results = {}
    results['out'] = {}
    results['between'] = {}

    for s, kad in t.history_k.iteritems():
        kd, action = project_KD_A(kad)

        c_o = nx.out_degree_centrality(action)
        # c=nx.closeness_centrality(action,normalized=True)
        c_b = nx.betweenness_centrality(action, normalized=True)
        for k in t.last_action:
            results['out'].setdefault(k, {})
            results['between'].setdefault(k, {})
            for e in t.last_action[k]:
                n_list = t.last_action[k][e]
                tc_o = 0.0
                tc_b = 0.0
                inc_b = 0.0
                for n in n_list:
                    if n in c_o:
                        tc_o += c_o[n]

                    if n in c_b:
                        tc_b += c_b[n]
                        inc_b = c_b[n]
                        # else:
                        # continue

                add_o = results['out'][k].setdefault(e, [])
                add_o.append(tc_o)

                add_b = results['between'][k].setdefault(e, [])
                add_b.append(tc_b)

    results['hits-a'] = {}
    results['hits-h'] = {}
    for s, kad in t.history_k.iteritems():
        kd, action = project_KD_A(kad)

        if s != 0:
            (h, a) = nx.hits(action, max_iter=10000, normalized=True)  # (hubs,authorities)
        else:
            h = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            a = dict.fromkeys(t.history_k[s].nodes(), 0.0)
        # c=nx.betweenness_centrality(kd,normalized=True)
        for k in t.last_action:
            results['hits-a'].setdefault(k, {})
            results['hits-h'].setdefault(k, {})
            for e in t.last_action[k]:
                n_list = t.last_action[k][e]
                ta = 0.0
                th = 0.0
                inc_a = 0.0
                inc_h = 0.0

                for n in n_list:
                    if n in a:
                        inc_a = a[n]
                        ta += a[n]
                    if n in h:
                        inc_h = h[n]
                        th += h[n]
                        # else:
                        # continue

                add = results['hits-a'][k].setdefault(e, [])
                add.append(ta)  # ta #use ta for accumulated, inc_a for timestep

                add = results['hits-h'][k].setdefault(e, [])
                add.append(th)  # th #use th for accumulated, inc_h for timestep

    return results


# Ranks through time
def cent_rank_by_time(t):
    # Track the centrality rankings of actions by age

    # Get nodes of decision
    d_nodes = t.last_action['dec'][0]
    # del d_nodes[0] #0th index is starting node-'no decision'

    age_dict = {}
    # Decisions tracking
    # print t.history_k
    for step, kad in t.history_k.iteritems():  # iterate through kad maps
        d_in_kad = []
        for n in d_nodes:
            if n in kad.nodes():
                d_in_kad.append(n)

        if d_in_kad:  # only procceed if a decision node exists
            kd, action = project_KD_A(kad)  # get action projection
            bc = nx.betweenness_centrality(action, normalized=True)  # calc betweenness centrality
            bc_array = np.array(bc.values())  # set up array
            bc_array = np.sort(bc_array)  # sort by array by centrality

            for n in d_in_kad:  # go through decision nodes
                n_step = action.node[n]['step']  # get node step
                age = step - n_step  # get total age
                bc_rank = bc_array.tolist().index(bc[n]) / float(
                    action.number_of_nodes())  # get rank of decision in between centrality
                # print step,n,bc_rank
                add = age_dict.setdefault(age, [])  # if age does not exist, create
                add.append(bc_rank)  # append rank

    return age_dict


# Ranks through time
def cent_rank_by_time_KAD(t):
    # Track the centrality rankings of actions by age

    # Get nodes of decision
    d_nodes = t.last_action['dec'][0]
    # del d_nodes[0] #0th index is starting node-'no decision'

    age_dict = {}
    # Decisions tracking
    # print t.history_k
    for step, kad in t.history_k.iteritems():  # iterate through kad maps
        d_in_kad = []
        for n in d_nodes:
            if n in kad.nodes():
                d_in_kad.append(n)

        if d_in_kad:  # only procceed if a decision node exists
            # kd, action = project_KD_A(kad) #get action projection
            #(h, a) = nx.hits(kad, max_iter=10000, normalized=True)
            bc = nx.betweenness_centrality(kad, normalized=True)  # calc betweenness centrality
            bc_array = np.array(bc.values())  # set up array
            bc_array = np.sort(bc_array)  # sort by array by centrality

            for n in d_in_kad:  # go through decision nodes
                n_step = kad.node[n]['step']  # get node step
                age = step - n_step  # get total age
                bc_rank = bc_array.tolist().index(bc[n]) / float(
                    kad.number_of_nodes())  # get rank of decision in between centrality
                # print step,n,bc_rank
                add = age_dict.setdefault(age, [])  # if age does not exist, create
                add.append(bc_rank)  # append rank

    return age_dict


# Criticality through time
def crit_rank_by_prob(t, routes):
    # Track critical node rankings of actions by probability it creates part of a solution

    prob_dict = {}

    for step, kad in t.history_k.iteritems():
        if step == 0:
            continue
        kd, action = project_KD_A(kad)  # get kd projection
        odc = nx.out_degree_centrality(kd)  # calc out degree centrality
        odc_array = np.array(odc.values())  # set up array
        odc_array = np.sort(odc_array)  # sort by array by centrality


# Decision ID
def decision_ID(tsp_object, routes):
    kd, action = project_KD_A(t.history_k[len(tsp_object.history_k) - 1])  # check last KAD
    bc = nx.betweenness_centrality(action, normalized=True)
    cc = nx.closeness_centrality(action, normalized=True)
    odc = nx.out_degree_centrality(action)
    n_nodes = float(action.number_of_nodes())
    # sored array of all centrality ml's
    bc_rank_list = []
    cc_rank_list = []
    odc_rank_list = []
    bc_array = np.array(bc.values())
    bc_array = np.sort(bc_array)[::-1]
    cc_array = np.array(cc.values())
    cc_array = np.sort(cc_array)[::-1]
    odc_array = np.array(odc.values())
    odc_array = np.sort(odc_array)[::-1]
    for n in tsp_object.last_action['dec'][0]:  # go through paths with max pheromone ml
        # print n, action.node[n]
        bc_rank = bc_array.tolist().index(bc[n])  # get rank of decision in between centrality
        bc_rank_list.append(bc_rank)
        cc_rank = cc_array.tolist().index(cc[n])  # get rank of decision in closeness centrality
        cc_rank_list.append(cc_rank)
        odc_rank = odc_array.tolist().index(odc[n])  # get rank of decision in outdegree centrality
        odc_rank_list.append(odc_rank)

    # Check critical moves
    r = routes[-1][0]
    e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
    del e_r[-1]
    maxes = {}
    print e_r
    for i in xrange(len(e_r)):
        bc_route = []
        cc_route = []
        odc_route = []
        # for n in tsp_object.last_action['route'][i]:

        for n in tsp_object.last_action['dist'][e_r[i]]:
            # for e,n in tsp_object.last_action['dist'].iteritems():
            # print n

            # if action.node[n]['e']==e_r[i]: #for route
            if action.node[n]['label'] == e_r[i]:  # for distance
                # if action.node[n]['label'] == e: #for all distance
                print n, action.node[n], e_r[i], bc_array.tolist().index(bc[n]), cc_array.tolist().index(
                    cc[n]), odc_array.tolist().index(odc[n])
                bc_route.append(bc_array.tolist().index(bc[n]))  # get rank of decision in between centrality
                cc_route.append(cc_array.tolist().index(cc[n]))  # get rank of decision in closeness centrality
                odc_route.append(odc_array.tolist().index(odc[n]))  # get rank of decision in outdegree centrality
        maxes[(i, e_r[i])] = (min(bc_route) / n_nodes, min(cc_route) / n_nodes, min(odc_route) / n_nodes)

    # Centrality rank by edge
    bar_dict = {}
    for e, n_list in tsp_object.last_action['dist'].iteritems():
        e_rank = []
        for n in n_list:
            e_rank.append(odc_array.tolist().index(odc[n]))  # get rank of decision in outdegree centrality
        if e_rank:
            bar_dict[e] = min(e_rank)
    plt.figure()
    plt.bar(range(len(bar_dict)), bar_dict.values(), align="center")
    plt.xticks(range(len(bar_dict)), list(bar_dict.keys()))

    # print 'betweenness ranks:', bc_rank_list
    # print 'closeness ranks:', cc_rank_list
    # print 'out degree ranks:', odc_rank_list
    # for n,c in bc.iteritems():
    #     print 'n:{}, rank:{}, data:{}'.format(n, bc_array.tolist().index(bc[n]),action.node[n])
    return np.mean(bc_rank_list) / n_nodes, np.mean(cc_rank_list) / n_nodes, np.mean(odc_rank_list) / n_nodes, maxes


def plot_results(results, KAD_type='KD', start_on=True, dec_on=True, rest_on=True, results_to_plot='hits-h'):
    # Plotting results
    # start_on=True
    # dec_on=True
    # rest_on=True
    cmap = cm.get_cmap('jet')
    x = range(t.step)
    # results to plot
    plot_results = results_to_plot

    e_r = [tuple(t.d[i:i + 2]) for i in xrange(0, len(t.d), 1)]
    del e_r[-1]

    for k, ele in results[plot_results].iteritems():  # ['hits']['a'] ['hits']['h'] ['out']
        plt.figure(figsize=(12, 7))
        N = len(ele.keys())
        i = 0

        for e, series in ele.iteritems():
            # print series
            if N > 1:
                c = cmap(float(i) / (N - 1))
            else:
                c = cmap(0)

            if type(e) == tuple:
                if (e in e_r) and dec_on:
                    plt.plot(x, series, '--', linewidth=2, label='{}'.format(e), color=c, )
                elif (e[0] == 0) and start_on:
                    plt.plot(x, series, ':', linewidth=2, label='{}'.format(e), color=c)
                elif rest_on:
                    plt.plot(x, series, label='{}'.format(e), color=c)

            else:
                plt.plot(x, series, label='{}'.format(e), color=c)
            i += 1

        for i in xrange(len(t.d_change)):
            # print t.d_change[i]
            plt.axvline(x=t.d_change[i], color='r', label='decision')  # , label='decision {}'.format(i))

        # plt.axvline(x=t.d_change[0], color='r', label='decision')

        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        if k == 'dist':
            k_label = 'distance'
        elif k == 'dec':
            k_label = 'decision'
        elif k == 'p':
            k_label = 'preference'
        else:
            k_label = k

        plt.xlabel('Time')
        plt.ylabel('Centrality')
        # _ = plt.ylim([0.0, 1.1])#max(series)+.1])

        # plt.title('Final solution = {}, Accumulated {} Centrality, {}'.format(t.d,plot_results,k_label))
        plt.title('Final solution = {}, {}:{} Centrality, {}'.format(t.d, KAD_type, plot_results, k_label))


def project_KD_A(KAD):
    # input KAD and return KD and A networks
    # KD is projection of A layer
    # A is projection of KD layer

    # Flatten to KD
    # iterate through nodes, if edge points to action->flatten, if edge points to otherwise->keep
    KD = KAD.copy()
    for i in KD.nodes():
        if KD.node[i]['layer'] == 'a':
            step = KD.node[i]['step']
            # get successors
            tails = KD.predecessors(i)

            # get head
            heads = KD.successors(i)

            # Add edges
            for tail in tails:
                for head in heads:
                    KD.add_edge(tail, head, step=step)

            KD.remove_node(i)

    A = KAD.copy()
    for i in A.nodes():
        if A.node[i]['layer'] in ['k', 'd']:
            # get successors
            preds = A.predecessors(i)
            tails = []
            for p in preds:
                if A.node[p]['layer'] not in ['k', 'd']:
                    tails.append(p)

            # get head
            sucs = A.successors(i)
            heads = []
            for s in sucs:
                if A.node[s]['layer'] not in ['k', 'd']:
                    heads.append(s)

            # Add edges
            for tail in tails:
                for head in heads:
                    step = A.node[head]['step']  # step related to head
                    A.add_edge(tail, head, step=step)

            # Add edges if initialized k
            if not tails:  # no preds
                e_bad_direction = list(itertools.combinations(heads, 2))  # ID edges
                for e in e_bad_direction:  # iterate through edgs
                    if e[1] < e[0]:  # Flip if reverse order
                        e = (e[1], e[0])
                    step = A.node[e[0]]['step']  # step related to head
                    # print i, e
                    A.add_edge(e[0], e[1], step=step)

            A.remove_node(i)
    return KD, A


def ML_path_comparison_KAD(tsp_object, results_series, title,comparison_base=0):
    # comparison_base: 0=compare to pheromone, 1=compare to centrality
    # Compare the maximum likelihood paths between a set of results and pheromones
    tsp_series = tsp_object.history_s

    # Get mle(s) for pheromones
    ml_pheromone = {}
    ml_centrality = {}
    prob_influence = {} #track probability that an edge influences a solution
    ml_error = {}
    average_prob_error = {}
    average_rank_error = {}
    ml_paths = {}
    s_paths={}
    s_paths['c']={}
    s_paths['ph']={}

    for s, tsp in tsp_series.iteritems():  # go through time steps
        ml_pheromone[s] = {}
        ml_centrality[s] = {}
        prob_influence[s]=dict.fromkeys(tsp.edges(), [])
        ml_error[s] = {}
        error_list = []
        nodes = list(set(tsp.nodes()) - set([0]))

        #Get states edge information
        state_edges={}
        state_edges['c']={}
        state_edges['ph']={}
        total_c=0.0

        #track starting conditions
        cent={}
        ph={}


        for u,v,d in tsp.edges(data=True):
            if d['dist']==0.0:
                e_h=tsp_object.explore
            else:
                e_h = 1.0 / tsp[u][v]['dist']
            e_ph = np.power(tsp[u][v]['p'], tsp_object.alpha)* np.power(e_h,0.0) #tsp_object.beta)
            state_edges['ph'][(u,v)]=e_ph

            # for last_node in tsp_object.last['dist'][(u,v)][::-1]:  # iterate through nodes to get centrality results
            #     if last_node in results_series[s]:
            #         e_cent = results_series[s][last_node]
            #         break  # Only use the most recent and exit

            e_cent=0.0
            for last_node in tsp_object.last['dist'][(u, v)]:  # iterate through nodes to get centrality results
                if last_node in results_series[s]:
                    e_cent += results_series[s][last_node]
            state_edges['c'][(u, v)] = e_cent
            total_c+=e_cent

            # if u==0:
            #     cent[(u,v)]=e_cent
            #     ph[(u,v)]=e_ph

        # print s
        # for n in tsp.nodes():
        #     cent = {}
        #     ph = {}
        #     for e in tsp.edges():
        #         if e[0] == n:
        #             cent[e]=state_edges['c'][e]
        #             ph[e]=state_edges['ph'][e]
            # print '{} Cent:'.format(n), cent
            # print '{} Ph:'.format(n), ph

        # print s
        # print 'Cent:',cent
        # print 'Ph:', ph



        #Normalize centrality
        # if total_c!=0:
        #     for u,v,d in tsp.edges(data=True):
        #         state_edges['c'][(u,v)]=state_edges['c'][(u,v)]/total_c

        for route in itertools.permutations(nodes):  # calc mle for each route
            r = [0] + list(route)
            likelihood_ph = 1.0
            likelihood_cent = 1.0

            # Go through each step in the solution and calc prob that it occurred
            e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
            del e_r[-1]
            examined_edges = set(e_r)  # track which edges influence solution
            tabu = [0]
            for e in e_r:  # selected step
                e_ph=state_edges['ph'][e]
                e_cent=state_edges['c'][e]


                tot_ph = 0.0
                tot_cent = 0.0
                num_nodes = 0
                #print e_r, e, tabu
                for tail in tsp.nodes():
                    #print '  ',tail
                    if tail in tabu: # or tail==e[0]:
                        continue
                    num_nodes += 1
                    # get pheromones and centralities for available edges
                    e_tail=(e[0],tail)
                    tot_ph += state_edges['ph'][e_tail]
                    tot_cent += state_edges['c'][e_tail]
                    if state_edges['ph'][e_tail]!=0.0:
                        examined_edges.add(e_tail)

                if tot_ph == 0.0:
                    prob_ph = 0.0
                else:
                    prob_ph = e_ph / tot_ph

                if tot_cent == 0.0:
                    prob_cent=0.0
                    #prob_cent = 1.0 / num_nodes
                else:
                    prob_cent = e_cent / tot_cent
                likelihood_ph *= prob_ph
                likelihood_cent *= prob_cent
                tabu.append(e[1])


            # Compare likelihoods
            ml_pheromone[s][route] = likelihood_ph
            ml_centrality[s][route] = likelihood_cent
            error = (likelihood_cent - likelihood_ph)
            ml_error[s][route] = error
            error_list.append(error)

            #add influence to each edge
            # for edge in examined_edges:
            #     prob_influence[s][edge].append(likelihood_ph)

        # Probability check
        # print '{}: total probability, ph={}, cent={}'.format(s, sum(ml_pheromone[s].values()),
        #                                                          sum(ml_centrality[s].values()))
        # for e,p_inf in prob_influence[s].iteritems():
        #     print s, e, p_inf


        # Get ml paths
        mls = (ml_pheromone, ml_centrality)
        base_index = comparison_base  # which dict in tuple to use as base
        comp_index = 1 - base_index
        path_ph = max(mls[base_index][s].iteritems(), key=operator.itemgetter(1))[0]
        path_cent = max(mls[comp_index][s].iteritems(), key=operator.itemgetter(1))[0]

        # print 'P({},ph)={},  P({},c)={}'.format(path_ph,mls[base_index][s][path_ph],path_cent,mls[comp_index][s][path_cent])

        # print path_ph
        keys = []
        for p, ml in mls[base_index][s].iteritems():
            if ml == mls[base_index][s][path_ph]:
                keys.append(p)
                #print p,mls[base_index][s][path_ph]

        # Get rank error of paths
        error_list_rank = []
        error_list_prob = []

        # Strongest Paths
        tabu_p = [0]
        tabu_c = [0]
        unique = True
        for i in xrange(t.tsp.number_of_nodes() - 1):  # iterate through nodes
            # print unique
            # if unique == False:
            #     break
            step_e_p = {}
            step_e_c = {}
            for u, v in t.tsp.edges():
                # cent
                if u == tabu_c[-1] and (v not in tabu_c):
                    step_e_c[(u, v)] = state_edges['c'][(u, v)]
                # ph
                if u == tabu_p[-1] and (v not in tabu_p):
                    step_e_p[(u, v)] = state_edges['ph'][(u, v)]
                    # print (u,v),tabu_c, step_e_c, tabu_p, step_e_p

            # get maxes
            max_e_c = max(step_e_c.iteritems(), key=operator.itemgetter(1))[0]
            # print max_e_c
            c_keys = []
            for e_c, val in step_e_c.iteritems():
                if val == step_e_c[max_e_c]:
                    c_keys.append(e_c)
            if len(c_keys) > 1:
                unique = False
                tabu_c.append(max_e_c[1])

            else:
                tabu_c.append(max_e_c[1])

            max_e_p = max(step_e_p.iteritems(), key=operator.itemgetter(1))[0]
            p_keys = []
            for e_p, val in step_e_p.iteritems():
                if val == step_e_p[max_e_p]:
                    p_keys.append(e_p)
            if len(p_keys) > 1:
                unique = False
                tabu_p.append(max_e_p[1])

            else:
                tabu_p.append(max_e_p[1])

        # if unique == False:
        #     if s==0:
        #         path_temp=list(xrange(t.tsp.number_of_nodes()))
        #         e_path_ph = [tuple(path_temp[i:i + 2]) for i in xrange(0, len(path_temp), 1)]
        #         del e_path_ph[-1]
        #         e_path_cent = [tuple(path_temp[i:i + 2]) for i in xrange(0, len(path_temp), 1)]
        #         del e_path_ph[-1]
        #         shared = list(set(e_path_ph).intersection(e_path_cent))
        #         error_path = 1.0 - float(len(shared)) / len(e_path_cent)
        #     else:
        #         path_temp = list(xrange(t.tsp.number_of_nodes()))
        #         e_path_ph = [tuple(path_temp[i:i + 2]) for i in xrange(0, len(path_temp), 1)]
        #         del e_path_ph[-1]
        #         path_temp = path_temp[::-1]
        #         e_path_cent = [tuple(path_temp[i:i + 2]) for i in xrange(0, len(path_temp), 1)]
        #         del e_path_ph[-1]
        #         shared = list(set(e_path_ph).intersection(e_path_cent))
        #         error_path = 1.0 - float(len(shared)) / len(e_path_cent)
        #
        # else:
        #     # compare agreeance of paths
        #     e_path_ph = [tuple(tabu_p[i:i + 2]) for i in xrange(0, len(tabu_p), 1)]
        #     del e_path_ph[-1]
        #
        #     e_path_cent = [tuple(tabu_c[i:i + 2]) for i in xrange(0, len(tabu_c), 1)]
        #     del e_path_cent[-1]
        #
        #     shared = list(set(e_path_ph).intersection(e_path_cent))
        #     error_path = 1.0 - float(len(shared)) / len(e_path_cent)
        #     # print s, 'ph={}, cent={}, shared={}, percent difference={}'.format(e_path_ph, e_path_cent, shared,
        #     #                                                                    error_path)

        # compare agreeance of paths
        e_path_ph = [tuple(tabu_p[i:i + 2]) for i in xrange(0, len(tabu_p), 1)]
        del e_path_ph[-1]

        e_path_cent = [tuple(tabu_c[i:i + 2]) for i in xrange(0, len(tabu_c), 1)]
        del e_path_cent[-1]

        shared = list(set(e_path_ph).intersection(e_path_cent))
        error_path = 1.0 - float(len(shared)) / len(e_path_cent)
        # print s, 'ph={}, cent={}, shared={}, percent difference={}'.format(e_path_ph, e_path_cent, shared,
        #

        error_list_prob.append(error_path)

        # # compare agreeance of paths
        # path_ph_full=[0] + list(path_ph)
        # e_path_ph = [tuple(path_ph_full[i:i + 2]) for i in xrange(0, len(path_ph_full), 1)]
        # del e_path_ph[-1]
        #
        # path_cent_full = [0] + list(path_cent)
        # e_path_cent = [tuple(path_cent_full[i:i + 2]) for i in xrange(0, len(path_cent_full), 1)]
        # del e_path_cent[-1]
        #
        # shared = list(set(e_path_ph).intersection(e_path_cent))
        # error_path=1.0-float(len(shared)) / len(e_path_cent)
        # print s , 'ph={}, cent={}, shared={}, percent difference={}'.format(e_path_ph,e_path_cent,shared,error_path)
        # error_list_prob.append(error_path)

        # sored array of all centrality ml's
        ml_comp_array = np.array(mls[comp_index][s].values())
        ml_comp_array = np.sort(ml_comp_array)[::-1]
        for p in keys:  # go through paths with max pheromone ml
            rank = ml_comp_array.tolist().index(mls[comp_index][s][p])  # get rank of centrality ml
            error_list_rank.append(rank)
            # print s,p, 'centrality={}'.format(ml_centrality[s][p]), 'pheromone={}'.format(ml_pheromone[s][p])
            # error_list_prob.append(abs(mls[comp_index][s][p] - mls[base_index][s][p]) / mls[base_index][s][p])
        ml_paths[s] = keys
        average_rank_error[s] = np.mean(error_list_rank)
        average_prob_error[s] = np.mean(error_list_prob)

    # plot prob ratio and rank difference at 0-offset
    gen = xrange(len(average_rank_error))
    error = [average_rank_error[g] for g in gen]  # / np.math.factorial(tsp_object.n_cities + 1)

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Rank difference of most likely paths, average={}'.format(title, np.mean((error))))

    error = [average_prob_error[g] for g in gen]

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Prob difference of most likely paths, average={}'.format(title, np.mean((error))))

    # check prediction at different offsets
    off = []
    e_prob_off = []
    e_rank_off = []
    for offset in xrange(-15, 16):  # amount to offset centrality results
        e_prob_s = []
        e_rank_s = []
        for s, ml_ph in mls[base_index].iteritems():  # go over each state
            s_off = s + offset
            # Track errors at this state
            e_prob_path = []
            e_rank_path = []
            if s_off not in mls[comp_index]:  # offset doesn't exist
                continue
            else:
                # sorted array of all centrality ml's
                ml_c_array = np.array(mls[comp_index][s_off].values())
                ml_c_array = np.sort(ml_c_array)[::-1]
                for p in ml_paths[s]:  # go through paths with max pheromone ml
                    rank = ml_c_array.tolist().index(mls[comp_index][s_off][p])  # get rank of centrality ml
                    e_rank_path.append(rank)
                    e_prob_path.append(
                        abs(mls[comp_index][s_off][p] - mls[base_index][s][p]))# / mls[base_index][s][p])
            e_prob_s.append(np.mean(e_prob_path))
            e_rank_s.append(np.mean(e_rank_path))  # / np.math.factorial(tsp_object.n_cities + 1))
        e_prob_off.append(np.mean(e_prob_s))
        e_rank_off.append(np.mean(e_rank_s))
        off.append(offset)

    plt.figure()
    plt.scatter(off, e_rank_off)
    plt.title('{} Rank difference of most likely paths at different offsets'.format(title))

    plt.figure()
    plt.scatter(off, e_prob_off)
    plt.title('{} Prob difference of most likely paths at different offsets'.format(title))

    return e_rank_off, e_prob_off, average_prob_error, ml_error, ml_pheromone, ml_centrality


def ML_path_comparison(tsp_object, results_series, title,comparison_base=0):
    #comparison_base: 0=compare to pheromone, 1=compare to centrality
    # Compare the maximum likelihood paths between a set of results and pheromones
    tsp_series = tsp_object.history_s
    # Get mle(s) for pheromones
    ml_pheromone = {}
    ml_centrality = {}
    ml_error = {}
    prob_influence={}
    average_prob_error = {}
    average_rank_error = {}
    ml_paths = {}
    for s, tsp in tsp_series.iteritems():  # go through time steps
        ml_pheromone[s] = {}
        ml_centrality[s] = {}
        prob_influence[s] = dict.fromkeys(t.history_k[s].edges(), 0.0)
        ml_error[s] = {}
        error_list = []
        nodes = list(set(tsp.nodes()) - set([0]))
        for route in itertools.permutations(nodes):  # calc mle for each route
            r = [0] + list(route)
            likelihood_ph = 1.0
            likelihood_cent = 1.0

            # Go through each step in the solution and calc prob that it occurred
            e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
            del e_r[-1]
            tabu = [0]
            for e in e_r:  # selected step
                if tsp[e[0]][e[1]]['dist']==0.0:
                    e_h=tsp_object.explore
                else:
                    e_h=1.0/tsp[e[0]][e[1]]['dist']
                e_ph=np.power(tsp[e[0]][e[1]]['p'],tsp_object.alpha)*np.power(e_h,tsp_object.beta)  #tsp[e[0]][e[1]]['p']
                # e_ph = tsp[e[0]][e[1]]['p']
                e_cent = results_series['dist'][e][s]
                tot_ph = 0.0
                tot_cent = 0.0
                num_nodes = 0
                for tail in tsp.nodes():
                    if tail in tabu:
                        continue
                    num_nodes += 1
                    # get pheromones and centralities for available edges

                    if tsp[e[0]][tail]['dist'] == 0.0:
                        e_h = tsp_object.explore
                    else:
                        e_h = 1.0 / tsp[e[0]][tail]['dist']

                    tot_ph += np.power(tsp[e[0]][tail]['p'],tsp_object.alpha)*np.power(e_h,tsp_object.beta) #tsp[e[0]][tail]['p']
                    # tot_ph += tsp[e[0]][tail]['p']
                    tot_cent += results_series['dist'][(e[0], tail)][s]

                if tot_ph == 0.0:
                    prob_ph = 0.0
                else:
                    prob_ph = e_ph / tot_ph

                if tot_cent == 0.0:
                    prob_cent=0.0
                    #prob_cent = 1.0 / num_nodes
                else:
                    prob_cent = e_cent / tot_cent
                likelihood_ph *= prob_ph
                likelihood_cent *= prob_cent
                tabu.append(e[1])

            # Compare likelihoods
            ml_pheromone[s][route] = likelihood_ph
            ml_centrality[s][route] = likelihood_cent
            error = (likelihood_cent - likelihood_ph)
            ml_error[s][route] = error
            error_list.append(error)

        # #Probability check
        #print '{}: total probability, ph={}, cent={}'.format(s,sum(ml_pheromone[s].values()),sum(ml_centrality[s].values()))

        # Get ml paths
        mls=(ml_pheromone,ml_centrality)
        base_index=comparison_base #which dict in tuple to use as base
        comp_index=1-base_index
        path_ph = max(mls[base_index][s].iteritems(), key=operator.itemgetter(1))[0]
        # print path_ph
        keys = []
        for p, ml in mls[base_index][s].iteritems():
            if ml == mls[base_index][s][path_ph]:
                keys.append(p)

        # Get rank error of paths
        error_list_rank = []
        error_list_prob = []
        # sored array of all centrality ml's
        ml_comp_array = np.array(mls[comp_index][s].values())
        ml_comp_array = np.sort(ml_comp_array)[::-1]
        for p in keys:  # go through paths with max pheromone ml
            rank = ml_comp_array.tolist().index(mls[comp_index][s][p])  # get rank of centrality ml
            error_list_rank.append(rank)
            # print s,p, 'centrality={}'.format(ml_centrality[s][p]), 'pheromone={}'.format(ml_pheromone[s][p])
            error_list_prob.append(abs(mls[comp_index][s][p] - mls[base_index][s][p]) )# / mls[base_index][s][p])
        ml_paths[s] = keys
        average_rank_error[s] = np.mean(error_list_rank)
        average_prob_error[s] = np.mean(error_list_prob)

    # plot prob ratio and rank difference at 0-offset
    gen = xrange(len(average_rank_error))
    error = [average_rank_error[g] for g in gen]  # / np.math.factorial(tsp_object.n_cities + 1)

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Rank difference of most likely paths, average={}'.format(title, np.mean((error))))

    error = [average_prob_error[g] for g in gen]

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Prob difference of most likely paths, average={}'.format(title, np.mean((error))))

    # check prediction at different offsets
    off = []
    e_prob_off = []
    e_rank_off = []
    for offset in xrange(-15, 16):  # amount to offset centrality results
        e_prob_s = []
        e_rank_s = []
        for s, ml_ph in mls[base_index].iteritems():  # go over each state
            s_off = s + offset
            # Track errors at this state
            e_prob_path = []
            e_rank_path = []
            if s_off not in mls[comp_index]:  # offset doesn't exist
                continue
            else:
                # sorted array of all centrality ml's
                ml_c_array = np.array(mls[comp_index][s_off].values())
                ml_c_array = np.sort(ml_c_array)[::-1]
                for p in ml_paths[s]:  # go through paths with max pheromone ml
                    rank = ml_c_array.tolist().index(mls[comp_index][s_off][p])  # get rank of centrality ml
                    e_rank_path.append(rank)
                    e_prob_path.append(abs(mls[comp_index][s_off][p] - mls[base_index][s][p]) )# / mls[base_index][s][p])
            e_prob_s.append(np.mean(e_prob_path))
            e_rank_s.append(np.mean(e_rank_path))# / np.math.factorial(tsp_object.n_cities + 1))
        e_prob_off.append(np.mean(e_prob_s))
        e_rank_off.append(np.mean(e_rank_s))
        off.append(offset)

    plt.figure()
    plt.scatter(off, e_rank_off)
    plt.title('{} Rank difference of most likely paths at different offsets'.format(title))

    plt.figure()
    plt.scatter(off, e_prob_off)
    plt.title('{} Prob difference of most likely paths at different offsets'.format(title))

    return e_rank_off, e_prob_off, average_prob_error, ml_error, ml_pheromone, ml_centrality


def test(results):
    return results.keys()

def tsp_list_analysis(tsp_list):
    cent_l = []
    path_l = []
    for t in tsp_list:
        cent_r, path_r, cent_params, path_params = analysis_suite(t, interval=interval, offset_range=0)
        cent_l.append(cent_r)
        path_l.append(path_r)

    cent_o = pd.concat(cent_l)
    # cent_o=cent_o.fillna(0.0)
    # print cent_o['centrality']
    # print cent_params
    cent_data_columns = [col for col in cent_o.columns if col not in cent_params]
    # print cent_data_columns
    groupby_list=['centrality', 'layer', 's','tol']
    cent_o[cent_data_columns] = cent_o[cent_data_columns].apply(pd.to_numeric)
    cent_mean = cent_o.groupby(groupby_list, as_index=False)[cent_data_columns].mean()
    # print cent_mean.head()
    cent_error = cent_o.groupby(groupby_list, as_index=False)[cent_data_columns].agg(st.sem)

    # cent_error=cent_o.groupby(['centrality','layer','s'],as_index=False)[cent_data_columns].var()
    # cent_error[cent_data_columns]=cent_error[cent_data_columns].apply(lambda x: np.sqrt(x))
    # print cent_error.head()

    # print cent_error.head()
    # cent_o=cent_o.fillna(0.0)
    # print cent_o.index
    path_o = pd.concat(path_l)
    path_data_columns = [col for col in path_o.columns if col not in path_params]
    # print path_data_columns
    path_groupby_list=['centrality', 'offset', 's','tol']
    path_mean = path_o.groupby(path_groupby_list, as_index=False)[path_data_columns].mean()
    path_min = path_o.groupby(path_groupby_list, as_index=False)[path_data_columns].min()
    path_max = path_o.groupby(path_groupby_list, as_index=False)[path_data_columns].max()
    path_error = path_o.groupby(path_groupby_list, as_index=False)[path_data_columns].agg(st.sem)  # np.var
    path_mean['std'] = path_error[path_data_columns]  # .apply(lambda x: np.sqrt(x))
    # print path_mean.apply(lambda row: pd.Series(st.t.interval(0.95,iters-1,loc=row['Strongest Path Error'], scale=row['std'])),axis=1)
    path_mean['Min Error'], path_mean['Max Error'] = zip(
        *path_mean.apply(lambda row: st.t.interval(0.95, iters - 1, loc=row['Strongest Path Error'], scale=row['std']),
                         axis=1))
    path_mean['Min Error'].fillna(path_mean['Strongest Path Error'], inplace=True)
    path_mean['Max Error'].fillna(path_mean['Strongest Path Error'], inplace=True)
    path_mean['Min Error'][path_mean['Min Error'] < 0.0] = 0.0
    path_mean['Max Error'][path_mean['Max Error'] > 1.0] = 1.0

    return cent_o, cent_mean, cent_error, path_mean #Total dataframe, mean df, std df, path error df

num_cities = 5
steps=60
interval=1
tick_interval=5
rug=1.0
iters=5
save=False
dec_line_on=True
dc='r' #Decision Line Color
dw=2 #Decision Line Width
dl='Earliest Decision'
figsize=(10,6)
tsp_list=[]
scores=[]
for i in xrange(iters):
    tol=1.0
    tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities, explore=1.0, tolerance=tol, rug=rug)
    tsp_list.append(tsp)
    for r,s in zip(routes,xrange(len(routes))):
        scores.append([tol,s,r[1]])


t_d_contex=[] #Track when the first decision is made
for i in xrange(iters):
    tol=0.7
    tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities, explore=1.0, tolerance=tol,rug=rug)
    tsp_list.append(tsp)
    for r,s in zip(routes,xrange(len(routes))):
        scores.append([tol,s,r[1]])

    if tsp.d_change: #Context decision made
        t_d_contex.append(tsp.d_change[0]) #Time decision occured
    # else:
    #     t_d_contex.append(steps-1) #Decision not made, set to max time of convergence


#Score Dataframe
score_df=pd.DataFrame(scores,columns=['tol','t','s'])
score_mean = score_df.groupby(['tol','t'], as_index=False)['s'].mean()
score_error = score_df.groupby(['tol', 't'], as_index=False)['s'].agg(st.sem)

#Time of context decision
# dec_time=np.mean(t_d_contex)
dec_time=min(t_d_contex)-1
print 'Context Decision={}'.format(dec_time)

#Centrality Dataframe
cent_o,cent_mean,cent_error,path_mean=tsp_list_analysis(tsp_list)

# cent_l=[]
# path_l=[]
# for t in tsp_list:
#     cent_r,path_r,cent_params,path_params=analysis_suite(t,interval=interval,offset_range=0)
#     cent_l.append(cent_r)
#     path_l.append(path_r)
#
# cent_o=pd.concat(cent_l)
# # cent_o=cent_o.fillna(0.0)
# # print cent_o['centrality']
# # print cent_params
# cent_data_columns=[col for col in cent_o.columns if col not in cent_params]
# # print cent_data_columns
# cent_o[cent_data_columns]=cent_o[cent_data_columns].apply(pd.to_numeric)
# cent_mean=cent_o.groupby(['centrality','layer','s'],as_index=False)[cent_data_columns].mean()
# # print cent_mean.head()
# cent_error=cent_o.groupby(['centrality','layer','s'],as_index=False)[cent_data_columns].agg(st.sem)
#
# #calculate 95% confidence bound
# conf_stat=st.t.ppf(1.95/2.0,iters-1)
#
# # cent_error=cent_o.groupby(['centrality','layer','s'],as_index=False)[cent_data_columns].var()
# # cent_error[cent_data_columns]=cent_error[cent_data_columns].apply(lambda x: np.sqrt(x))
# # print cent_error.head()
#
# # print cent_error.head()
# # cent_o=cent_o.fillna(0.0)
# # print cent_o.index
# path_o=pd.concat(path_l)
# path_data_columns=[col for col in path_o.columns if col not in path_params]
# # print path_data_columns
#
# path_mean=path_o.groupby(['centrality','offset','s'],as_index=False)[path_data_columns].mean()
# path_min=path_o.groupby(['centrality','offset','s'],as_index=False)[path_data_columns].min()
# path_max=path_o.groupby(['centrality','offset','s'],as_index=False)[path_data_columns].max()
# path_error=path_o.groupby(['centrality','offset','s'],as_index=False)[path_data_columns].agg(st.sem) #np.var
# path_mean['std']=path_error[path_data_columns]#.apply(lambda x: np.sqrt(x))
# # print path_mean.apply(lambda row: pd.Series(st.t.interval(0.95,iters-1,loc=row['Strongest Path Error'], scale=row['std'])),axis=1)
# path_mean['Min Error'], path_mean['Max Error']=zip(*path_mean.apply(lambda row: st.t.interval(0.95,iters-1,loc=row['Strongest Path Error'], scale=row['std']),axis=1))
# path_mean['Min Error'].fillna(path_mean['Strongest Path Error'],inplace=True)
# path_mean['Max Error'].fillna(path_mean['Strongest Path Error'],inplace=True)
# path_mean['Min Error'][path_mean['Min Error']<0.0]=0.0
# path_mean['Max Error'][path_mean['Max Error']>1.0]=1.0

# path_mean['Min Error']=path_min[path_data_columns]
# path_mean['Max Error']=path_max[path_data_columns]
# print path_error.head()
# print 'Path Mean'
# print path_mean.head()
# print cent_o['centrality'].unique()

##Analysis parameters
layers=['a','k']
layer_label={'a':'A', 'k':'KD'}

ana_label={'out':'Out Degree', 'in':'In Degree', 'between':'Betweenness',
           'hits-a':'HITS-Authorities', 'hits-h':'HITS-Hubs'}

seg_labels=['Decision','Score','Prototyping','Context', 'TSP Paths']
sub_labels=['Start Non-Solution', 'Start Solution', 'TSP Solution', 'TSP Non-Solution']

save_dir='C:\Users\colinsh\Documents\DesignSim'

# calculate 95% confidence bound
conf_stat = st.t.ppf(1.95 / 2.0, iters - 1)


#Scores history
col='s'

#score
df1=score_mean.loc[(score_mean['tol']==1.0)][col].reset_index(drop=True)
df2=score_mean.loc[(score_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
df1=score_error.loc[(score_error['tol']==1.0)][col].reset_index(drop=True)
df2=score_error.loc[(score_error['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=score_mean['t'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(score_mean['t'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=score_mean['t'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(score_mean['t'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)
plt.axhline(y=num_cities, color='r', linewidth=2, label='Optimal = {}'.format(num_cities))

plt.ylim([num_cities-1.0,num_cities*2])

plt.ylabel('Prototype Score Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Designer\'s Solution Quality, {} City C+TSP'.format(num_cities))
plt.legend(['No Decision', 'Decision', 'Optimal = {}'.format(num_cities)], loc='best')

if save:
    plt.savefig('{}\score.pdf'.format(save_dir))

#Segment comparison parameters
w=1.0 #width
ec='none' #edgecolor
alpha=.5 #alpha
cl=['maroon','orangered','orange','b','g'] #colorlist

#HITS-K Knowledge Segment Comparison no Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-h') & (cent_mean['layer']=='k')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0 # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[2:]
color_list = cl[2:]#['orange','b','g']
plot_df.plot.bar(x='s',
                 y=['Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, color=color_list, edgecolor=ec, ax=a)
plt.ylabel('HITS-Hubs Centrality Sum')
plt.xlabel('Time')
plt.title('Knowledge Layer: Segment Comparison')
plt.legend(['Prototypes','Context', 'TSP Paths'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\hh_k_seg_comp_.pdf'.format(save_dir))

#HITS-A Action Segment Comparison no Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0  # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[1:]
color_list = cl[1:]#['orangered','orange','b','g']
plot_df.plot.bar(x='s',
                 y=['Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, color=color_list, edgecolor=ec, ax=a)
plt.ylabel('HITS-Authorities Centrality Sum')
plt.xlabel('Time')
plt.title('Action Layer: Segment Comparison')
plt.legend(['Scoring','Prototyping','Context Learning', 'TSP Paths Learning'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\ha_a_seg_comp.pdf'.format(save_dir))

#In Context Action Sum
main_plot=cent_mean.loc[(cent_mean['centrality']=='in') & (cent_mean['layer']=='a')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)]['Start Edges Sum'].reset_index(drop=True)
plot_df.columns=['Context Learning']
plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Context Learning'], ax=a, alpha=alpha,
                 color='b', figsize=figsize, width=w, edgecolor=ec)
plt.ylabel('In-Degree Sum')
plt.xlabel('Time')
plt.title('Action Layer: Context Learning')
# plt.legend(seg_labels, loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\in_a_c_sum.pdf'.format(save_dir))

#HITS-A Context A Sum
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)]['Start Edges Sum'].reset_index(drop=True)
plot_df.columns=['Learned Context']
plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Learned Context'], ax=a, alpha=alpha,
                 color='b', figsize=figsize, width=w, edgecolor=ec)
plt.ylabel('HITS-Authorities Centrality Sum')
plt.xlabel('Time')
plt.title('Action Layer: Context Learning')
# plt.legend(seg_labels, loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\ha_a_c_sum.pdf'.format(save_dir))

#HITS-A Context A Mean
col='Start Edges Mean'
ana='hits-a'
layer='a'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
plot_e=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)

comp=pd.concat([plot_df,plot_e],axis=1)
comp.columns=['v','std']
# print 'HA-A Context'
# print comp

plt.figure(figsize=figsize)
a=plt.gca()

comp.plot(x=cent_mean['s'].unique(), y=['v'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), comp['v']- conf_stat * comp['std'],
                 comp['v'] + conf_stat * comp['std'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plt.ylabel('HITS-Authorities Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Action Layer: Context Learning')
plt.legend(['Context Learning'], loc='best')

if save:
    plt.savefig('{}\ha_a_context_mean.pdf'.format(save_dir))

#Out Context K Mean
col='Start Edges Mean'
ana='out'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
plot_e=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
# print plot_e

comp=pd.concat([plot_df,plot_e],axis=1)
comp.columns=['v','std']
# print 'OUT-K Context'
# print comp

plt.figure(figsize=figsize)
a=plt.gca()

comp.plot(x=cent_mean['s'].unique(), y=['v'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), comp['v']- conf_stat * comp['std'],
                 comp['v'] + conf_stat * comp['std'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plt.ylabel('Out-Degree Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context')
plt.legend(['Context'], loc='best')

if save:
    plt.savefig('{}\out_k_context_mean.pdf'.format(save_dir))

#HITS-Hubs Context K Mean
col='Start Edges Mean'
ana='hits-h'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
plot_e=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
# print plot_e

comp=pd.concat([plot_df,plot_e],axis=1)
comp.columns=['v','std']
# print 'HH-K Context'
# print comp

plt.figure(figsize=figsize)
a=plt.gca()

comp.plot(x=cent_mean['s'].unique(), y=['v'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), comp['v']- conf_stat * comp['std'],
                 comp['v'] + conf_stat * comp['std'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context')
plt.legend(['Context'], loc='best')

if save:
    plt.savefig('{}\hh_k_context_mean.pdf'.format(save_dir))


#HITS-A Action Sum Context and TSP
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0  # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[3:]
# color_list = ['r','y','b','g','m']
plot_df.plot.bar(x='s',
                 y=['Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, edgecolor=ec, color=['b','g'], ax=a)
plt.ylabel('HITS-Authorities Centrality Sum')
plt.xlabel('Time')
plt.title('Action Layer: Context and TSP Paths Learning')
plt.legend(['Context Learning', 'TSP Paths Learning'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\ha_a_CT_sum_comp.pdf'.format(save_dir))


#HITS-A Action Mean Context and TSP
col1='Start Edges Mean'
col2='TSP Edges Mean'
ana='hits-a'
layer='a'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col1].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==1.0)][col2].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['Context', 'TSP Paths']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col1].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==1.0)][col2].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['Context', 'TSP Paths']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['Context'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Context'] - conf_stat * plot_e['Context'],
                 plot_df['Context'] + conf_stat * plot_e['Context'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['TSP Paths'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['TSP Paths'] - conf_stat * plot_e['TSP Paths'],
                 plot_df['TSP Paths'] + conf_stat * plot_e['TSP Paths'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

plt.ylabel('HITS-Authorities Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Action Layer: Context and TSP Paths Learning')
# plt.legend(seg_labels, loc='best')

if save:
    plt.savefig('{}\ha_a_CT_mean_comp.pdf'.format(save_dir))

#HITS-H Knowledge Mean Context and TSP
col1='Start Edges Mean'
col2='TSP Edges Mean'
ana='hits-h'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col1].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==1.0)][col2].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['Context', 'TSP Paths']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col1].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==1.0)][col2].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['Context', 'TSP Paths']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['Context'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Context'] - conf_stat * plot_e['Context'],
                 plot_df['Context'] + conf_stat * plot_e['Context'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['TSP Paths'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['TSP Paths'] - conf_stat * plot_e['TSP Paths'],
                 plot_df['TSP Paths'] + conf_stat * plot_e['TSP Paths'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

plt.ylabel('HITS-Hub Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context and TSP Paths')
# plt.legend(seg_labels, loc='best')

if save:
    plt.savefig('{}\hh_k_CT_mean_comp.pdf'.format(save_dir))


#HITS-A Knowledge Sum Context and TSP
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='k')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0  # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[3:]
# color_list = ['r','y','b','g','m']
plot_df.plot.bar(x='s',
                 y=['Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, edgecolor=ec, color=['b','g'], ax=a)
plt.ylabel('HITS-Authorities Centrality Sum')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context and TSP Paths')
plt.legend(['Context', 'TSP Paths'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\ha_k_CT_sum_comp.pdf'.format(save_dir))


#HITS-H Knowledge Sum Context and TSP
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-h') & (cent_mean['layer']=='k')]
plot_df=main_plot.loc[(cent_mean['tol']==1.0)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0  # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[3:]
# color_list = ['r','y','b','g','m']
plot_df.plot.bar(x='s',
                 y=['Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, edgecolor=ec, color=['b','g'], ax=a)
plt.ylabel('HITS-Hubs Centrality Sum')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context and TSP Paths')
plt.legend(['Context', 'TSP Paths'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\hh_k_CT_sum_comp.pdf'.format(save_dir))



#Out K Mean Context Sol, Context Non-Sol
col1='Start Edges Solution Mean'
col2='Start Edges Non-Solution Mean'
col_names=['Solution', 'Non-Solution']
ana='out'
layer='k'


#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col1].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==1.0)][col2].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=col_names

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col1].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==1.0)][col2].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=col_names

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
                 plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
                 plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

plt.ylabel('Out-Degree Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context, Solution and Non-Solution')
# plt.legend(seg_labels, loc='best')

if save:
    plt.savefig('{}\out_k_C_SNS_mean_comp.pdf'.format(save_dir))


#Out K Mean TSP Sol, Context Non-Sol
col1='TSP Edges Solution Mean'
col2='TSP Edges Non-Solution Mean'
col_names=['Solution', 'Non-Solution']
ana='out'
layer='k'


#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col1].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==1.0)][col2].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=col_names

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col1].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==1.0)][col2].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=col_names

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
                 plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
                 plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

plt.ylabel('Out-Degree Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: TSP Paths, Solution and Non-Solution')
# plt.legend(seg_labels, loc='best')

if save:
    plt.savefig('{}\out_k_TSP_SNS_mean_comp.pdf'.format(save_dir))


#HITS-H K Mean Context Sol, Context Non-Sol
col1='Start Edges Solution Mean'
col2='Start Edges Non-Solution Mean'
col_names=['Solution', 'Non-Solution']
ana='hits-h'
layer='k'


#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col1].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==1.0)][col2].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=col_names

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col1].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==1.0)][col2].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=col_names

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
                 plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
                 plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context, Solution and Non-Solution')
# plt.legend(seg_labels, loc='best')

if save:
    plt.savefig('{}\hh_k_C_SNS_mean_comp.pdf'.format(save_dir))


#HITS-Hubs K Mean TSP Sol, Context Non-Sol
col1='TSP Edges Solution Mean'
col2='TSP Edges Non-Solution Mean'
col_names=['Solution', 'Non-Solution']
ana='hits-h'
layer='k'


#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col1].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==1.0)][col2].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=col_names

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col1].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==1.0)][col2].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=col_names

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
                 plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
                 plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: TSP Paths, Solution and Non-Solution')
# plt.legend(seg_labels, loc='best')

if save:
    plt.savefig('{}\hh_k_TSP_SNS_mean_comp.pdf'.format(save_dir))


############################
############################
###Decision Comparisions####
############################
############################

#HITS-K Knowledge Segment Comparison w/ Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-h') & (cent_mean['layer']=='k')]
plot_df=main_plot.loc[(cent_mean['tol']==0.7)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0 # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[2:]
color_list = cl[2:]#['orange','b','g']
plot_df.plot.bar(x='s',
                 y=['Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, edgecolor=ec, color=color_list, ax=a)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Sum')
plt.xlabel('Time')
plt.title('Knowledge Layer: Segment Comparison with Decision')
plt.legend([dl, 'Prototypes','Context', 'TSP Paths'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\hh_k_seg_comp_dec.pdf'.format(save_dir))

#HITS-A Action Segment Comparison w/ Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
plot_df=main_plot.loc[(cent_mean['tol']==0.7)].reset_index(drop=True)
# plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()

cmap = plt.cm.hsv  # jet
range_bottom = 0.0  # 0.2
range_top = 0.625  # .85

plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))
color_list = cl#['firebrick','orangered','orange','b','g']
plot_df.plot.bar(x='s',
                 y=['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
                 stacked=True, width=w, edgecolor=ec, color=color_list, ax=a)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Authorities Centrality Sum')
plt.xlabel('Time')
plt.title('Action Layer: Segment Comparison with Decision')
plt.legend([dl, 'Decision-making','Scoring','Prototyping','Context Learning', 'TSP Paths Learning'], loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\ha_a_seg_comp_dec.pdf'.format(save_dir))

#In Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='in') & (cent_mean['layer']=='a')]
plot_df=main_plot.loc[(cent_mean['tol']==0.7)]['Decision'].reset_index(drop=True)
plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a, alpha=alpha,
                 color='r', figsize=figsize, width=w, edgecolor=ec)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('In-Degree Sum')
plt.xlabel('Time')
plt.title('Action Layer: Decision')
plt.legend(loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\in_a_dec.pdf'.format(save_dir))

# HITS-A Action Decision
main_plot = cent_mean.loc[(cent_mean['centrality'] == 'hits-a') & (cent_mean['layer'] == 'a')]
plot_df = main_plot.loc[(cent_mean['tol'] == 0.7)]['Decision'].reset_index(drop=True)
plot_df.columns = ['Decision']
plt.figure(figsize=figsize)
a = plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a, alpha=alpha,
                 color='r', figsize=figsize, width=w, edgecolor=ec)

if dec_line_on:
    plt.axvline(x=dec_time, color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Authorities Centrality Sum')
plt.xlabel('Time')
plt.title('Action Layer: Decision')
plt.legend(loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\ha_a_dec.pdf'.format(save_dir))

#Out Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='out') & (cent_mean['layer']=='k')]
plot_df=main_plot.loc[(cent_mean['tol']==0.7)]['Decision'].reset_index(drop=True)
plot_df.columns=['Decision']
plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a, alpha=alpha,
                 color='r', figsize=figsize, width=w, edgecolor=ec)
if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('Out-Degree Sum')
plt.xlabel('Time')
plt.title('Decision Layer: Decision')
plt.legend(loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\out_d_dec.pdf'.format(save_dir))

#Out Context Decision vs. No Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='out') & (cent_mean['layer']=='k')]
df1=main_plot.loc[(cent_mean['tol']==1.0)]['Start Edges Sum'].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)]['Start Edges Sum'].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']
plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['No Decision'], ax=a,
                 color='b', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a,
                 color='g', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('Out-Degree Sum')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context')
plt.legend(loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\out_k_context_comparison.pdf'.format(save_dir))


#Out Transition Decision vs. No Decision
main_plot=cent_mean.loc[(cent_mean['centrality']=='out') & (cent_mean['layer']=='k')]
df1=main_plot.loc[(cent_mean['tol']==1.0)]['Step Sum'].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)]['Step Sum'].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']
plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['No Decision'], ax=a,
                 color='b', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)
plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a,
                 color='g', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('Out-Degree Sum')
plt.xlabel('Time')
plt.title('Knowledge Layer: Prototyping')
plt.legend(loc='best')
plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
if save:
    plt.savefig('{}\out_k_proto_comparison_sum.pdf'.format(save_dir))

#Out Context Mean Decision vs. No Decision
col='Start Edges Mean'
ana='out'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('Out-Degree Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context')
plt.legend(loc='best')

if save:
    plt.savefig('{}\out_k_context_comparison_mean.pdf'.format(save_dir))


#HITS-H Context Mean Decision vs. No Decision
col='Start Edges Mean'
ana='hits-h'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context')
plt.legend(loc='best')

if save:
    plt.savefig('{}\hh_k_context_comparison_mean.pdf'.format(save_dir))


#Out TSP Mean Decision vs. No Decision
col='TSP Edges Mean'
ana='out'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('Out-Degree Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: TSP Paths')
plt.legend( loc='best')

if save:
    plt.savefig('{}\out_k_TSP_comparison_mean.pdf'.format(save_dir))

#HITS-Hubs TSP Mean Decision vs. No Decision
col='TSP Edges Mean'
ana='hits-h'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: TSP Paths')
plt.legend( loc='best')

if save:
    plt.savefig('{}\hh_k_TSP_comparison_mean.pdf'.format(save_dir))

#HITS-H Context Sol Mean Decision vs. No Decision
col='Start Edges Solution Mean'
ana='hits-h'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context, Solution')
plt.legend(loc='best')

if save:
    plt.savefig('{}\hh_k_CS_comparison_mean.pdf'.format(save_dir))

#HITS-H Context Non-Sol Mean Decision vs. No Decision
col='Start Edges Non-Solution Mean'
ana='hits-h'
layer='k'

#Centrality
main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
df1=main_plot.loc[(cent_mean['tol']==1.0)][col].reset_index(drop=True)
df2=main_plot.loc[(cent_mean['tol']==0.7)][col].reset_index(drop=True)
plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_df.columns=['No Decision', 'Decision']

#error
error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
df1=error_plot.loc[(error_plot['tol']==1.0)][col].reset_index(drop=True)
df2=error_plot.loc[(error_plot['tol']==0.7)][col].reset_index(drop=True)
plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
plot_e.columns=['No Decision', 'Decision']

plt.figure(figsize=figsize)
a=plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
                 color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
                 color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: Context, Non-Solution')
plt.legend( loc='best')

if save:
    plt.savefig('{}\hh_k_CNS_comparison_mean.pdf'.format(save_dir))

# HITS-H TSP Sol Mean Decision vs. No Decision
col = 'TSP Edges Solution Mean'
ana = 'hits-h'
layer = 'k'

# Centrality
main_plot = cent_mean.loc[(cent_mean['centrality'] == ana) & (cent_mean['layer'] == layer)]
df1 = main_plot.loc[(cent_mean['tol'] == 1.0)][col].reset_index(drop=True)
df2 = main_plot.loc[(cent_mean['tol'] == 0.7)][col].reset_index(drop=True)
plot_df = pd.concat([df1, df2], axis=1, ignore_index=False)
plot_df.columns = ['No Decision', 'Decision']

# error
error_plot = cent_error.loc[(cent_error['centrality'] == ana) & (cent_error['layer'] == layer)]
df1 = error_plot.loc[(error_plot['tol'] == 1.0)][col].reset_index(drop=True)
df2 = error_plot.loc[(error_plot['tol'] == 0.7)][col].reset_index(drop=True)
plot_e = pd.concat([df1, df2], axis=1, ignore_index=False)
plot_e.columns = ['No Decision', 'Decision']

plt.figure(figsize=figsize)
a = plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
             color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
             color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: TSP Paths, Solution')
plt.legend(loc='best')

if save:
    plt.savefig('{}\hh_k_TS_comparison_mean.pdf'.format(save_dir))

# HITS-H TSP N-Sol Mean Decision vs. No Decision
col = 'TSP Edges Non-Solution Mean'
ana = 'hits-h'
layer = 'k'

# Centrality
main_plot = cent_mean.loc[(cent_mean['centrality'] == ana) & (cent_mean['layer'] == layer)]
df1 = main_plot.loc[(cent_mean['tol'] == 1.0)][col].reset_index(drop=True)
df2 = main_plot.loc[(cent_mean['tol'] == 0.7)][col].reset_index(drop=True)
plot_df = pd.concat([df1, df2], axis=1, ignore_index=False)
plot_df.columns = ['No Decision', 'Decision']

# error
error_plot = cent_error.loc[(cent_error['centrality'] == ana) & (cent_error['layer'] == layer)]
df1 = error_plot.loc[(error_plot['tol'] == 1.0)][col].reset_index(drop=True)
df2 = error_plot.loc[(error_plot['tol'] == 0.7)][col].reset_index(drop=True)
plot_e = pd.concat([df1, df2], axis=1, ignore_index=False)
plot_e.columns = ['No Decision', 'Decision']

plt.figure(figsize=figsize)
a = plt.gca()
# plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
#                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
             color='b', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
                 plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
                 facecolor='b', alpha=.5,
                 linewidth=0.0)

plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
             color='g', figsize=figsize, linewidth=3.0)

plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
                 plot_df['Decision'] + conf_stat * plot_e['Decision'],
                 facecolor='g', alpha=.5,
                 linewidth=0.0)

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
plt.xlabel('Time')
plt.title('Knowledge Layer: TSP Paths, Non-Solution')
plt.legend( loc='best')

if save:
    plt.savefig('{}\hh_k_TNS_comparison_mean.pdf'.format(save_dir))

full_analysis=False #Every centrality metric for every layer
if full_analysis:
    for ana in cent_o['centrality'].unique():
        # print ana
        # path=path_mean.copy()
        # # path_e=path_error.copy()
        # plot_path = path.loc[path['centrality'] == ana]  # .groupby('offset').mean()
        # plot_offset = path_o.loc[path_o['centrality'] == ana]
        # # plot_e=path_e.loc[path_e['centrality'] == ana]
        # # for off in plot_path['offset'].unique():
        # plot_0 = plot_path.loc[plot_path['offset'] == 0]
        # print plot_0[['Min Error','Max Error']]
        # plot_0_e = plot_e.loc[plot_e['offset'] == 0]
        # plot_0.join(plot_0_e['std'])
        # print plot_0
        # ax=plot_0.plot(x='s',y='error',yerr='std' ,title='Analysis-{}: Path Error at 0 Offset'.format(ana))
        # plot_0.plot(x='s',y=['Strongest Path Error','Min Error','Max Error'],title='Analysis-{}: Path Error at 0 Offset'.format(ana))

        ### Plot error
        # plot_0.plot(x='s', y=['Strongest Path Error'], linewidth=3.0, title='Analysis-{}: Path Error at 0 Offset'.format(ana))
        # plt.fill_between(plot_0['s'].values,plot_0['Min Error'],plot_0['Max Error'],
        #                  facecolor='blue', linewidth=0.0, alpha=.5, label='Confidence Interval')
        ### Plot error

        # print plot_offset.head(10)
        # plot_offset.boxplot(column=['error'],by=['offset'])
        # plot_offset.boxplot(column=['error'], by=['offset'])#, kind='box', title='Analysis-{}: Error by Offset'.format(ana))
        # plot_off = plot_path.groupby('offset').mean()
        # plot_off_error = plot_path.groupby('offset').std()
        # # print plot_error
        # # print plot_path['error']
        #
        # plot_off.plot(y='error',yerr=plot_off_error['error'],title='Analysis-{}: Average Error by Offset'.format(ana))

        for l in layers:
            # print ana,l
            cent=cent_mean.copy()
            cent_e=cent_error.copy()
            plot_df=cent.loc[(cent['centrality']==ana) & (cent['layer']==l)]
            plot_e=cent_e.loc[(cent_e['centrality']==ana) & (cent_e['layer']==l)]
            # print plot_df.head()
            # print plot_e.head()
            # print ana, l
            # plot_df.fillna(0.0)
            # print plot_df[['Decision Mean','Score Mean']]
            cmap=plt.cm.hsv#jet
            range_bottom=0.0#0.2
            range_top=0.625#.85

            plot_list = ['Decision Mean', 'Score Mean', 'Step Mean', 'Start Edges Mean', 'TSP Edges Mean']
            color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))
            # color_list = ['r','y','b','g','m']
            plot_df.plot.bar(x='s',
                             y=['Decision Sum','Score Sum','Step Sum','Start Edges Sum','TSP Edges Sum'],
                             stacked=True,width=.75, color=color_list, edgecolor='None',figsize=figsize)
                             # title='Analysis-{}, Layer-{}: Segment Influence Sum'.format(ana,l))#,edgecolor='None')
            plt.ylabel('{} Centrality Sum, {} Layer'.format(ana_label[ana],layer_label[l]))
            plt.xlabel('Time')
            plt.legend(seg_labels,loc='best')
            plt.xticks(np.arange(0, steps + 1,float(tick_interval)),range(0, steps + 1, tick_interval))
            # print np.arange(0,steps+1,5.0)
            if save:
                plt.savefig('{}\{}_{}_seg_sum.pdf'.format(save_dir, ana, l))
            # If all score == 0 skip it

            # print color_list
            plot_df.plot(x='s',
                         y=plot_list,#'Score Mean',
                         color=color_list,
                         linewidth=3,figsize=figsize)
                         # title='Analysis-{}, Layer-{}: Segment Influence Mean'.format(ana, l))

            plt.ylabel('{} Centrality Mean, {} Layer'.format(ana_label[ana], layer_label[l]))
            plt.xlabel('Time')
            plt.legend(seg_labels, loc='best')
            # plt.xticks(np.arange(0, steps + 1, 5.0))

            if ana=='out' and l=='a':
                plt.ylim(top=1.05)

            plt.ylim(bottom=0.0)
            for v in plot_list:
                color=color_list[plot_list.index(v)]
                plt.fill_between(plot_df['s'].values, plot_df[v]-conf_stat*plot_e[v],
                                 plot_df[v]+conf_stat*plot_e[v],
                                 facecolor=color, alpha=.5,
                                 linewidth=0.0)

            if save:
                plt.savefig('{}\{}_{}_seg_mean.pdf'.format(save_dir, ana, l))
            # plt.fill_between(plot_df['s'].values, plot_df['Score Mean'] - conf_stat * plot_e['Score Mean'],
            #                  plot_df['Score Mean'] + conf_stat * plot_e['Score Mean'], facecolor='blue', alpha=.2)
            # plt.fill_between(plot_df['s'].values, plot_df['Step Mean'] - conf_stat * plot_e['Step Mean'],
            #                  plot_df['Step Mean'] + conf_stat * plot_e['Score Mean'], facecolor='blue', alpha=.2)
            # plt.fill_between(plot_df['s'].values, plot_df['Start Edges Mean'] - conf_stat * plot_e['Start Edges Mean'],
            #                  plot_df['Score Mean'] + conf_stat * plot_e['Score Mean'], facecolor='blue', alpha=.2)

            plot_list=['Start Edges Non-Solution Mean', 'Start Edges Solution Mean',
                       'TSP Edges Solution Mean', 'TSP Edges Non-Solution Mean']


            cmap = plt.cm.hsv
            edge_range_bottom = 0.0
            edge_range_top = 0.5

            # edge_range_bottom=4.0*(range_top-range_bottom)/5.0+range_bottom
            # edge_range_top=5.0*(range_top-range_bottom)/5.0+range_bottom
            color_list = cmap(np.linspace(edge_range_bottom, edge_range_top, len(plot_list)))
            # color_list = ['y', 'b', 'g', 'm']

            plot_df.plot(x='s',
                         y=plot_list,
                         color=color_list,
                         linewidth=3,
                         kind='line',figsize=figsize)
                         # title='Analysis-{}, Layer-{}: Edge Influence Mean'.format(ana, l))

            plt.ylabel('{} Centrality Mean, {} Layer'.format(ana_label[ana], layer_label[l]))
            plt.xlabel('Time')
            plt.legend(sub_labels, loc='best')
            # plt.xticks(np.arange(0, steps + 1, 5.0))
            plt.ylim(bottom=0.0)
            for v in plot_list:
                color = color_list[plot_list.index(v)]
                plt.fill_between(plot_df['s'].values, plot_df[v] - conf_stat * plot_e[v],
                                 plot_df[v] + conf_stat * plot_e[v],
                                 facecolor=color, alpha=.5,
                                 linewidth=0.0)

            if save:
                plt.savefig('{}\{}_{}_sub_mean.pdf'.format(save_dir, ana, l))

            plot_df.plot.bar(x='s',
                            y=['Start Edges Non-Solution Sum', 'Start Edges Solution Sum',
                            'TSP Edges Solution Sum', 'TSP Edges Non-Solution Sum'],
                            stacked=True,width=.75, color=color_list, edgecolor='None',figsize=figsize)
                            # title='Analysis-{}, Layer-{}: Edge Influence Sum'.format(ana, l))#,edgecolor='None')
            plt.ylabel('{} Centrality Sum, {} Layer'.format(ana_label[ana], layer_label[l]))
            plt.xlabel('Time')
            plt.legend(sub_labels, loc='best')
            plt.xticks(np.arange(0, steps + 1, float(tick_interval)),range(0, steps + 1, tick_interval))
            if save:
                plt.savefig('{}\{}_{}_sub_sum.pdf'.format(save_dir, ana, l))



#Path error no decision
path_0_offset = path_mean.loc[(path_mean['offset'] == 0) & (path_mean['tol'] == 1.0) ]

plt.figure(figsize=figsize)
ax=plt.gca()

cmap = plt.cm.hsv
edge_range_bottom = 0.0
edge_range_top = 0.75

color_list = cmap(np.linspace(edge_range_bottom, edge_range_top, len(cent_o['centrality'].unique())))[2:]
color_list=['b','g']

for ana,color in zip(['hits-h','out'],color_list):
    path_0_a = path_0_offset.loc[path_0_offset['centrality'] == ana]
    line,=ax.plot(path_0_a['s'].values, path_0_a['Strongest Path Error'], linewidth=3.0, color=color, label=ana)
    ax.fill_between(path_0_a['s'].values, path_0_a['Min Error'], path_0_a['Max Error'],
                     facecolor=color, linewidth=0.0, alpha=.5) #line.get_color()
plt.ylabel('Strongest Path Edge Error (with 95% CI)')
plt.xlabel('Time')
plt.title('Strongest Path: Pheromone and Centrality Comparison')
plt.legend(['HITS-Hub','Out Degree'],loc='best')
if save:
    plt.savefig('{}\error.pdf'.format(save_dir))


#Path error with Decision
path_0_offset = path_mean.loc[(path_mean['offset'] == 0) & (path_mean['tol'] == 0.7) ]

plt.figure(figsize=figsize)
ax=plt.gca()

cmap = plt.cm.hsv
edge_range_bottom = 0.0
edge_range_top = 0.75

color_list = cmap(np.linspace(edge_range_bottom, edge_range_top, len(cent_o['centrality'].unique())))[2:]
color_list=['b','g']

for ana,color in zip(['hits-h','out'],color_list):
    path_0_a = path_0_offset.loc[path_0_offset['centrality'] == ana]
    line,=ax.plot(path_0_a['s'].values, path_0_a['Strongest Path Error'], linewidth=3.0, color=color, label=ana)
    ax.fill_between(path_0_a['s'].values, path_0_a['Min Error'], path_0_a['Max Error'],
                     facecolor=color, linewidth=0.0, alpha=.5) #line.get_color()

if dec_line_on:
    plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)

plt.ylabel('Strongest Path Edge Error (with 95% CI)')
plt.xlabel('Time')
plt.title('Strongest Path: Pheromone and Centrality Comparison with Decision')
plt.legend(['HITS-Hub','Out Degree',dl],loc='best')
if save:
    plt.savefig('{}\error_d.pdf'.format(save_dir))



#Behavior analysis
# print t.history_k[len(t.history_k)-1]
# A=np.array(nx.adjacency_matrix(t.history_k[len(t.history_k)-1]).todense())
# A_t=A.transpose()
# w,v=np.linalg.eig(np.dot(A,A_t))
# w_m=max(w)
# for i_v in xrange(len(w)):
#     if w[i_v]==w_m:
#         print w[i_v], np.sort(v[i_v])[::-1]

# KAD
# kad_results = KAD_analysis(t)
# macro,edge, proto=influence_chart(t,kad_results)
# # print kad_results['hits-h'] is None
# # for time in kad_results['hits-h'].keys():
# #     print test(kad_results['hits-h'][time])
#
# # for i, nl in t.last['route'].iteritems():
# #     for n in nl:
# #         print i, n, kad_results['hits-a'][len(kad_results['hits-a'])-1][n]
# e_rank_off, e_prob_off, ave_prob_error, ml_error, ml_ph, ml_cent = ML_path_comparison_KAD(t, kad_results['out'],
#                                                                                           title='out', comparison_base=0)
#
# e_rank_off, e_prob_off, ave_prob_error, ml_error, ml_ph, ml_cent = ML_path_comparison_KAD(t, kad_results['hits-h'],
#                                                                                           title='hits-h',
#                                                                                           comparison_base=0)

# K Analysis
# k_results = KD_analysis(t)
# # plot_results(k_results, KAD_type='KD',results_to_plot='hits-a')
# e_rank_off, e_prob_off, ave_prob_error, ml_error, ml_ph, ml_cent = ML_path_comparison(t, k_results['hits-h'], title='K',
#                                                                                       comparison_base=0)

# A Analysis
# a_results=A_analysis(t)
# # print a_results
# plot_results(a_results, KAD_type='A', results_to_plot='out',rest_on=True,start_on=True)

# Decision ID
# bc,cc,odc,maxes=decision_ID(t,routes)
# print 'bc={}, cc={}, obc={}'.format(bc,cc,odc)
# for m in maxes:
#     print m,maxes[m]

# age_dict = cent_rank_by_time(t)
# age = []
# rank = []
# for a, ranks in age_dict.iteritems():
#     # print a, age_dict[a]
#     age.append(a)
#     rank.append(np.mean(ranks))
# plt.figure()
# plt.scatter(age, rank)
# plt.title('Rank by Age, A')

# print len(routes),routes
# print len(t.history_s.keys())
# print t.d
# t.problem_difficulty(t.history_d[0])


# print \num cities:\, n_cities

# for i in range(0,len(t.d)-1):
#     print 'delta t={}'.format(time[i+1]-time[i])
#     t.problem_difficulty(t.d[:i+1])


# age_dict = cent_rank_by_time_KAD(t)
# age = []
# rank = []
# for a, ranks in age_dict.iteritems():
#     # print a, age_dict[a]
#     age.append(a)
#     rank.append(np.mean(ranks))
# plt.figure()
# plt.scatter(age, rank)
# plt.title('Rank by Age, KAD')

# print len(routes),routes
# print len(t.history_s.keys())
# print t.d
# t.problem_difficulty(t.history_d[0])

# time = [0]
# time.extend(t.d_change)

# s = [i[1] for i in routes]
# gen = xrange(len(s))
# plt.figure()
# plt.scatter(gen, s)
# _ = plt.ylim([0.0, max(s) + 1.0])

plt.show()






# print results['hits-a']
# average_prob_error,ml_error,ml_pheromone,ml_centrality=ML_path_comparison(t,k_results['hits-h'])
# gen=xrange(len(average_prob_error))
# error=[average_prob_error[g]/np.math.factorial(num_cities+1) for g in gen]
# # for g in gen:
# #     print g, average_prob_error[g]
# # for g in gen:
# #     for r in ml_centrality[g]:
# #         print g,r, 'ph={},cent={}'.format(ml_pheromone[g][r],ml_centrality[g][r])
#
#
# plt.figure()
# plt.scatter(gen,error)
# plt.title('Rank difference of most likely paths, average={}'.format(np.mean((error))))


# for n,d in t.k.nodes(data=True):
#     print n,d
#
# for u,v,d in t.k.edges(data=True):
#     print (u,v),'{}:{},'.format(u,t.k.node[u]),'{}:{},'.format(v,t.k.node[v]), 'data={}'.format(d)

# KD,A=project_KD_A(t.k)
# print 'kad'
# for u,v,d in t.k.edges(data=True):
#     print (u,v),'{}:{},'.format(u,t.k.node[u]),'{}:{},'.format(v,t.k.node[v]), 'data={}'.format(d)
#
# print 'kd'
# for u,v,d in KD.edges(data=True):
#     print (u,v),'{}:{},'.format(u,KD.node[u]),'{}:{},'.format(v,KD.node[v]), 'data={}'.format(d)
#
# print 'a'
# for u,v,d in A.edges(data=True):
#     print (u,v),'{}:{},'.format(u,A.node[u]),'{}:{},'.format(v,A.node[v]), 'data={}'.format(d)

# print 'last',t.last
# print 'last action', t.last_action