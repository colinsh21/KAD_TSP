import networkx as nx
import numpy as np

import copy

import itertools

from bisect import bisect

"""
v1: Decide to not a path if the probability it gets used is lower than some tolerance. (test, with difference)
TO DO:
1) Track centrality of decided paths vs. undecided paths
"""

class TSP_2diss(object):
    def __init__(self, start=0, diss_old=.2, diss_new=.2, tolerance=.2, alpha=1.0, beta=1.0, explore=1.0, n_cities=4, force=None):
        # inputs
        self.n_cities = n_cities
        self.diss_old = diss_old
        self.diss_new = diss_new
        self.explore = explore
        self.tolerance = tolerance
        self.force = force
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.step = 0

        # setup
        self.history_d = {}  # holds decisions about node order
        self.history_ed = {}    #holds decisions about edges
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

        # d is list of edges that have been eliminated
        self.d = [self.start]
        self.d_edges=[] #list of eliminated edges
        self.history_ed[self.step] = self.d_edges

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
                    tsp.add_edge(i, j, dist=float(abs(i - j)))

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
        tabu=[int(self.start)]

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
                        h_l.append(1.0/self.explore)  # 10.0
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

            for n in n_l:  # n_l is the list of possible nodes to be selected
                # if n not in tabu: #only non tabu
                if n != tabu[-1]:  # no self-edge
                    e = (tabu[-1], n)
                    if (e[0], e[1]) not in self.tsp.edges():  # self.last_visited.keys():
                        e = (e[1], e[0])

                    if e not in self.d_edges:
                        # self.k.add_edge(self.last['p'][e][-1], n_label, step=self.step, t=2) #edges from pheromones
                        # if e[0] != self.start:
                        self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)
                    else:
                        last_index = len(self.last['dec'][0])
                        last = self.last['dec'][0][last_index - 1]
                        # last = self.last['dec'][0][e_i + 1]
                        self.k.add_edge(last, a_label, step=self.step, t=1)

            for e_i in xrange(len(tabu) - 1):  # iterate through route selections
                # add constraining, e_i is "edge selection i"
                # last = self.last['route'][e_i][-1]
                # self.k.add_edge(last, a_label, step=self.step, t=1)
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
        p_old = float(self.diss_old)
        p_new = float(self.diss_new)
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
            g_t[u][v]['p'] = float(g[u][v]['p']) * (1.0 - p_old)

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
            g_t[e[0]][e[1]]['p'] = float(g_t[e[0]][e[1]]['p']) + t_update * p_new

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
        self.history_ed[self.step] = self.d_edges
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
                    h_l.append(1.0/self.explore)
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

        # print tolerance, self.step

        #How to track which edges are defined?

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
                if (n not in self.d) and (n != dec_node) and (g_t[dec_node][n]['p']!=0.0):
                    n_l.append(n)
                    p_l.append(g[dec_node][n]['p'])
                    # print dec_node,n, n!=dec_node

                    if g[dec_node][n]['dist'] == 0.0:  # only look at explored nodes
                        # h_l.append(0.0)
                        h_l.append(1.0/self.explore)
                        # print dec_node,n
                        # n_l.append(n)
                        # p_l.append(g[dec_node][n]['p'])

                    else:
                        # h_l.append(1.0)
                        h_l.append(1.0 / g[dec_node][n]['dist'])

            ph_l = np.power(p_l, self.alpha) * np.power(h_l, 0.0)  # only use pheromone preference
            if sum(ph_l) == 0.0:
                perc_l = [1.0 / len(ph_l)] * len(ph_l)
            else:
                perc_l = [float(i) / sum(ph_l) for i in ph_l]
            l = list(perc_l)
            # print perc_l
            m_1 = l.pop(l.index(max(l)))
            # if l:
            #     print n_l,l,m_1,m_1-min(l)


            if not l: # only one option
                # print 'decision'
                dec_index = perc_l.index(max(perc_l))  # decision index
                node = n_l[dec_index]
                #self.d_edges.append((dec_node, node))
                self.d.append(node)  # add node to locked-in edges
                # print 'decision',self.step,dec_node,node,self.d_change

                # # Eliminate other edge options
                # # skip=[int(self.start),dec_node]
                # for n in g.nodes():
                #     if n not in self.d:
                #         if (dec_node, n) in g_t.edges():
                #             g_t[dec_node][n]['p'] = 0.0  # now prob of taking that edges is 0
                #
                # # Add action for decision
                # a_label = self.k.number_of_nodes() + 1
                # self.k.add_node(a_label, layer='a', label='dec', d=list(self.d), step=self.step)
                # self.last_action['dec'][0].append(a_label)
                #
                # # add edges from pheromone knowledge to action
                # for n in n_l:
                #     e = (dec_node, n)
                #     self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)
                #
                # # add edges from decision knowledge to action
                # self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)
                #
                # # update k for decision from action
                # n_label = self.k.number_of_nodes() + 1
                # self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
                # self.k.add_edge(self.last['dec'][0][-1], n_label, step=self.step, t=3)
                # self.k.add_edge(a_label, n_label, step=self.step, t=3)
                # self.last['dec'][0].append(n_label)
                # self.d_change.append(self.step)

            elif (m_1 - min(l)) >= tolerance:  # prob gap is larger than tolerance
            #elif (min(l)) < tolerance:  # prob gap is larger than tolerance
                dec_index = perc_l.index(min(perc_l))  # decision index
                # print n_l,dec_index
                node = n_l[dec_index]
                self.d_edges.append((dec_node,node))  # add node to decisions
                # print dec_node,self.d
                # print 'decision',self.step,dec_node,node,self.d_change
                # print 'decision, {}'.format(m_1 - min(l)), self.step, (dec_node, node)

                # Eliminate edge option
                g_t[dec_node][node]['p'] = 0.0  # now prob of taking that edges is 0

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


class TSP_1diss(object):
    def __init__(self, start=0, dissipation=.2, tolerance=.2, criteria='diff', alpha=1.0, beta=1.0, explore=1.0,
                 n_cities=4, force=None, rug=1.0, run_ID=0, label='SBD'):
        # method
        self.method='conv'

        #ID
        self.run_ID=run_ID
        self.label=label


        # inputs
        self.n_cities = n_cities
        self.dissipation = dissipation
        self.explore = explore
        self.tolerance = tolerance
        self.criteria=criteria
        self.force = force
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.rug=rug #ruggedness of landscape (exponentially scales distances)
        self.step = 0

        # setup
        self.history_d = {}  # holds decisions about node order
        self.history_ed = {}    #holds decisions about edges
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

        # d is list of edges that have been eliminated
        self.d = [self.start]
        self.d_edges=[] #list of eliminated edges
        self.history_ed[self.step] = copy.deepcopy(self.d_edges)

        # self.d=[]
        self.d_change = []
        self.history_d[self.step] = copy.deepcopy(self.d)
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
        tabu=[int(self.start)]

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
                        h_l.append(1.0/self.explore)  # 10.0
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

            for n in n_l:  # n_l is the list of possible nodes to be selected
                # if n not in tabu: #only non tabu
                if n != tabu[-1]:  # no self-edge
                    e = (tabu[-1], n)
                    if (e[0], e[1]) not in self.tsp.edges():  # self.last_visited.keys():
                        e = (e[1], e[0])

                    if e not in self.d_edges:
                        # self.k.add_edge(self.last['p'][e][-1], n_label, step=self.step, t=2) #edges from pheromones
                        # if e[0] != self.start:
                        self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)
                    else:
                        last_index = len(self.last['dec'][0])
                        last = self.last['dec'][0][last_index - 1]
                        # last = self.last['dec'][0][e_i + 1]
                        self.k.add_edge(last, a_label, step=self.step, t=1)

            for e_i in xrange(len(tabu) - 1):  # iterate through route selections
                # add constraining, e_i is "edge selection i"
                # last = self.last['route'][e_i][-1]
                # self.k.add_edge(last, a_label, step=self.step, t=1)
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
        self.history_ed[self.step] = copy.deepcopy(self.d_edges)
        self.history_d[self.step] = copy.deepcopy(self.d)
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
                    h_l.append(1.0/self.explore)
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

        # print tolerance, self.step

        #How to track which edges are defined?

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
                if (n not in self.d) and (n != dec_node) and (g_t[dec_node][n]['p']!=0.0):
                    n_l.append(n)
                    p_l.append(g[dec_node][n]['p'])
                    # print dec_node,n, n!=dec_node

                    if g[dec_node][n]['dist'] == 0.0:  # only look at explored nodes
                        # h_l.append(0.0)
                        h_l.append(1.0/self.explore)
                        # print dec_node,n
                        # n_l.append(n)
                        # p_l.append(g[dec_node][n]['p'])

                    else:
                        # h_l.append(1.0)
                        h_l.append(1.0 / g[dec_node][n]['dist'])

            ph_l = np.power(p_l, self.alpha) * np.power(h_l, 0.0)  # only use pheromone preference
            if sum(ph_l) == 0.0:
                perc_l = [1.0 / len(ph_l)] * len(ph_l)
            else:
                perc_l = [float(i) / sum(ph_l) for i in ph_l]
            l = list(perc_l)
            # print perc_l
            m_1 = l.pop(l.index(max(l)))
            # if l:
            #     print n_l,l,m_1,m_1-min(l)

            if self.criteria=='diff':
                if l:
                    dec_value=(m_1 - min(l))
            elif self.criteria=='abs':
                if l:
                    dec_value=1.0-min(l)


            if not l: # only one option
                # print 'decision'
                dec_index = perc_l.index(max(perc_l))  # decision index
                node = n_l[dec_index]
                #self.d_edges.append((dec_node, node))
                self.d.append(node)  # add node to locked-in edges
                # print 'decision',self.step,dec_node,node,self.d_change

                # # Eliminate other edge options
                # # skip=[int(self.start),dec_node]
                # for n in g.nodes():
                #     if n not in self.d:
                #         if (dec_node, n) in g_t.edges():
                #             g_t[dec_node][n]['p'] = 0.0  # now prob of taking that edges is 0
                #
                # # Add action for decision
                # a_label = self.k.number_of_nodes() + 1
                # self.k.add_node(a_label, layer='a', label='dec', d=list(self.d), step=self.step)
                # self.last_action['dec'][0].append(a_label)
                #
                # # add edges from pheromone knowledge to action
                # for n in n_l:
                #     e = (dec_node, n)
                #     self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)
                #
                # # add edges from decision knowledge to action
                # self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)
                #
                # # update k for decision from action
                # n_label = self.k.number_of_nodes() + 1
                # self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
                # self.k.add_edge(self.last['dec'][0][-1], n_label, step=self.step, t=3)
                # self.k.add_edge(a_label, n_label, step=self.step, t=3)
                # self.last['dec'][0].append(n_label)
                # self.d_change.append(self.step)

            elif dec_value >= tolerance:  # prob gap is larger than tolerance
            #elif (min(l)) < tolerance:  # prob gap is larger than tolerance
                dec_index = perc_l.index(min(perc_l))  # decision index
                # print n_l,dec_index
                node = n_l[dec_index]
                self.d_edges.append((dec_node,node))  # add node to decisions
                # print dec_node,self.d
                # print 'decision',self.step,dec_node,node,self.d_change
                # print 'decision, {}'.format(m_1 - min(l)), self.step, (dec_node, node)

                # Eliminate edge option
                g_t[dec_node][node]['p'] = 0.0  # now prob of taking that edges is 0

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