import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import random
import itertools
import operator
import timeit
from bisect import bisect


class TSP(object):
    def __init__(self, start=1, dissipation=.2, tolerance=.2, alpha=1.0, beta=1.0, explore=1.0, n_cities=4, force=None):
        # inputs
        self.n_cities = n_cities
        self.dissipation = dissipation
        self.explore = explore
        self.tolerance = tolerance
        self.force = force
        self.alpha = alpha
        self.beta = beta
        self.start = start
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

        self.last['score'][0] = []
        self.last['dec'][0] = []  # zero is placeholder

        #create last_action
        self.last_action=copy.deepcopy(self.last)

        self.d = [int(self.start)]
        # self.d=[]
        self.d_change = []
        self.history_d[self.step] = list(self.d)
        self.state = self.init_graph(self.tsp)
        self.history_s[self.step] = self.state.copy()
        self.k = self.init_k(self.state)
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
        n_label = k.number_of_nodes() + 1
        k.add_node(n_label, label='decision', layer='d', d=list(self.d), step=self.step)
        self.last['dec'][0].append(n_label)

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

            #Combine pheromones and heuristic
            c_l = np.power(p_l, self.alpha) * np.power(h_l, self.beta)

            #Select next step
            n_index = self.make_decision(c_l)
            new_n = n_l[n_index]

            # Add action node for route
            a_label = self.k.number_of_nodes() + 1
            self.k.add_node(a_label, layer='a', label='selection', i=i, e=(tabu[-1], new_n), step=self.step)
            self.last_action['route'][i].append(a_label)

            # Add edges from knowledge to action
            for e_i in xrange(len(tabu) - 1):  # iterate through route selections
                # add constraining, e_i is "edge selection i"
                last = self.last['route'][e_i][-1]
                self.k.add_edge(last, a_label, step=self.step, t=1)

            # if influenced by decision, edge from decision not other edge info
            if (dec_point or (len(n_l) == 1 and len(self.d) == self.tsp.number_of_nodes())):
                # only edge from decision
                self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=1)

            else: # add edges from distances and pheromones
                for n in n_l:  #n_l is the list of possible nodes to be selected
                    # if n not in tabu: #only non tabu
                    if n != tabu[-1]:  # no self-edge
                        e = (tabu[-1], n)
                        if (e[0], e[1]) not in self.tsp.edges():  # self.last_visited.keys():
                            e = (e[1], e[0])

                        #self.k.add_edge(self.last['p'][e][-1], n_label, step=self.step, t=2) #edges from pheromones
                        #if e[0] != self.start:
                        self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)
            # Action addition complete

            # Add new knowledge and edge from actions
            n_label = self.k.number_of_nodes() + 1

            # Node label is edge number in solution, e is edge added
            self.k.add_node(n_label, layer='k', label='selection', i=i, e=(tabu[-1], new_n), step=self.step)
            self.last['route'][i].append(n_label)
            self.k.add_edge(a_label,n_label, step=self.step) #edge from action to new k

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
                a_label = self.k.number_of_nodes() + 1
                self.k.add_node(a_label, layer='a',label='learn',e=(e[0], e[1]), dist=float(dist), step=self.step)
                self.k.add_edge(self.last['route'][sel_index][-1], a_label, step=self.step, t=1)
                self.last_action['dist'][(e[0], e[1])].append(a_label)

                # update k for changed distance from action
                n_label = self.k.number_of_nodes() + 1
                last_n = self.last['dist'][(e[0], e[1])][-1]
                self.k.add_node(n_label, layer='k', label='learn',e=(e[0], e[1]), dist=float(dist), step=self.step)
                self.k.add_edge(last_n, n_label, step=self.step, t=2) #edge from refinement
                self.k.add_edge(a_label,n_label, step=self.step) #edge from action

                self.last['dist'][(e[0], e[1])].append(n_label)

                # Update edge history
                new_edges.append(e)

        self.e_change[self.step] = new_edges
        old_edges = list(self.history_e[self.step - 1])
        all_edges = old_edges + new_edges
        self.history_e[self.step] = all_edges

        # Add action for score
        a_label = self.k.number_of_nodes()+1
        self.k.add_node(a_label, layer='a', label='score',score=score,step=self.step)
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
        # Action added

        # add k for score
        n_label = self.k.number_of_nodes() + 1
        self.k.add_node(n_label, layer='k', label='score', score=score, step=self.step)
        self.k.add_edge(a_label, n_label, step=self.step) # add edge from action to knowledge

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

        #Update pheromone
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
        a_label=self.k.number_of_nodes()+1
        self.k.add_node(a_label, layer='a', label='dec', d=list(self.d), step=self.step)
        self.last_action['dec'][0].append(a_label)

        #add edges from pheromone knowledge to action
        for n in n_l:
            e = (dec_node, n)
            #self.k.add_edge(self.last['p'][e][-1], n_label, step=self.step, t=2)
            self.k.add_edge(self.last['dist'][e][-1], a_label, step=self.step, t=2)

        #add edges from decision knowledge to action
        self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)


        # update k for decision from action
        n_label = self.k.number_of_nodes() + 1
        self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
        self.k.add_edge(self.last['dec'][0][-1], n_label, step=self.step, t=3)
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

                #Add action for decision
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
                self.k.add_edge(self.last['dec'][0][-1], a_label, step=self.step, t=3)

                # update k for decision from action
                n_label = self.k.number_of_nodes() + 1
                self.k.add_node(n_label, layer='d', label='dec', d=list(self.d), step=self.step)
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


def run_ACO(steps=1000,cities=5,tolerance=0.7,alpha=1.0,beta=1.0,dissipation=0.2,explore=1.0,force=None):

    t=TSP(start=0,tolerance=tolerance,alpha=alpha,beta=beta,dissipation=dissipation,explore=explore,n_cities=cities,force=force)


    routes = []
    converge = 0
    decided=0
    for i in xrange(steps):
        r,s=t.walk()
        routes.append((r,s))
        t.state=t.update_edges(r,s)

        #check convergence

        if len(routes)>=2:
            if routes[-1]==routes[-2]:
                #print converge
                converge+=1
            else:
                converge=0

        if len(t.d)==(cities+1) and decided==0:
            converge=10
            decided=1

        if converge>=20:
            break

        #if len(t.d)==t.state.number_of_nodes():
            #break
        #print t.edges(data=True)

    #print routes
    #print t.state.edges(data=True)
    print t.d
    #print t.d_change
    return t, routes

def KAD_analysis(t):
    results = {}

    analysis_type=['out','close','between','katz','eigen','hits-a','hits-h']
    for a_type in analysis_type:
        results[a_type]={}

    for s,kad in t.history_k.iteritems():
        if s==0:
            results['hits-h'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['hits-a'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['out'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['close'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['between'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            results['katz'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)
            #results['eigen'][s] = dict.fromkeys(t.history_k[s].nodes(), 0.0)

        if s!=0:
            (h, a) = nx.hits(kad, max_iter=10000, normalized=True)  # (hubs,authorities)
            results['hits-h'][s] = h
            results['hits-a'][s] = a
            results['out'][s] = nx.out_degree_centrality(kad)
            results['close'][s] = nx.closeness_centrality(kad,normalized=True)
            results['between'][s] = nx.betweenness_centrality(kad,normalized=True)
            results['katz'][s] = nx.katz_centrality(kad.reverse(), normalized=True)
            #results['eigen'][s] = nx.eigenvector_centrality(kad.reverse(),max_iter=100000)

    return results


def KD_analysis(t):
    ### Test Results ###
    results={}
    results['out']={}

    for s,kad in t.history_k.iteritems():
        kd, action = project_KD_A(kad)

        c=nx.out_degree_centrality(kd)
        #c=nx.closeness_centrality(kd,normalized=True)
        #c=nx.betweenness_centrality(kd,normalized=True)
        for k in t.last:
            results['out'].setdefault(k,{})
            for e in t.last[k]:
                n_list=t.last[k][e]
                tc=0.0
                for n in n_list:
                    if n in c:
                        tc+=c[n]
                    #else:
                        #continue

                add=results['out'][k].setdefault(e,[])
                add.append(tc)

    results['hits-a']={}
    results['hits-h']={}
    for s,kad in t.history_k.iteritems():
        kd,action=project_KD_A(kad)

        if s != 0:
            (h,a)=nx.hits(kd,max_iter=10000,normalized=True) #(hubs,authorities)
        else:
            h = dict.fromkeys(t.history_k[s].nodes(),0.0)
            a = dict.fromkeys(t.history_k[s].nodes(), 0.0)
        #c=nx.betweenness_centrality(kd,normalized=True)
        for k in t.last:
            results['hits-a'].setdefault(k,{})
            results['hits-h'].setdefault(k,{})
            for e in t.last[k]:
                n_list=t.last[k][e]
                ta=0.0
                th=0.0

                for n in n_list:
                    if n in a:
                        inc_a = a[n]
                        ta+=a[n]
                    if n in h:
                        inc_h = h[n]
                        th+=h[n]
                    #else:
                        #continue

                add=results['hits-a'][k].setdefault(e,[])
                add.append(ta) #ta #use ta for accumulated, inc_a for timestep

                add=results['hits-h'][k].setdefault(e,[])
                add.append(th) #th #use th for accumulated, inc_h for timestep


    return results

def A_analysis(t):
    ### Test Results ###
    results={}
    results['out']={}
    results['between']={}

    for s,kad in t.history_k.iteritems():
        kd, action = project_KD_A(kad)

        c_o=nx.out_degree_centrality(action)
        #c=nx.closeness_centrality(action,normalized=True)
        c_b=nx.betweenness_centrality(action,normalized=True)
        for k in t.last_action:
            results['out'].setdefault(k,{})
            results['between'].setdefault(k, {})
            for e in t.last_action[k]:
                n_list=t.last_action[k][e]
                tc_o=0.0
                tc_b=0.0
                inc_b=0.0
                for n in n_list:
                    if n in c_o:
                        tc_o+=c_o[n]

                    if n in c_b:
                        tc_b+=c_b[n]
                        inc_b=c_b[n]
                    #else:
                        #continue

                add_o=results['out'][k].setdefault(e,[])
                add_o.append(tc_o)

                add_b = results['between'][k].setdefault(e, [])
                add_b.append(tc_b)

    results['hits-a']={}
    results['hits-h']={}
    for s,kad in t.history_k.iteritems():
        kd,action=project_KD_A(kad)

        if s != 0:
            (h,a)=nx.hits(action,max_iter=10000,normalized=True) #(hubs,authorities)
        else:
            h = dict.fromkeys(t.history_k[s].nodes(),0.0)
            a = dict.fromkeys(t.history_k[s].nodes(), 0.0)
        #c=nx.betweenness_centrality(kd,normalized=True)
        for k in t.last_action:
            results['hits-a'].setdefault(k,{})
            results['hits-h'].setdefault(k,{})
            for e in t.last_action[k]:
                n_list=t.last_action[k][e]
                ta=0.0
                th=0.0
                inc_a=0.0
                inc_h=0.0

                for n in n_list:
                    if n in a:
                        inc_a = a[n]
                        ta+=a[n]
                    if n in h:
                        inc_h = h[n]
                        th+=h[n]
                    #else:
                        #continue

                add=results['hits-a'][k].setdefault(e,[])
                add.append(ta) #ta #use ta for accumulated, inc_a for timestep

                add=results['hits-h'][k].setdefault(e,[])
                add.append(th) #th #use th for accumulated, inc_h for timestep

    return results

#Ranks through time
def cent_rank_by_time(t):
    #Track the centrality rankings of actions by age

    #Get nodes of decision
    d_nodes=t.last_action['dec'][0]
    #del d_nodes[0] #0th index is starting node-'no decision'

    age_dict={}
    #Decisions tracking
    # print t.history_k
    for step,kad in t.history_k.iteritems(): #iterate through kad maps
        d_in_kad=[]
        for n in d_nodes:
            if n in kad.nodes():
                d_in_kad.append(n)

        if d_in_kad: #only procceed if a decision node exists
            kd, action = project_KD_A(kad) #get action projection
            bc = nx.betweenness_centrality(action, normalized=True) #calc betweenness centrality
            bc_array = np.array(bc.values()) #set up array
            bc_array = np.sort(bc_array) # sort by array by centrality

            for n in d_in_kad: #go through decision nodes
                n_step=action.node[n]['step'] #get node step
                age=step-n_step #get total age
                bc_rank = bc_array.tolist().index(bc[n])/float(action.number_of_nodes()) # get rank of decision in between centrality
                #print step,n,bc_rank
                add=age_dict.setdefault(age,[]) #if age does not exist, create
                add.append(bc_rank) #append rank

    return age_dict

#Ranks through time
def cent_rank_by_time_KAD(t):
    #Track the centrality rankings of actions by age

    #Get nodes of decision
    d_nodes=t.last_action['dec'][0]
    #del d_nodes[0] #0th index is starting node-'no decision'

    age_dict={}
    #Decisions tracking
    # print t.history_k
    for step,kad in t.history_k.iteritems(): #iterate through kad maps
        d_in_kad=[]
        for n in d_nodes:
            if n in kad.nodes():
                d_in_kad.append(n)

        if d_in_kad: #only procceed if a decision node exists
            #kd, action = project_KD_A(kad) #get action projection
            bc = nx.betweenness_centrality(kad, normalized=True) #calc betweenness centrality
            bc_array = np.array(bc.values()) #set up array
            bc_array = np.sort(bc_array) # sort by array by centrality

            for n in d_in_kad: #go through decision nodes
                n_step=kad.node[n]['step'] #get node step
                age=step-n_step #get total age
                bc_rank = bc_array.tolist().index(bc[n])/float(kad.number_of_nodes()) # get rank of decision in between centrality
                #print step,n,bc_rank
                add=age_dict.setdefault(age,[]) #if age does not exist, create
                add.append(bc_rank) #append rank

    return age_dict

#Criticality through time
def crit_rank_by_prob(t,routes):
    # Track critical node rankings of actions by probability it creates part of a solution

    prob_dict={}

    for step,kad in t.history_k.iteritems():
        if step==0:
            continue
        kd,action=project_KD_A(kad) #get kd projection
        odc=nx.out_degree_centrality(kd) #calc out degree centrality
        odc_array = np.array(odc.values())  # set up array
        odc_array = np.sort(odc_array)  # sort by array by centrality



# Decision ID
def decision_ID(tsp_object,routes):
    kd, action = project_KD_A(t.history_k[len(tsp_object.history_k) - 1])  # check last KAD
    bc = nx.betweenness_centrality(action, normalized=True)
    cc = nx.closeness_centrality(action, normalized=True)
    odc = nx.out_degree_centrality(action)
    n_nodes=float(action.number_of_nodes())
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
        #print n, action.node[n]
        bc_rank = bc_array.tolist().index(bc[n])  # get rank of decision in between centrality
        bc_rank_list.append(bc_rank)
        cc_rank = cc_array.tolist().index(cc[n])  # get rank of decision in closeness centrality
        cc_rank_list.append(cc_rank)
        odc_rank = odc_array.tolist().index(odc[n])  # get rank of decision in outdegree centrality
        odc_rank_list.append(odc_rank)

    #Check critical moves
    r=routes[-1][0]
    e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
    del e_r[-1]
    maxes={}
    print e_r
    for i in xrange(len(e_r)):
        bc_route = []
        cc_route = []
        odc_route = []
        #for n in tsp_object.last_action['route'][i]:

        for n in tsp_object.last_action['dist'][e_r[i]]:
        # for e,n in tsp_object.last_action['dist'].iteritems():
            #print n

            #if action.node[n]['e']==e_r[i]: #for route
            if action.node[n]['label'] == e_r[i]: #for distance
            # if action.node[n]['label'] == e: #for all distance
                print n, action.node[n], e_r[i], bc_array.tolist().index(bc[n]),cc_array.tolist().index(cc[n]),odc_array.tolist().index(odc[n])
                bc_route.append(bc_array.tolist().index(bc[n]))  # get rank of decision in between centrality
                cc_route.append(cc_array.tolist().index(cc[n]))  # get rank of decision in closeness centrality
                odc_route.append(odc_array.tolist().index(odc[n]))  # get rank of decision in outdegree centrality
        maxes[(i,e_r[i])]=(min(bc_route)/n_nodes,min(cc_route)/n_nodes,min(odc_route)/n_nodes)

    #Centrality rank by edge
    bar_dict={}
    for e, n_list in tsp_object.last_action['dist'].iteritems():
        e_rank=[]
        for n in n_list:
            e_rank.append(odc_array.tolist().index(odc[n]))  # get rank of decision in outdegree centrality
        if e_rank:
            bar_dict[e]=min(e_rank)
    plt.figure()
    plt.bar(range(len(bar_dict)), bar_dict.values(), align="center")
    plt.xticks(range(len(bar_dict)), list(bar_dict.keys()))

    # print 'betweenness ranks:', bc_rank_list
    # print 'closeness ranks:', cc_rank_list
    # print 'out degree ranks:', odc_rank_list
    # for n,c in bc.iteritems():
    #     print 'n:{}, rank:{}, data:{}'.format(n, bc_array.tolist().index(bc[n]),action.node[n])
    return np.mean(bc_rank_list)/n_nodes, np.mean(cc_rank_list)/n_nodes, np.mean(odc_rank_list)/n_nodes,maxes

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
                    plt.plot(x, series, '--',linewidth=2,label='{}'.format(e), color=c,)
                elif (e[0] == 0) and start_on:
                    plt.plot(x, series, ':',linewidth=2,label='{}'.format(e), color=c)
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
    #input KAD and return KD and A networks
    #KD is projection of A layer
    #A is projection of KD layer

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

    A=KAD.copy()
    for i in A.nodes():
        if A.node[i]['layer'] in ['k','d']:
            #get successors
            preds=A.predecessors(i)
            tails=[]
            for p in preds:
                if A.node[p]['layer'] not in ['k','d']:
                    tails.append(p)

            #get head
            sucs=A.successors(i)
            heads = []
            for s in sucs:
                if A.node[s]['layer'] not in ['k', 'd']:
                    heads.append(s)

            #Add edges
            for tail in tails:
                for head in heads:
                    step = A.node[head]['step'] #step related to head
                    A.add_edge(tail,head,step=step)

            #Add edges if initialized k
            if not tails: #no preds
                e_bad_direction=list(itertools.combinations(heads, 2)) #ID edges
                for e in e_bad_direction: #iterate through edgs
                    if e[1]<e[0]: #Flip if reverse order
                        e=(e[1],e[0])
                    step=A.node[e[0]]['step'] #step related to head
                    #print i, e
                    A.add_edge(e[0],e[1],step=step)

            A.remove_node(i)
    return KD, A

def ML_path_comparison_KAD(tsp_object, results_series,title):
    # Compare the maximum likelihood paths between a set of results and pheromones
    tsp_series=tsp_object.history_s

    #Get mle(s) for pheromones
    ml_pheromone={}
    ml_centrality={}
    ml_error={}
    average_prob_error={}
    average_rank_error = {}
    ml_paths={}
    for s,tsp in tsp_series.iteritems():#go through time steps
        ml_pheromone[s]={}
        ml_centrality[s]={}
        ml_error[s]={}
        error_list=[]
        nodes=list(set(tsp.nodes())-set([0]))
        for route in itertools.permutations(nodes): #calc mle for each route
            r = [0]+list(route)
            likelihood_ph=1.0
            likelihood_cent=1.0

            #Go through each step in the solution and calc prob that it occurred
            e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
            del e_r[-1]
            tabu=[0]
            for e in e_r: #selected step
                if tsp[e[0]][e[1]]['dist']==0.0:
                    e_h=tsp_object.explore
                else:
                    e_h=1.0/tsp[e[0]][e[1]]['dist']
                e_ph=np.power(tsp[e[0]][e[1]]['p'],tsp_object.alpha)*np.power(e_h,tsp_object.beta)  #tsp[e[0]][e[1]]['p']
                #e_ph=tsp[e[0]][e[1]]['p']

                e_cent=0.0
                # for last_node in tsp_object.last['dist'][e]:# iterate through nodes to get centrality results
                #     if last_node in results_series[s]:
                #         e_cent+=results_series[s][last_node]

                for last_node in tsp_object.last['dist'][e][::-1]:  # iterate through nodes to get centrality results
                    if last_node in results_series[s]:
                        e_cent += results_series[s][last_node]
                        break # Only use the most recent and exit

                tot_ph=0.0
                tot_cent=0.0
                num_nodes=0
                for tail in tsp.nodes():
                    if tail in tabu:
                        continue
                    num_nodes+=1
                    # get pheromones and centralities for available edges

                    if tsp[e[0]][tail]['dist'] == 0.0:
                        e_h = tsp_object.explore
                    else:
                        e_h = 1.0 / tsp[e[0]][tail]['dist']

                    tot_ph += np.power(tsp[e[0]][tail]['p'],tsp_object.alpha)*np.power(e_h,tsp_object.beta) #tsp[e[0]][tail]['p']
                    #tot_ph += tsp[e[0]][tail]['p']
                    # for last_node in tsp_object.last['dist'][(e[0],tail)]:  # iterate through nodes to get centrality results
                    #     if last_node in results_series[s]:
                    #         tot_cent += results_series[s][last_node]

                    for last_node in tsp_object.last['dist'][(e[0], tail)][::-1]:  # iterate through nodes to get centrality results
                        if last_node in results_series[s]:
                            tot_cent += results_series[s][last_node]
                            break

                if tot_ph==0.0:
                    prob_ph=0.0
                else:
                    prob_ph=e_ph/tot_ph

                if tot_cent==0.0:
                    prob_cent=1.0/num_nodes
                else:
                    prob_cent=e_cent/tot_cent
                likelihood_ph*=prob_ph
                likelihood_cent*=prob_cent
                tabu.append(e[1])

            #Compare likelihoods
            ml_pheromone[s][route]=likelihood_ph
            ml_centrality[s][route]=likelihood_cent
            error = (likelihood_cent-likelihood_ph)
            ml_error[s][route]=error
            error_list.append(error)


        #Get ml paths
        path_ph=max(ml_pheromone[s].iteritems(), key=operator.itemgetter(1))[0]
        #print path_ph
        keys=[]
        for p,ml in ml_pheromone[s].iteritems():
            if ml==ml_pheromone[s][path_ph]:
                keys.append(p)

        #Get rank error of paths
        error_list_rank = []
        error_list_prob = []
        #sored array of all centrality ml's
        ml_c_array=np.array(ml_centrality[s].values())
        ml_c_array=np.sort(ml_c_array)[::-1]
        for p in keys: #go through paths with max pheromone ml
            rank=ml_c_array.tolist().index(ml_centrality[s][p]) #get rank of centrality ml
            error_list_rank.append(rank)
            #print s,p, 'centrality={}'.format(ml_centrality[s][p]), 'pheromone={}'.format(ml_pheromone[s][p])
            error_list_prob.append(abs(ml_centrality[s][p]-ml_pheromone[s][p])/ml_pheromone[s][p])
        ml_paths[s]=keys
        average_rank_error[s] = np.mean(error_list_rank)
        average_prob_error[s] = np.mean(error_list_prob)


    #plot prob ratio and rank difference at 0-offset
    gen = xrange(len(average_rank_error))
    error = [average_rank_error[g]  for g in gen]#/ np.math.factorial(tsp_object.n_cities + 1)

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Rank difference of most likely paths, average={}'.format(title, np.mean((error))))

    error = [average_prob_error[g] for g in gen]

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Prob difference of most likely paths, average={}'.format(title,np.mean((error))))

    #check prediction at different offsets

    off=[]
    e_prob_off = []
    e_rank_off = []
    for offset in xrange(-16,15): #amount to offset centrality results
        e_prob_s = []
        e_rank_s = []
        for s,ml_ph in ml_pheromone.iteritems(): #go over each state
            s_off=s+offset
            #Track errors at this state
            e_prob_path=[]
            e_rank_path=[]
            if s_off not in ml_centrality: #offset doesn't exist
                continue
            else:
                # sorted array of all centrality ml's
                ml_c_array = np.array(ml_centrality[s_off].values())
                ml_c_array = np.sort(ml_c_array)[::-1]
                for p in ml_paths[s]:  # go through paths with max pheromone ml
                    rank = ml_c_array.tolist().index(ml_centrality[s_off][p])  # get rank of centrality ml
                    e_rank_path.append(rank)
                    e_prob_path.append(abs(ml_centrality[s_off][p] - ml_pheromone[s][p])/ ml_pheromone[s][p])
            e_prob_s.append(np.mean(e_prob_path))
            e_rank_s.append(np.mean(e_rank_path))#/np.math.factorial(tsp_object.n_cities + 1))
        e_prob_off.append(np.mean(e_prob_s))
        e_rank_off.append(np.mean(e_rank_s))
        off.append(offset)

    plt.figure()
    plt.scatter(off, e_rank_off)
    plt.title('{} Rank difference of most likely paths at different offsets'.format(title))

    plt.figure()
    plt.scatter(off, e_prob_off)
    plt.title('{} Prob difference of most likely paths at different offsets'.format(title))



    return e_rank_off,e_prob_off,average_prob_error,ml_error,ml_pheromone,ml_centrality

def ML_path_comparison(tsp_object, results_series, title):
    # Compare the maximum likelihood paths between a set of results and pheromones
    tsp_series=tsp_object.history_s
    #Get mle(s) for pheromones
    ml_pheromone={}
    ml_centrality={}
    ml_error={}
    average_prob_error={}
    average_rank_error = {}
    ml_paths={}
    for s,tsp in tsp_series.iteritems():#go through time steps
        ml_pheromone[s]={}
        ml_centrality[s]={}
        ml_error[s]={}
        error_list=[]
        nodes=list(set(tsp.nodes())-set([0]))
        for route in itertools.permutations(nodes): #calc mle for each route
            r = [0]+list(route)
            likelihood_ph=1.0
            likelihood_cent=1.0

            #Go through each step in the solution and calc prob that it occurred
            e_r = [tuple(r[i:i + 2]) for i in xrange(0, len(r), 1)]
            del e_r[-1]
            tabu=[0]
            for e in e_r: #selected step
                # if tsp[e[0]][e[1]]['dist']==0.0:
                #     e_h=tsp_object.explore
                # else:
                #     e_h=1.0/tsp[e[0]][e[1]]['dist']
                # e_ph=np.power(tsp[e[0]][e[1]]['p'],tsp_object.alpha)*np.power(e_h,tsp_object.beta)  #tsp[e[0]][e[1]]['p']
                e_ph=tsp[e[0]][e[1]]['p']
                e_cent=results_series['dist'][e][s]
                tot_ph=0.0
                tot_cent=0.0
                num_nodes=0
                for tail in tsp.nodes():
                    if tail in tabu:
                        continue
                    num_nodes+=1
                    # get pheromones and centralities for available edges

                    # if tsp[e[0]][tail]['dist'] == 0.0:
                    #     e_h = tsp_object.explore
                    # else:
                    #     e_h = 1.0 / tsp[e[0]][tail]['dist']
                    #
                    # tot_ph += np.power(tsp[e[0]][tail]['p'],tsp_object.alpha)*np.power(e_h,tsp_object.beta) #tsp[e[0]][tail]['p']
                    tot_ph += tsp[e[0]][tail]['p']
                    tot_cent += results_series['dist'][(e[0],tail)][s]

                if tot_ph==0.0:
                    prob_ph=0.0
                else:
                    prob_ph=e_ph/tot_ph

                if tot_cent==0.0:
                    prob_cent=1.0/num_nodes
                else:
                    prob_cent=e_cent/tot_cent
                likelihood_ph*=prob_ph
                likelihood_cent*=prob_cent
                tabu.append(e[1])

            #Compare likelihoods
            ml_pheromone[s][route]=likelihood_ph
            ml_centrality[s][route]=likelihood_cent
            error = (likelihood_cent-likelihood_ph)
            ml_error[s][route]=error
            error_list.append(error)


        #Get ml paths
        path_ph=max(ml_pheromone[s].iteritems(), key=operator.itemgetter(1))[0]
        #print path_ph
        keys=[]
        for p,ml in ml_pheromone[s].iteritems():
            if ml==ml_pheromone[s][path_ph]:
                keys.append(p)

        #Get rank error of paths
        error_list_rank = []
        error_list_prob = []
        #sored array of all centrality ml's
        ml_c_array=np.array(ml_centrality[s].values())
        ml_c_array=np.sort(ml_c_array)[::-1]
        for p in keys: #go through paths with max pheromone ml
            rank=ml_c_array.tolist().index(ml_centrality[s][p]) #get rank of centrality ml
            error_list_rank.append(rank)
            #print s,p, 'centrality={}'.format(ml_centrality[s][p]), 'pheromone={}'.format(ml_pheromone[s][p])
            error_list_prob.append(abs(ml_centrality[s][p]-ml_pheromone[s][p])/ml_pheromone[s][p])
        ml_paths[s]=keys
        average_rank_error[s] = np.mean(error_list_rank)
        average_prob_error[s] = np.mean(error_list_prob)

    #plot prob ratio and rank difference at 0-offset
    gen = xrange(len(average_rank_error))
    error = [average_rank_error[g]  for g in gen]#/ np.math.factorial(tsp_object.n_cities + 1)

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Rank difference of most likely paths, average={}'.format(title,np.mean((error))))

    error = [average_prob_error[g] for g in gen]

    plt.figure()
    plt.scatter(gen, error)
    plt.title('{} Prob difference of most likely paths, average={}'.format(title, np.mean((error))))

    #check prediction at different offsets
    off=[]
    e_prob_off = []
    e_rank_off = []
    for offset in xrange(-10,11): #amount to offset centrality results
        e_prob_s = []
        e_rank_s = []
        for s,ml_ph in ml_pheromone.iteritems(): #go over each state
            s_off=s+offset
            #Track errors at this state
            e_prob_path=[]
            e_rank_path=[]
            if s_off not in ml_centrality: #offset doesn't exist
                continue
            else:
                # sorted array of all centrality ml's
                ml_c_array = np.array(ml_centrality[s_off].values())
                ml_c_array = np.sort(ml_c_array)[::-1]
                for p in ml_paths[s]:  # go through paths with max pheromone ml
                    rank = ml_c_array.tolist().index(ml_centrality[s_off][p])  # get rank of centrality ml
                    e_rank_path.append(rank)
                    e_prob_path.append(abs(ml_centrality[s_off][p] - ml_pheromone[s][p])/ ml_pheromone[s][p])
            e_prob_s.append(np.mean(e_prob_path))
            e_rank_s.append(np.mean(e_rank_path)/np.math.factorial(tsp_object.n_cities + 1))
        e_prob_off.append(np.mean(e_prob_s))
        e_rank_off.append(np.mean(e_rank_s))
        off.append(offset)

    plt.figure()
    plt.scatter(off, e_rank_off)
    plt.title('{} Rank difference of most likely paths at different offsets'.format(title))

    plt.figure()
    plt.scatter(off, e_prob_off)
    plt.title('{} Prob difference of most likely paths at different offsets'.format(title))


    return e_rank_off,e_prob_off,average_prob_error,ml_error,ml_pheromone,ml_centrality

def test(results):
    return results.keys()

num_cities=5
t,routes=run_ACO(steps=1000,cities=num_cities,explore=.2,tolerance=.9)

#KAD
kad_results=KAD_analysis(t)
# print kad_results['hits-h'] is None
# for time in kad_results['hits-h'].keys():
#     print test(kad_results['hits-h'][time])
e_rank_off,e_prob_off,ave_prob_error,ml_error,ml_ph,ml_cent=ML_path_comparison_KAD(t,kad_results['hits-h'],title='KAD')

#K Analysis
k_results=KD_analysis(t)
# plot_results(k_results, KAD_type='KD',results_to_plot='hits-a')
e_rank_off,e_prob_off,ave_prob_error,ml_error,ml_ph,ml_cent=ML_path_comparison(t,k_results['hits-h'],title='K')

#A Analysis
# a_results=A_analysis(t)
# # print a_results
# plot_results(a_results, KAD_type='A', results_to_plot='out',rest_on=True,start_on=True)

#Decision ID
# bc,cc,odc,maxes=decision_ID(t,routes)
# print 'bc={}, cc={}, obc={}'.format(bc,cc,odc)
# for m in maxes:
#     print m,maxes[m]

age_dict=cent_rank_by_time(t)
age=[]
rank=[]
for a,ranks in age_dict.iteritems():
    #print a, age_dict[a]
    age.append(a)
    rank.append(np.mean(ranks))
plt.figure()
plt.scatter(age,rank)
plt.title('Rank by Age, K')

#print len(routes),routes
#print len(t.history_s.keys())
#print t.d
#t.problem_difficulty(t.history_d[0])


#print \num cities:\, n_cities

# for i in range(0,len(t.d)-1):
#     print 'delta t={}'.format(time[i+1]-time[i])
#     t.problem_difficulty(t.d[:i+1])


age_dict=cent_rank_by_time_KAD(t)
age=[]
rank=[]
for a,ranks in age_dict.iteritems():
    #print a, age_dict[a]
    age.append(a)
    rank.append(np.mean(ranks))
plt.figure()
plt.scatter(age,rank)
plt.title('Rank by Age, KAD')

#print len(routes),routes
#print len(t.history_s.keys())
#print t.d
#t.problem_difficulty(t.history_d[0])
time=[0]
time.extend(t.d_change)

s=[i[1] for i in routes]
gen=xrange(len(s))
plt.figure()
plt.scatter(gen,s)
_=plt.ylim([0.0,max(s)+1.0])

plt.show()



#print results['hits-a']
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

#KD,A=project_KD_A(t.k)
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