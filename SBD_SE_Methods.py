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
import seaborn as sns

#SAVING
import os
import cPickle as pickle

from SE_Agent import TSP_1diss as TSP_SE
from SBD_Agent import TSP_1diss as TSP_SBD

import timeit
from bisect import bisect

"""
v1: both iterative and convergent decicions (test, with difference->decided difference, not absolute)
    SKIPPED ******dissipation for old pheromones and new pheronoes****
TO DO:
1) Update Agents and Dataframe for Simulation 'Label'
"""



def run_ACO(steps=100, to_step_lim=False, cities=5,
            tolerance=0.7, criteria='diff', alpha=1.0, beta=1.0, dissipation=0.2,
            explore=1.0, force=None, rug=1.0,run=0, label='NA',
            method='iter'):

    if method=='iter':
        t = TSP_SE(start=0, tolerance=tolerance, criteria=criteria, alpha=alpha, beta=beta, dissipation=dissipation, explore=explore,
            n_cities=cities, force=force, rug=rug, run_ID=run, label=label)
    if method == 'conv':
        t = TSP_SBD(start=0, tolerance=tolerance, criteria=criteria, alpha=alpha, beta=beta, dissipation=dissipation, explore=explore,
                   n_cities=cities, force=force, rug=rug, run_ID=run, label=label)

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
    # print t.d
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
        if (s % interval != 0) and (s != 1):
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

def analysis_suite(t, interval=1,offset_range=0):
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
    param_columns=['label','run','N','rug','p','exp','tol','criteria','method','force','alpha','beta',
                   'tsp_sol','tsp_nsol','start_sol','start_nsol','centrality','Time']

    # columns_names
    cent_columns = param_columns+['layer']
    cent_param_columns=list(cent_columns)

    #parameters for Df
    params=[t.label,t.run_ID,t.n_cities,t.rug,t.dissipation,t.explore,t.tolerance,t.criteria,t.method,t.force,t.alpha,t.beta,
            t.tsp_sol_e,t.tsp_nonsol_e,t.start_sol_e,t.start_nonsol_e]
    cent_output = []

    #start path dictionary
    s_paths = {}

    #possible paths
    tsp_cities=t.tsp.nodes()
    tsp_cities.remove(0) #routes without start city
    temp_paths=itertools.permutations(tsp_cities) #routes without start city
    poss_paths=[] #routes with start city
    poss_paths_e=[]
    for path in temp_paths:
        path=[0]+list(path)
        poss_paths.append(path)
        path_e=[tuple(path[i:i + 2]) for i in xrange(0, len(path), 1)]
        del path_e[-1]
        poss_paths_e.append(path_e)

    # process
    for s,tsp in t.history_s.iteritems():
        if (s % interval!=0) and (s!=1):
            continue

        if s==0:
            continue
        #centrality
        #Df: params;centrality type;s;a/k;macro;edge;prototype;
        #Macro -> average by type -> average by nodes
        #M-D,A-D,M-Proto,Sep-Proto,A-Proto, M-Start,A-Sol Start,A-NSol Start, M-TSP, A-TSPSol, A-TSP NSol
        s_paths[s]={}

        #Get eliminated, undecided, locked-in edges
        all_e=t.tsp.edges()
        dec_e=t.history_ed[s] #edges that have decision
        lock_nodes=t.history_d[s] #nodes that are locked-into the solution
        lock_e=[tuple(lock_nodes[i:i + 2]) for i in xrange(0, len(lock_nodes), 1)]
        del lock_e[-1] #edges tht are locked in to solution


        possible = set() #set of edges that could still be part of the solution
        for r in poss_paths_e:
            if r[:len(lock_e)] == lock_e:
                possible.update(r)
        unlock_e = possible - set(lock_e) #set of viable edges not in the locked path
        undec_e = unlock_e - set(dec_e) #set of unlocked edges that have not been decided on
        elim_e = set(all_e) - set(lock_e) - undec_e
        # print 's', s,'dec',dec_e
        # print 'unlock', unlock_e
        # print 'lock', set(lock_e)
        # print 'undec', undec_e
        # print 'elim', elim_e

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
                # start-elmin, start-undec, start-lock, tsp-elim, tsp-undec, tsp-lock
                tot_start = []
                tot_tsp = []

                s_ns = []  # start non solution
                s_s = []  # start solution
                tsp_s = []  # tsp solution
                tsp_ns = []  # tsp non solution

                s_e = [] #start elim
                s_ud = [] #start undecided
                s_l = [] #start locked in

                tsp_e = []  # tsp elim
                tsp_ud = []  # tsp undecided
                tsp_l = []  # tsp locked in

                for sel, nodes in last['dist'].iteritems():
                    for n in nodes:
                        if n in results_dict:
                            if sel[0] == 0:
                                tot_start.append(results_dict[n])
                                if sel in sol_edges:  # in the solution
                                    s_s.append(results_dict[n])
                                else:  # not in solution
                                    s_ns.append(results_dict[n])

                                #decision state
                                if sel in elim_e:
                                    s_e.append(results_dict[n])
                                if sel in undec_e:
                                    s_ud.append(results_dict[n])
                                if sel in set(lock_e):
                                    s_l.append(results_dict[n])

                            else:
                                tot_tsp.append(results_dict[n])
                                if sel in sol_edges:  # in the solution
                                    tsp_s.append(results_dict[n])
                                else:  # not in solution
                                    tsp_ns.append(results_dict[n])

                                # decision state
                                if sel in elim_e:
                                    tsp_e.append(results_dict[n])
                                if sel in undec_e:
                                    tsp_ud.append(results_dict[n])
                                if sel in set(lock_e):
                                    tsp_l.append(results_dict[n])

                column_add=['Start Edges Sum','Start Edges Non-Solution Sum','Start Edges Solution Sum',
                            'Start Edges Eliminated Sum', 'Start Edges Undecided Sum', 'Start Edges Locked-In Sum',
                            'Start Edges Mean','Start Edges Non-Solution Mean', 'Start Edges Solution Mean',
                            'Start Edges Eliminated Mean', 'Start Edges Undecided Mean', 'Start Edges Locked-In Mean',
                            'TSP Edges Sum', 'TSP Edges Non-Solution Sum', 'TSP Edges Solution Sum',
                            'TSP Edges Eliminated Sum', 'TSP Edges Undecided Sum', 'TSP Edges Locked-In Sum',
                            'TSP Edges Mean','TSP Edges Non-Solution Mean', 'TSP Edges Solution Mean',
                            'TSP Edges Eliminated Mean', 'TSP Edges Undecided Mean', 'TSP Edges Locked-In Mean',]
                if not (set(column_add) < set(cent_columns)):
                    cent_columns.extend(column_add)

                cent_state.extend([np.sum(tot_start),np.sum(s_ns),np.sum(s_s),
                                   np.sum(s_e), np.sum(s_ud), np.sum(s_l),
                                   np.mean(tot_start),np.mean(s_ns),np.mean(s_s),
                                   np.mean(s_e), np.mean(s_ud), np.mean(s_l),
                                   np.sum(tot_tsp), np.sum(tsp_ns), np.sum(tsp_s),
                                   np.sum(tsp_e), np.sum(tsp_ud), np.sum(tsp_l),
                                   np.mean(tot_tsp), np.mean(tsp_ns), np.mean(tsp_s),
                                   np.mean(tsp_e), np.mean(tsp_ud), np.mean(tsp_l)])

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
                max_e_c = max(step_e_c.iteritems(), key=operator.itemgetter(1))[0]
                # print max_e_c
                c_keys = []
                for e_c, val in step_e_c.iteritems():
                    if val == step_e_c[max_e_c]:
                        c_keys.append(e_c)
                if len(c_keys) > 1:
                    unique = False
                    # tabu_c.append(max_e_c[1])
                    tabu_c.append(random.choice(c_keys)[1])

                else:
                    tabu_c.append(max_e_c[1])

                max_e_p = max(step_e_p.iteritems(), key=operator.itemgetter(1))[0]
                p_keys = []
                for e_p, val in step_e_p.iteritems():
                    if val == step_e_p[max_e_p]:
                        p_keys.append(e_p)
                if len(p_keys) > 1:
                    unique = False
                    # tabu_p.append(max_e_p[1])
                    tabu_p.append(random.choice(p_keys)[1])

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


def test(results):
    return results.keys()

def tsp_list_analysis(tsp_list,iters,interval=5):
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
    groupby_list=['centrality', 'layer', 'Time','tol']
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
    path_groupby_list=['centrality', 'offset', 'Time','tol']
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

    return cent_o, cent_mean, cent_error, path_o, path_mean #Total dataframe, mean df, std df, path error df

def condition_comparison(df, analysis='out',layer='k',value='Start Edges Mean',
                         save=False, save_title='test',figsize=(10,6),
                         save_dir='C:\Users\colinsh\Documents\DesignSim' ):
    p_data=df.loc[(df['centrality']==analysis) & (df['layer']==layer)].reset_index(drop=True).fillna(0.0)

    # print p_data
    f, a = plt.subplots(1, 1, figsize=figsize)
    sns.tsplot(time='Time', value=value, condition='label', unit='run', data=p_data, ci=95, ax=a)

    if save:
        plt.savefig('{}\{}.pdf'.format(save_dir,save_title))


def run_tests_bulk(steps=60, iters=30, interval=1,
              exp_l=[1.0], tol_l=[.6],
              rug_l=[1.0], diss_l=[.1],
              cities_l=[5], criteria_l=['abs'],
              meth_l=['iter', 'conv'],save_title='test', save_dir='C:\Users\colinsh\Documents\DesignSim' ):

    # exp_l = [0.1, 1.0, 2.0]
    # tol_l = [.4, .6, .8]
    # rug_l = [0.0, 1.0, 2.0]
    # diss_l = [.1, .2, .3]
    # meth_l = ['iter', 'conv']
    # criteria_l=['abs','diff']
    tsp_list = []
    scores = []
    for crit_test in criteria_l:
        for cities_test in cities_l:
            for meth_test in meth_l:
                for rug_test in rug_l:
                    for diss_test in diss_l:
                        for tol_test in tol_l:
                            for exp_test in exp_l:
                                label = '{}_R{}_P{}_E{}_T{}'.format(meth_test, rug_test,
                                                                    diss_test, exp_test,
                                                                    tol_test)
                                print label
                                for i in xrange(iters):
                                    tsp, routes = run_ACO(steps=steps, to_step_lim=True,
                                                          cities=cities_test,
                                                          explore=exp_test,
                                                          tolerance=tol_test,
                                                          criteria=crit_test,
                                                          method=meth_test,
                                                          dissipation=diss_test,
                                                          rug=rug_test, run=i, label=label)
                                    # print tsp.d_edges
                                    tsp_list.append(tsp)
                                    for r, s in zip(routes, xrange(len(routes))):
                                        scores.append([label, i, s, r[1]])

    # Score Dataframe
    score_df = pd.DataFrame(scores, columns=['label', 'run', 'Time', 'Score'])


    # Centrality Dataframe
    cent_o, cent_mean, cent_error, path_o, path_mean = tsp_list_analysis(tsp_list,iters,interval=interval)
    cent_o['ID'] = cent_o.apply(
        lambda row: '{}_r{}_e{}_p{}_{}_t{}'.format(row['N'], row['rug'], row['exp'],
                                                   row['p'], row['method'], row['tol']),axis=1)

    # Save dataframes
    cent_o.to_pickle('{}\{}_cent_df.pkl'.format(save_dir,save_title))
    path_o.to_pickle('{}\{}_path_df.pkl'.format(save_dir,save_title))
    score_df.to_pickle('{}\{}_score_df.pkl'.format(save_dir,save_title))

    return cent_o,path_o,score_df

def run_tests(steps=60, iters=30, interval=1,
                  exp_l=[1.0], tol_l=[.6],
                  rug_l=[1.0], diss_l=[.1],
                  cities_l=[5], criteria_l=['abs'],
                  meth_l=['iter', 'conv'],
                  save_nets=False,
                  save_title='test',
                  save_dir='C:\Users\colinsh\Documents\DesignSim'):
    """
    For each run,
    Run network analysis, save results as in a list
    save network as a pickle and delete it
    """

    #Create Directory if it does not exist
    dest='{}\{}'.format(save_dir,save_title)
    # print dest
    if not os.path.exists(dest):
        os.makedirs(dest)

    dest_net='{}\{}'.format(dest,'nets')
    # print dest_net
    if not os.path.exists(dest_net):
        os.makedirs(dest_net)

    # exp_l = [0.1, 1.0, 2.0]
    # tol_l = [.4, .6, .8]
    # rug_l = [0.0, 1.0, 2.0]
    # diss_l = [.1, .2, .3]
    # meth_l = ['iter', 'conv']
    # criteria_l=['abs','diff']
    # tsp_list = []
    cent_l = []
    path_l = []
    scores = []
    opt=[]
    for crit_test in criteria_l:
        for cities_test in cities_l:
            for meth_test in meth_l:
                for rug_test in rug_l:
                    for diss_test in diss_l:
                        for tol_test in tol_l:
                            for exp_test in exp_l:
                                label = '{}_R{}_P{}_E{}_T{}'.format(meth_test, rug_test,
                                                                    diss_test, exp_test,
                                                                    tol_test)
                                print label
                                opt_any=0.0
                                opt_end=0.0
                                for i in xrange(iters):
                                    tsp, routes = run_ACO(steps=steps, to_step_lim=True,
                                                          cities=cities_test,
                                                          explore=exp_test,
                                                          tolerance=tol_test,
                                                          criteria=crit_test,
                                                          method=meth_test,
                                                          dissipation=diss_test,
                                                          rug=rug_test, run=i, label=label)
                                    # print tsp.d_edges

                                    #Calculate run data
                                    cent_r, path_r, cent_params, path_params = analysis_suite(tsp, interval=interval,
                                                                                              offset_range=0)
                                    cent_l.append(cent_r)
                                    path_l.append(path_r)

                                    is_opt=False
                                    for r, s in zip(routes, xrange(len(routes))):
                                        scores.append([label, i, s, r[1]])
                                        if r[-1]==cities_test and not is_opt:
                                            opt_any+=1.0
                                            is_opt=True
                                    if scores[-1][-1]==cities_test:
                                        opt_end+=1.0

                                    #SAVE TSP
                                    # tsp_list.append(tsp)
                                    if save_nets:
                                        dest_run='{}\{}_{}.pkl'.format(dest_net,label,i)
                                        # print dest_run
                                        pickle.dump(tsp,open(dest_run,'wb'))
                                opt.append([label,opt_any/iters,opt_end/iters])


    # Score Dataframe
    score_df = pd.DataFrame(scores, columns=['label', 'run', 'Time', 'Score'])

    # Opt Dataframe
    opt_df=pd.DataFrame(opt,columns=['label','Any','End'])
    # print opt_df

    # Centrality Dataframe
    # cent_o, cent_mean, cent_error, path_o, path_mean = tsp_list_analysis(tsp_list, iters, interval=interval)
    # cent_o['ID'] = cent_o.apply(
    #     lambda row: '{}_r{}_e{}_p{}_{}_t{}'.format(row['N'], row['rug'], row['exp'],
    #                                                row['p'], row['method'], row['tol']), axis=1)

    cent_o=pd.concat(cent_l)
    path_o = pd.concat(path_l)

    # Save dataframes
    cent_o.to_pickle('{}\{}\cent_df.pkl'.format(save_dir, save_title))
    path_o.to_pickle('{}\{}\path_df.pkl'.format(save_dir, save_title))
    score_df.to_pickle('{}\{}\score_df.pkl'.format(save_dir, save_title))
    opt_df.to_pickle('{}\{}\optimal_df.pkl'.format(save_dir, save_title))

    return cent_o, path_o, score_df