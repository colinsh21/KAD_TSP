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

def tsp_list_analysis(tsp_list,interval=5):
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
                         save=False, save_title='test',
                         save_dir='C:\Users\colinsh\Documents\DesignSim' ):
    p_data=df.loc[(df['centrality']==analysis) & (df['layer']==layer)].reset_index(drop=True).fillna(0.0)

    # print p_data
    f, a = plt.subplots(1, 1, figsize=figsize)
    sns.tsplot(time='Time', value=value, condition='label', unit='run', data=p_data, ci=95, ax=a)

    if save:
        plt.savefig('{}\{}.pdf'.format(save_dir,save_title))

steps=70
interval=5
tick_interval=5
iters=2
save=False
dec_line_on=True
dc='r' #Decision Line Color
dw=2 #Decision Line Width
dl='Earliest Decision'
figsize=(10,6)
tsp_list=[]
scores=[]


#params
exp=1.0
num_cities=5
diss=0.1
rug=1.0

# print 'IT decision Baseline'
# tol=0.5
# meth='iter'
# label='iter_diff'
# for i in xrange(iters):
#     tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
#                           explore=exp, tolerance=tol,criteria='diff', method=meth,
#                           dissipation=diss, rug=rug, run=i, label=label)
#     print tsp.d_edges
#     tsp_list.append(tsp)
#     for r,s in zip(routes,xrange(len(routes))):
#         # scores.append([label,i,num_cities,rug,exp, diss, meth,tol,s,r[1]])
#         scores.append([label, i, s, r[1]])
test_iter_tol=False
if test_iter_tol:
    print 'IT decision tolerance'

    tol=0.3
    meth='iter'
    label='iter_T.3'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol,criteria='diff', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r,s in zip(routes,xrange(len(routes))):
            # scores.append([label,i,num_cities,rug,exp, diss, meth,tol,s,r[1]])
            scores.append([label, i, s, r[1]])

    tol=0.4
    meth='iter'
    label='iter_T.4'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol,criteria='diff', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r,s in zip(routes,xrange(len(routes))):
            # scores.append([label,i,num_cities,rug,exp, diss, meth,tol,s,r[1]])
            scores.append([label, i, s, r[1]])

    tol=0.5
    meth='iter'
    label='iter_T.5'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol,criteria='diff', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r,s in zip(routes,xrange(len(routes))):
            # scores.append([label,i,num_cities,rug,exp, diss, meth,tol,s,r[1]])
            scores.append([label, i, s, r[1]])

    tol=0.6
    meth='iter'
    label='iter_T.6'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol,criteria='diff', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r,s in zip(routes,xrange(len(routes))):
            # scores.append([label,i,num_cities,rug,exp, diss, meth,tol,s,r[1]])
            scores.append([label, i, s, r[1]])


# print 'CON decision baseline'
# t_d_contex=[] #Track when the first path is locked in!!
# tol=.9
# meth='conv'
# label='con_T.9'
# for i in xrange(iters):
#     tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
#                           explore=exp, tolerance=tol, criteria='abs', method=meth,
#                           dissipation=diss, rug=rug, run=i, label=label)
#     print tsp.d_edges
#     tsp_list.append(tsp)
#     for r,s in zip(routes,xrange(len(routes))):
#         # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
#         scores.append([label, i, s, r[1]])
#
#
#     if tsp.d_change:  # Context decision made
#         t_d_contex.append(tsp.d_change[0]) #Time decision occured
#     else:
#         t_d_contex.append(steps-1) #Decision not made, set to max time of convergence
#
#     # if tsp.d_change: #lock-in
#     #     if len(tsp.d_change)>=(num_cities-1): #can't get if not locked-in
#     #         t_d_contex.append(tsp.d_change[num_cities-2])  # Time lock-in occured, index of first lock-in is n-2


# print 'CON decision difference'
# t_d_contex=[] #Track when the first path is locked in!!
# tol=.9
# meth='conv'
# label='con_diff'
# for i in xrange(iters):
#     tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
#                           explore=exp, tolerance=tol, criteria='diff', method=meth,
#                           dissipation=diss, rug=rug, run=i, label=label)
#     print tsp.d_edges
#     tsp_list.append(tsp)
#     for r,s in zip(routes,xrange(len(routes))):
#         # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
#         scores.append([label, i, s, r[1]])

test_con_tol=False
if test_con_tol:
    print 'CON decision toleranc'
    t_d_contex = []  # Track when the first path is locked in!!
    tol = .9
    meth = 'conv'
    label = 'con_T.9'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol, criteria='abs', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r, s in zip(routes, xrange(len(routes))):
            # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
            scores.append([label, i, s, r[1]])


    t_d_contex = []  # Track when the first path is locked in!!
    tol = .7
    meth = 'conv'
    label = 'con_T.7'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol, criteria='abs', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r, s in zip(routes, xrange(len(routes))):
            # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
            scores.append([label, i, s, r[1]])

    tol = .8
    meth = 'conv'
    label = 'con_T.8'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol, criteria='abs', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r, s in zip(routes, xrange(len(routes))):
            # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
            scores.append([label, i, s, r[1]])

    tol = .6
    meth = 'conv'
    label = 'con_T.6'
    for i in xrange(iters):
        tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                              explore=exp, tolerance=tol, criteria='abs', method=meth,
                              dissipation=diss, rug=rug, run=i, label=label)
        print tsp.d_edges
        tsp_list.append(tsp)
        for r, s in zip(routes, xrange(len(routes))):
            # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
            scores.append([label, i, s, r[1]])

test_con_rug=False
if test_con_rug:
    print 'CON ruggedness'
    rug_l=[0.0,1.0,2.0]
    tol_l=[.6,.7,.8]
    for tol_test in tol_l:
        for rug_test in rug_l:
            meth = 'conv'
            label = 'con_R{}_T{}'.format(rug_test,tol_test)
            for i in xrange(iters):
                tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                                      explore=exp, tolerance=tol_test, criteria='abs', method=meth,
                                      dissipation=diss, rug=rug_test, run=i, label=label)
                print tsp.d_edges
                tsp_list.append(tsp)
                for r, s in zip(routes, xrange(len(routes))):
                    # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
                    scores.append([label, i, s, r[1]])

    # tol = .7
    # meth = 'conv'
    # label = 'con_R1'
    # rug=1.0
    #
    # for i in xrange(iters):
    #     tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
    #                           explore=exp, tolerance=tol, criteria='abs', method=meth,
    #                           dissipation=diss, rug=rug, run=i, label=label)
    #     print tsp.d_edges
    #     tsp_list.append(tsp)
    #     for r, s in zip(routes, xrange(len(routes))):
    #         # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
    #         scores.append([label, i, s, r[1]])
    #
    # tol = .7
    # meth = 'conv'
    # label = 'con_R2'
    # rug=2.0
    # for i in xrange(iters):
    #     tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
    #                           explore=exp, tolerance=tol, criteria='abs', method=meth,
    #                           dissipation=diss, rug=rug, run=i, label=label)
    #     print tsp.d_edges
    #     tsp_list.append(tsp)
    #     for r, s in zip(routes, xrange(len(routes))):
    #         # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
    #         scores.append([label, i, s, r[1]])


test_lists=True
if test_lists:
    steps=70
    iters=1
    test_con_learn = True
    if test_con_learn:
        print 'CON learn'
        exp_l = [1.0] #[0.1, 1.0, 2.0]
        tol_l = [.6] #[.4, .6, .8]
        rug_l = [1.0] #[0.0, 1.0, 2.0]
        diss_l = [.1,.2,.3]
        meth_l = ['iter','conv']
        for meth_test in meth_l:
            for rug_test in rug_l:
                for diss_test in diss_l:
                    for tol_test in tol_l:
                        for exp_test in exp_l:
                            label = '{}_R{}_P{}_E{}_T{}'.format(meth_test,rug_test,diss_test,exp_test, tol_test)
                            print label
                            for i in xrange(iters):
                                tsp, routes = run_ACO(steps=steps, to_step_lim=True, cities=num_cities,
                                                      explore=exp_test, tolerance=tol_test, criteria='abs', method=meth_test,
                                                      dissipation=diss_test, rug=rug_test, run=i, label=label)
                                print tsp.d_edges
                                tsp_list.append(tsp)
                                for r, s in zip(routes, xrange(len(routes))):
                                    # scores.append([label,i, num_cities, rug, exp, diss, meth, tol, s, r[1]])
                                    scores.append([label, i, s, r[1]])


def run_tests(steps=60, iters=30, interval=1, exp_l=[1.0], tol_l=[.6], rug_l=[1.0], diss_l=[.1],
              meth_l=['iter', 'conv'],save_title='test'):

    # exp_l = [0.1, 1.0, 2.0]
    # tol_l = [.4, .6, .8]
    # rug_l = [0.0, 1.0, 2.0]
    # diss_l = [.1, .2, .3]
    # meth_l = ['iter', 'conv']
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
                                                  cities=num_cities,
                                                  explore=exp_test,
                                                  tolerance=tol_test,
                                                  criteria='abs',
                                                  method=meth_test,
                                                  dissipation=diss_test,
                                                  rug=rug_test, run=i, label=label)
                            print tsp.d_edges
                            tsp_list.append(tsp)
                            for r, s in zip(routes, xrange(len(routes))):
                                scores.append([label, i, s, r[1]])

    # Score Dataframe
    score_df = pd.DataFrame(scores, columns=['label', 'run', 'Time', 'Score'])


    # Centrality Dataframe
    cent_o, cent_mean, cent_error, path_o, path_mean = tsp_list_analysis(tsp_list,interval)
    cent_o['ID'] = cent_o.apply(
        lambda row: '{}_r{}_e{}_p{}_{}_t{}'.format(row['N'], row['rug'], row['exp'],
                                                   row['p'], row['method'], row['tol']),axis=1)

    # Save dataframes
    cent_o.to_pickle('C:\Users\colinsh\Documents\DesignSim\{}_cent_df.pkl'.format(save_title))
    path_o.to_pickle('C:\Users\colinsh\Documents\DesignSim\{}_path_df.pkl'.format(save_title))
    score_df.to_pickle('C:\Users\colinsh\Documents\DesignSim\{}_score_df.pkl'.format(save_title))

    return cent_o,path_o,score_df

# cent_o,path_o,score_df=run_tests(steps=70,iters=2,interval=5)

#Score Dataframe
# score_df=pd.DataFrame(scores,columns=['label','run','N','rug','exp','p','method','tol','Time','Score'])

# score_df['ID']=score_df.apply(lambda row: '{}_r{}_e{}_p{}_{}_t{}'.format(row['N'],row['rug'],row['exp'],
#                                                                          row['p'],row['method'],row['tol']),axis=1)

score_df=pd.DataFrame(scores,columns=['label','run','Time','Score'])
f,a = plt.subplots(1,1,figsize=figsize)
sns.tsplot(time='Time',value='Score', condition='label', unit='run', data=score_df, ci=95, ax=a)



#Time of context decision
# dec_time=np.mean(t_d_contex)
# dec_time=min(t_d_contex)-1
# print 'Context Decision={}'.format(dec_time)

# Centrality Dataframe
cent_o,cent_mean,cent_error,path_o,path_mean=tsp_list_analysis(tsp_list)
cent_o['ID'] = cent_o.apply(lambda row: '{}_r{}_e{}_p{}_{}_t{}'.format(row['N'], row['rug'], row['exp'],
                                                                           row['p'], row['method'], row['tol']), axis=1)

"""
'Start Edges Sum', 'Start Edges Non-Solution Sum', 'Start Edges Solution Sum',
 'Start Edges Eliminated Sum', 'Start Edges Undecided Sum', 'Start Edges Locked-In Sum',
 'Start Edges Mean', 'Start Edges Non-Solution Mean', 'Start Edges Solution Mean',
 'Start Edges Eliminated Mean', 'Start Edges Undecided Mean', 'Start Edges Locked-In Mean',
 'TSP Edges Sum', 'TSP Edges Non-Solution Sum', 'TSP Edges Solution Sum',
 'TSP Edges Eliminated Sum', 'TSP Edges Undecided Sum', 'TSP Edges Locked-In Sum',
 'TSP Edges Mean', 'TSP Edges Non-Solution Mean', 'TSP Edges Solution Mean',
 'TSP Edges Eliminated Mean', 'TSP Edges Undecided Mean', 'TSP Edges Locked-In Mean'
 """

#Save dataframes
# cent_o.to_pickle('C:\Users\colinsh\Documents\DesignSim\cent_all_df.pkl')
# score_df.to_pickle('C:\Users\colinsh\Documents\DesignSim\score_all_df.pkl')

condition_comparison(cent_o,analysis='hits-h',layer='k',value='Start Edges Undecided Mean')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='Start Edges Eliminated Mean')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='Start Edges Locked-In Mean')
condition_comparison(cent_o,analysis='hits-h',layer='k',value='TSP Edges Undecided Mean')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='TSP Edges Eliminated Mean')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='TSP Edges Locked-In Mean')

condition_comparison(cent_o,analysis='hits-h',layer='k',value='Start Edges Undecided Sum')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='Start Edges Eliminated Sum')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='Start Edges Locked-In Sum')
condition_comparison(cent_o,analysis='hits-h',layer='k',value='TSP Edges Undecided Sum')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='TSP Edges Eliminated Sum')
# condition_comparison(cent_o,analysis='hits-h',layer='k',value='TSP Edges Locked-In Sum')

# p_data=cent_o.loc[(cent_o['centrality']=='hits-a') & (cent_o['layer']=='a') & (cent_o['method']=='iter')].reset_index(drop=True)
# p_data=cent_o.loc[(cent_o['centrality']=='hits-a') & (cent_o['layer']=='a')].reset_index(drop=True)
# # print p_data[['run','s','method','Start Edges Mean']]
# f,a = plt.subplots(1,1,figsize=figsize)
# sns.tsplot(time='s',value='Start Edges Mean', condition='ID', unit='run', data=p_data, ax=a)



# def condition_comparison(df, analysis='out',layer='k',value='Start Edges Mean'):
#     p_data=df.loc[(df['centrality']==analysis) & (cent_o['layer']==layer)].reset_index(drop=True)
#     f, a = plt.subplots(1, 1, figsize=figsize)
#     sns.tsplot(time='s', value=value, condition='ID', unit='run', data=p_data, ax=a)



# ##Analysis parameters
# layers=['a','k']
# layer_label={'a':'A', 'k':'KD'}
#
# ana_label={'out':'Out Degree', 'in':'In Degree', 'between':'Betweenness',
#            'hits-a':'HITS-Authorities', 'hits-h':'HITS-Hubs'}
#
# seg_labels=['Decision','Score','Prototyping','Context', 'TSP Paths']
# sub_labels=['Start Non-Solution', 'Start Solution', 'TSP Solution', 'TSP Non-Solution']
#
# save_dir='C:\Users\colinsh\Documents\DesignSim'
#
# # calculate 95% confidence bound
# conf_stat = st.t.ppf(1.95 / 2.0, iters - 1)
#
#
# #Scores history
# col='s'
#
# #score
# df1=score_mean.loc[(score_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=score_mean.loc[(score_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# df1=score_error.loc[(score_error['tol']==tol_nd)][col].reset_index(drop=True)
# df2=score_error.loc[(score_error['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=score_mean['t'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(score_mean['t'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=score_mean['t'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(score_mean['t'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
# plt.axhline(y=num_cities, color='r', linewidth=2, label='Optimal = {}'.format(num_cities))
#
# plt.ylim([num_cities-1.0,num_cities*2])
#
# plt.ylabel('Prototype Score Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Designer\'s Solution Quality, {} City C+TSP'.format(num_cities))
# plt.legend(['No Decision', 'Decision', 'Optimal = {}'.format(num_cities)], loc='best')
#
# if save:
#     plt.savefig('{}\score.pdf'.format(save_dir))
#
# #Segment comparison parameters
# w=1.0 #width
# ec='none' #edgecolor
# alpha=.5 #alpha
# cl=['maroon','orangered','orange','b','g'] #colorlist
#
# #HITS-K Knowledge Segment Comparison no Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-h') & (cent_mean['layer']=='k')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0 # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[2:]
# color_list = cl[2:]#['orange','b','g']
# plot_df.plot.bar(x='s',
#                  y=['Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, color=color_list, edgecolor=ec, ax=a)
# plt.ylabel('HITS-Hubs Centrality Sum')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Segment Comparison')
# plt.legend(['Prototypes','Context', 'TSP Paths'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\hh_k_seg_comp_.pdf'.format(save_dir))
#
# #HITS-A Action Segment Comparison no Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0  # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[1:]
# color_list = cl[1:]#['orangered','orange','b','g']
# plot_df.plot.bar(x='s',
#                  y=['Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, color=color_list, edgecolor=ec, ax=a)
# plt.ylabel('HITS-Authorities Centrality Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Segment Comparison')
# plt.legend(['Scoring','Prototyping','Context Learning', 'TSP Paths Learning'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\ha_a_seg_comp.pdf'.format(save_dir))
#
# #In Context Action Sum
# main_plot=cent_mean.loc[(cent_mean['centrality']=='in') & (cent_mean['layer']=='a')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)]['Start Edges Sum'].reset_index(drop=True)
# plot_df.columns=['Context Learning']
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Context Learning'], ax=a, alpha=alpha,
#                  color='b', figsize=figsize, width=w, edgecolor=ec)
# plt.ylabel('In-Degree Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Context Learning')
# # plt.legend(seg_labels, loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\in_a_c_sum.pdf'.format(save_dir))
#
# #HITS-A Context A Sum
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)]['Start Edges Sum'].reset_index(drop=True)
# plot_df.columns=['Learned Context']
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Learned Context'], ax=a, alpha=alpha,
#                  color='b', figsize=figsize, width=w, edgecolor=ec)
# plt.ylabel('HITS-Authorities Centrality Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Context Learning')
# # plt.legend(seg_labels, loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\ha_a_c_sum.pdf'.format(save_dir))
#
# #HITS-A Context A Mean
# col='Start Edges Mean'
# ana='hits-a'
# layer='a'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# plot_e=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
#
# comp=pd.concat([plot_df,plot_e],axis=1)
# comp.columns=['v','std']
# # print 'HA-A Context'
# # print comp
#
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# comp.plot(x=cent_mean['s'].unique(), y=['v'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), comp['v']- conf_stat * comp['std'],
#                  comp['v'] + conf_stat * comp['std'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('HITS-Authorities Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Action Layer: Context Learning')
# plt.legend(['Context Learning'], loc='best')
#
# if save:
#     plt.savefig('{}\ha_a_context_mean.pdf'.format(save_dir))
#
# #Out Context K Mean
# col='Start Edges Mean'
# ana='out'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# plot_e=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# # print plot_e
#
# comp=pd.concat([plot_df,plot_e],axis=1)
# comp.columns=['v','std']
# # print 'OUT-K Context'
# # print comp
#
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# comp.plot(x=cent_mean['s'].unique(), y=['v'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), comp['v']- conf_stat * comp['std'],
#                  comp['v'] + conf_stat * comp['std'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context')
# plt.legend(['Context'], loc='best')
#
# if save:
#     plt.savefig('{}\out_k_context_mean.pdf'.format(save_dir))
#
# #HITS-Hubs Context K Mean
# col='Start Edges Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# plot_e=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# # print plot_e
#
# comp=pd.concat([plot_df,plot_e],axis=1)
# comp.columns=['v','std']
# # print 'HH-K Context'
# # print comp
#
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# comp.plot(x=cent_mean['s'].unique(), y=['v'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), comp['v']- conf_stat * comp['std'],
#                  comp['v'] + conf_stat * comp['std'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context')
# plt.legend(['Context'], loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_context_mean.pdf'.format(save_dir))
#
#
# #HITS-A Action Sum Context and TSP
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0  # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[3:]
# # color_list = ['r','y','b','g','m']
# plot_df.plot.bar(x='s',
#                  y=['Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, edgecolor=ec, color=['b','g'], ax=a)
# plt.ylabel('HITS-Authorities Centrality Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Context and TSP Paths Learning')
# plt.legend(['Context Learning', 'TSP Paths Learning'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\ha_a_CT_sum_comp.pdf'.format(save_dir))
#
#
# #HITS-A Action Mean Context and TSP
# col1='Start Edges Mean'
# col2='TSP Edges Mean'
# ana='hits-a'
# layer='a'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['Context', 'TSP Paths']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['Context', 'TSP Paths']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['Context'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Context'] - conf_stat * plot_e['Context'],
#                  plot_df['Context'] + conf_stat * plot_e['Context'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['TSP Paths'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['TSP Paths'] - conf_stat * plot_e['TSP Paths'],
#                  plot_df['TSP Paths'] + conf_stat * plot_e['TSP Paths'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('HITS-Authorities Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Action Layer: Context and TSP Paths Learning')
# # plt.legend(seg_labels, loc='best')
#
# if save:
#     plt.savefig('{}\ha_a_CT_mean_comp.pdf'.format(save_dir))
#
#
# #HITS-A Knowledge Sum Context and TSP
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='k')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0  # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[3:]
# # color_list = ['r','y','b','g','m']
# plot_df.plot.bar(x='s',
#                  y=['Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, edgecolor=ec, color=['b','g'], ax=a)
# plt.ylabel('HITS-Authorities Centrality Sum')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context and TSP Paths')
# plt.legend(['Context', 'TSP Paths'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\ha_k_CT_sum_comp.pdf'.format(save_dir))
#
#
# #HITS-H Knowledge Sum Context and TSP
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-h') & (cent_mean['layer']=='k')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_nd)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0  # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[3:]
# # color_list = ['r','y','b','g','m']
# plot_df.plot.bar(x='s',
#                  y=['Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, edgecolor=ec, color=['b','g'], ax=a)
# plt.ylabel('HITS-Hubs Centrality Sum')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context and TSP Paths')
# plt.legend(['Context', 'TSP Paths'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\hh_k_CT_sum_comp.pdf'.format(save_dir))
#
#
#
# #Out K Mean Context Sol, Context Non-Sol
# col1='Start Edges Solution Mean'
# col2='Start Edges Non-Solution Mean'
# col_names=['Solution', 'Non-Solution']
# ana='out'
# layer='k'
#
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=col_names
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=col_names
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
#                  plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
#                  plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context, Solution and Non-Solution')
# # plt.legend(seg_labels, loc='best')
#
# if save:
#     plt.savefig('{}\out_k_C_SNS_mean_comp.pdf'.format(save_dir))
#
#
# #Out K Mean TSP Sol, Context Non-Sol
# col1='TSP Edges Solution Mean'
# col2='TSP Edges Non-Solution Mean'
# col_names=['Solution', 'Non-Solution']
# ana='out'
# layer='k'
#
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=col_names
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=col_names
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
#                  plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
#                  plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: TSP Paths, Solution and Non-Solution')
# # plt.legend(seg_labels, loc='best')
#
# if save:
#     plt.savefig('{}\out_k_TSP_SNS_mean_comp.pdf'.format(save_dir))
#
#
# #HITS-H K Mean Context Sol, Context Non-Sol
# col1='Start Edges Solution Mean'
# col2='Start Edges Non-Solution Mean'
# col_names=['Solution', 'Non-Solution']
# ana='hits-h'
# layer='k'
#
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=col_names
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=col_names
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
#                  plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
#                  plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context, Solution and Non-Solution')
# # plt.legend(seg_labels, loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_C_SNS_mean_comp.pdf'.format(save_dir))
#
#
# #HITS-Hubs K Mean TSP Sol, Context Non-Sol
# col1='TSP Edges Solution Mean'
# col2='TSP Edges Non-Solution Mean'
# col_names=['Solution', 'Non-Solution']
# ana='hits-h'
# layer='k'
#
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=col_names
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col1].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_nd)][col2].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=col_names
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[0]], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[0]] - conf_stat * plot_e[col_names[0]],
#                  plot_df[col_names[0]] + conf_stat * plot_e[col_names[0]],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=[col_names[1]], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df[col_names[1]] - conf_stat * plot_e[col_names[1]],
#                  plot_df[col_names[1]] + conf_stat * plot_e[col_names[1]],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: TSP Paths, Solution and Non-Solution')
# # plt.legend(seg_labels, loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_TSP_SNS_mean_comp.pdf'.format(save_dir))
#
#
# ############################
# ############################
# ###Decision Comparisions####
# ############################
# ############################
#
# #HITS-K Knowledge Segment Comparison w/ Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-h') & (cent_mean['layer']=='k')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_d)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0 # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))[2:]
# color_list = cl[2:]#['orange','b','g']
# plot_df.plot.bar(x='s',
#                  y=['Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, edgecolor=ec, color=color_list, ax=a)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Sum')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Segment Comparison with Decision')
# plt.legend([dl, 'Prototypes','Context', 'TSP Paths'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\hh_k_seg_comp_dec.pdf'.format(save_dir))
#
# #HITS-A Action Segment Comparison w/ Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='hits-a') & (cent_mean['layer']=='a')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_d)].reset_index(drop=True)
# # plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
#
# cmap = plt.cm.hsv  # jet
# range_bottom = 0.0  # 0.2
# range_top = 0.625  # .85
#
# plot_list = ['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum']
# color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))
# color_list = cl#['firebrick','orangered','orange','b','g']
# plot_df.plot.bar(x='s',
#                  y=['Decision Sum', 'Score Sum', 'Step Sum', 'Start Edges Sum', 'TSP Edges Sum'], alpha=alpha,
#                  stacked=True, width=w, edgecolor=ec, color=color_list, ax=a)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Authorities Centrality Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Segment Comparison with Decision')
# plt.legend([dl, 'Decision-making','Scoring','Prototyping','Context Learning', 'TSP Paths Learning'], loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\ha_a_seg_comp_dec.pdf'.format(save_dir))
#
# #In Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='in') & (cent_mean['layer']=='a')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_d)]['Decision'].reset_index(drop=True)
# plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a, alpha=alpha,
#                  color='r', figsize=figsize, width=w, edgecolor=ec)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('In-Degree Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Decision')
# plt.legend(loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\in_a_dec.pdf'.format(save_dir))
#
# # HITS-A Action Decision
# main_plot = cent_mean.loc[(cent_mean['centrality'] == 'hits-a') & (cent_mean['layer'] == 'a')]
# plot_df = main_plot.loc[(cent_mean['tol'] == tol_d)]['Decision'].reset_index(drop=True)
# plot_df.columns = ['Decision']
# plt.figure(figsize=figsize)
# a = plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a, alpha=alpha,
#                  color='r', figsize=figsize, width=w, edgecolor=ec)
#
# if dec_line_on:
#     plt.axvline(x=dec_time, color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Authorities Centrality Sum')
# plt.xlabel('Time')
# plt.title('Action Layer: Decision')
# plt.legend(loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\ha_a_dec.pdf'.format(save_dir))
#
# #Out Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='out') & (cent_mean['layer']=='k')]
# plot_df=main_plot.loc[(cent_mean['tol']==tol_d)]['Decision'].reset_index(drop=True)
# plot_df.columns=['Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a, alpha=alpha,
#                  color='r', figsize=figsize, width=w, edgecolor=ec)
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Sum')
# plt.xlabel('Time')
# plt.title('Decision Layer: Decision')
# plt.legend(loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\out_d_dec.pdf'.format(save_dir))
#
# #Out Context Decision vs. No Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='out') & (cent_mean['layer']=='k')]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)]['Start Edges Sum'].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)]['Start Edges Sum'].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['No Decision'], ax=a,
#                  color='b', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a,
#                  color='g', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Sum')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context')
# plt.legend(loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\out_k_context_comparison.pdf'.format(save_dir))
#
#
# #Out Transition Decision vs. No Decision
# main_plot=cent_mean.loc[(cent_mean['centrality']=='out') & (cent_mean['layer']=='k')]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)]['Step Sum'].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)]['Step Sum'].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['No Decision'], ax=a,
#                  color='b', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)
# plot_df.plot.bar(x=cent_mean['s'].unique(), y=['Decision'], ax=a,
#                  color='g', figsize=figsize, width=w, edgecolor=ec, alpha=alpha)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Sum')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Prototyping')
# plt.legend(loc='best')
# plt.xticks(np.arange(0, steps + 1, float(tick_interval)), range(0, steps + 1, tick_interval))
# if save:
#     plt.savefig('{}\out_k_proto_comparison_sum.pdf'.format(save_dir))
#
# #Out Context Mean Decision vs. No Decision
# col='Start Edges Mean'
# ana='out'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\out_k_context_comparison_mean.pdf'.format(save_dir))
#
#
# #HITS-H Context Mean Decision vs. No Decision
# col='Start Edges Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_context_comparison_mean.pdf'.format(save_dir))
#
#
# #Out TSP Mean Decision vs. No Decision
# col='TSP Edges Mean'
# ana='out'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: TSP Paths')
# plt.legend( loc='best')
#
# if save:
#     plt.savefig('{}\out_k_TSP_comparison_mean.pdf'.format(save_dir))
#
# #HITS-Hubs TSP Mean Decision vs. No Decision
# col='TSP Edges Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: TSP Paths')
# plt.legend( loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_TSP_comparison_mean.pdf'.format(save_dir))
#
# #HITS-H Context Sol Mean Decision vs. No Decision
# col='Start Edges Solution Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context, Solution')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_CS_comparison_mean.pdf'.format(save_dir))
#
# #HITS-H Context Non-Sol Mean Decision vs. No Decision
# col='Start Edges Non-Solution Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context, Non-Solution')
# plt.legend( loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_CNS_comparison_mean.pdf'.format(save_dir))
#
# # HITS-H TSP Sol Mean Decision vs. No Decision
# col = 'TSP Edges Solution Mean'
# ana = 'hits-h'
# layer = 'k'
#
# # Centrality
# main_plot = cent_mean.loc[(cent_mean['centrality'] == ana) & (cent_mean['layer'] == layer)]
# df1 = main_plot.loc[(cent_mean['tol'] == tol_nd)][col].reset_index(drop=True)
# df2 = main_plot.loc[(cent_mean['tol'] == tol_d)][col].reset_index(drop=True)
# plot_df = pd.concat([df1, df2], axis=1, ignore_index=False)
# plot_df.columns = ['No Decision', 'Decision']
#
# # error
# error_plot = cent_error.loc[(cent_error['centrality'] == ana) & (cent_error['layer'] == layer)]
# df1 = error_plot.loc[(error_plot['tol'] == tol_nd)][col].reset_index(drop=True)
# df2 = error_plot.loc[(error_plot['tol'] == tol_d)][col].reset_index(drop=True)
# plot_e = pd.concat([df1, df2], axis=1, ignore_index=False)
# plot_e.columns = ['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a = plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#              color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#              color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: TSP Paths, Solution')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_TS_comparison_mean.pdf'.format(save_dir))
#
# # HITS-H TSP N-Sol Mean Decision vs. No Decision
# col = 'TSP Edges Non-Solution Mean'
# ana = 'hits-h'
# layer = 'k'
#
# # Centrality
# main_plot = cent_mean.loc[(cent_mean['centrality'] == ana) & (cent_mean['layer'] == layer)]
# df1 = main_plot.loc[(cent_mean['tol'] == tol_nd)][col].reset_index(drop=True)
# df2 = main_plot.loc[(cent_mean['tol'] == tol_d)][col].reset_index(drop=True)
# plot_df = pd.concat([df1, df2], axis=1, ignore_index=False)
# plot_df.columns = ['No Decision', 'Decision']
#
# # error
# error_plot = cent_error.loc[(cent_error['centrality'] == ana) & (cent_error['layer'] == layer)]
# df1 = error_plot.loc[(error_plot['tol'] == tol_nd)][col].reset_index(drop=True)
# df2 = error_plot.loc[(error_plot['tol'] == tol_d)][col].reset_index(drop=True)
# plot_e = pd.concat([df1, df2], axis=1, ignore_index=False)
# plot_e.columns = ['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a = plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#              color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#              color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: TSP Paths, Non-Solution')
# plt.legend( loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_TNS_comparison_mean.pdf'.format(save_dir))
#
# ############################
# ############################
# ####Lock-in Comparisions####
# ############################
# ############################
#
# """
# 'Start Edges Eliminated Mean', 'Start Edges Undecided Mean', 'Start Edges Locked-In Mean'
# """
#
# #Out Start-Elim Mean Decision vs. No Decision
# col='Start Edges Eliminated Mean'
# ana='out'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context Eliminated')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\out_k_s_elim_comparison_mean.pdf'.format(save_dir))
#
# #Out Start-Undec Mean Decision vs. No Decision
# col='Start Edges Undecided Mean'
# ana='out'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context Undecided')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\out_k_s_undec_comparison_mean.pdf'.format(save_dir))
#
#
# #Out Start-Lock Mean Decision vs. No Decision
# col='Start Edges Locked-In Mean'
#
# ana='out'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Out-Degree Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context Locked-in')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\out_k_s_lock_comparison_mean.pdf'.format(save_dir))
#
# #HH Start-Elim Mean Decision vs. No Decision
# col='Start Edges Eliminated Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context Eliminated')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_s_elim_comparison_mean.pdf'.format(save_dir))
#
# #HH Start-Undec Mean Decision vs. No Decision
# col='Start Edges Undecided Mean'
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context Undecided')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_s_undec_comparison_mean.pdf'.format(save_dir))
#
#
# #HITS-Hubs Start-Lock Mean Decision vs. No Decision
# col='Start Edges Locked-In Mean'
#
# ana='hits-h'
# layer='k'
#
# #Centrality
# main_plot=cent_mean.loc[(cent_mean['centrality']==ana) & (cent_mean['layer']==layer)]
# df1=main_plot.loc[(cent_mean['tol']==tol_nd)][col].reset_index(drop=True)
# df2=main_plot.loc[(cent_mean['tol']==tol_d)][col].reset_index(drop=True)
# plot_df=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_df.columns=['No Decision', 'Decision']
#
# #error
# error_plot=cent_error.loc[(cent_error['centrality']==ana) & (cent_error['layer']==layer)]
# df1=error_plot.loc[(error_plot['tol']==tol_nd)][col].reset_index(drop=True)
# df2=error_plot.loc[(error_plot['tol']==tol_d)][col].reset_index(drop=True)
# plot_e=pd.concat([df1,df2],axis=1, ignore_index=False)
# plot_e.columns=['No Decision', 'Decision']
#
# plt.figure(figsize=figsize)
# a=plt.gca()
# # plot_df.plot.bar(x=cent_mean['s'].unique(),y=['No Decision','Decision'],
# #                  stacked=False, edgecolor='None',figsize=figsize, width=1.0)
# plot_df.plot(x=cent_mean['s'].unique(), y=['No Decision'], ax=a, kind='line',
#                  color='b', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['No Decision'] - conf_stat * plot_e['No Decision'],
#                  plot_df['No Decision'] + conf_stat * plot_e['No Decision'],
#                  facecolor='b', alpha=.5,
#                  linewidth=0.0)
#
# plot_df.plot(x=cent_mean['s'].unique(), y=['Decision'], ax=a, kind='line',
#                  color='g', figsize=figsize, linewidth=3.0)
#
# plt.fill_between(cent_mean['s'].unique(), plot_df['Decision'] - conf_stat * plot_e['Decision'],
#                  plot_df['Decision'] + conf_stat * plot_e['Decision'],
#                  facecolor='g', alpha=.5,
#                  linewidth=0.0)
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('HITS-Hubs Centrality Mean (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Knowledge Layer: Context Locked-in')
# plt.legend(loc='best')
#
# if save:
#     plt.savefig('{}\hh_k_s_lock_comparison_mean.pdf'.format(save_dir))
#
#
# full_analysis=False #Every centrality metric for every layer
# if full_analysis:
#     for ana in cent_o['centrality'].unique():
#         # print ana
#         # path=path_mean.copy()
#         # # path_e=path_error.copy()
#         # plot_path = path.loc[path['centrality'] == ana]  # .groupby('offset').mean()
#         # plot_offset = path_o.loc[path_o['centrality'] == ana]
#         # # plot_e=path_e.loc[path_e['centrality'] == ana]
#         # # for off in plot_path['offset'].unique():
#         # plot_0 = plot_path.loc[plot_path['offset'] == 0]
#         # print plot_0[['Min Error','Max Error']]
#         # plot_0_e = plot_e.loc[plot_e['offset'] == 0]
#         # plot_0.join(plot_0_e['std'])
#         # print plot_0
#         # ax=plot_0.plot(x='s',y='error',yerr='std' ,title='Analysis-{}: Path Error at 0 Offset'.format(ana))
#         # plot_0.plot(x='s',y=['Strongest Path Error','Min Error','Max Error'],title='Analysis-{}: Path Error at 0 Offset'.format(ana))
#
#         ### Plot error
#         # plot_0.plot(x='s', y=['Strongest Path Error'], linewidth=3.0, title='Analysis-{}: Path Error at 0 Offset'.format(ana))
#         # plt.fill_between(plot_0['s'].values,plot_0['Min Error'],plot_0['Max Error'],
#         #                  facecolor='blue', linewidth=0.0, alpha=.5, label='Confidence Interval')
#         ### Plot error
#
#         # print plot_offset.head(10)
#         # plot_offset.boxplot(column=['error'],by=['offset'])
#         # plot_offset.boxplot(column=['error'], by=['offset'])#, kind='box', title='Analysis-{}: Error by Offset'.format(ana))
#         # plot_off = plot_path.groupby('offset').mean()
#         # plot_off_error = plot_path.groupby('offset').std()
#         # # print plot_error
#         # # print plot_path['error']
#         #
#         # plot_off.plot(y='error',yerr=plot_off_error['error'],title='Analysis-{}: Average Error by Offset'.format(ana))
#
#         for l in layers:
#             # print ana,l
#             cent=cent_mean.copy()
#             cent_e=cent_error.copy()
#             plot_df=cent.loc[(cent['centrality']==ana) & (cent['layer']==l)]
#             plot_e=cent_e.loc[(cent_e['centrality']==ana) & (cent_e['layer']==l)]
#             # print plot_df.head()
#             # print plot_e.head()
#             # print ana, l
#             # plot_df.fillna(0.0)
#             # print plot_df[['Decision Mean','Score Mean']]
#             cmap=plt.cm.hsv#jet
#             range_bottom=0.0#0.2
#             range_top=0.625#.85
#
#             plot_list = ['Decision Mean', 'Score Mean', 'Step Mean', 'Start Edges Mean', 'TSP Edges Mean']
#             color_list = cmap(np.linspace(range_bottom, range_top, len(plot_list)))
#             # color_list = ['r','y','b','g','m']
#             plot_df.plot.bar(x='s',
#                              y=['Decision Sum','Score Sum','Step Sum','Start Edges Sum','TSP Edges Sum'],
#                              stacked=True,width=.75, color=color_list, edgecolor='None',figsize=figsize)
#                              # title='Analysis-{}, Layer-{}: Segment Influence Sum'.format(ana,l))#,edgecolor='None')
#             plt.ylabel('{} Centrality Sum, {} Layer'.format(ana_label[ana],layer_label[l]))
#             plt.xlabel('Time')
#             plt.legend(seg_labels,loc='best')
#             plt.xticks(np.arange(0, steps + 1,float(tick_interval)),range(0, steps + 1, tick_interval))
#             # print np.arange(0,steps+1,5.0)
#             if save:
#                 plt.savefig('{}\{}_{}_seg_sum.pdf'.format(save_dir, ana, l))
#             # If all score == 0 skip it
#
#             # print color_list
#             plot_df.plot(x='s',
#                          y=plot_list,#'Score Mean',
#                          color=color_list,
#                          linewidth=3,figsize=figsize)
#                          # title='Analysis-{}, Layer-{}: Segment Influence Mean'.format(ana, l))
#
#             plt.ylabel('{} Centrality Mean, {} Layer'.format(ana_label[ana], layer_label[l]))
#             plt.xlabel('Time')
#             plt.legend(seg_labels, loc='best')
#             # plt.xticks(np.arange(0, steps + 1, 5.0))
#
#             if ana=='out' and l=='a':
#                 plt.ylim(top=1.05)
#
#             plt.ylim(bottom=0.0)
#             for v in plot_list:
#                 color=color_list[plot_list.index(v)]
#                 plt.fill_between(plot_df['s'].values, plot_df[v]-conf_stat*plot_e[v],
#                                  plot_df[v]+conf_stat*plot_e[v],
#                                  facecolor=color, alpha=.5,
#                                  linewidth=0.0)
#
#             if save:
#                 plt.savefig('{}\{}_{}_seg_mean.pdf'.format(save_dir, ana, l))
#             # plt.fill_between(plot_df['s'].values, plot_df['Score Mean'] - conf_stat * plot_e['Score Mean'],
#             #                  plot_df['Score Mean'] + conf_stat * plot_e['Score Mean'], facecolor='blue', alpha=.2)
#             # plt.fill_between(plot_df['s'].values, plot_df['Step Mean'] - conf_stat * plot_e['Step Mean'],
#             #                  plot_df['Step Mean'] + conf_stat * plot_e['Score Mean'], facecolor='blue', alpha=.2)
#             # plt.fill_between(plot_df['s'].values, plot_df['Start Edges Mean'] - conf_stat * plot_e['Start Edges Mean'],
#             #                  plot_df['Score Mean'] + conf_stat * plot_e['Score Mean'], facecolor='blue', alpha=.2)
#
#             plot_list=['Start Edges Non-Solution Mean', 'Start Edges Solution Mean',
#                        'TSP Edges Solution Mean', 'TSP Edges Non-Solution Mean']
#
#
#             cmap = plt.cm.hsv
#             edge_range_bottom = 0.0
#             edge_range_top = 0.5
#
#             # edge_range_bottom=4.0*(range_top-range_bottom)/5.0+range_bottom
#             # edge_range_top=5.0*(range_top-range_bottom)/5.0+range_bottom
#             color_list = cmap(np.linspace(edge_range_bottom, edge_range_top, len(plot_list)))
#             # color_list = ['y', 'b', 'g', 'm']
#
#             plot_df.plot(x='s',
#                          y=plot_list,
#                          color=color_list,
#                          linewidth=3,
#                          kind='line',figsize=figsize)
#                          # title='Analysis-{}, Layer-{}: Edge Influence Mean'.format(ana, l))
#
#             plt.ylabel('{} Centrality Mean, {} Layer'.format(ana_label[ana], layer_label[l]))
#             plt.xlabel('Time')
#             plt.legend(sub_labels, loc='best')
#             # plt.xticks(np.arange(0, steps + 1, 5.0))
#             plt.ylim(bottom=0.0)
#             for v in plot_list:
#                 color = color_list[plot_list.index(v)]
#                 plt.fill_between(plot_df['s'].values, plot_df[v] - conf_stat * plot_e[v],
#                                  plot_df[v] + conf_stat * plot_e[v],
#                                  facecolor=color, alpha=.5,
#                                  linewidth=0.0)
#
#             if save:
#                 plt.savefig('{}\{}_{}_sub_mean.pdf'.format(save_dir, ana, l))
#
#             plot_df.plot.bar(x='s',
#                             y=['Start Edges Non-Solution Sum', 'Start Edges Solution Sum',
#                             'TSP Edges Solution Sum', 'TSP Edges Non-Solution Sum'],
#                             stacked=True,width=.75, color=color_list, edgecolor='None',figsize=figsize)
#                             # title='Analysis-{}, Layer-{}: Edge Influence Sum'.format(ana, l))#,edgecolor='None')
#             plt.ylabel('{} Centrality Sum, {} Layer'.format(ana_label[ana], layer_label[l]))
#             plt.xlabel('Time')
#             plt.legend(sub_labels, loc='best')
#             plt.xticks(np.arange(0, steps + 1, float(tick_interval)),range(0, steps + 1, tick_interval))
#             if save:
#                 plt.savefig('{}\{}_{}_sub_sum.pdf'.format(save_dir, ana, l))
#
#
#
# #Path error no decision
# path_0_offset = path_mean.loc[(path_mean['offset'] == 0) & (path_mean['tol'] == tol_nd) ]
#
# plt.figure(figsize=figsize)
# ax=plt.gca()
#
# cmap = plt.cm.hsv
# edge_range_bottom = 0.0
# edge_range_top = 0.75
#
# color_list = cmap(np.linspace(edge_range_bottom, edge_range_top, len(cent_o['centrality'].unique())))[2:]
# color_list=['b','g']
#
# for ana,color in zip(['hits-h','out'],color_list):
#     path_0_a = path_0_offset.loc[path_0_offset['centrality'] == ana]
#     line,=ax.plot(path_0_a['s'].values, path_0_a['Strongest Path Error'], linewidth=3.0, color=color, label=ana)
#     ax.fill_between(path_0_a['s'].values, path_0_a['Min Error'], path_0_a['Max Error'],
#                      facecolor=color, linewidth=0.0, alpha=.5) #line.get_color()
# plt.ylabel('Strongest Path Edge Error (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Strongest Path: Pheromone and Centrality Comparison')
# plt.legend(['HITS-Hub','Out Degree'],loc='best')
# if save:
#     plt.savefig('{}\error.pdf'.format(save_dir))
#
#
# #Path error with Decision
# path_0_offset = path_mean.loc[(path_mean['offset'] == 0) & (path_mean['tol'] == tol_d) ]
#
# plt.figure(figsize=figsize)
# ax=plt.gca()
#
# cmap = plt.cm.hsv
# edge_range_bottom = 0.0
# edge_range_top = 0.75
#
# color_list = cmap(np.linspace(edge_range_bottom, edge_range_top, len(cent_o['centrality'].unique())))[2:]
# color_list=['b','g']
#
# for ana,color in zip(['hits-h','out'],color_list):
#     path_0_a = path_0_offset.loc[path_0_offset['centrality'] == ana]
#     line,=ax.plot(path_0_a['s'].values, path_0_a['Strongest Path Error'], linewidth=3.0, color=color, label=ana)
#     ax.fill_between(path_0_a['s'].values, path_0_a['Min Error'], path_0_a['Max Error'],
#                      facecolor=color, linewidth=0.0, alpha=.5) #line.get_color()
#
# if dec_line_on:
#     plt.axvline(x=dec_time,color=dc, linewidth=2, label=dl)
#
# plt.ylabel('Strongest Path Edge Error (with 95% CI)')
# plt.xlabel('Time')
# plt.title('Strongest Path: Pheromone and Centrality Comparison with Decision')
# plt.legend(['HITS-Hub','Out Degree',dl],loc='best')
# if save:
#     plt.savefig('{}\error_d.pdf'.format(save_dir))
#


plt.show()