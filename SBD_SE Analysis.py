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

#LOADING
import os
import cPickle as pickle

from SE_Agent import TSP_1diss as TSP_SE
from SBD_Agent import TSP_1diss as TSP_SBD


def condition_comparison(df, analysis='out', layer='k', value='Start Edges Mean',
                         save=False, save_title='test',figsize=(10,6),
                         save_dir='C:\Users\colinsh\Documents\DesignSim'):


    p_data = df.loc[(df['centrality'] == analysis) & (df['layer'] == layer)].reset_index(drop=True).fillna(0.0)

    # print p_data
    f, a = plt.subplots(1, 1, figsize=figsize)
    sns.tsplot(time='Time', value=value, condition='label', unit='run', data=p_data, ci=95, ax=a)

    if save:
        plt.savefig('{}\{}.pdf'.format(save_dir, save_title))
#Get df's
#load dataframes
dir='C:\Users\colinsh\Documents\DesignSim'
exp='test'
cent_o=pd.read_pickle('{}\{}\{}.pkl'.format(dir,exp,'cent_df'))
score_df=pd.read_pickle('{}\{}\{}.pkl'.format(dir,exp,'score_df'))

#load TSPs
load_TSP=False
if load_TSP:
    net_dir='{}\{}\{}'.format(dir,exp,'nets')
    tsp_list=[]
    for filename in os.listdir(net_dir):
        # print filename
        p_path='{}\{}'.format(net_dir,filename)
        tsp=pickle.load(open(p_path,'rb'))
        tsp_list.append(tsp)

    # print tsp_list

figsize=(10,6)

f,a = plt.subplots(1,1,figsize=figsize)
sns.tsplot(time='Time',value='Score', condition='label', unit='run', data=score_df, ci=95, ax=a)

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

plt.show()