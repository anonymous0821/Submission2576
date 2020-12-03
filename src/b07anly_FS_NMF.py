#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 00:26:24 2019

@author: 




"""
# %% rec
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import NMF
# %% parms
import argparse
parser = argparse.ArgumentParser(description='')


parser.add_argument('--scratch', default=r"../results/run_1", type=str, help='results dir')
parser.add_argument('--base_scratch', default=r"../results/run_1", type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input', type=str, help='input file path')
parser.add_argument('--input_test',  type=str, help='input file path')
parser.add_argument('--dim', default=50, type=int, help='NMF dimension')
parser.add_argument('--alpha', default=0, type=float, help='NMF preciation')
parser.add_argument('--nmfinit', default='nndsvda', type=str, help='NMF initialization methods')
parser.add_argument('--nmfitr', default=10, type=int, help='NMF initialization methods')
parser.add_argument('--account_summerized', default=True, type=bool, help='use exposured account name as label')

args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch

# %% data 
fsdata=pd.read_pickle(args.input)
fsdata_test=pd.read_pickle(args.input_test)
# %% transcripter

transcripter=pd.read_csv(args.base_scratch+"/transcripter_summerized_account.csv",header=0, index_col=0)

    
initseed=42
# %% NMF
nmf = NMF(n_components=args.dim, init=args.nmfinit, random_state=initseed,alpha=args.alpha, l1_ratio=0)
fsdata_dim_reduced_nmf = pd.DataFrame(nmf.fit_transform(fsdata.fillna(0)))
nmf_comp = pd.DataFrame(nmf.components_.transpose())
if args.account_summerized:
    nmf_comp.index=fsdata.columns
else:
    nmf_comp.index=transcripter.loc[fsdata.columns,"冗長ラベル（日本語）"]
    

fsdata_dim_reduced_nmf.to_csv(SCRATCH + "/proc_data/out07_NMFscore.csv",index=False,header=False)
nmf_comp.to_csv(SCRATCH + "/proc_data/out07_NMFcomp.csv",index=False,header=False)


fsdata_test_dim_reduced_nmf = pd.DataFrame(nmf.transform(fsdata_test.fillna(0)))
fsdata_test_dim_reduced_nmf.to_csv(SCRATCH + "/proc_data/out07_NMFscore_test.csv",index=False,header=False)


# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
coler_increase='tomato'
coler_decrease='darkblue'

bar_num=10
for num in range(args.dim):
    fig=plt.figure(figsize=(4,3))
    ax=fig.add_subplot(1,2,2)
    tmp=nmf_comp.loc[:,num].copy()
    tmp=tmp.sort_values(ascending=False)
    tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    col_mask=tmp.index.str.contains("up")
    col=pd.Series([coler_increase,]*bar_num)
    col[~col_mask]=coler_decrease
    #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（日本語）"].str.strip()
    #tmp.index.str.strip()
    
    
    ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    plt.yticks(range(bar_num), tmp.index.to_list())
    ax.grid(True)
    fig.suptitle(" #" + str(num))
    fig.savefig(SCRATCH + "/NMF_JPN_barh/comp_" + str(num) + ".pdf", transparent=True)
    
    
bar_num=10
for num in range(args.dim):
    fig=plt.figure(figsize=(12,3))
    ax=fig.add_subplot(1,4,4)
    tmp=nmf_comp.loc[:,num].copy()
    tmp=tmp.sort_values(ascending=False)
    tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    col_mask=tmp.index.str.contains("up")
    col=pd.Series([coler_increase,]*bar_num)
    col[~col_mask]=coler_decrease
    #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（英語）"].str.strip()
    #tmp.index.str.strip()
    
    
    ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    plt.yticks(range(bar_num), tmp.index.to_list())
    ax.grid(True)
    fig.suptitle(" #" + str(num))
    fig.savefig(SCRATCH + "/NMF_ENG_barh/comp_" + str(num) + ".pdf", transparent=True)
