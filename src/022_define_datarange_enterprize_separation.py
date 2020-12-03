#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:51:29 2020

@author: 
"""


# %% ##########################################################################
# Requirements
###############################################################################


import pandas as pd
import numpy as np
import sys

import joblib
import datetime

# %% ##########################################################################
# args
###############################################################################

import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--scratch', default=r"../results/run_61", type=str, help='results dir')
parser.add_argument('--base_scratch', default=r"../results/run_61", type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input', default=r"../results/run_1/proc_data/out02_diff_data_all.pkl.cmp", type=str, help='input file path')
args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch


# %% ##########################################################################
# load
###############################################################################


responce_data_masked=pd.read_pickle(args.base_scratch+"/proc_data/out02_docIDinfo.pkl")


filename=PROJDIR+"../metadata/EdinetcodeDlInfo1903.csv" 

edinetcode=pd.read_csv(filename,header=1,index_col=False, engine="python", encoding="cp932")
mask=edinetcode.loc[:,'上場区分']=='上場'
edinetcode_filtered=edinetcode.loc[mask,:].copy()

filename=args.input
results = joblib.load(filename)
fsdata=pd.concat(results,axis=1,sort=False)
results=[]

# %% ##########################################################################
# masking out non JGAAP or non-consolidated
###############################################################################

mask_JGAAP=fsdata.loc['AccountingStandards_DEI',:]=='Japan GAAP'
mask_consolidated=fsdata.loc['WhetherConsolidatedFinancialStatementsArePrepared_DEI',:]==True
# %% ##########################################################################
# Year filter
###############################################################################

docID2info=responce_data_masked.copy()

EDNETCODE2industry=edinetcode_filtered.set_index('ＥＤＩＮＥＴコード')
docID2info['industry']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"提出者業種"].values
docID2info['companyName']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"提出者名"].values
docID2info['termDate']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"決算日"].values
docID2info['SECcode']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"証券コード"].values.astype(int)


termindex=[]
for y in range(2010,2021):
    for m in range(12):
        termindex.append(y*100+m+1)

def fnc_date_formatted(x):
    x_date_format=datetime.datetime.strptime(x,"%Y-%m-%d")
    return x_date_format.year*100+x_date_format.month

score_t=pd.DataFrame()
score_t['periodStart']=docID2info['periodStart'].map(fnc_date_formatted)
score_t['periodEnd']=docID2info['periodEnd'].map(fnc_date_formatted)

mask_gt_201503=score_t['periodEnd']>=201503
mask_lt_202003=score_t['periodEnd']<202003
mask_eq_202003=score_t['periodEnd']==202003

mask_le_202003=score_t['periodEnd']<=202003

# %% ##########################################################################
# make mask to divide into train and test data
###############################################################################
mask_all=mask_JGAAP & mask_consolidated & mask_gt_201503 & mask_le_202003

companies_pool=docID2info.loc[mask_all,'secCode'].value_counts()
companies_shuffle=companies_pool.sample(frac=1,random_state=0)

test_comp=companies_shuffle.iloc[:round(len(companies_shuffle)/8)-1].index.to_list()
train_comp=companies_shuffle.iloc[round(len(companies_shuffle)/8):].index.to_list()

mask_test=docID2info.secCode.isin(test_comp)
mask_train=docID2info.secCode.isin(train_comp)

mask_train=mask_all & mask_train
mask_test=mask_all & mask_test
print("all data size: ",mask_all.sum())
print("train data size: ",mask_train.sum())
print("test data size: ",mask_test.sum())


mask_all.to_csv(SCRATCH + "/proc_data/out021_mask_all.csv")
mask_train.to_csv(SCRATCH + "/proc_data/out021_mask_train.csv")
mask_test.to_csv(SCRATCH + "/proc_data/out021_mask_test.csv")


