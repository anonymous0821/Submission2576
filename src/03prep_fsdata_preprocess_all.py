#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 07:12:57 2019
@author: 

Changed 2020/10/04 #1004:
    filter out net revinue and title account (with no value).

Python 3.6
"""

# %% ##########################################################################
# requirements
###############################################################################
import joblib

import pandas as pd
import numpy as np

# %% ##########################################################################
# params
###############################################################################
import argparse
parser = argparse.ArgumentParser(description='')


runkey="_1"
runkey_org="_1"
parser.add_argument('--fs_normalization', default="NetAsset", type=str, help='NetAsset(d), FrobeniusNorm, binalize, hierarchy')
parser.add_argument('--drop_general_account', default=r"../data/metadata/Yuho_dummy_account_list.csv", type=str, help='nan or file path')
parser.add_argument('--fs_nan_treatment', default="zero", type=str, help='zero(d), nan')
parser.add_argument('--scratch', default=r"../results/run_1", type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input', default=r"../results/run"+runkey_org+"/proc_data/out02_diff_data_all.pkl.cmp", type=str, help='input file path')
parser.add_argument('--input_mask', default=r"../results/run"+runkey+"/proc_data/out021_mask_all.csv", type=str, help='input file path 1903')
parser.add_argument('--account_summerized', default=True, type=bool, help='use exposured account name as label')
parser.add_argument('--PLdiff', default=True, type=bool, help='True; use PL difference')
parser.add_argument('--cmax', default=10, type=float, help='Clipped threshold')
parser.add_argument('--input_mask_train', default=r"../results/run_1/proc_data/out021_mask_train.csv", type=str, help='input file path')
parser.add_argument('--input_mask_test', default=r"../results/run_1/proc_data/out021_mask_test.csv", type=str, help='input file path')


args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch
import warnings
warnings.simplefilter('ignore')

# %% ##########################################################################
# data load
###############################################################################
filename=args.input
results = joblib.load(filename)

fsdata=pd.concat(results,axis=1,sort=False)

mask_all=pd.read_csv(args.input_mask,index_col=0,header=None).values[:,0]
fsdata=fsdata.loc[:,mask_all].copy()

# %% ##########################################################################
# data load 
# TODO: rerighted into function
###############################################################################

sheet_BS=pd.DataFrame()
sheet_PL=pd.DataFrame()

    
y_list=[2019,2018,2017,2016,2015]


for year in y_list:
    filename=PROJDIR+"/data/metadata/"+str(year)+"account_list.xls"
    book=pd.ExcelFile(filename)
    for itr in range(2,len(book.sheet_names)-1):
        sheet_t=book.parse(sheet_name=itr,header=1,index_col=1)
        
        sheet_doc_name_row=book.parse(sheet_name=itr,header=1,index_col=0)
        
        mask=sheet_doc_name_row.index.str.extract('(.+科目一覧)').isna()
        startpoint=np.where(~mask)[0]
        
        sheet_t["BSkey"]=False
        sheet_t["BSkey"].iloc[:startpoint[0]-1]=True
        mask=sheet_t["BSkey"]==True
        sheet_BS_t=sheet_t.loc[mask,:].copy()
        
        sheet_t["PLkey"]=False
        sheet_t["PLkey"].iloc[startpoint[0]:startpoint[1]-1]=True
        mask=sheet_t["PLkey"]==True
        sheet_PL_t=sheet_t.loc[mask,:].copy()
        
        sheet_BS=pd.concat([sheet_BS,sheet_BS_t],axis=0)
        sheet_PL=pd.concat([sheet_PL,sheet_PL_t],axis=0)
# %% convert duplicated account name

if args.account_summerized:
    sheet_BS[sheet_BS.index.name]=sheet_BS.index
    sheet_BS.loc[sheet_BS.index=='貸倒引当金',"標準ラベル（日本語）"]=sheet_BS.loc[sheet_BS.index=='貸倒引当金',"冗長ラベル（日本語）"].copy()
    sheet_BS.loc[sheet_BS.index=='その他',"標準ラベル（日本語）"]=sheet_BS.loc[sheet_BS.index=='その他',"冗長ラベル（日本語）"].copy()
    sheet_BS.loc[sheet_BS.index=='減価償却累計額',"標準ラベル（日本語）"]=sheet_BS.loc[sheet_BS.index=='減価償却累計額',"冗長ラベル（日本語）"].copy()
    
    sheet_BS.loc[sheet_BS.index=='貸倒引当金',"標準ラベル（英語）"]=sheet_BS.loc[sheet_BS.index=='貸倒引当金',"冗長ラベル（英語）"].copy()
    sheet_BS.loc[sheet_BS.index=='その他',"標準ラベル（英語）"]=sheet_BS.loc[sheet_BS.index=='その他',"冗長ラベル（英語）"].copy()
    sheet_BS.loc[sheet_BS.index=='減価償却累計額',"標準ラベル（英語）"]=sheet_BS.loc[sheet_BS.index=='減価償却累計額',"冗長ラベル（英語）"].copy()

    filename='../data/metadata/OneYearAccounts.csv'
    OneYearAccounts=pd.read_csv(filename,header=0)
    for i in range(len(OneYearAccounts)):
        sheet_BS.loc[sheet_BS['標準ラベル（日本語）']==OneYearAccounts.loc[i,'標準ラベル（日本語）'],'標準ラベル（日本語）']=OneYearAccounts.loc[i,'標準ラベル（日本語）変更後']
    # Japanese        
    sheet_PL[sheet_PL.index.name]=sheet_PL.index
    sheet_PL.loc[sheet_PL.index=='その他',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='その他',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='合計',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='合計',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='小計',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='小計',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='差引',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='差引',"冗長ラベル（日本語）"].copy()
    ## Name duplicated
    sheet_PL.loc[sheet_PL.index=='その他有価証券評価差額金',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='その他有価証券評価差額金',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='土地再評価差額金',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='土地再評価差額金',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='株式交付費',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='株式交付費',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='為替換算調整勘定',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='為替換算調整勘定',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='社債発行費',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='社債発行費',"冗長ラベル（日本語）"].copy()
    sheet_PL.loc[sheet_PL.index=='繰延ヘッジ損益',"標準ラベル（日本語）"]=sheet_PL.loc[sheet_PL.index=='繰延ヘッジ損益',"冗長ラベル（日本語）"].copy()

    # English
    sheet_PL.loc[sheet_PL.loc[:,"標準ラベル（英語）"]=='Other',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.loc[:,"標準ラベル（英語）"]=='Other',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.loc[:,"標準ラベル（英語）"]=='Options',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.loc[:,"標準ラベル（英語）"]=='Options',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='合計',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='合計',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='小計',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='小計',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='差引',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='差引',"冗長ラベル（英語）"].copy()
    ## Name duplicated
    sheet_PL.loc[sheet_PL.index=='その他有価証券評価差額金',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='その他有価証券評価差額金',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='土地再評価差額金',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='土地再評価差額金',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='株式交付費',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='株式交付費',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='為替換算調整勘定',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='為替換算調整勘定',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='社債発行費',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='社債発行費',"冗長ラベル（英語）"].copy()
    sheet_PL.loc[sheet_PL.index=='繰延ヘッジ損益',"標準ラベル（英語）"]=sheet_PL.loc[sheet_PL.index=='繰延ヘッジ損益',"冗長ラベル（英語）"].copy()
    
    
# %% drop the account that is not necessary

if args.drop_general_account!="nan":
    filename=args.drop_general_account
    drop_account=pd.read_csv(filename,header=None)
    dropwords=drop_account[0].to_list()
    ## 1004
    mask=~sheet_BS.index.isna()
    sheet_BS=sheet_BS.loc[mask,:].copy()
    mask_BS0=~sheet_BS.index.isin(dropwords)
    mask_BS1=~sheet_BS.loc[:,'冗長ラベル（日本語）'].str.contains('タイトル')
    mask_BS=mask_BS0 & mask_BS1
    
    mask=~sheet_PL.index.isna()
    sheet_PL=sheet_PL.loc[mask,:].copy()
    
    mask_PL0=~sheet_PL.index.isin(dropwords)
    mask_PL1=~sheet_PL.index.str.contains('（△）')# drop net revinue
    mask_PL2=~sheet_PL.loc[:,'冗長ラベル（日本語）'].str.contains('タイトル')
    mask_PL3=~sheet_PL.loc[:,'冗長ラベル（日本語）'].str.contains('その他の包括利益')
    mask_PL=mask_PL0 & mask_PL1 & mask_PL2 & mask_PL3
    ## 1004
    transcripter_BS=sheet_BS.loc[mask_BS,["要素名","balance","冗長ラベル（日本語）","標準ラベル（日本語）","冗長ラベル（英語）","標準ラベル（英語）"]].groupby("要素名").first()
    transcripter_PL=sheet_PL.loc[mask_PL,["要素名","balance","冗長ラベル（日本語）","標準ラベル（日本語）","冗長ラベル（英語）","標準ラベル（英語）"]].groupby("要素名").first()

else:
    transcripter_BS=sheet_BS.loc[:,["要素名","balance","冗長ラベル（日本語）","標準ラベル（日本語）","冗長ラベル（英語）","標準ラベル（英語）"]].groupby("要素名").first()
    transcripter_PL=sheet_PL.loc[:,["要素名","balance","冗長ラベル（日本語）","標準ラベル（日本語）","冗長ラベル（英語）","標準ラベル（英語）"]].groupby("要素名").first()
    

if args.PLdiff:
    transcripter_PL.index=transcripter_PL.index.values
else:
    transcripter_PL.index=transcripter_PL.index.values+"_CYvalue"

transcripter=pd.concat([transcripter_BS,transcripter_PL],axis=0)

if args.account_summerized:
    
    transcripter_summerized_account=transcripter.groupby('標準ラベル（日本語）').first()
    transcripter_summerized_account[transcripter_summerized_account.index.name]=transcripter_summerized_account.index
    transcripter_minus=transcripter_summerized_account.copy()+" "
    transcripter_minus.index=transcripter_summerized_account.index.values+"_down"
    transcripter_minus["color"]="cyan"
    
    transcripter_plus=" "+transcripter_summerized_account.copy()
    transcripter_plus.index=transcripter_summerized_account.index.values+"_up"
    transcripter_plus["color"]="tomato"
    
    transcripter=pd.concat([transcripter_plus,transcripter_minus],axis=0)
    transcripter.to_csv(args.scratch+"/transcripter_summerized_account.csv")
else:
    
    transcripter_minus=transcripter.copy()+" "
    transcripter_minus.index=transcripter.index.values+"_down"
    transcripter_minus["color"]="cyan"
    transcripter_plus=" "+transcripter.copy()
    transcripter_plus.index=transcripter.index.values+"_up"
    transcripter_plus["color"]="tomato"
    transcripter=pd.concat([transcripter_plus,transcripter_minus],axis=0)
    transcripter.to_csv(args.scratch+"/transcripter.csv")


# %% ##########################################################################
# Extract BS and PL value
###############################################################################



mask_BS=fsdata.index.isin(transcripter_BS.index.values)
mask_PL=fsdata.index.isin(transcripter_PL.index.values)

if args.account_summerized:
    fsdata_tmp=fsdata.loc[mask_BS,:].copy()
    fsdata_tmp['key']=transcripter_BS.loc[fsdata_tmp.index.values,"標準ラベル（日本語）"]
    fsdata_BS_diff=fsdata_tmp.groupby('key').agg(np.sum).transpose()
    fsdata_tmp=fsdata.loc[mask_PL,:].copy()
    fsdata_tmp['key']=transcripter_PL.loc[fsdata_tmp.index.values,"標準ラベル（日本語）"]
    fsdata_PL=fsdata_tmp.groupby('key').agg(np.sum).transpose()
    
else:
    fsdata_BS_diff=fsdata.loc[mask,:].transpose().copy()
    fsdata_PL=fsdata.loc[mask,:].transpose().copy()

fsdata_NetAssets=fsdata.loc["NetAssets_CYvalue",:].copy()


# %% ##########################################################################
# fsdata normalization then flip for BS down
###############################################################################


fsdata_diff_minus=fsdata_BS_diff.copy()*-1
fsdata_diff_minus.columns=fsdata_diff_minus.columns+"_down"
fsdata_diff_minus=fsdata_diff_minus[fsdata_diff_minus>0].copy()

fsdata_diff_plus=fsdata_BS_diff.copy()
fsdata_diff_plus.columns=fsdata_diff_plus.columns+"_up"
fsdata_diff_plus=fsdata_diff_plus[fsdata_diff_plus>0].copy()

fsdata_PL_minus=fsdata_PL.copy()*-1
fsdata_PL_minus.columns=fsdata_PL_minus.columns+"_down"
fsdata_PL_minus=fsdata_PL_minus[fsdata_PL_minus>0].copy()

fsdata_PL_plus=fsdata_PL.copy()
fsdata_PL_plus.columns=fsdata_PL_plus.columns+"_up"
fsdata_PL_plus=fsdata_PL_plus[fsdata_PL_plus>0].copy()


fsdata_out=pd.concat([fsdata_diff_plus,fsdata_diff_minus,fsdata_PL_plus,fsdata_PL_minus],axis=1)

# %% plane
fsdata_out.to_pickle(SCRATCH + "/proc_data/out03_fsdata_plane.pkl")

# %% 1. Normalized by Current Net Asset
fsdata_out=fsdata_out.div(fsdata_NetAssets.abs(),axis=0).copy()
#fsdata_out.to_pickle(SCRATCH + "/proc_data/out03_fsdata_prep1.pkl")


# %% 2. clipping outlier
def ClippedNormalization(x, c_min, c_max, axis = None):
    x_clipped=(np.clip(x,c_min,c_max)-c_min)/(c_max-c_min)
    return x_clipped

fsdata_normed=fsdata_out.apply(ClippedNormalization,axis=0, c_min=0, c_max=args.cmax).copy()
#fsdata_normed.to_pickle(SCRATCH + "/proc_data/out03_fsdata_prep2.pkl")


# %% 3. Normalized by Current Net Asset, and by average
fsdata_mean=fsdata_normed.mean(axis=0,skipna=True)
fsdata_normed=fsdata_normed.loc[:,fsdata_mean!=0]/fsdata_mean[fsdata_mean!=0]
#fsdata_normed.to_pickle(SCRATCH + "/proc_data/out03_fsdata_prep3.pkl")

# %% 4. neglog transform
def neglog(x):
    return np.sign(x)*np.log1p(x+1)
fsdata_normed=fsdata_normed.fillna(0).apply(neglog,axis=0)
fsdata_normed.to_pickle(SCRATCH + "/proc_data/out03_fsdata_prep4.pkl")

# load training and test data slabel
mask_train=pd.read_csv(args.input_mask_train,index_col=0,header=None).values[:,0]
mask_test=pd.read_csv(args.input_mask_test,index_col=0,header=None).values[:,0]
# divided into the data (thus, training and test data are through the same scaling preprocesses)
mask_train=mask_train[mask_all]
mask_test=mask_test[mask_all]
fsdata_normed.loc[mask_train].to_pickle(SCRATCH + "/proc_data/out03_fsdata_prep4_train.pkl")
fsdata_normed.loc[mask_test].to_pickle(SCRATCH + "/proc_data/out03_fsdata_prep4_test.pkl")

