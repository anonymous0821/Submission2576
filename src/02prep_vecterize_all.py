#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:35:31 2020

@author: 

We used parser of xbrl available below (Japanese site only). 
https://srbrnote.work/archives/1430


"""
# %% ##########################################################################
# requirements
###############################################################################

import re
import unicodedata
import string
import pandas as pd
import numpy as np
import sys
from bs4 import BeautifulSoup
import os
import warnings
warnings.filterwarnings('ignore')
import workdays
from dateutil.relativedelta import relativedelta
from time import sleep
import datetime
import joblib
import os
from tqdm import tqdm

# We used parser of xbrl available below (Japanese site only). 
# https://srbrnote.work/archives/1430
sys.path.append(r'../Shiraberu')
from xbrl_proc import read_xbrl_from_zip

# %% ##########################################################################
# Params
###############################################################################

import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--scratch', default=r"../results/run_1", type=str, help='results dir')
parser.add_argument('--cwd',  type=str, help='base dir')
parser.add_argument('--gyoushu', default="", type=str, help='Filtered into gyoushu JPN')
parser.add_argument('--n_parallel', default=2, type=int, help='run parallelly n jobs')



args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch



# %% ##########################################################################
# Gathering the downloaded metadata and documents
###############################################################################

filename=PROJDIR+"../metadata/EdinetcodeDlInfo1903.csv" 
edinetcode=pd.read_csv(filename,header=1,index_col=False, engine="python", encoding="cp932")

filename=PROJDIR+"../metadata/doclist.txt"
docid1803=pd.read_csv(filename,sep='\t',header=None).values

filename=PROJDIR+"../metadata/doclist3.txt"
docid1903=pd.read_csv(filename,sep='\t',header=None).values

filename=PROJDIR+"../metadata/doclist0904.txt"
docid2003=pd.read_csv(filename,sep='\t',header=None).values


# 2015-1803
filename=PROJDIR+"../metadata/EDINET_responce.csv"
responce_data1803=pd.read_csv(filename,encoding='utf-8')

# 1903
filename=PROJDIR+"../metadata/EDINET_responce_1903.csv"
responce_data1903=pd.read_csv(filename,encoding='utf-8')

edinet_res_file=PROJDIR+'../metadata/EDINET_responce_0621.csv'
responce_data1903_0621=pd.read_csv(edinet_res_file,encoding='utf-8')
# 2020
edinet_res_file=PROJDIR+'../metadata/EDINET_responce_20200904.csv'
responce_data_20200904=pd.read_csv(edinet_res_file,encoding='utf-8')

responce_data=pd.concat([responce_data1803,responce_data1903,responce_data1903_0621,responce_data_20200904],sort=False)


print('record size: ', len(responce_data), 'columns: ', responce_data.shape[1])
# %% ##########################################################################
# Filter the documents
###############################################################################


mask=edinetcode.loc[:,'上場区分']=='上場'

edinetcode_filtered=edinetcode.loc[mask,:].copy()
classification=edinetcode_filtered.loc[:,'提出者業種'].values
print(np.unique(classification))
edinetcodelist=edinetcode_filtered.loc[:,'ＥＤＩＮＥＴコード'].values
print(edinetcodelist.shape)

docid=np.concatenate([docid1803,docid1903,docid2003])
print(docid.shape)


print("All data: ",len(responce_data))
mask=(responce_data.loc[:,'docTypeCode']==120).values # # only annual reports
print("Annual reports: ",mask.sum())
mask=(responce_data.loc[:,'edinetCode'].isin(edinetcodelist)).values * mask # exceptong unlisted companies
print("Annual reports by listed companies: ",mask.sum())
mask=(responce_data.loc[:,'docID'].isin(docid[:,0])).values * mask # excepting download error
print("Annual reports by listed companies that were sucsessfully downloaded: ",mask.sum())
responce_data_masked=responce_data.loc[mask,:]
print("duplicated file (download date double-covered): ",responce_data_masked['docID'].duplicated().sum())
mask=~responce_data_masked['docID'].duplicated(keep='first')
responce_data_masked=responce_data_masked.loc[mask,:].copy()
print("after drop duplicated",len(responce_data_masked))
responce_data_masked=responce_data_masked.drop(axis=0,index=301940).copy() # the file is blocken
print("finaly",len(responce_data_masked))



docID2info=responce_data_masked.set_index('docID')

EDNETCODE2industry=edinetcode_filtered.set_index('ＥＤＩＮＥＴコード')
docID2info['industry']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"提出者業種"].values
docID2info['companyName']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"提出者名"].values
docID2info['termDate']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"決算日"].values
docID2info['SECcode']=EDNETCODE2industry.loc[docID2info['edinetCode'].values,"証券コード"].values.astype(int)

# save the information of filtered documnents
docID2info.to_pickle(SCRATCH+"/proc_data/out02_docIDinfo.pkl")

# %% ##########################################################################
# Define the functions
###############################################################################


def formatXBRL_texts(df_list):
    """
    Extracting text in annual reports by the list of taxonomi of textual information, "Yuho_textblocks".
    The list was generated by copying the cell including [text block] in "taxonomi.xles"
    """
    filename=PROJDIR+"/data/metadata/Yuho_textblocks.xlsx"
    
    tagdata=df_list[0].copy()
    tagdata=tagdata.set_index('tag', drop=True)
    book_textblock=pd.ExcelFile(filename)
    sheet=book_textblock.parse(sheet_name=0,header=0,index_col=0)
    textblocks=tagdata.loc[sheet["標準ラベル（英語）"],:]
    textblocks=textblocks[textblocks["値"].isna()==False].copy()
    texts=textblocks["値"].copy()
    return texts

def formatXBRL_CY(df_list):
    """
    Extracting the current year of the documnents and consolidated account
    """
    
    mask=((df_list[0].loc[:,'終了日']==df_list[0].loc[1,'報告対象期間期末日']))
    mask=((df_list[0].loc[:,'期末日']==df_list[0].loc[1,'報告対象期間期末日']))|mask
    # not included no-consolidated values
    mask=~(df_list[0].loc[:,'context'].str.contains('NonConsolidated'))&mask
    # CurrentYear
    mask2=df_list[0].loc[:,'context']=='CurrentYearInstant'
    mask2=(df_list[0].loc[:,'context']=='CurrentYearDuration')|mask2
    
    mask=mask2&mask
    
    tagdata=df_list[0].loc[mask,:]
    tagdata=tagdata.set_index('tag', drop=True)
    return tagdata

def formatXBRL_PY(df_list):
     """
    Extracting the previous year of the documnents and consolidated account
    """
    
    mask=(df_list[0].loc[:,'終了日']==df_list[0].loc[1,'報告対象期間期末日']-relativedelta(years=1))
    mask=((df_list[0].loc[:,'期末日']==df_list[0].loc[1,'報告対象期間期末日']-relativedelta(years=1)))|mask
    
    # not included no-consolidated values
    mask=~(df_list[0].loc[:,'context'].str.contains('NonConsolidated'))&mask
    
    # Prior1Year
    mask2=df_list[0].loc[:,'context']=='Prior1YearInstant'
    mask2=(df_list[0].loc[:,'context']=='Prior1YearDuration')|mask2
    mask=mask2&mask
    tagdata=df_list[0].loc[mask,:]
    tagdata=tagdata.set_index('tag', drop=True)
    return tagdata

def series_dtype(x):
    return type(x)

def GetDummyNames():
    """
    version of taxonomi in this analysis is 20190228.
    For new version, you should correct partition label; sheet_name, startpoint and endpoint.
    
    """
    filename=PROJDIR+"/data/metadata/taxonomi.xls"
    book_tax=pd.ExcelFile(filename)
    sheet=book_tax.parse(sheet_name="9",header=1,index_col=1)
    startpoint=sheet.index.get_loc("250000 経理の状況 ( jpcrp_000500-000_2019-02-28_pre.xml )")
    endpoint=sheet.index.get_loc("400010 連結財務諸表作成のための基本となる重要な事項 ( jpcrp_000600-000_2019-02-28_pre.xml )")
    sheet["key"]=False
    sheet["key"].iloc[startpoint:endpoint]=True
    mask=sheet["key"]==True
    sheet_not_KeiriState=sheet.loc[~mask,:].copy()
    mask=sheet_not_KeiriState["要素名"].apply(series_dtype)==str
    dummy_names=sheet_not_KeiriState.loc[mask,"要素名"].values
    return dummy_names

def AccountFilter(df,dummy):
    """
    Taxonomi being not accout are dropped
    
    """
    mask=df.index.isin(dummy)
    mask=(df.index.str.contains('NumberOf'))|mask
    mask=(df.index.str.contains('BusinessResults'))|mask
    return df.loc[~mask,:].groupby(level=0).first()
    
def vecterize_TB(doc):
    sleep(3)
    warnings.filterwarnings('ignore')

    dummy=GetDummyNames()
    
    try:
        
        zip_file = r'../data/testdata/'+doc+'.zip'
        if os.path.exists(zip_file):
            df_list = read_xbrl_from_zip(zip_file)
        
        zip_file = r'../data/testdata2003/'+doc+'.zip'
        if os.path.exists(zip_file):
            df_list = read_xbrl_from_zip(zip_file)
        
        zip_file = r'../data/Docs/'+doc+'.zip'
        if os.path.exists(zip_file):
            df_list = read_xbrl_from_zip(zip_file)
            
        
        texts=formatXBRL_texts(df_list)
        rcd_PY=formatXBRL_PY(df_list)
        rcd_PY["type"]=rcd_PY.loc[:,"値"].apply(series_dtype)
        
        rcd_PY_int=rcd_PY.loc[rcd_PY.loc[:,"type"]==int,:]
        rcd_CY=formatXBRL_CY(df_list)
        rcd_CY["type"]=rcd_CY.loc[:,"値"].apply(series_dtype)
        
        rcd_CY_int=rcd_CY.loc[rcd_CY.loc[:,"type"]==int,:]
        
        rcd_CY_int_filtered=AccountFilter(rcd_CY_int,dummy)
        rcd_PY_int_filtered=AccountFilter(rcd_PY_int,dummy)
        diff_vec=rcd_CY_int_filtered["値"].subtract(rcd_PY_int_filtered["値"],fill_value=0)
        rcd_CY_int_filtered.index=rcd_CY_int_filtered.index+"_CYvalue"
        rcd=pd.concat([diff_vec,rcd_CY_int_filtered["値"],texts],axis=0)
    except:
        rcd=pd.Series()
    
    rcd["id"]=doc
    try:
        
        rcd['AmendmentFlag_DEI']=df_list[0].query('tag == "AmendmentFlagDEI" ')['値'].values[0]
        rcd['AccountingStandards_DEI']=df_list[0].query('tag == "AccountingStandardsDEI" ')['値'].values[0]
        rcd['WhetherConsolidatedFinancialStatementsArePrepared_DEI']=df_list[0].query('tag == "WhetherConsolidatedFinancialStatementsArePreparedDEI" ')['値'].values[0]
        rcd['IndustryCodeWhenConsolidatedFinancialStatementsArePreparedInAccordanceWithIndustrySpecificRegulations_DEI']=df_list[0].query('tag == "IndustryCodeWhenConsolidatedFinancialStatementsArePreparedInAccordanceWithIndustrySpecificRegulationsDEI" ')['値'].values[0]
        rcd['IdentificationOfDocumentSubjectToAmendment_DEI']=df_list[0].query('tag == "IdentificationOfDocumentSubjectToAmendmentDEI" ')['値'].values[0]
        rcd['ReportAmendmentFlag_DEI']=df_list[0].query('tag == "ReportAmendmentFlagDEI" ')['値'].values[0]
        rcd['XBRLAmendmentFlag_DEI']=df_list[0].query('tag == "XBRLAmendmentFlagDEI" ')['値'].values[0]        
    except:        
        pass

    return rcd



# %% ##########################################################################
# run 
###############################################################################
    
doclist=responce_data_masked.loc[:,'docID'].values
results = joblib.Parallel(n_jobs=2,verbose=2)(
            [
                joblib.delayed(vecterize_TB)(doclist[itr])
                for itr in range(len(doclist))
            ]
        )


# %% ##########################################################################
# save
###############################################################################

def set_name(series):
    if series is None:
        return series
    else:
        series.name=series['id']
        return series

results2=list(map(set_name,results))

out_filename=SCRATCH + "/proc_data/out02_diff_data_all.pkl.cmp"
joblib.dump(results2, out_filename, compress=True)

fsdata=pd.concat(results2,axis=1,sort=False)
fsdata.to_csv(SCRATCH+"/proc_data/out02_fsdata.csv")
