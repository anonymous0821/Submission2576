#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:51:17 2019

@author: 

- It is best to avoid the term that many entities submit the documents, mid to late June daytime.
- EDINET API often stop in night
- sleep time should be longer.

"""

# %% Requirements

import requests
import json
import pandas as pd
import numpy as np
import datetime
from time import sleep
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
# %% parms
import argparse
parser = argparse.ArgumentParser(description='')

runkey=str(10)
parser.add_argument('--cwd', type=str, help='base dir')

args = parser.parse_args()
PROJDIR=args.cwd

# %% ##########################################################################
# Params
###############################################################################
# set the terms of ducument submission by entities
start_date=datetime.datetime(2019,9,1) # later than five years ago
end_date=datetime.datetime(2020,8,31) # earlyer than now

PATHTOSAVE=args.cwd+'/data/testdata2003/'
# %% ##########################################################################
# Requesting meta data
###############################################################################

def EdinetQueryPD(params):
    EDINET_API_url = "https://disclosure.edinet-fsa.go.jp/api/v1/documents.json"
    res = requests.get(EDINET_API_url, params=params, verify=False)
    res_parsed=json.loads(res.text)
    responce_table=pd.read_json(json.dumps(res_parsed['results']))
    return responce_table

itr=0
target_date=start_date + datetime.timedelta(days=itr)
params = {
    "date" : target_date.strftime("%Y-%m-%d"),
    "type" : 2
}
output=EdinetQueryPD(params)
edinet_res_file=args.cwd+'../metadata/EDINET_responce_20200904.csv'
output.to_csv(edinet_res_file,encoding='utf-8') # When start a download
#output.to_csv('../data/EDINET_responce.csv',encoding='utf-8',mode='a',header=False) # If the download has stoppped and resume
sleep(1)
for itr in tqdm(range(1,(end_date-start_date).days)):
    
    target_date=start_date + datetime.timedelta(days=itr)
    params = {
        "date" : target_date.strftime("%Y-%m-%d"),
        "type" : 2
    }
    responce=EdinetQueryPD(params)
    responce.to_csv(edinet_res_file,encoding='utf-8',mode='a',header=False)
    sleep(1)
    
    
# %% ##########################################################################
# select download documents
###############################################################################

edinetcode=pd.read_csv(args.cwd+'../metadata/EdinetcodeDlInfo1903.csv', \
                   header=1,index_col=False, engine="python", encoding="cp932")

# filter the data to the documents of the listed company
mask=edinetcode.loc[:,'上場区分']=='上場' 
edinetcode_filterd=edinetcode.loc[mask,:].copy()

# filter the data to the annual reports and quarterly reports, the ratter were not used in this study.
edinetcodelist=edinetcode_filterd.loc[:,'ＥＤＩＮＥＴコード'].values
responce_data=pd.read_csv(edinet_res_file,encoding='utf-8')

mask=((responce_data.loc[:,'docTypeCode']==120)|(responce_data.loc[:,'docTypeCode']==140)).values
mask=(responce_data.loc[:,'edinetCode'].isin(edinetcodelist)).values * mask

docIDs=responce_data.loc[mask,'docID'].values
# %% ##########################################################################
# download via EDINET API
###############################################################################
def GetDocEdinet(docid,url,params):

    filename =PATHTOSAVE + docid + '.zip'
    res = requests.get(url, params=params, verify=False)
    if res.status_code == 200:
      with open(filename, 'wb') as f:
        for chunk in res.iter_content(chunk_size=1024):
          f.write(chunk)

        
for itr in tqdm(range(len(docIDs))):
    print(itr)
    docid = docIDs[itr]
    url = 'https://disclosure.edinet-fsa.go.jp/api/v1/documents/' + docid
    params = {
      "type" : 1
    }
    GetDocEdinet(docid,url,params)
    sleep(1)




