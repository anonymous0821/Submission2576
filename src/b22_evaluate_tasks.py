#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 02:40:35 2020

@author: 


"""

# %% rec
import json

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from collections import defaultdict
from nltk import FreqDist
import matplotlib.pyplot as plt
import wordcloud as wc 
import gensim

# %% parms
import argparse
parser = argparse.ArgumentParser(description='')

runkey='_1'
parser.add_argument('--scratch', default="../results/run"+runkey, type=str, help='results dir')
parser.add_argument('--base_scratch', default="../results/run_61", type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input', default=r"../results/run_61/proc_data/out17_text_test.json", type=str, help='base dir')
parser.add_argument('--input_fsdata', default=r"../results/run" + runkey + "/proc_data/out03_fsdata_prep4_test.pkl", type=str, help='fsdata')
parser.add_argument('--input_fsdata_train', default=r"../results/run" + runkey + "/proc_data/out03_fsdata_prep4_train.pkl", type=str, help='fsdata')

parser.add_argument('--input_dict', default=r"../results/run" + runkey + "/proc_data/out06_dictionary.dict", type=str, help='lda dict')
parser.add_argument('--input_lda', default=r"../results/run" + runkey + "/proc_data/out06_lda.model", type=str, help='lda model')
parser.add_argument('--no_below', default=3, type=int, help='drop words observed fewer documents than no_below')
parser.add_argument('--no_above', default=0.5, type=float, help='drop words observed more frequent than no_avobe')
parser.add_argument('--dim', default=20, type=int, help='cca dimension; number of topics')
parser.add_argument('--ldadim', default=400, type=int, help='LDA dimension; number of topics')

args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch


# %% ##########################################################################
# start
###############################################################################


lda_score=np.load(SCRATCH+"/proc_data/out18_lda_score_test.npy")
# %% ##########################################################################
# Scaling
###############################################################################
nmf_score=pd.read_csv(SCRATCH + "/proc_data/out07_NMFscore_test.csv",header=None, index_col=None)
nmf_score_train=pd.read_csv(SCRATCH + "/proc_data/out07_NMFscore.csv",header=None, index_col=None)

fs_score=(nmf_score-nmf_score_train.mean())/np.sqrt(nmf_score_train.var())

lda_score_train=np.load(SCRATCH+"/proc_data/out06_lda_score.npy")
lda_score_scaled=(lda_score-lda_score_train.mean())/np.sqrt(lda_score_train.var())


# %% ##########################################################################
# ncca test
###############################################################################



# ncca train

fs_ncca_cor_train=pd.read_csv(SCRATCH + "/proc_data/out08_ncca_cor.csv",encoding='utf-8',index_col=0)
fs_ncca_cor_train.columns=['NCCA_train']

# ncca test
fsdata=pd.read_pickle(args.input_fsdata)

fs_ncca_comp=pd.read_csv(SCRATCH + '/proc_data/out08_ncca_fscomp.csv',
                         encoding='utf-8',index_col=0)

texts_ncca_comp=pd.read_csv(SCRATCH + '/proc_data/out08_ncca_ldacomp.csv',
                            encoding='utf-8',index_col=0)

ncca_proj_fs=pd.DataFrame(np.dot(fs_score,fs_ncca_comp),index=fsdata.index,columns=texts_ncca_comp.columns)
ncca_proj_text=pd.DataFrame(np.dot(lda_score_scaled,texts_ncca_comp),index=fsdata.index,columns=texts_ncca_comp.columns)

corlist=[]
for itr in range(20):
    corlist.append(np.corrcoef(ncca_proj_fs.iloc[:,itr],ncca_proj_text.iloc[:,itr])[0,1])

fs_ncca_cor_test=pd.Series(corlist,index=fs_ncca_cor_train.index,name='NCCA_test')

# dcca train
modelname='dcca_opt'
output_nmf_train=np.load(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_output_train.npy")
#np.load(SCRATCH+"/proc_data/out20_nmf_dcca_output_train.npy")
output_lda_train=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_output_train.npy")
#np.load(SCRATCH+"/proc_data/out20_lda_dcca_output_train.npy")

corr_score_train=[]
for itr_dim in range(20):
    corr_score_train.append(np.corrcoef(output_nmf_train[:,itr_dim],output_lda_train[:,itr_dim])[1,0])

fs_dcca_cor_train=pd.Series(corr_score_train,index=fs_ncca_cor_train.index,name='DCCA_train')

# dcca test
#output_nmf_test=np.load(SCRATCH+"/proc_data/out20_nmf_dcca_output_test.npy")
#output_lda_test=np.load(SCRATCH+"/proc_data/out20_lda_dcca_output_test.npy")

#output_nmf_test=np.load(SCRATCH+"/proc_data/out20_nmf_dcca_NNoutput_test.npy")
#output_lda_test=np.load(SCRATCH+"/proc_data/out20_lda_dcca_NNoutput_test.npy")

output_nmf_test=np.load(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_NNoutput_test.npy")
output_lda_test=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_NNoutput_test.npy")

w0=np.load(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_train_weight.npy")
w1=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_train_weight.npy")



# %% null


corlist_null=np.zeros((10000,20))
for null_sim in range(10000):
    ncca_proj_fs_parm=ncca_proj_fs.sample(frac=1,random_state=null_sim)

    for itr in range(20):
        corlist_null[null_sim,itr]=np.corrcoef(ncca_proj_fs_parm.iloc[:,itr],ncca_proj_text.iloc[:,itr])[0,1]
    
## %% 95%
#fs_ncca_cor_null=pd.DataFrame(corlist_null).sum()
tmp=corlist_null.max(axis=1)
tmp2=np.quantile(tmp,q=0.95,axis=0)
print(tmp2)

# %%

output_nmf_test=np.dot(output_nmf_test-output_nmf_test.mean(axis=0).reshape([1, -1]).repeat(len(output_nmf_test), axis=0),w0)
output_lda_test=np.dot(output_lda_test-output_lda_test.mean(axis=0).reshape([1, -1]).repeat(len(output_lda_test), axis=0),w1)

#.reshape([1, -1]).repeat(len(output_nmf_test), axis=0)
# %%
corr_score_test=[]
for itr_dim in range(20):
    corr_score_test.append(np.corrcoef(output_nmf_test[:,itr_dim],output_lda_test[:,itr_dim])[1,0])
# %%
fs_dcca_cor_test=pd.Series(corr_score_test,index=fs_ncca_cor_train.index,name='DCCA_test')

# vis
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
FONT=r"/Users/noro/Documents/Projects/XBRLanalysis/Arial/Arial/Arial.ttf"

cors=pd.concat([fs_ncca_cor_train,fs_ncca_cor_test,fs_dcca_cor_train,fs_dcca_cor_test],axis=1)
fig=plt.figure(figsize=(6,8))
ax=plt.plot(cors)
plt.legend(cors.columns)
fig.savefig(SCRATCH + "/NCCA_DCCA_cor.pdf", transparent=True)


# sort by train
ncca_sortkey=cors.sort_values(by='NCCA_train',ascending=False).index.to_list()
dcca_sortkey=cors.sort_values(by='DCCA_train',ascending=False).index.to_list()

cors.loc[ncca_sortkey,'NCCA_train'].reset_index()
cors_sort=pd.concat([cors.loc[ncca_sortkey,'NCCA_train'].reset_index(drop=True),
                     cors.loc[ncca_sortkey,'NCCA_test'].reset_index(drop=True),
                     cors.loc[dcca_sortkey,'DCCA_train'].reset_index(drop=True),
                     cors.loc[dcca_sortkey,'DCCA_test'].reset_index(drop=True)],axis=1)

fig=plt.figure(figsize=(6,8))
ax=plt.plot(cors_sort)
plt.legend(cors.columns)
fig.savefig(SCRATCH + "/NCCA_DCCA_cor_sort.pdf", transparent=True)

# %% ##########################################################################
# Evaluation
###############################################################################

result_table=pd.DataFrame([],index=['LiDA_fs','PCA_fs','NMF400_fs','NCCA_fs','DCCA_fs','LiDA_all','PCA_all','NCCA_all','DCCA_all','fs_orig'],
             columns=['BusinessSecterPrediction_k20','BusinessSecterPrediction_k40','CompanyIdentification_k1','CompanyIdentification_k5'])

# %% ##########################################################################
# bussiness secter prediction
###############################################################################

from sklearn.neighbors import NearestNeighbors

def kNN_eval(k,train,train_label,test,test_label):
    """
    An evaluation of kNN classification
    # Arguments:
        k = k of kNN
        train = [training sample x dim], pd.DataFrame
        train_label =[training sample x dim], pd.DataFrame of consistent index with train
        test = [test sample x dim], pd.DataFrame
        test_label =[test sample x dim], pd.DataFrame of consistent index with test
    # Return:
        arculacy rate of winner take all selection in kNN
    
    """
    nn =  NearestNeighbors(n_neighbors=k)
    nn.fit(train)
    dist_test, ind = nn.kneighbors(test)

    acc=np.zeros((len(test_label)))
    for itr in range(len(test_label)):
        prob_class=train_label.iloc[ind[itr],:].label.value_counts() #find k label
        acc[itr]=(prob_class.index[0]==test_label.iloc[itr,:].label)*1 # If top label is ans label, 1, otherwize, 0.
    return acc.sum()/len(acc)
    
# %%
fs_score_train=pd.read_csv(SCRATCH + '/proc_data/out08_ncca_fs_scaled.csv',encoding='utf-8',index_col=0)
lda_score_train=pd.read_csv(SCRATCH + '/proc_data/out08_ncca_lda_scaled.csv',encoding='utf-8',index_col=0)

# %% fs_nmf_data projection

# defining training data
fsdata_train=pd.read_pickle(args.input_fsdata_train)
ncca_proj_fs_train=pd.DataFrame(np.dot(fs_score_train,fs_ncca_comp),index=fsdata_train.index,columns=texts_ncca_comp.columns)
docIDinfo=pd.read_pickle(args.base_scratch+"/proc_data/out02_docIDinfo.pkl")

# %% ##################################  label
training_data_label=pd.concat([ncca_proj_fs_train, docIDinfo[['industry']]], axis=1, join_axes=[ncca_proj_fs_train.index])
training_data_label.rename(columns={'industry':'label'},inplace=True)

# answer label
ans_label=pd.concat([ncca_proj_fs, docIDinfo[['industry']]], axis=1, join_axes=[ncca_proj_fs.index])
ans_label.rename(columns={'industry':'label'},inplace=True)

# %%
# CCAUV proj
result_table.loc['NCCA_fs','BusinessSecterPrediction_k20']=kNN_eval(20,ncca_proj_fs_train,training_data_label,ncca_proj_fs,ans_label)
result_table.loc['NCCA_fs','BusinessSecterPrediction_k40']=kNN_eval(40,ncca_proj_fs_train,training_data_label,ncca_proj_fs,ans_label)

# NCCA

# test data
CCAUV_test=pd.DataFrame(0.5*np.abs(np.dot(fs_score,fs_ncca_comp)+np.dot(lda_score_scaled,texts_ncca_comp)),index=fsdata.index,columns=texts_ncca_comp.columns)
# defining training data
CCAUV=pd.DataFrame(0.5*np.abs(np.dot(lda_score_train,texts_ncca_comp)+np.dot(fs_score_train,fs_ncca_comp)),index=fsdata_train.index,columns=texts_ncca_comp.columns)

result_table.loc['NCCA_all','BusinessSecterPrediction_k20']=kNN_eval(20,CCAUV,training_data_label,CCAUV_test,ans_label)
result_table.loc['NCCA_all','BusinessSecterPrediction_k40']=kNN_eval(40,CCAUV,training_data_label,CCAUV_test,ans_label)

#  DCCA

#output_nmf_train=np.load(SCRATCH+"/proc_data/out20_nmf_dcca_output_train.npy")
modelname='dcca_l512_b800_2'
output_nmf_train=np.load(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_output_train.npy")
#output_lda_train=np.load(SCRATCH+"/proc_data/out20_lda_dcca_output_train.npy")
output_lda_train=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_output_train.npy")
dcca_train=pd.DataFrame(0.5*(output_nmf_train+output_lda_train),index=fsdata_train.index)

#output_nmf_test=np.load(SCRATCH+"/proc_data/out20_nmf_dcca_output_test.npy")
output_nmf_test=np.load(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_output_test.npy")
#output_lda_test=np.load(SCRATCH+"/proc_data/out20_lda_dcca_output_test.npy")
output_lda_test=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_output_test.npy")
dcca_test=pd.DataFrame(0.5*(output_nmf_test+output_lda_test),index=fsdata.index)
#  normalize
#output_nmf_train_n=output_nmf_train-output_nmf_train.mean(axis=1).reshape([1, -1]).repeat(20, axis=0).T/output_nmf_train.std(axis=1).reshape([1, -1]).repeat(20, axis=0).T
#output_nmf_test_n=output_nmf_test-output_nmf_test.mean(axis=1).reshape([1, -1]).repeat(20, axis=0).T/output_nmf_test.std(axis=1).reshape([1, -1]).repeat(20, axis=0).T

# 
result_table.loc['DCCA_all','BusinessSecterPrediction_k20']=kNN_eval(20,dcca_train,training_data_label,dcca_test,ans_label)
result_table.loc['DCCA_all','BusinessSecterPrediction_k40']=kNN_eval(40,dcca_train,training_data_label,dcca_test,ans_label)

# DCCA fs only
result_table.loc['DCCA_fs','BusinessSecterPrediction_k20']=kNN_eval(20,output_nmf_train,training_data_label,output_nmf_test,ans_label)
result_table.loc['DCCA_fs','BusinessSecterPrediction_k40']=kNN_eval(40,output_nmf_train,training_data_label,output_nmf_test,ans_label)

# fsdata
result_table.loc['fs_orig','BusinessSecterPrediction_k20']=kNN_eval(20,fsdata_train,training_data_label,fsdata,ans_label)
result_table.loc['fs_orig','BusinessSecterPrediction_k40']=kNN_eval(40,fsdata_train,training_data_label,fsdata,ans_label)

# nmf
result_table.loc['NMF400_fs','BusinessSecterPrediction_k20']=kNN_eval(20,fs_score_train,training_data_label,fs_score,ans_label)
result_table.loc['NMF400_fs','BusinessSecterPrediction_k40']=kNN_eval(40,fs_score_train,training_data_label,fs_score,ans_label)

# LiDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# fs
clf_bsc = LinearDiscriminantAnalysis(n_components=20)
clf_bsc.fit(fsdata_train,training_data_label['label'])
fsdata_lida_train=clf_bsc.transform(fsdata_train)
fsdata_lida=clf_bsc.transform(fsdata)

result_table.loc['LiDA_fs','BusinessSecterPrediction_k20']=kNN_eval(20,fsdata_lida_train,training_data_label,fsdata_lida,ans_label)
result_table.loc['LiDA_fs','BusinessSecterPrediction_k40']=kNN_eval(40,fsdata_lida_train,training_data_label,fsdata_lida,ans_label)


#  PCA fs
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca.fit(fsdata_train)
fsdata_pca_train=pca.transform(fsdata_train)
fsdata_pca_test=pca.transform(fsdata)
result_table.loc['PCA_fs','BusinessSecterPrediction_k20']=kNN_eval(20,fsdata_pca_train,training_data_label,fsdata_pca_test,ans_label)
result_table.loc['PCA_fs','BusinessSecterPrediction_k40']=kNN_eval(40,fsdata_pca_train,training_data_label,fsdata_pca_test,ans_label)

# %% ##########################################################################
# company fingerprinting
###############################################################################

# %% Label define

training_data_label=pd.concat([ncca_proj_fs_train, docIDinfo[['secCode']]], axis=1, join_axes=[ncca_proj_fs_train.index])
training_data_label.rename(columns={'secCode':'label'},inplace=True)
ans_label=pd.concat([CCAUV_test, docIDinfo[['secCode']]], axis=1, join_axes=[CCAUV_test.index])
ans_label.rename(columns={'secCode':'label'},inplace=True)
# %%
result_table.loc['DCCA_fs','CompanyIdentification_k1']=kNN_eval(1,output_nmf_train,training_data_label,output_nmf_test,ans_label)
result_table.loc['DCCA_fs','CompanyIdentification_k5']=kNN_eval(5,output_nmf_train,training_data_label,output_nmf_test,ans_label)
# DCCA
result_table.loc['DCCA_all','CompanyIdentification_k1']=kNN_eval(1,dcca_train,training_data_label,dcca_test,ans_label)
result_table.loc['DCCA_all','CompanyIdentification_k5']=kNN_eval(5,dcca_train,training_data_label,dcca_test,ans_label)

# base line fsdata dimension unfare
result_table.loc['fs_orig','CompanyIdentification_k1']=kNN_eval(1,fsdata_train,training_data_label,fsdata,ans_label)
result_table.loc['fs_orig','CompanyIdentification_k5']=kNN_eval(5,fsdata_train,training_data_label,fsdata,ans_label)
# nmf dimension unfare
result_table.loc['NMF400_fs','CompanyIdentification_k1']=kNN_eval(1,fs_score_train,training_data_label,fs_score,ans_label)
result_table.loc['NMF400_fs','CompanyIdentification_k5']=kNN_eval(5,fs_score_train,training_data_label,fs_score,ans_label)

# NCCA_proj
result_table.loc['NCCA_fs','CompanyIdentification_k1']=kNN_eval(1,ncca_proj_fs_train,training_data_label,ncca_proj_fs,ans_label)
result_table.loc['NCCA_fs','CompanyIdentification_k5']=kNN_eval(5,ncca_proj_fs_train,training_data_label,ncca_proj_fs,ans_label)

# NCCA
result_table.loc['NCCA_all','CompanyIdentification_k1']=kNN_eval(1,CCAUV,training_data_label,CCAUV_test,ans_label)
result_table.loc['NCCA_all','CompanyIdentification_k5']=kNN_eval(5,CCAUV,training_data_label,CCAUV_test,ans_label)

# PCA
result_table.loc['PCA_fs','CompanyIdentification_k1']=kNN_eval(1,fsdata_pca_train,training_data_label,fsdata_pca_test,ans_label)
result_table.loc['PCA_fs','CompanyIdentification_k5']=kNN_eval(5,fsdata_pca_train,training_data_label,fsdata_pca_test,ans_label)



# LDA positive control

clf_ci = LinearDiscriminantAnalysis(n_components=20)
clf_ci.fit(fsdata_train,training_data_label['label'].astype(int))

fsdata_lida_train=clf_ci.transform(fsdata_train)
fsdata_lida=clf_ci.transform(fsdata)

result_table.loc['LiDA_fs','CompanyIdentification_k1']=kNN_eval(1,fsdata_lida_train,training_data_label,fsdata_lida,ans_label)
result_table.loc['LiDA_fs','CompanyIdentification_k5']=kNN_eval(5,fsdata_lida_train,training_data_label,fsdata_lida,ans_label)

# %% ALL
# %%
# all
fs_data_concat=pd.concat([fsdata_train,lsi_mat_train.fillna(0)],axis=1,join_axes=[fsdata_train.index])
fs_data_concat_test=pd.concat([fsdata,lsi_mat_test.fillna(0)],axis=1,join_axes=[fsdata.index])

# 
clf_bsc = LinearDiscriminantAnalysis(n_components=20)
clf_bsc.fit(fs_data_concat,training_data_label['label'])
fsdata_lida_concat_train=clf_bsc.transform(fs_data_concat)
fsdata_lida_concat_test=clf_bsc.transform(fs_data_concat_test)

result_table.loc['LiDA_all','BusinessSecterPrediction_k20']=kNN_eval(20,fsdata_lida_concat_train,training_data_label,fsdata_lida_concat_test,ans_label)
result_table.loc['LiDA_all','BusinessSecterPrediction_k40']=kNN_eval(40,fsdata_lida_concat_train,training_data_label,fsdata_lida_concat_test,ans_label)




#  PCA concat
pca = PCA(n_components=20)
pca.fit(fs_data_concat)
fsdata_pca_concat_train=pca.transform(fs_data_concat)
fsdata_pca_concat_test=pca.transform(fs_data_concat_test)
# 
result_table.loc['PCA_all','BusinessSecterPrediction_k20']=kNN_eval(20,fsdata_pca_concat_train,training_data_label,fsdata_pca_concat_test,ans_label)
result_table.loc['PCA_all','BusinessSecterPrediction_k40']=kNN_eval(40,fsdata_pca_concat_train,training_data_label,fsdata_pca_concat_test,ans_label)



# PCA all
result_table.loc['PCA_all','CompanyIdentification_k1']=kNN_eval(1,fsdata_pca_concat_train,training_data_label,fsdata_pca_concat_test,ans_label)
result_table.loc['PCA_all','CompanyIdentification_k5']=kNN_eval(5,fsdata_pca_concat_train,training_data_label,fsdata_pca_concat_test,ans_label)



#  LiDA all
clf_ci = LinearDiscriminantAnalysis(n_components=20)
clf_ci.fit(fs_data_concat,training_data_label['label'].astype(int))

fsdata_lida_concat_train=clf_ci.transform(fs_data_concat)
fsdata_lida_concat_test=clf_ci.transform(fs_data_concat_test)

result_table.loc['LiDA_all','CompanyIdentification_k1']=kNN_eval(1,fsdata_lida_concat_train,training_data_label,fsdata_lida_concat_test,ans_label)
result_table.loc['LiDA_all','CompanyIdentification_k5']=kNN_eval(5,fsdata_lida_concat_train,training_data_label,fsdata_lida_concat_test,ans_label)


result_table.to_csv(SCRATCH+'/results_table1008.csv')

