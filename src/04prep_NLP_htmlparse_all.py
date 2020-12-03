#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 07:41:19 2019

@author: 

Change
Restriction to Noaccount Bug was fixed.2020/5/31



"""
# %% rec
import re
import unicodedata
import string
import pandas as pd
import numpy as np
import sys
from bs4 import BeautifulSoup


import joblib

import sys
import MeCab
import os
#import urllib.request
from collections import Counter

from gensim import corpora
# %% parms
import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--Yuho', default="textblocks", type=str, help='textblocs(d), NoAccount, current, NoRisk')
parser.add_argument('--fs_nan_treatment', default="zero", type=str, help='zero(d), nan')
parser.add_argument('--scratch', default="../results/run_1", type=str, help='results dir')
parser.add_argument('--org', default=r"/data", type=str, help='base dir')
parser.add_argument('--cwd',  type=str, help='base dir')
parser.add_argument('--input', default=r"../results/run_1/proc_data/out02_diff_data_all.pkl.cmp", type=str, help='input file path')
parser.add_argument('--input_mask', default=r"../results/run_1/proc_data/out021_mask_all.csv", type=str, help='input file path')
parser.add_argument('--input_mask_train', default=r"../results/run_1/proc_data/out021_mask_train.csv", type=str, help='input file path')
parser.add_argument('--input_mask_test', default=r"../results/run_1/proc_data/out021_mask_test.csv", type=str, help='input file path')
#parser.add_argument('--output', default=r"/Users/noro/Documents/Projects/XBRLanalysis/data/proc_data/text_all2.csv", type=str, help='input file path')


args = parser.parse_args()
PROJDIR=args.org
SCRATCH=args.scratch
import warnings
warnings.simplefilter('ignore')


# %% funcs

def RtnDroper(text):
    replaced_text=text.replace('\n\n','\n')
    replaced_text=replaced_text.replace('\n \n','\n')
    if replaced_text==text:
        return replaced_text
    else:
        return RtnDroper(replaced_text)
    
    
def drop_number(text):
    """
    pattern = r'\d+'
    replacer = re.compile(pattern)
    result = replacer.sub('0', text)
    """
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '', text)
    
    return replaced_text

def preproc_nlp(text):
    # unicode
    replaced_text = unicodedata.normalize("NFKC", text)
    # drop number
    replaced_text = drop_number(replaced_text)
    # drop signature 1
    replaced_text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', replaced_text)
    # drop signature 2
    replaced_text = re.sub(r'\(', '', replaced_text)
    
    # drop signature 3
    table = str.maketrans("", "", string.punctuation  + "◆■※【】)(「」、。・")
    replaced_text = replaced_text.translate(table)
    # drop return (recursive)
    replaced_text=RtnDroper(replaced_text)    
    
    return replaced_text


def htmlparce(text):
    
    htmlsoup = BeautifulSoup(text, "lxml")
    p_tags = htmlsoup.find_all("p")
    output=p_tags[0].text
    for itr in range(1,len(p_tags)):
        output=output+'\n'+p_tags[itr].text
    output=preproc_nlp(output)
    
    return output

def htmlparce_try(text):
    try:
        htmlsoup = BeautifulSoup(text, "lxml")
        p_tags = htmlsoup.find_all("p")
        output=p_tags[0].text
        for itr in range(1,len(p_tags)):
            output=output+'\n'+p_tags[itr].text
        output=preproc_nlp(output)
        
        return output
    except:
        pass

# %% 
filename=args.input
results = joblib.load(filename)
mask_all=pd.read_csv(args.input_mask,index_col=0,header=None).values[:,0]

# %%
def series_dtype(x):
    return type(x)

if args.Yuho=="textblocks":
    filename=PROJDIR + "/metadata/Yuho_textblocks2.xlsx" ## all textblock
    book=pd.ExcelFile(filename)
    sheet=book.parse(sheet_name=0,header=0,index_col=0)

if args.Yuho=="NoAccount":
    filename=PROJDIR + "/metadata/Yuho_textblocks_not_accounts.xlsx"
    book=pd.ExcelFile(filename)
    sheet=book.parse(sheet_name=0,header=0,index_col=0)

if args.Yuho=="current":
    filename=PROJDIR + "/metadata/Yuho_textblocks_current.xlsx"
    book=pd.ExcelFile(filename)
    sheet=book.parse(sheet_name=0,header=0,index_col=0)

if args.Yuho=="NoRisk":
    filename=PROJDIR + "/metadata/Yuho_textblocks_not_risk.xlsx"
    book=pd.ExcelFile(filename)
    sheet=book.parse(sheet_name=0,header=0,index_col=0)

# %%

def FormatTexts(input_texts):
    if input_texts is None:
        return ""
    else:
        try:
            input_texts=input_texts.loc[sheet["標準ラベル（英語）"]].copy()
        except KeyError:
            return "missing"
        mask=pd.Series(input_texts).apply(series_dtype)==str
        texts=input_texts[mask]
        texts_parced=texts.apply(htmlparce_try)
        texts_list=map(str,pd.Series(texts_parced).to_list())
        out_texts='¥n'.join(texts_list)
        return out_texts

results_cleaned=list(map(FormatTexts,results))
results_cleaned=list(filter(lambda x:len(x)>0, results_cleaned)) # remove empty
data=pd.Series(results_cleaned) #not concat bat difine dataframe


# %%
data_all=data.loc[mask_all].copy()

mask_train=pd.read_csv(args.input_mask_train,index_col=0,header=None).values[:,0]
mask_test=pd.read_csv(args.input_mask_test,index_col=0,header=None).values[:,0]

data_train=data.loc[mask_train].copy()
data_test=data.loc[mask_test].copy()


# %% save
data_all.to_csv(SCRATCH+"/proc_data/out04_text_all.csv")
data_train.to_csv(SCRATCH+"/proc_data/out04_text_train.csv")
data_test.to_csv(SCRATCH+"/proc_data/out04_text_test.csv")
