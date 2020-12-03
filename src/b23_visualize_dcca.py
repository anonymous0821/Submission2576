#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:21:47 2020

@author: noro

visualize DCCA  
"""

# %%
import pandas as pd
import numpy as np
import datetime

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf

from tqdm import tqdm
import wordcloud as wc 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

# %% parms
import argparse
parser = argparse.ArgumentParser(description='')

runkey="_1"
parser.add_argument('--scratch', default="../results/run"+runkey, type=str, help='results dir')
parser.add_argument('--base_scratch', default="../results/run"+runkey, type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input_fsdata', default=r"../results/run" + runkey + "/proc_data/out03_fsdata_prep4_train.pkl", type=str, help='fsdata')
parser.add_argument('--input_lda', default=r"../results/run" + runkey + "/proc_data/out06_lda.model", type=str, help='lda model')
parser.add_argument('--input_dict', default=r"../results/run" + runkey + "/proc_data/out06_dictionary.dict", type=str, help='lda dict')
parser.add_argument('--input_nmf_comp', default=r"../results/run" + runkey + "/proc_data/out07_NMFcomp.csv", type=str, help='lda dict')
parser.add_argument('--ldadim', default=400, type=int, help='LDA dimension; number of topics')
parser.add_argument('--nccadim', default=20, type=int, help='ncca dimension')
parser.add_argument('--account_summerized', default=True, type=bool, help='use exposured account name as label')

args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch
# %%
# nmf comp dccadim x 
nmf_comp=pd.read_csv(args.input_nmf_comp,index_col=None,header=None)


lda = Nmf.load(args.input_lda)

dictionary=Dictionary.load(args.input_dict)

lda_comp=pd.DataFrame()
for itr in range(args.ldadim):
    lda_comp_t=pd.DataFrame(lda.get_topic_terms(itr,topn=500))
    lda_comp_t=lda_comp_t.set_index(0)
    lda_comp=pd.concat([lda_comp,lda_comp_t],axis=1,sort=False)

lda_comp=lda_comp.fillna(0)
## %%

## %% DCCA comp



fs_dcca_comp=pd.read_csv(SCRATCH + '/proc_data/out21_DCCAscore_fs.csv',
                         encoding='utf-8',header=None,index_col=None)

texts_dcca_comp=pd.read_csv(SCRATCH + '/proc_data/out21_DCCAscore_text.csv',
                            encoding='utf-8',header=None,index_col=None)

fs_dcca_comp_m=pd.read_csv(SCRATCH + '/proc_data/out21_DCCAscore_fs_m.csv',
                         encoding='utf-8',header=None,index_col=None)

texts_dcca_comp_m=pd.read_csv(SCRATCH + '/proc_data/out21_DCCAscore_text_m.csv',
                            encoding='utf-8',header=None,index_col=None)

## %% ##########################################################################
# Projection score 
###############################################################################
## %% loading account transcripter
if args.account_summerized:
    transcripter=pd.read_csv(args.base_scratch+"/transcripter_summerized_account.csv",header=0, index_col=0)
else:
    transcripter=pd.read_csv(args.base_scratch+"/transcripter.csv",header=0, index_col=0)

fsdata=pd.read_pickle(args.input_fsdata)

texts_nmf_projection=pd.DataFrame(np.dot(lda_comp.values,texts_dcca_comp.values),index=lda_comp.index)
texts_nmf_projection_m=pd.DataFrame(np.dot(lda_comp.values,texts_dcca_comp_m.values),index=lda_comp.index)

if args.account_summerized:
    fsdata_nmf_projection=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp.values),index=transcripter.loc[fsdata.columns,"標準ラベル（日本語）"])
    fsdata_nmf_projection_barh=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp.values),index=fsdata.columns)
    fsdata_nmf_projection_eng=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp.values),index=transcripter.loc[fsdata.columns,"標準ラベル（英語）"])
else:
    fsdata_nmf_projection=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp.values),index=transcripter.loc[fsdata.columns,"冗長ラベル（日本語）"])
    fsdata_nmf_projection_eng=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp.values),index=transcripter.loc[fsdata.columns,"冗長ラベル（日本語）"])
if args.account_summerized:
    fsdata_nmf_projection_m=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp_m.values),index=transcripter.loc[fsdata.columns,"標準ラベル（日本語）"])
    fsdata_nmf_projection_barh_m=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp_m.values),index=fsdata.columns)
    fsdata_nmf_projection_eng_m=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp_m.values),index=transcripter.loc[fsdata.columns,"標準ラベル（英語）"])
else:
    fsdata_nmf_projection_m=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp_m.values),index=transcripter.loc[fsdata.columns,"冗長ラベル（日本語）"])
    fsdata_nmf_projection_eng_m=pd.DataFrame(np.dot(nmf_comp.values,fs_dcca_comp_m.values),index=transcripter.loc[fsdata.columns,"冗長ラベル（日本語）"])
## %% ##########################################################################
# Extract top-score words for word-cloud visualizing
###############################################################################


tmp=texts_nmf_projection.copy()
for itr in tqdm(range(len(tmp))):
    tmp=tmp.rename(index={tmp.index[itr]:dictionary[tmp.index[itr]]})
words_cca_projection=tmp.copy()
#JPNwords=words_cca_projection.index.tolist()
tmp=texts_nmf_projection_m.copy()
for itr in tqdm(range(len(tmp))):
    tmp=tmp.rename(index={tmp.index[itr]:dictionary[tmp.index[itr]]})
words_cca_projection_m=tmp.copy()

# %% ##########################################################################
# Word cloud setting
###############################################################################

FONT=r"/Users/noro/Documents/analysis/2019.3/書き出されたフォント/Hiragino Maru Gothic ProN/ヒラギノ丸ゴ ProN W4.ttc"
# color setting
# https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html
# Copyright 2020, Andreas Mueller
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


# difineing increase/decrease color mapping
if args.account_summerized:
    list_minus=transcripter.loc[transcripter.index.str.contains("down"),"標準ラベル（日本語）"].to_list()
    list_plus=transcripter.loc[transcripter.index.str.contains("up"),"標準ラベル（日本語）"].to_list()
    list_minus_eng=transcripter.loc[transcripter.index.str.contains("down"),"標準ラベル（英語）"].to_list()
    list_plus_eng=transcripter.loc[transcripter.index.str.contains("up"),"標準ラベル（英語）"].to_list()
    
    
else:
    list_minus=transcripter.loc[transcripter.index.str.contains("down"),"冗長ラベル（日本語）"].to_list()
    list_plus=transcripter.loc[transcripter.index.str.contains("up"),"冗長ラベル（日本語）"].to_list()
    list_minus_eng=transcripter.loc[transcripter.index.str.contains("down"),"冗長ラベル（英語）"].to_list()
    list_plus_eng=transcripter.loc[transcripter.index.str.contains("up"),"冗長ラベル（英語）"].to_list()

coler_increase='tomato'
coler_decrease='darkblue'#'#00ff00'

color_to_words = {coler_decrease: list_minus, coler_increase: list_plus}
default_color = 'grey'


## %% ##########################################################################
# Visualize summary of NCCA components
###############################################################################

# Create a color function with single tone
# grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

# Create a color function with multiple tones
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from matplotlib.backends.backend_pdf import PdfPages
max_words_text=100
max_words_fs=50
for num in range(args.nccadim):
    fig=plt.figure(figsize=(10,20))
    ax1=fig.add_subplot(2,1,1)
    tmp=words_cca_projection.loc[:,num].copy()
    
    tmp=tmp.sort_values(ascending=False)
    x=tmp.to_dict()
    
    
    wc_text=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=max_words_text,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(x)
    
    ax1.imshow(wc_text)
    ax1.axis("off")
    #.title("Topic #" + str(num))
    
    ax2=fig.add_subplot(2,1,2)
    weight_plus=fsdata_nmf_projection.loc[:,num].copy()
    wc_fs=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=max_words_fs,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(weight_plus.to_dict())
    wc_fs.recolor(color_func=grouped_color_func)
    ax2.imshow(wc_fs)
    ax2.axis("off")
    fig.suptitle(" #" + str(num))
              
    fig.savefig(SCRATCH + "/DCCA_results_JPN/comp_" + str(num) + ".pdf", transparent=True)

## %% ##########################################################################
# Visualize NCCA FS increase/decrease components
###############################################################################

bar_num=20
for num in range(args.nccadim):
    fig=plt.figure(figsize=(8,10))
    ax=fig.add_subplot(1,2,2)
    tmp=fsdata_nmf_projection_barh.loc[:,num].copy()
    tmp=tmp.sort_values(ascending=False)
    tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    col_mask=tmp.index.str.contains("up")
    col=pd.Series([coler_increase,]*20)
    col[~col_mask]=coler_decrease
    #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（日本語）"].str.strip()
    ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    plt.yticks(range(bar_num), tmp.index.to_list())
    ax.grid(True)
    fig.suptitle(" #" + str(num))
    fig.savefig(SCRATCH + "/DCCA_results_JPN_barh/comp_" + str(num) + ".pdf", transparent=True)


## %% ##########################################################################
# Visualize NCCA Summary (English)
###############################################################################

FONT=r"/Users/noro/Documents/Projects/XBRLanalysis/Arial/Arial/Arial.ttf"
from time import sleep
from googletrans import Translator

color_to_words = {coler_decrease: list_minus_eng,'tomato': list_plus_eng}
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

for num in range(args.nccadim):
    fig=plt.figure(figsize=(10,20))
    ax1=fig.add_subplot(2,1,1)
    tmp=words_cca_projection.loc[:,num].copy()
    
    tmp=tmp.sort_values(ascending=False)
    expolist=tmp.iloc[:50]
    jpnw='\n'.join(expolist.index)
    translator = Translator()
    engw=translator.translate(jpnw).text
    engwlist=engw.split('\n')
    expolist.index=engwlist
    x=expolist.to_dict()
    
    wc_text=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(x)
    
    ax1.imshow(wc_text)
    ax1.axis("off")
    #.title("Topic #" + str(num))
    
    ax2=fig.add_subplot(2,1,2)
    weight_plus=fsdata_nmf_projection_eng.loc[:,num].copy()
    wc_fs=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(weight_plus.to_dict())
    wc_fs.recolor(color_func=grouped_color_func)
    ax2.imshow(wc_fs)
    ax2.axis("off")
    fig.suptitle(" #" + str(num))
    sleep(1)
    fig.savefig(SCRATCH + "/DCCA_results_ENG/comp_" + str(num) + ".pdf", transparent=True)

## %% ##########################################################################
# Visualize NCCA FS increase/decrease components (English)
###############################################################################
    
bar_num=20
for num in range(args.nccadim):
    fig=plt.figure(figsize=(8,10))
    ax=fig.add_subplot(1,2,2)
    tmp=fsdata_nmf_projection_barh.loc[:,num].copy()
    tmp=tmp.sort_values(ascending=False)
    tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    col_mask=tmp.index.str.contains("up")
    col=pd.Series([coler_increase,]*20)
    col[~col_mask]=coler_decrease
    #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（英語）"].str.strip()
    
    
    ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    plt.yticks(range(bar_num), tmp.index.to_list())
    ax.grid(True)
    
    
    
    fig.suptitle(" #" + str(num))
    fig.savefig(SCRATCH + "/DCCA_results_ENG_barh/comp_" + str(num) + ".pdf", transparent=True)



## %%minus ######################

FONT=r"/PathToFont/ヒラギノ丸ゴ ProN W4.ttc"

color_to_words = {coler_decrease: list_minus, coler_increase: list_plus}
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

max_words_text=100
max_words_fs=50
for num in range(args.nccadim):
    fig=plt.figure(figsize=(10,20))
    ax1=fig.add_subplot(2,1,1)
    tmp=words_cca_projection_m.loc[:,num].copy()
    
    tmp=tmp.sort_values(ascending=False)
    x=tmp.to_dict()
    
    
    wc_text=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=max_words_text,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(x)
    
    ax1.imshow(wc_text)
    ax1.axis("off")
    #.title("Topic #" + str(num))
    
    ax2=fig.add_subplot(2,1,2)
    weight_plus=fsdata_nmf_projection_m.loc[:,num].copy()
    wc_fs=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=max_words_fs,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(weight_plus.to_dict())
    wc_fs.recolor(color_func=grouped_color_func)
    ax2.imshow(wc_fs)
    ax2.axis("off")
    fig.suptitle(" #" + str(num))
              
    fig.savefig(SCRATCH + "/DCCA_results_JPN/minus_comp_" + str(num) + ".pdf", transparent=True)

## %% ##########################################################################
# Visualize NCCA FS increase/decrease components
###############################################################################

bar_num=20
for num in range(args.nccadim):
    fig=plt.figure(figsize=(8,10))
    ax=fig.add_subplot(1,2,2)
    tmp=fsdata_nmf_projection_barh_m.loc[:,num].copy()
    tmp=tmp.sort_values(ascending=False)
    tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    col_mask=tmp.index.str.contains("up")
    col=pd.Series([coler_increase,]*20)
    col[~col_mask]=coler_decrease
    #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（日本語）"].str.strip()
    ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    plt.yticks(range(bar_num), tmp.index.to_list())
    ax.grid(True)
    fig.suptitle(" #" + str(num))
    fig.savefig(SCRATCH + "/DCCA_results_JPN_barh/minus_comp_" + str(num) + ".pdf", transparent=True)


## %% ##########################################################################
# Visualize NCCA Summary (English)
###############################################################################

FONT=r"PathToFont/Arial.ttf"
from time import sleep
from googletrans import Translator

color_to_words = {coler_decrease: list_minus_eng,'tomato': list_plus_eng}
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

for num in range(args.nccadim):
    fig=plt.figure(figsize=(10,20))
    ax1=fig.add_subplot(2,1,1)
    tmp=words_cca_projection_m.loc[:,num].copy()
    
    tmp=tmp.sort_values(ascending=False)
    expolist=tmp.iloc[:50]
    jpnw='\n'.join(expolist.index)
    translator = Translator()
    engw=translator.translate(jpnw).text
    engwlist=engw.split('\n')
    expolist.index=engwlist
    x=expolist.to_dict()
    
    wc_text=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(x)
    
    ax1.imshow(wc_text)
    ax1.axis("off")
    #.title("Topic #" + str(num))
    
    ax2=fig.add_subplot(2,1,2)
    weight_plus=fsdata_nmf_projection_eng_m.loc[:,num].copy()
    wc_fs=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(weight_plus.to_dict())
    wc_fs.recolor(color_func=grouped_color_func)
    ax2.imshow(wc_fs)
    ax2.axis("off")
    fig.suptitle(" #" + str(num))
    sleep(1)
    fig.savefig(SCRATCH + "/DCCA_results_ENG/minus_comp_" + str(num) + ".pdf", transparent=True)

## %% ##########################################################################
# Visualize NCCA FS increase/decrease components (English)
###############################################################################
    
bar_num=20
for num in range(args.nccadim):
    fig=plt.figure(figsize=(8,10))
    ax=fig.add_subplot(1,2,2)
    tmp=fsdata_nmf_projection_barh_m.loc[:,num].copy()
    tmp=tmp.sort_values(ascending=False)
    tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    col_mask=tmp.index.str.contains("up")
    col=pd.Series([coler_increase,]*20)
    col[~col_mask]=coler_decrease
    #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（英語）"].str.strip()
    
    
    ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    plt.yticks(range(bar_num), tmp.index.to_list())
    ax.grid(True)
    
    
    
    fig.suptitle(" #" + str(num))
    fig.savefig(SCRATCH + "/DCCA_results_ENG_barh/minus_comp_" + str(num) + ".pdf", transparent=True)
