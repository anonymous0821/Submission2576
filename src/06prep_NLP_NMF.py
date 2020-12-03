#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:23:21 2020

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

runkey='run_1'
parser.add_argument('--scratch', default="../results/"+runkey, type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input', default=r"../results/"+runkey+"/proc_data/out05_text_all.json", type=str, help='base dir')
parser.add_argument('--no_below', default=3, type=int, help='drop words observed fewer documents than no_below')
parser.add_argument('--no_above', default=0.5, type=float, help='drop words observed more frequent than no_avobe')
parser.add_argument('--dim', default=50, type=int, help='LDA dimension; number of topics')
parser.add_argument('--tfidf', default=0, type=int, help='True, appried tf-idf')


args = parser.parse_args()
PROJDIR=args.cwd
SCRATCH=args.scratch
# %%

## Unstable depending on json.load automaticaly to 2d list

file=open(args.input)
text_all=json.load(file)
file.close()

dictionary = Dictionary(text_all)


# %%

# Frequency filter
dictionary.filter_extremes(no_below=args.no_below, no_above=args.no_above,keep_n=400000)
# drop words obserbed fewer documents than no_below
# drop words obserbed more frequent than no_avobe
dictionary.compactify() # Reassign id
corpus = [dictionary.doc2bow(itrtext) for itrtext in text_all]

# %% tf idf
from gensim import corpora, models
if args.tfidf>0:
    print('TFIDF')
    tfidf = models.TfidfModel(corpus)
    corpus = tfidf[corpus]
# %%
gensim.corpora.MmCorpus.serialize(SCRATCH+"/proc_data/out06_corpus.mm", corpus)
dictionary.save(SCRATCH+"/proc_data/out06_dictionary.dict")

# %%
NUM_TOPICS = args.dim
from gensim.models.nmf import Nmf
nmf = Nmf(corpus, num_topics=NUM_TOPICS, id2word=dictionary)

nmf.save(SCRATCH+"/proc_data/out06_lda.model")
# nmf = Nmf.load(temp_file)
# %%
scores=np.zeros((len(corpus),NUM_TOPICS))
for itr in range(len(corpus)):
    itrtext=corpus[itr]
    #print(itrtext)
    for topic, score in nmf.get_document_topics(itrtext,normalize=False):
        scores[itr,topic]=float(score)

# %%
np.save(SCRATCH+"/proc_data/out06_lda_score.npy",scores)
np.savetxt(SCRATCH+"/proc_data/out06_lda_score.csv",scores,delimiter=',')


# %% Visualization


FONT=r"/Users/noro/Documents/analysis/2019.3/書き出されたフォント/Hiragino Maru Gothic ProN/ヒラギノ丸ゴ ProN W4.ttc"

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from matplotlib.backends.backend_pdf import PdfPages


for t in range(nmf.num_topics):
    #plt.subplot(10,1,t+1)
    fig=plt.figure(figsize=(10,20))
    x = dict(nmf.show_topic(t,200))
    wc_pc0=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(x)
    ax=fig.add_subplot(1,1,1)
    ax.imshow(wc_pc0)
    ax.axis("off")
    fig.savefig(SCRATCH + "/LDA_results/topic_" + str(t) + ".pdf", transparent=True)

# %%
FONT=r"PathToFont/Arial/Arial/Arial.ttf"
from time import sleep
from googletrans import Translator
for num in range(nmf.num_topics):
    fig=plt.figure(figsize=(10,10))
    
    expolist=pd.Series(dict(nmf.show_topic(num,50)))
    jpnw='\n'.join(expolist.index)
    translator = Translator()
    engw=translator.translate(jpnw).text
    engwlist=engw.split('\n')
    expolist.index=engwlist
    x=expolist.to_dict()
    
    wc_pc0=wc.WordCloud(
            background_color="white",
            font_path=FONT,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(x)
    ax=fig.add_subplot(1,1,1)
    ax.imshow(wc_pc0)
    ax.axis("off")
    sleep(2)
    fig.savefig(SCRATCH + "/LDA_ENG/topic_" + str(num) + ".pdf", transparent=True)

