#!/bin/sh

SRCPATH=/Users/noro/Documents/Projects/XBRLanalysis/src
DATAPATH=/Users/noro/Documents/Projects/XBRLanalysis/results/run_1
ORGPATH=/Users/noro/Documents/Projects/XBRLanalysis/data
CWD=/Users/noro/Documents/Projects/XBRLanalysis
BASE_SCRATCH=${CWD}/results/run_1
SCRATCH=${CWD}/results/run_2 ############ should be change !!
mkdir ${SCRATCH}

WORD_thre=0.3
LDADIM=400
NMFDIM=400
NCCADIM=20
CMAX=2
NMFalpha=0.1
LMBDA=0
NMF_ITR=10
NMF_INIT=nndsvda
NRESTART=200

echo "base, data corrected" > ${SCRATCH}/logfile.txt
echo "word frequency bound=" ${WORD_thre} >> ${SCRATCH}/logfile.txt
echo "NCCADIM=" ${NCCADIM} >> ${SCRATCH}/logfile.txt
echo "NMFDIM=" ${NMFDIM} >> ${SCRATCH}/logfile.txt
echo "LDADIM=" ${LDADIM} >> ${SCRATCH}/logfile.txt
echo "L1 NCCA=" ${LMBDA} >> ${SCRATCH}/logfile.txt
echo "FS_thre=" ${CMAX} >> ${SCRATCH}/logfile.txt
echo "NMF initialization methods=" ${NMF_INIT} >> ${SCRATCH}/logfile.txt
echo "scale=TRUE" >> ${SCRATCH}/logfile.txt
echo "company division text tfidf NMF" >> ${SCRATCH}/logfile.txt

mkdir -p  ${SCRATCH}/proc_data

# 02

#python ${SRCPATH}/02prep_vecterize_all.py --cwd ${CWD} --scratch ${SCRATCH} --n_parallel 8

echo 02
#python ${SRCPATH}/021_define_datarange.py --cwd ${CWD} --scratch ${SCRATCH} --base_scratch ${BASE_SCRATCH} --input ${DATAPATH}/proc_data/out02_diff_data_all.pkl.cmp
#python ${SRCPATH}/022_define_datarange_enterprize_separation.py --cwd ${CWD} --scratch ${SCRATCH} --base_scratch ${BASE_SCRATCH} --input ${DATAPATH}/proc_data/out02_diff_data_all.pkl.cmp
echo 02_2

###### 03 --input is locked to test run
#python ${SRCPATH}/03prep_fsdata_preprocess_all.py --cwd ${CWD} --scratch ${SCRATCH} --fs_normalization NetAsset --fs_nan_treatment zero --input ${DATAPATH}/proc_data/out02_diff_data_all.pkl.cmp --input_mask ${SCRATCH}/proc_data/out021_mask_all.csv --drop_general_account ${ORGPATH}/metadata/Yuho_dummy_account_list2.csv --account_summerized True --PLdiff True --cmax ${CMAX} --input_mask_train ${SCRATCH}/proc_data/out021_mask_train.csv --input_mask_test ${SCRATCH}/proc_data/out021_mask_test.csv

#######python ${SRCPATH}/03prep_PCA.py --cwd ${CWD} --scratch ${SCRATCH} --fs_normalization NetAsset --fs_nan_treatment zero --input ${SCRATCH}/proc_data/out02_diff_data.pkl.cmp --drop_general_account ${ORGPATH}/metadata/Yuho_dummy_account_list.csv --account_summerized True
echo 03

####### 04 --input is locked to test run
#python ${SRCPATH}/04prep_NLP_htmlparse_all.py --cwd ${CWD} --scratch ${SCRATCH} --Yuho textblocks --input ${DATAPATH}/proc_data/out02_diff_data_all.pkl.cmp --input_mask ${SCRATCH}/proc_data/out021_mask_all.csv --input_mask_train ${SCRATCH}/proc_data/out021_mask_train.csv --input_mask_test ${SCRATCH}/proc_data/out021_mask_test.csv

#######python ${SRCPATH}/04prep_NLP_htmlparse.py --cwd ${CWD} --scratch ${SCRATCH} --Yuho NoAccount --input ${SCRATCH}/proc_data/out02_diff_data.pkl.cmp

echo 04

######### 042 filter 201503-201803
###python ${SRCPATH}/042prep_unfilter_doc.py --cwd ${CWD} --scratch ${SCRATCH} --input_text ${BASE_SCRATCH}/proc_data/out04_text_all.csv --input_fsdata ${BASE_SCRATCH}/proc_data/out03_fsdata_prep4.pkl


# 05 parse by mecab and filter out stopwords
# [1] N, NA, NAV
# [2] drop key words file path
# [3] Input file path
# [4] Output file path
#/usr/local/bin/R --vanilla --slave --args N ${ORGPATH}/metadata/Yuho_dummy_words.csv ${SCRATCH}/proc_data/out04_text_train.csv ${SCRATCH} < ${SRCPATH}/05prep_NLP_mecab_parce_stopwords_filter.R

echo 05

####### 06
mkdir -p ${SCRATCH}/LDA_results
mkdir -p ${SCRATCH}/LDA_ENG

python ${SRCPATH}/06prep_NLP_NMF.py --cwd ${CWD} --scratch ${SCRATCH} --input ${SCRATCH}/proc_data/out05_text_all.json --no_below 2 --no_above ${WORD_thre} --dim ${LDADIM} --tfidf True

echo 06

####### b07 NMF
mkdir -p ${SCRATCH}/NMF_JPN_barh
mkdir -p ${SCRATCH}/NMF_ENG_barh

#python ${SRCPATH}/b07anly_FS_NMF.py --cwd ${CWD} --scratch ${SCRATCH} --base_scratch ${BASE_SCRATCH} --input ${SCRATCH}/proc_data/out03_fsdata_prep4_train.pkl --input_test ${SCRATCH}/proc_data/out03_fsdata_prep4_test.pkl --dim ${NMFDIM} --alpha ${NMFalpha} --nmfinit ${NMF_INIT} --nmfitr ${NMF_ITR} --account_summerized True

echo 07

# b08
# [1] input FS_NMF score
# [2] input lda score
# [3] output dir
# [4] dim ncca
# [5] sparse x
# [6] sparse y
mkdir -p ${SCRATCH}/ncca_Xp
mkdir -p ${SCRATCH}/ncca_Yp

/usr/local/bin/R --vanilla --slave --args ${SCRATCH}/proc_data/out07_NMFscore.csv ${SCRATCH}/proc_data/out06_lda_score.csv ${SCRATCH} ${NCCADIM} ${LAMBDA} ${LAMBDA} ${NRESTART} < ${SRCPATH}/b08anly_ncca_scaled.R

echo 08

mkdir -p ${SCRATCH}/NCCA_results_JPN
mkdir -p  ${SCRATCH}/NCCA_results_ENG
mkdir -p ${SCRATCH}/NCCA_results_ENG_barh
mkdir -p ${SCRATCH}/NCCA_results_JPN_barh
mkdir -p ${SCRATCH}/NCCA_results_JPN_score
mkdir -p ${SCRATCH}/NCCA_results_ENG_score
mkdir -p ${SCRATCH}/NCCA_year_effect

# b09
python ${SRCPATH}/b09anly_NCCA_vis_all.py --cwd ${CWD} --scratch ${SCRATCH} --base_scratch ${BASE_SCRATCH} --input_fsdata ${SCRATCH}/proc_data/out03_fsdata_prep4_train.pkl --input_lda ${SCRATCH}/proc_data/out06_lda.model --input_dict ${SCRATCH}/proc_data/out06_dictionary.dict --input_nmf_comp ${SCRATCH}/proc_data/out07_NMFcomp.csv --ldadim ${LDADIM} --nccadim ${NCCADIM} --account_summerized True

echo 09


#python ${SRCPATH}/b10evaluate_prep_gyoshu.py --cwd ${CWD} --scratch ${SCRATCH} --input ${SCRATCH}/proc_data/out03_fsdata_prep4_train.pkl

echo 10

#/usr/local/bin/R --vanilla --slave --args ${SCRATCH}/proc_data/out07_NMFscore.csv ${SCRATCH}/proc_data/out10_ind_score.csv ${SCRATCH} ${NCCADIM} ${LAMBDA} ${LAMBDA} ${NRESTART} < ${SRCPATH}/b11evaluate_NCCA_scaled_gyoshu.R

echo 11

# 12
#python ${SRCPATH}/b12eval_ncca_results_all.py --cwd ${CWD} --scratch ${SCRATCH} --input_fsdata ${SCRATCH}/proc_data/out03_fsdata_prep4_train.pkl --input_lda ${SCRATCH}/proc_data/out06_lda.model --input_dict ${SCRATCH}/proc_data/out06_dictionary.dict --input_nmf_comp ${SCRATCH}/proc_data/out07_NMFcomp.csv --nccadim ${NCCADIM}

echo 12

# 13

# 13
#for itr in `seq 0 70`
#do
#echo ${itr}
#/usr/local/bin/R --vanilla --slave --args N ${ORGPATH}/metadata/Yuho_dummy_words.csv ${SCRATCH}/proc_data/out13_text_${itr}.csv ${SCRATCH} ${itr} < ${SRCPATH}/b14prep_NLP_mecab_parce_sep.R
#done


################
# test data
################

# text preprocessing
#/usr/local/bin/R --vanilla --slave --args N ${ORGPATH}/metadata/Yuho_dummy_words.csv ${SCRATCH}/proc_data/out04_text_test.csv ${SCRATCH} < ${SRCPATH}/17prep_NLP_mecab_parce_stopwords_filter_test.R

python ${SRCPATH}/b18_lda_test.py --cwd ${CWD} --scratch ${SCRATCH} --base_scratch ${BASE_SCRATCH} --input ${SCRATCH}/proc_data/out17_text_test.json --input_lda ${SCRATCH}/proc_data/out06_lda.model --input_dict ${SCRATCH}/proc_data/out06_dictionary.dict --ldadim ${LDADIM}