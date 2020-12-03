# ----
# encoded utf-8
# 
# Input: 
# Output:
# R mecab
# Refferences:
# http://rstudio-pubs-static.s3.amazonaws.com/85463_ab84b5964c4c4c129d8601dc495b4e51.html
# http://www.ic.daito.ac.jp/~mizutani/mining/rmecab_func.html
# https://qiita.com/hujuu/items/314a64a50875cdabf755
# https://heavywatal.github.io/rstats/stringr.html
# neologd on RMeCab
# https://y-mattu.hatenablog.com/entry/2016/08/12/014841

# python
# shellscript
# http://www.mwsoft.jp/programming/munou/mecab_command.html
# input-buffer overflow. The line is split. use -b #SIZE option.
# ----
args <- commandArgs(trailingOnly = T)
default_args <- c("N","/Users/noro/Documents/Projects/XBRLanalysis/data/metadata/Yuho_dummy_words.csv",
                  "/Users/noro/Documents/Projects/XBRLanalysis/results/run01/proc_data/out04_text_all.csv",
                  '/Users/noro/Documents/Projects/XBRLanalysis/data/proc_data')
default_flg <- is.na(args[1:4])
args[default_flg] <- default_args[default_flg]



library(tidyverse)

data=read.csv(args[3],header=FALSE)
filename='/Users/noro/Documents/Projects/FSmodeling/scripts/stopwords.csv'
stopwords=read.csv(filename,header=FALSE)

library(RMeCab)
library(stringr)

data_processed<-lapply(data$V2,function(y) {
  if (str_length(y) < 1){
    return("nan")
  }else{
  res <- RMeCabC(y,dic = "/Users/noro/Documents/Projects/XBRLanalysis/src/mecab-ipadic-neologd/build/mecab-ipadic-2.7.0-20070801-neologd-20191226/mecab-user-dict-seed.20191226.csv.dic")
  tmp<-unlist(res)
  if(args[1]=="N"){
    tmp<-tmp[names(tmp)=="名詞"]
  }else if(args[1]=="NA"){
    tmp<-tmp[(names(tmp)=="名詞") || (names(tmp)=="形容詞")]
  }else if(args[1]=="NAV"){
    tmp<-tmp[(names(tmp)=="名詞") || (names(tmp)=="形容詞") || (names(tmp)=="動詞")]
  }else{print("Not Filtered")}
  tmp<-setdiff(tmp,stopwords)
  if(args[2]=="nan"){}else{
    filename=args[2]
    dummywords=read.csv(filename,header=FALSE,stringsAsFactors=FALSE)
    #tmp<-setdiff(tmp,dummywords)
    for(i in 1:dim(dummywords)[1]){
      dropwords=str_subset(tmp,dummywords$V2[i])
      tmp<-setdiff(tmp,dropwords)
    }
    
  }
  return(tmp)
  }
})

#out_filename=args[4]
out_filename=paste(args[4],"",sep="/proc_data/out05_text_all.json")
require(rlist)
list.save(data_processed,out_filename)

# test ----
#filename="/Users/noro/Documents/Projects/XBRLanalysis/data/metadata/Yuho_dummy_words.csv"
#dummywords=read.csv(filename,header=FALSE,stringsAsFactors=FALSE)
#i=1
#res <- RMeCabC(data$V2[1])
#tmp<-unlist(res)
#dropwords=str_subset(tmp,pattern =  dummywords$V2[i])
#tmp<-setdiff(tmp,dropwords)





  
  
#  res <- RMeCabC(data$V2[44])
#res <- RMeCabFreq(data$V2[1])
#tmp=unlist (res)
#tmp=tmp[names(tmp)=="名詞"]
#setdiff(tmp,stopwords)
#tmp[tmp %in% stopwords]

#tmp = map(str,data_processed[1])
#tmp = paste(data_processed[1],sep = " ")
#tmp=str_c(data_processed[1],sep=" ")
#write.csv(data_processed,out_filename)