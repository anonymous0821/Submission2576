# ----
# encoded utf-8
# Nonnegative CCA
# 
# Input: 
# Output:
# 
# Refferences:
# https://rdrr.io/cran/nscancor/man/nscancor.html
# https://cran.rstudio.com/web/packages/nscancor/nscancor.pdf
# https://sigg-iten.ch/learningbits/2014/01/20/canonical-correlation-analysis-under-constraints/
# Clear global workspace
# http://htsuda.net/archives/1545
# rm( list = ls( envir = globalenv() ), envir = globalenv() )
# ----

args <- commandArgs(trailingOnly = T)
default_args <- c('/Users/noro/Documents/Projects/XBRLanalysis/data/proc_data/out07_NMFscore_dim50.csv',
                  '/Users/noro/Documents/Projects/XBRLanalysis/data/proc_data/out06_2_lda_scores.csv',
                  '/Users/noro/Documents/Projects/XBRLanalysis/data',
                  '10',
                  '0',
                  '0',
                  '5')
default_flg <- is.na(args[1:7])
args[default_flg] <- default_args[default_flg]


library(tidyverse)
filename=args[1]
fs_score=read.csv(filename,header=FALSE)

filename=args[2]
doc_score=read.csv(filename,header=FALSE)


require(MASS)
ypredict <- function(x, yc, cc) {
  en <- glmnet::glmnet(x, yc, 
                       alpha = as.numeric(args[5]),
                       intercept = FALSE,
                       lower.limits = 0)
  W <- coef(en)
  return(W[2:nrow(W), ncol(W)])
}
xpredict <- function(y, xc, cc) {
  en <- glmnet::glmnet(y, xc,
                       alpha = as.numeric(args[6]),
                       intercept = FALSE,
                       lower.limits = 0 # optional
                       )
  V <- coef(en)
  return(V[2:nrow(V), ncol(V)])
}
require(glmnet)
library(nscancor)
source('~/Documents/Projects/XBRLanalysis/src/b08anly_NCCA_null.R', echo=TRUE)

nscc <- nscancor_test(args,fs_score, doc_score, nvar = as.integer(args[4]),
                 xcenter = TRUE, ycenter = TRUE,
                 xscale = TRUE, yscale = TRUE,
                 #xscale = FALSE, yscale = FALSE,
                 nrestart = 200,
                 minimumitr=5,
                 verbosity=1,
                 xpredict = xpredict, ypredict = ypredict)

out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_fscomp.csv")
write.csv(nscc$xcoef,out_filename)
out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_ldacomp.csv")
write.csv(nscc$ycoef,out_filename)
out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_fsscore.csv")
write.csv(nscc$xp,out_filename)
out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_ldascore.csv")
write.csv(nscc$yp,out_filename)

out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_fs_scaled.csv")
write.csv(scale(fs_score,center = nscc$xcenter
                ,scale =nscc$xscale
                ),out_filename)
#write.csv(fs_score,out_filename)
out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_lda_scaled.csv")
write.csv(scale(doc_score,center = nscc$ycenter
                ,scale =nscc$yscale
                ),out_filename)
#write.csv(doc_score,out_filename)


out_filename=paste(args[3],"",sep="/proc_data/out08_ncca_cor.csv")
write.csv(nscc$cor,out_filename)


## test ----

