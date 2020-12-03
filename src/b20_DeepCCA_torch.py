#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:04:05 2020




@author:
"""


# %% ##########################################################################
# Requirements
###############################################################################

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

torch.set_default_tensor_type(torch.DoubleTensor)


import time
import logging

import matplotlib.pyplot as plt

# %% ##########################################################################
# args
###############################################################################


import argparse
parser = argparse.ArgumentParser(description='')
runkey='_1'

parser.add_argument('--scratch', default="../results/run"+runkey, type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input_fsdata_train', default=r"../results/run" + runkey + "/proc_data/out07_NMFscore.csv", type=str, help='fsdata')
parser.add_argument('--input_fsdata_test', default=r"../results/run" + runkey + "//proc_data/out07_NMFscore_test.csv", type=str, help='fsdata')

parser.add_argument('--input_textdata_train', default=r'../results/run' + runkey + '/proc_data/out06_lda_score.csv', type=str, help='input file path')
parser.add_argument('--input_textdata_test', default=r'../results/run' + runkey + '/proc_data/out06_lda_score.csv', type=str, help='input file path')

parser.add_argument('--num_epochs', default="50", type=int, help='fast, full')
parser.add_argument('--batch_size', default="800", type=int, help='fast, full')
parser.add_argument('--nccadim', default=20, type=int, help='ncca dimension')


args = parser.parse_args()
SCRATCH=args.scratch
modelname='dcca_opt'
checkpoint_filename=SCRATCH+'/DCCA/'+modelname+'DCCA_checkpoint.model'
num_epochs=args.num_epochs

log_filename=SCRATCH+'/DCCA/'+modelname+'DCCA.log'

random_seed=None



# %% ##########################################################################
# Define model
###############################################################################

# Referense
# The following code is a modification of the following code
# https://github.com/Michaelvll/DeepCCA
#
# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu



class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MlpNetC1(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNetC1, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers) # parameter as list layers are not be renew so that convert modulelist.

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MlpNetC2(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNetC2, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers) # parameter as list layers are not be renew so that convert modulelist.

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNetC1(layer_sizes1, input_size1).double()
        self.model2 = MlpNetC2(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2



class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu'),log_filename=log_filename):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler(log_filename)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint=checkpoint_filename):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        #x1.to(self.device)
        #x2.to(self.device)

        data_size = x1.size(0)
        best_val_loss = 0

        #if vx1 is not None and vx2 is not None:
            #best_val_loss = 0
            #vx1.to(self.device)
            #vx2.to(self.device)
        #if tx1 is not None and tx2 is not None:
            #tx1.to(self.device)
            #tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs_lcca = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs_lcca, outputs
            else:
                return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs



class linear_cca():
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r1 = 1e-4
        r2 = 1e-4

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m[0] = np.mean(H1, axis=0)
        self.m[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m[0], (m, 1))
        H2bar = H2 - np.tile(self.m[1], (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T,
                                                 H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T,
                                                 H2bar) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(
            np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(
            np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv,
                                   SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w[idx])
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)



# %% ##########################################################################
# Set conditions
###############################################################################

# Parameters Section
device=torch.device("cpu")
#net.to(device)
#device = torch.device('cuda')

#save_to = './new_features.gz'

# the size of the new space learned by the model (number of the new features)
outdim_size =args.nccadim
batch_size=1100
# size of the input for view 1 and view 2
input_shape1 = 400
input_shape2 = 400

l_size1=600
l_size2=400
# number of layers with nodes in each one
#layer_sizes1 = [l_size, l_size, l_size, outdim_size]
layer_num1=5
layer_sizes1 = []
for itr in range(layer_num1):
    layer_sizes1.append(l_size1)
layer_sizes1.append(outdim_size)

#layer_sizes2 = [l_size2, l_size2, l_size2, outdim_size]
layer_num2=7
layer_sizes2 = []
for itr in range(layer_num2):
    layer_sizes2.append(l_size2)
layer_sizes2.append(outdim_size)
# the parameters for training the network
learning_rate = 0.0009
epoch_num = args.num_epochs
#batch_size = args.batch_size

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 8.5*1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False

# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
apply_linear_cca = True
# end of parameters section
############

# %% ##########################################################################
# Data
###############################################################################

#filename1=args.input_fsdata
#fs_data=pd.read_csv(filename1,encoding='utf-8',index_col=None,header=None).values

#fs_score_train=pd.read_csv(SCRATCH + '/proc_data/out08_ncca_fs_scaled.csv',encoding='utf-8',index_col=0)

nmf_score_test=pd.read_csv(SCRATCH + "/proc_data/out07_NMFscore_test.csv",header=None, index_col=None).values
nmf_score_train=pd.read_csv(SCRATCH + "/proc_data/out07_NMFscore.csv",header=None, index_col=None).values
#nmf_score_test=nmf_score_test/nmf_score_train.max()
#nmf_score_test=(nmf_score_test-nmf_score_train.mean())/np.sqrt(nmf_score_train.var())
#nmf_score_train=nmf_score_train/nmf_score_train.max()
#nmf_score_train=(nmf_score_train-nmf_score_train.mean())/np.sqrt(nmf_score_train.var())
#fs_score_test=(nmf_score-nmf_score_train.mean())/np.sqrt(nmf_score_train.var())

lda_score_train=pd.read_csv(SCRATCH + '/proc_data/out06_lda_score.csv',header=None,index_col=None).values
lda_score_test=np.load(SCRATCH+"/proc_data/out18_lda_score_test.npy")

#lda_score_test=(lda_score_test-lda_score_train.mean())/np.sqrt(lda_score_train.var())
#lda_score_train=(lda_score_train-lda_score_train.mean())/np.sqrt(lda_score_train.var())



# %% ##########################################################################
# Set solver and data
###############################################################################


model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                input_shape2, outdim_size, use_all_singular_values, device=device).double()

        
l_cca = None
if apply_linear_cca:
    l_cca = linear_cca()
solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                learning_rate, reg_par, device=device)

def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)
    return data_x, data_y


train1, train2 = make_tensor((nmf_score_train, lda_score_train))
val1, val2 = make_tensor((nmf_score_test,lda_score_test))
test1, test2 = make_tensor((nmf_score_test,lda_score_test))


# %% ##########################################################################
# fit (requiring computational power)
###############################################################################

solver.fit(train1, train2, val1, val2, test1, test2)
torch.save(model.state_dict(), SCRATCH + "/DCCA/"+modelname+"out_b20_trained.model")
np.save(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_train_weight.npy",solver.linear_cca.w[0])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_train_weight.npy",solver.linear_cca.w[1])


# %% ##########################################################################
# Evaluate train test canonical correlations
###############################################################################
set_size = [0, train1.size(0), train1.size(
    0) + val1.size(0), train1.size(0) + val1.size(0) + test1.size(0)]

# test outputs are NN output projected onto canonical space
loss, outputs, NN_outputs = solver.test(torch.cat([test1], dim=0), torch.cat(
    [test2], dim=0), apply_linear_cca)


np.save(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_output_test.npy",outputs[0])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_output_test.npy",outputs[1])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_NNoutput_test.npy",NN_outputs[0])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_NNoutput_test.npy",NN_outputs[1])



import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

corr_score=[]
for itr_dim in range(20):
    corr_score.append(np.corrcoef(outputs[0][:,itr_dim],outputs[1][:,itr_dim])[1,0])
fig=plt.figure(figsize=(10,20))
plt.plot(corr_score)
plt.ylim([0,1])
fig.savefig(SCRATCH + "/DCCA_score_tests", transparent=True)

# train
loss, outputs, NN_outputs = solver.test(torch.cat([train1], dim=0), torch.cat(
    [train2], dim=0), apply_linear_cca)

np.save(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_output_train.npy",outputs[0])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_output_train.npy",outputs[1])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_NNoutput_train.npy",NN_outputs[0])
np.save(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_NNoutput_train.npy",NN_outputs[1])

corr_score=[]
for itr_dim in range(20):
    corr_score.append(np.corrcoef(outputs[0][:,itr_dim],outputs[1][:,itr_dim])[1,0])
fig=plt.figure(figsize=(10,20))
plt.plot(corr_score)
plt.ylim([0,1])
fig.savefig(SCRATCH + "/DCCA_scores_train", transparent=True)




