#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 00:06:35 2020

@author: 
References;
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py

"""

# %% params

import argparse
parser = argparse.ArgumentParser(description='')

runkey='_1'
runkey_b='_1'
parser.add_argument('--scratch', default="../results/run"+runkey, type=str, help='results dir')
parser.add_argument('--cwd', type=str, help='base dir')
parser.add_argument('--input_fsdata', default=r'../results/run'+runkey+'/proc_data/out07_NMFscore.csv', type=str, help='input file path')

parser.add_argument('--input_textdata', default=r'../results/run'+runkey+'/proc_data/out06_lda_score.csv', type=str, help='input file path')

parser.add_argument('--num_epochs', default="50", type=int, help='fast, full')
parser.add_argument('--batch_size', default="1024", type=int, help='fast, full')
parser.add_argument('--nccadim', default=20, type=int, help='ncca dimension')

parser.add_argument('--input_fsdata_org', default=r'../results/run'+runkey+'/proc_data/out03_fsdata_prep4_train.pkl', type=str, help='fsdata')
parser.add_argument('--input_nmf_comp', default=r'../results/run'+runkey+'/proc_data/out07_NMFcomp.csv', type=str, help='lda dict')

parser.add_argument('--input_dict', default=r'../run'+runkey+'/proc_data/out06_dictionary.dict', type=str, help='lda dict')
parser.add_argument('--input_lda', default=r'../results/run'+runkey+'/proc_data/out06_lda.model', type=str, help='lda model')
parser.add_argument('--base_scratch', default='../results/run'+runkey_b+'', type=str, help='results dir')
parser.add_argument('--account_summerized', default=True, type=bool, help='use exposured account name as label')

args = parser.parse_args()
SCRATCH=args.scratch



# %%
# Reference
# The following code is a modification of the following code
# https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py
# Copyright (c) 2017 Utku Ozbulak


import os
import numpy as np
import torch.nn as nn

import torch
from torch.optim import Adam
#from torchvision import models

#from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,weights):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.weights=weights
        # Create the folder to export images if not exists
        
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        #random_image = np.uint8(np.random.uniform(0, 2, 400))
        random_input = np.uint8(np.random.uniform(0, 2, 400),requires_grad=True)
        # Process image and return variable
        #processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([random_input], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = random_input
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            
    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        #random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        random_input = np.random.rand(1, 400)*2#.float()
        # Process image and return variable
        x_input=torch.tensor(random_input,requires_grad=True)
        #processed_image = preprocess_image(random_image, False)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input], lr=0.1, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = x_input
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            #self.conv_output = x[0, self.selected_filter]
            x_n=x-torch.ones(1,x.size(1))*torch.mean(x[0,:])
            #self.conv_output = torch.abs(torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter]))) # must fix to substitute mean
            self.conv_output = torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter])) # must fix to substitute mean
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.sum( -torch.mean(self.conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
            #target = torch.sum( -torch.mean(self.conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
            #loss=-torch.autograd.grad(target, x[0,:], create_graph=True,allow_unused=True)[0]
            #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            savelist.append(self.conv_output)
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
        return(x_input,loss)
    
    def visualise_layer_without_hooks_m(self):
        random_input = np.random.rand(1, 400)*2#.float()
        x_input=torch.tensor(random_input,requires_grad=True)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input], lr=0.1, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            x = x_input
            for index, layer in enumerate(self.model):
                x = layer(x)
                if index == self.selected_layer:
                    break
            x_n=x-torch.ones(1,x.size(1))*torch.mean(x[0,:])
            self.conv_output = -torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter])) # must fix to substitute mean
            loss = torch.sum( -torch.mean(self.conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
            #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            savelist.append(self.conv_output)
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
        return(x_input)
    def visualise_layer_without_hooks_target(self,target):
        random_input = np.random.rand(1, 400)*2#.float()
        x_input=torch.tensor(random_input,requires_grad=True)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input], lr=0.05, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            x = x_input
            for index, layer in enumerate(self.model):
                x = layer(x)
                if index == self.selected_layer:
                    break
            x_n=x-torch.ones(1,x.size(1))*torch.mean(x[0,:])
            self.conv_output = -torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter])) # must fix to substitute mean
            loss = torch.sum(torch.pow((self.conv_output-torch.tensor(target,requires_grad=True)),2)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
            print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            savelist.append(self.conv_output)
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
        return(x_input)



class CNNLayerVisualization_mvl():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model1, model2, selected_layer1, selected_filter1, weights1, selected_layer2, selected_filter2,weights2):
        self.model1 = model1
        self.model1.eval()
        self.selected_layer1 = selected_layer1
        self.selected_filter1 = selected_filter1
        self.weights1=weights1
        
        self.model2 = model2
        self.model2.eval()
        self.selected_layer2 = selected_layer2
        self.selected_filter2 = selected_filter2
        self.weights2=weights2
        
        self.conv_output = 0
        
    def visualise_layer_without_hooks(self):
        random_input1 = np.random.rand(1, 400)*2
        random_input2 = np.random.rand(1, 400)*2
        # Process image and return variable
        x_input1=torch.tensor(random_input1,requires_grad=True)
        x_input2=torch.tensor(random_input2,requires_grad=True)
        #processed_image = preprocess_image(random_image, False)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input1,x_input2], lr=0.02, weight_decay=1e-6)
        for i in range(1, 501):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x1 = x_input1
            for index1, layer1 in enumerate(self.model1):
                x1 = layer1(x1)
                if index1 == self.selected_layer1:
                    break
            x2 = x_input2
            for index2, layer2 in enumerate(self.model2):
                x2 = layer2(x2)
                if index2 == self.selected_layer2:
                    break
            x_n1=x1-torch.ones(1,x1.size(1))*torch.mean(x1[0,:])
            x_n2=x2-torch.ones(1,x2.size(1))*torch.mean(x2[0,:])
            w_out1 = torch.dot(x_n1[0,:],torch.tensor(self.weights1[:,self.selected_filter1])) # must fix to substitute mean
            w_out2 = torch.dot(x_n2[0,:],torch.tensor(self.weights2[:,self.selected_filter2]))
            loss = torch.sum( -w_out1-w_out2+torch.pow((w_out1-w_out2),2)+0.1*torch.sum(torch.pow(x_input1,2))+0.1*torch.sum(torch.abs(x_input1))+0.1*torch.sum(torch.pow(x_input2,2))+0.1*torch.sum(torch.abs(x_input2)))
            #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            loss.backward()
            optimizer.step()
        return(x_input1,x_input2)

    def visualise_layer_without_hooks_m(self):
        random_input1 = np.random.rand(1, 400)*2
        random_input2 = np.random.rand(1, 400)*2
        # Process image and return variable
        x_input1=torch.tensor(random_input1,requires_grad=True)
        x_input2=torch.tensor(random_input2,requires_grad=True)
        #processed_image = preprocess_image(random_image, False)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input1,x_input2], lr=0.02, weight_decay=1e-6)
        for i in range(1, 501):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x1 = x_input1
            for index1, layer1 in enumerate(self.model1):
                x1 = layer1(x1)
                if index1 == self.selected_layer1:
                    break
            x2 = x_input2
            for index2, layer2 in enumerate(self.model2):
                x2 = layer2(x2)
                if index2 == self.selected_layer2:
                    break
            
            x_n1=x1-torch.ones(1,x1.size(1))*torch.mean(x1[0,:])
            x_n2=x2-torch.ones(1,x2.size(1))*torch.mean(x2[0,:])
            w_out1 = -torch.dot(x_n1[0,:],torch.tensor(self.weights1[:,self.selected_filter1])) # must fix to substitute mean
            w_out2 = -torch.dot(x_n2[0,:],torch.tensor(self.weights2[:,self.selected_filter2]))
            loss = torch.sum( -w_out1-w_out2+torch.pow((w_out1-w_out2),2)+0.1*torch.sum(torch.pow(x_input1,2))+0.1*torch.sum(torch.abs(x_input1))+0.1*torch.sum(torch.pow(x_input2,2))+0.1*torch.sum(torch.abs(x_input2)))
            #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            loss.backward()
            optimizer.step()
        return(x_input1,x_input2)

# %%
        
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


model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                input_shape2, outdim_size, use_all_singular_values, device=device).double()

# %%

#filter_pos = 5
#Weight_fs=solver.linear_cca.w[0]
modelname='dcca_opt'
Weight_fs=np.load(SCRATCH+"/DCCA/"+modelname+"out20_nmf_dcca_train_weight.npy")

model.load_state_dict(torch.load(SCRATCH + "/DCCA/"+modelname+"out_b20_trained.model"))

# Fully connected layer is not needed
#pretrained_model = models.vgg16(pretrained=True).features
# %%
num=1
cnn_layer = 5
layer_vis = CNNLayerVisualization(model.model1.layers, cnn_layer, num, Weight_fs)

#model1, model2, selected_layer1, selected_filter1, weights1, selected_layer2, selected_filter2,weights2)
# Layer visualization with pytorch hooks

tmp2=layer_vis.visualise_layer_without_hooks()
#pd.Series(tmp2.detach().numpy()[0]).hist()
# %%
sample_num=20
Weight_text=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_train_weight.npy")
text_layer=7
fs_layer=5
layer_vis = CNNLayerVisualization_mvl(model.model1.layers,model.model2.layers, fs_layer, num, Weight_fs,text_layer, num, Weight_text)
tmp2,tmp3=layer_vis.visualise_layer_without_hooks()
# %%
dcca_comp_tmp=pd.DataFrame()
for itr in range(sample_num):
    tmp2=layer_vis.visualise_layer_without_hooks()
    dcca_comp_tmp[itr]=tmp2.detach().numpy()[0]

# %%
import matplotlib.pyplot as plt
% matplotlib inline
fig=plt.figure(figsize=(5,5))
plt.imshow(np.corrcoef(dcca_comp_tmp.T))
fig.show()
# %%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


ncca_proj_link = linkage(dcca_comp_tmp, method='ward')

fig, ax = plt.subplots(figsize=(15,5))
plt.subplot(1,3,1)
ydendro_lda=dendrogram(ncca_proj_link, p=sample_num, truncate_mode='lastp', orientation='left',
                   leaf_font_size=4)

fig.savefig(SCRATCH+'/tmp.pdf')
clusters = fcluster(ncca_proj_link, t=2, criterion='maxclust')







# %%
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary

import matplotlib.pyplot as plt

from tqdm import tqdm
#import wordcloud as wc 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from matplotlib.backends.backend_pdf import PdfPages

nmf_comp=pd.read_csv(args.input_nmf_comp,index_col=None,header=None)
#tmp3=tmp2.detach().numpy()[0]
# %% LDA
fsdata=pd.read_pickle(args.input_fsdata_org)
transcripter=pd.read_csv(args.base_scratch+"/transcripter_summerized_account.csv",header=0, index_col=0)

#fsdata_nmf_projection=pd.DataFrame(np.dot(nmf_comp.values,tmp3),index=transcripter.loc[fsdata.columns,"標準ラベル（日本語）"])
#fsdata_nmf_projection_barh=pd.DataFrame(np.dot(nmf_comp.values,tmp3),index=fsdata.columns)
# %%
dcca_comp_fs=pd.DataFrame()
dcca_comp_text=pd.DataFrame()
cnn_layer = 5
for num in range(args.nccadim):
    
    #layer_vis = CNNLayerVisualization_mvl(model.model1.layers, cnn_layer, num, Weight_fs)
    layer_vis = CNNLayerVisualization_mvl(model.model1.layers,model.model2.layers, fs_layer, num, Weight_fs,text_layer, num, Weight_text)
    tmp2,tmp3=layer_vis.visualise_layer_without_hooks()
    dcca_comp_fs['dim'+str(num)]=tmp2.detach().numpy()[0]
    dcca_comp_text['dim'+str(num)]=tmp3.detach().numpy()[0]



dcca_comp_fs.to_csv(SCRATCH + "/proc_data/out21_DCCAscore_fs.csv",index=False,header=False)
dcca_comp_text.to_csv(SCRATCH + "/proc_data/out21_DCCAscore_text.csv",index=False,header=False)


dcca_comp_fs_m=pd.DataFrame()
dcca_comp_text_m=pd.DataFrame()
cnn_layer = 5
for num in range(args.nccadim):
    
    layer_vis = CNNLayerVisualization_mvl(model.model1.layers,model.model2.layers, fs_layer, num, Weight_fs,text_layer, num, Weight_text)
    tmp2,tmp3=layer_vis.visualise_layer_without_hooks_m()
    dcca_comp_fs_m['dim'+str(num)]=tmp2.detach().numpy()[0]
    dcca_comp_text_m['dim'+str(num)]=tmp3.detach().numpy()[0]

dcca_comp_fs_m.to_csv(SCRATCH + "/proc_data/out21_DCCAscore_fs_m.csv",index=False,header=False)
dcca_comp_text_m.to_csv(SCRATCH + "/proc_data/out21_DCCAscore_text_m.csv",index=False,header=False)
    # fsdata_nmf_projection_barh=pd.DataFrame(np.dot(nmf_comp.values,tmp3),index=fsdata.columns)
    
    # fig=plt.figure(figsize=(8,10))
    # ax=fig.add_subplot(1,2,2)
    # tmp=fsdata_nmf_projection_barh.loc[:,0].copy()
    # tmp=tmp.sort_values(ascending=False)
    # tmp=tmp.iloc[:bar_num].sort_values(ascending=True)
    # col_mask=tmp.index.str.contains("up")
    # col=pd.Series([coler_increase,]*20)
    # col[~col_mask]=coler_decrease
    # #tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
    # tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（日本語）"].str.strip()
    
    
    # ax.barh(range(bar_num), tmp.values, align = 'center',color=col)
    # plt.yticks(range(bar_num), tmp.index.to_list())
    # ax.grid(True)
    
    
    
    # fig.suptitle(" #" + str(num))
    # fig.savefig(SCRATCH + "/DCCA/comp_" + str(num) + ".pdf", transparent=True)

# %%

# for index, layer in enumerate(model.model1.layers):
#     print(index)
#     if index > 30:
#                     # (forward hook function triggered)
#         break


# %%



class CNNLayerVisualization_text():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,weights):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.weights=weights
        # Create the folder to export images if not exists
        
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

            
    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        #random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        random_input = np.random.rand(1, 400)*2#.float()
        # Process image and return variable
        x_input=torch.tensor(random_input,requires_grad=True)
        #processed_image = preprocess_image(random_image, False)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input], lr=0.1, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = x_input
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            #self.conv_output = x[0, self.selected_filter]
            x_n=x-torch.ones(1,x.size(1))*torch.mean(x[0,:])
            self.conv_output = torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter])) # must fix to substitute mean
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.sum( -torch.mean(self.conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
            #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            savelist.append(self.conv_output)
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
        return(x_input)


class CNNLayerVisualization_text_m():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,weights):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.weights=weights
        # Create the folder to export images if not exists
        
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

            
    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        #random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        random_input = np.random.rand(1, 400)*2#.float()
        # Process image and return variable
        x_input=torch.tensor(random_input,requires_grad=True)
        #processed_image = preprocess_image(random_image, False)
        savelist=[]
        # Define optimizer for the image
        optimizer = Adam([x_input], lr=0.1, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = x_input
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            #self.conv_output = x[0, self.selected_filter]
            x_n=x-torch.ones(1,x.size(1))*torch.mean(x[0,:])
            self.conv_output = -torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter])) # must fix to substitute mean
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.sum( -torch.mean(self.conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
            #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
            savelist.append(self.conv_output)
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
        return(x_input)

# %%
text_out_layer=7
#Weight_text
Weight_text=np.load(SCRATCH+"/DCCA/"+modelname+"out20_lda_dcca_train_weight.npy")
num=0
layer_vis = CNNLayerVisualization_text(model.model2.layers, text_out_layer, num, Weight_text)


tmp2=layer_vis.visualise_layer_without_hooks()
tmp3=tmp2.detach().numpy()[0]






# %%
coler_increase='tomato'
coler_decrease='darkblue'#'#00ff00'
bar_num=20
dcca_comp=pd.DataFrame()
for num in range(args.nccadim):
    
    layer_vis = CNNLayerVisualization_text(model.model2.layers, text_out_layer, num, Weight_text)
    
    tmp2=layer_vis.visualise_layer_without_hooks()
    dcca_comp['dim'+str(num)]=tmp2.detach().numpy()[0]
    
dcca_comp.to_csv(SCRATCH + "/proc_data/out21_DCCAscore_text.csv",index=False,header=False)

dcca_comp_m=pd.DataFrame()
for num in range(args.nccadim):
    
    layer_vis = CNNLayerVisualization_text(model.model2.layers, text_out_layer, num, Weight_text)
    
    tmp2=layer_vis.visualise_layer_without_hooks()
    dcca_comp_m['dim'+str(num)]=tmp2.detach().numpy()[0]
    
dcca_comp_m.to_csv(SCRATCH + "/proc_data/out21_DCCAscore_text_m.csv",index=False,header=False)



# %%

random_input = np.random.rand(1, 400)*2#.float()
# Process image and return variable
x_input=torch.tensor(random_input,requires_grad=True)
#processed_image = preprocess_image(random_image, False)
savelist=[]
# Define optimizer for the image
optimizer = Adam([x_input], lr=0.1, weight_decay=1e-6)
for i in range(1, 201):
    optimizer.zero_grad()
    # Assign create image to a variable to move forward in the model
    x = x_input
    for index, layer in enumerate(model.model1.layers):
        # Forward pass layer by layer
        
        x = layer(x)
        if index == 5:
            # Only need to forward until the selected layer is reached
            # Now, x is the output of the selected layer
            break
    # Here, we get the specific filter from the output of the convolution operation
    # x is a tensor of shape 1x512x28x28.(For layer 17)
    # So there are 512 unique filter outputs
    # Following line selects a filter from 512 filters so self.conv_output will become
    # a tensor of shape 28x28
    #self.conv_output = x[0, self.selected_filter]
    x_n=x-torch.ones(1,x.size(1),requires_grad=True)*torch.mean(x[0,:])
    #self.conv_output = torch.abs(torch.dot(x_n[0,:],torch.tensor(self.weights[:,self.selected_filter]))) # must fix to substitute mean
    conv_output = torch.dot(x_n[0,:],torch.tensor(Weight_fs[:,num],requires_grad=True)) # must fix to substitute mean
    # Loss function is the mean of the output of the selected layer/filter
    # We try to minimize the mean of the output of that specific filter
    #loss = torch.sum( -torch.mean(self.conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
    target = torch.sum( -torch.mean(conv_output)+torch.sum(torch.pow(x_input,2))+torch.sum(torch.abs(x_input)))
    #target = -torch.sum( -torch.mean(conv_output)+torch.sum(torch.pow(x_input,2)))
    #target=torch.sum(torch.pow(x_input[0,:],3))
    tt=x_input[0,:] ** 3
    target=torch.sum(tt)
    loss=torch.autograd.grad(target, x_input[0,:], create_graph=True)[0]
    #print('Iteration:', str(i), 'Loss:', "{0:.5f}".format(loss.data.numpy()))
    savelist.append(conv_output)
    # Backward
    loss.backward()
    # Update image
    optimizer.step()
    
    
# %%
#x = torch.tensor([4.]*2, requires_grad=True)
x_input=torch.tensor(random_input[0,:],requires_grad=True)
#x_input=torch.tensor(,requires_grad=True)
f = torch.sum(x_input ** 3)
g = torch.autograd.grad(f, x_input, create_graph=True,allow_unused=True)[0]
g.backward()

