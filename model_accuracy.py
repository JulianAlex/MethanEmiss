#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:16:23 2019

@author: jadolphs
"""
import os
import numpy as np                # numerics   
import matplotlib.pyplot as plt   # plotting


# create lists with 8 elements
drct           = [None] * 8
dirnam         = [None] * 8
emiss          = [None] * 8
emiss_mean     = [None] * 8
emiss_std      = [None] * 8
pred_err       = [None] * 8
pred_err_mean  = [None] * 8
pred_err_std   = [None] * 8
train_err      = [None] * 8
train_err_mean = [None] * 8
train_err_std  = [None] * 8
test_err       = [None] * 8
test_err_mean  = [None] * 8
test_err_std   = [None] * 8
pred_r2_mean   = [None] * 8
pred_rmse_mean = [None] * 8

drct[0] = 'GradBoost/'
#drct[1] = 'RandForest'
#drct[2] = 'Ridge'
#drct[3] = 'LinReg' 
#drct[4] = 'ANN'
#drct[5] = 'fixAnn'
#drct[6] = 'GaussProc'
#drct[7] = 'SVM'

fsyn  = '/' #'_nofs/'  # feature scaling yes/yo

#for i in range(8):
#    dirnam[i] = drct[i] + fsyn

exp_emiss = 11.6   # g / (h LU)


#for i in range(8):

i = 0

dirnam[i] = drct[i] + fsyn

emiss[i]         = np.load( dirnam[i] + 'emiss.npy',      allow_pickle=True )
emiss_mean[i]    = np.load( dirnam[i] + 'emiss_mean.npy', allow_pickle=True )
emiss_std[i]     = np.load( dirnam[i] + 'emiss_std.npy',  allow_pickle=True ) 

pred_err[i]      = np.load( dirnam[i] + 'pred_err.npy',      allow_pickle=True )
pred_err_mean[i] = np.load( dirnam[i] + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std[i]  = np.load( dirnam[i] + 'pred_err_std.npy',  allow_pickle=True ) 

train_err[i]      = np.load( dirnam[i] + 'train_err.npy',      allow_pickle=True )
train_err_mean[i] = np.load( dirnam[i] + 'train_err_mean.npy', allow_pickle=True )
train_err_std[i]  = np.load( dirnam[i] + 'train_err_std.npy',  allow_pickle=True ) 

test_err[i]      = np.load( dirnam[i] + 'test_err.npy',      allow_pickle=True )
test_err_mean[i] = np.load( dirnam[i] + 'test_err_mean.npy', allow_pickle=True )
test_err_std[i]  = np.load( dirnam[i] + 'test_err_std.npy',  allow_pickle=True ) 

pred_r2_mean[i]   = np.load( dirnam[i] + 'pred_r2_mean.npy', allow_pickle=True )
pred_rmse_mean[i] = np.load( dirnam[i] + 'pred_rmse_mean.npy', allow_pickle=True )

print("test_err_mae")
print(np.round(test_err_mean[i], 3))
print("")
print("pred_err_mae")
print(np.round(pred_err_mean[i], 3))
print("")
print("pred_err_r2")
print(np.round(pred_r2_mean[i], 3))
print("")
print("pred_err_rmse")
print(np.round(pred_rmse_mean[i], 3))

print("")
print("total absolute error")
print(np.round(np.abs( emiss_mean[i] - exp_emiss ), 4))

print("")
print("total absolute error [ % ]")
print(np.round(100*np.abs( emiss_mean[i] - exp_emiss )/exp_emiss, 2))

