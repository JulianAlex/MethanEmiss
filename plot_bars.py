#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:16:23 2019

@author: jadolphs
"""
import os
import numpy as np                # numerics   
import matplotlib.pyplot as plt   # plotting


name   = 'GradBoost'         
dirnam = 'GradBoost/'    
outdir = 'allFigures/' 
imtype = '.jpg'

exp_emiss = 11.6  #  [ g / (h LU) ]

if not os.path.exists(outdir): 
    os.makedirs(outdir)

emiss         = np.load( dirnam + 'emiss.npy',      allow_pickle=True )
emiss_mean    = np.load( dirnam + 'emiss_mean.npy', allow_pickle=True )
emiss_std     = np.load( dirnam + 'emiss_std.npy',  allow_pickle=True ) 

pred_err      = np.load( dirnam + 'pred_err.npy',      allow_pickle=True ) 
pred_err_mean = np.load( dirnam + 'pred_err_mean.npy', allow_pickle=True ) 
pred_err_std  = np.load( dirnam + 'pred_err_std.npy',  allow_pickle=True )

train_err      = np.load( dirnam + 'train_err.npy',      allow_pickle=True ) 
train_err_mean = np.load( dirnam + 'train_err_mean.npy', allow_pickle=True ) 
train_err_std  = np.load( dirnam + 'train_err_std.npy',  allow_pickle=True )

test_err      = np.load( dirnam + 'test_err.npy',      allow_pickle=True ) 
test_err_mean = np.load( dirnam + 'test_err_mean.npy', allow_pickle=True ) 
test_err_std  = np.load( dirnam + 'test_err_std.npy',  allow_pickle=True )

test_r2      = np.load( dirnam + 'test_r2.npy',      allow_pickle=True ) 
test_r2_mean = np.load( dirnam + 'test_r2_mean.npy', allow_pickle=True ) 
test_r2_std  = np.load( dirnam + 'test_r2_std.npy',  allow_pickle=True )

test_rmse      = np.load( dirnam + 'test_rmse.npy',      allow_pickle=True ) 
test_rmse_mean = np.load( dirnam + 'test_rmse_mean.npy', allow_pickle=True ) 
test_rmse_std  = np.load( dirnam + 'test_rmse_std.npy',  allow_pickle=True )


plt.close('all')


#x      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120']


# plt.legend(loc="best")


#------------------------------------------------------------------------------
# Emission and pred_error 
#------------------------------------------------------------------------------

x1      = np.array( range(0, len(emiss_mean)) )
x1sort  = np.flipud( np.argsort(emiss_mean) )
labels1 = x1sort + 1

x2      = np.array( range(0, len(pred_err_mean)) )
x2sort  = np.argsort(pred_err_mean) 
labels2 = x2sort + 1


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

im1 = ax1.plot([0, len(emiss_mean)-1], [exp_emiss, exp_emiss], '-', c='red', label="Experiment" )
im1 = ax1.bar(x1, emiss_mean[x1sort], alpha = 0.5, label='mean' )
im1 = ax1.bar(x1, emiss_mean[x1sort] + emiss_std[x1sort], alpha=0.5, label='std')
im1 = ax1.bar(x1, emiss_mean[x1sort] - emiss_std[x1sort], alpha=0.5)

ax1.tick_params(labelsize=11)
ax1.set_xticks( x1 )
ax1.set_xticklabels( labels1 )
ax1.set_ylim([10.6, 12.4])
ax1.set_xlabel('Scenario index', fontsize = 13)
ax1.set_ylabel('Emission   /   g h$^{-1}$ LU$^{-1}$', fontsize = 13)

im2 = ax2.bar(x2, pred_err_mean[x2sort], alpha = 0.5, label='mean' )
im2 = ax2.bar(x2, pred_err_mean[x2sort] + pred_err_std[x2sort], alpha=0.5, label='std')
im2 = ax2.bar(x2, pred_err_mean[x2sort] - pred_err_std[x2sort], alpha=0.5 )

ax2.tick_params(labelsize = 11)
ax2.set_xticks( x2 )
ax2.set_xticklabels( labels2 )
ax2.set_ylim([1, 2])
ax2.set_xlabel('Scenario index', fontsize = 13)
ax2.set_ylabel('Extrapolation error   /   g h$^{-1}$ LU$^{-1}$', fontsize = 13)

plt.savefig(outdir + 'bar_emiss_pred_err' + imtype, dpi=150, format='jpg')

plt.show(block=False)


#------------------------------------------------------------------------------
#  Predict_Error
#------------------------------------------------------------------------------
'''
xx     = np.array( range(0, len(pred_err_mean)) )
xsort  = np.flipud( np.argsort(pred_err_mean) )
labels = xsort + 1

fig = plt.figure( figsize = (10, 5) )
plt.bar(xx, pred_err_mean[xsort], alpha = 0.5, label='mean' )
plt.bar(xx, pred_err_mean[xsort] + pred_err_std[xsort], alpha=0.5, label='std')
plt.bar(xx, pred_err_mean[xsort] - pred_err_std[xsort], alpha=0.5 )
plt.ylim(0.3, 0.6)
plt.xticks(xx, labels)
plt.xlabel('Scenario index')
plt.ylabel('Prediction error')
plt.show(block=False)
#plt.savefig(outdir + 'bar_pred_err' + imtype, dpi=150, format='jpg')


#------------------------------------------------------------------------------
#  Test_Error

#------------------------------------------------------------------------------

xx     = np.array( range(0, len(test_err_mean)) )
xsort  = np.flipud( np.argsort(test_err_mean) )
labels = xsort + 1

fig = plt.figure( figsize = (10, 5) )
plt.bar(xx, test_err_mean[xsort], alpha = 0.5, label='mean' )
plt.bar(xx, test_err_mean[xsort] + test_err_std[xsort], alpha=0.5, label='std')
plt.bar(xx, test_err_mean[xsort] - test_err_std[xsort], alpha=0.5 )
plt.xticks(xx, labels)
plt.xlabel('Scenario index')
plt.ylabel('Test error')
plt.show(block=False)
#plt.savefig(outdir + 'bar_test_err' + imtype, dpi=150, format='jpg')



#------------------------------------------------------------------------------
# Emission  
#------------------------------------------------------------------------------

xx     = np.array( range(0, len(emiss_mean)) )
xsort  = np.flipud( np.argsort(emiss_mean) )
labels = xsort + 1

fig = plt.figure( figsize = (10, 5) )
plt.plot([0, len(emiss_mean)-1], [exp_emiss, exp_emiss], '-', c='red', label="Experiment" )
plt.bar(xx, emiss_mean[xsort], alpha = 0.5, label='mean' )
plt.bar(xx, emiss_mean[xsort] + emiss_std[xsort], alpha=0.5, label='std')
plt.bar(xx, emiss_mean[xsort] - emiss_std[xsort], alpha=0.5)
plt.ylim(6, 12)
plt.xticks(xx, labels)
plt.xlabel('Scenario index')
plt.ylabel('Emission')
#plt.legend()
plt.show(block=False)
#plt.savefig(outdir + 'bar_emiss' + imtype, dpi=150, format='jpg')
'''



#------------------------------------------------------------------------------
'''
#mnstd  = np.stack(( emiss_mean, emiss_std ))
#mnstd  = np.sort(mnstd, ) 

xx     = np.array( range(1, len(emiss_mean) + 1) )
xsort  = np.flipud( np.argsort(emiss_mean) + 1 )
labels = xsort

fig = plt.figure()
plt.plot([1, len(emiss_mean)], [exp_emiss, exp_emiss], '-', c='red', label="exp_emiss" )
plt.bar(xx, np.flipud( np.sort(emiss_mean) ))
plt.xticks(xx, labels)
plt.xlabel('Scenario index')
plt.ylabel('Emission')
plt.show(block=False)
plt.savefig(outdir + 'emiss' + imtype, dpi=150, format='jpg')
'''