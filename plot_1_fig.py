#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:16:23 2019

@author: jadolphs
"""

import numpy as np                # numerics   
import matplotlib.pyplot as plt   # plotting


name   = 'GradBoost' #'fixANN'#'LR'
dirnam = 'GradBoost/'#_nofs/'  # 1to14/'  # 'fixAnn_fs/' 

exp_emiss = 11.6   # g / (h LU)

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



x      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120']


fig = plt.figure()
plt.boxplot( np.transpose(emiss) )   # , whis=[0.5, 99.5] )
plt.plot([0, len(emiss) + 1], [exp_emiss, exp_emiss], '--', c='red', label="exp_emiss" )
#plt.xticks(x, labels)
plt.ylabel('Emission')
plt.title('Emission, ' + name) #, pad = -15 )
plt.show(block=False)
plt.savefig(dirnam + 'emiss' + '_Box' + '.png')

fig = plt.figure()
plt.boxplot( np.transpose(pred_err) )
#plt.xticks(x, labels)
plt.ylabel('Error')
plt.title('Prediction_Error, ' + name)
plt.show(block=False)
plt.savefig(dirnam + 'pred_err' + '_Box' + '.png')

fig = plt.figure()
plt.boxplot( np.transpose(train_err) )
#plt.xticks(x, labels)
plt.ylabel('Error')
plt.title('Train_Error, ' + name)
plt.show(block=False)
plt.savefig(dirnam + 'train_err' + '_Box' + '.png')

fig = plt.figure()
plt.boxplot( np.transpose(test_err) )
#plt.xticks(x, labels)
plt.ylabel('Error')
plt.title('Test_Error, ' + name)
plt.show(block=False)
plt.savefig(dirnam + 'test_err' + '_Box' + '.png')


xax = range(1, len(emiss_mean) + 1)

fig = plt.figure()
plt.plot( xax, emiss_mean, 'o-', c='blue', label="emiss_mean" )
plt.plot( xax, emiss_mean + emiss_std, '--', c='lightblue', label="emiss_1_sigm" )
plt.plot( xax, emiss_mean - emiss_std, '--', c='lightblue' )
plt.plot([1, len(emiss_mean)], [exp_emiss, exp_emiss], '-', c='red', label="exp_emiss" )
plt.plot([1, len(emiss_mean)], [np.mean(emiss_mean), np.mean(emiss_mean)], '-', c='green', label="mean_mean_emiss" )
plt.xlim(0.5, len(emiss_mean) + 0.5)
#plt.ylim(4, 15)
plt.xlabel('#')
plt.ylabel('NH3 emission')
plt.title('Predicted Emission, ' + name)
plt.legend(loc="best")
plt.show(block=False)
plt.savefig(dirnam + 'emission' + '.png')
#plt.close(fig)


fig = plt.figure()
plt.plot( xax, pred_err_mean, 'o-', c='red', label="pred_err_mean" )
plt.plot( xax, pred_err_mean + pred_err_std, '--', c='orange', label="pred_err_1_sigm" )
plt.plot( xax, pred_err_mean - pred_err_std, '--', c='orange' )
plt.xlabel('#')
plt.ylabel('mean absolute error')
plt.title('Prediction Error (MAE), ' + name)
plt.legend(loc="best")
plt.show(block=False)
plt.savefig(dirnam + 'pred_err' + '.png')
#plt.close(fig)


fig = plt.figure()
plt.plot( xax, train_err_mean , 'o-', c='blue', label="train_MAE_mean" )
plt.plot( xax, train_err_mean + train_err_std, '--', c='lightblue', label="train_MAE_1sigm" )
plt.plot( xax, train_err_mean - train_err_std, '--', c='lightblue' )
plt.plot( xax, test_err_mean , 'o-', c='red', label="test_MAE_mean" )
plt.plot( xax, test_err_mean + test_err_std, '--', c='orange', label="test_MAE_1sigm" )
plt.plot( xax, test_err_mean - test_err_std, '--', c='orange')
plt.xlabel('#')
plt.ylabel('error')
plt.title('Error Estimate (MAE) from Cross Validation')
plt.legend(loc="best")
plt.show(block=False)
plt.savefig(dirnam + 'train_test_err' + '.png')


fig = plt.figure()
plt.plot( xax, test_rmse_mean , 'o-', c='red', label="test_RMSE_mean" )
plt.plot( xax, test_rmse_mean + test_rmse_std, '--', c='orange', label="test_RMSE_1sigm" )
plt.plot( xax, test_rmse_mean - test_rmse_std, '--', c='orange')
plt.xlabel('#')
plt.ylabel('error')
plt.title('Error Estimate (RMSE) from Cross Validation')
plt.legend(loc="best")
plt.show(block=False)
plt.savefig(dirnam + 'test_err_rmse' + '.png')

fig = plt.figure()
plt.plot( xax, test_r2_mean , 'o-', c='red', label="test_R2_mean" )
plt.plot( xax, test_r2_mean + test_r2_std, '--', c='orange', label="test_R2_1sigm" )
plt.plot( xax, test_r2_mean - test_r2_std, '--', c='orange')
plt.xlabel('#')
plt.ylabel('error')
plt.title('Error Estimate (R2) from Cross Validation')
plt.legend(loc="best")
plt.show(block=False)
plt.savefig(dirnam + 'test_err_r2' + '.png')







