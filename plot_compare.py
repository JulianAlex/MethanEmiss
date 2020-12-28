#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:16:23 2019

@author: jadolphs
"""
import os
import numpy as np                # numerics   
import matplotlib.pyplot as plt   # plotting

plt.close('all')


dirnam = 'allFigures/'

if not os.path.exists(dirnam): 
    os.makedirs(dirnam)

dir_1 = 'GradBoost'
dir_2 = 'RandFor'
dir_3 = 'Ridge'
dir_4 = 'LinReg' 
dir_5 = 'ANN'
dir_6 = 'fixAnn'
dir_7 = 'GausProc'
dir_8 = 'SVM'

fs_n  = '/'  # feature scaling: no
fs_y  = '/'    # feature scaling: yes

imtype = '.jpg'   # imagetype for saving

fsy_1 = dir_1 + fs_y
fsy_2 = dir_2 + fs_y
fsy_3 = dir_3 + fs_y
fsy_4 = dir_4 + fs_y
fsy_5 = dir_5 + fs_y
fsy_6 = dir_6 + fs_y
fsy_7 = dir_7 + fs_y
fsy_8 = dir_8 + fs_y

fsn_1 = dir_1 + fs_n
fsn_2 = dir_2 + fs_n
fsn_3 = dir_3 + fs_n
fsn_4 = dir_4 + fs_n
fsn_5 = dir_5 + fs_n
fsn_6 = dir_6 + fs_n
fsn_7 = dir_7 + fs_n
fsn_8 = dir_8 + fs_n

exp_emiss = 9.401


emiss_1n         = np.load( fsn_1 + 'emiss.npy',      allow_pickle=True )
emiss_mean_1n    = np.load( fsn_1 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_1n     = np.load( fsn_1 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_2n         = np.load( fsn_2 + 'emiss.npy',      allow_pickle=True )
emiss_mean_2n    = np.load( fsn_2 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_2n     = np.load( fsn_2 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_3n         = np.load( fsn_3 + 'emiss.npy',      allow_pickle=True )
emiss_mean_3n    = np.load( fsn_3 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_3n     = np.load( fsn_3 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_4n         = np.load( fsn_4 + 'emiss.npy',      allow_pickle=True )
emiss_mean_4n    = np.load( fsn_4 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_4n     = np.load( fsn_4 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_5n         = np.load( fsn_5 + 'emiss.npy',      allow_pickle=True )
emiss_mean_5n    = np.load( fsn_5 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_5n     = np.load( fsn_5 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_6n         = np.load( fsn_6 + 'emiss.npy',      allow_pickle=True )
emiss_mean_6n    = np.load( fsn_6 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_6n     = np.load( fsn_6 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_7n         = np.load( fsn_7 + 'emiss.npy',      allow_pickle=True )
emiss_mean_7n    = np.load( fsn_7 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_7n     = np.load( fsn_7 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_8n         = np.load( fsn_8 + 'emiss.npy',      allow_pickle=True )
emiss_mean_8n    = np.load( fsn_8 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_8n     = np.load( fsn_8 + 'emiss_std.npy',  allow_pickle=True ) 

#..............................................................................

emiss_1y         = np.load( fsy_1 + 'emiss.npy',      allow_pickle=True )
emiss_mean_1y    = np.load( fsy_1 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_1y     = np.load( fsy_1 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_2y         = np.load( fsy_2 + 'emiss.npy',      allow_pickle=True )
emiss_mean_2y    = np.load( fsy_2 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_2y     = np.load( fsy_2 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_3y         = np.load( fsy_3 + 'emiss.npy',      allow_pickle=True )
emiss_mean_3y    = np.load( fsy_3 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_3y     = np.load( fsy_3 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_4y         = np.load( fsy_4 + 'emiss.npy',      allow_pickle=True )
emiss_mean_4y    = np.load( fsy_4 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_4y     = np.load( fsy_4 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_5y         = np.load( fsy_5 + 'emiss.npy',      allow_pickle=True )
emiss_mean_5y    = np.load( fsy_5 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_5y     = np.load( fsy_5 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_6y         = np.load( fsy_6 + 'emiss.npy',      allow_pickle=True )
emiss_mean_6y    = np.load( fsy_6 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_6y     = np.load( fsy_6 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_7y         = np.load( fsy_7 + 'emiss.npy',      allow_pickle=True )
emiss_mean_7y    = np.load( fsy_7 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_7y     = np.load( fsy_7 + 'emiss_std.npy',  allow_pickle=True ) 

emiss_8y         = np.load( fsy_8 + 'emiss.npy',      allow_pickle=True )
emiss_mean_8y    = np.load( fsy_8 + 'emiss_mean.npy', allow_pickle=True )
emiss_std_8y     = np.load( fsy_8 + 'emiss_std.npy',  allow_pickle=True ) 

#------------------------------------------------------------------------------

pred_err_1      = np.load( fsn_1 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_1 = np.load( fsn_1 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_1  = np.load( fsn_1 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_2      = np.load( fsn_2 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_2 = np.load( fsn_2 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_2  = np.load( fsn_2 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_3      = np.load( fsn_3 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_3 = np.load( fsn_3 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_3  = np.load( fsn_3 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_4      = np.load( fsn_4 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_4 = np.load( fsn_4 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_4  = np.load( fsn_4 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_5      = np.load( fsn_5 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_5 = np.load( fsn_5 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_5  = np.load( fsn_5 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_6      = np.load( fsn_6 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_6 = np.load( fsn_6 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_6  = np.load( fsn_6 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_7      = np.load( fsn_7 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_7 = np.load( fsn_7 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_7  = np.load( fsn_7 + 'pred_err_std.npy',  allow_pickle=True ) 

pred_err_8      = np.load( fsn_8 + 'pred_err.npy',      allow_pickle=True )
pred_err_mean_8 = np.load( fsn_8 + 'pred_err_mean.npy', allow_pickle=True )
pred_err_std_8  = np.load( fsn_8 + 'pred_err_std.npy',  allow_pickle=True )  


#==============================================================================
# Plot Emission Prediction for 4 Scenarios, with and without feature skaling
#==============================================================================

fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

# 1
ax = axs[0, 0] 
ax.plot( range(1, len(emiss_mean_1n) + 1), emiss_mean_1n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_1n) + 1), emiss_mean_1n + emiss_std_1n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_1n) + 1), emiss_mean_1n - emiss_std_1n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_1n)], [exp_emiss, exp_emiss], '--', c='red')
ax.plot([1, len(emiss_mean_1n)], [np.mean(emiss_mean_1n), np.mean(emiss_mean_1n)], '-', c='green' )
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7.5, 11.5)
ax.set_title(dir_1 + ', noFS',pad = -15.0)

ax = axs[0, 1]
ax.plot( range(1, len(emiss_mean_1y) + 1), emiss_mean_1y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_1y) + 1), emiss_mean_1y + emiss_std_1y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_1y) + 1), emiss_mean_1y - emiss_std_1y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_1y)], [exp_emiss, exp_emiss], '--', c='red')
ax.plot([1, len(emiss_mean_1y)], [np.mean(emiss_mean_1y), np.mean(emiss_mean_1y)], '-', c='green' )
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7.5, 11.5)
ax.set_title(dir_1 + ', FS',pad = -15.0)

# 2
ax = axs[1, 0]
ax.plot( range(1, len(emiss_mean_2n) + 1), emiss_mean_2n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_2n) + 1), emiss_mean_2n + emiss_std_2n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_2n) + 1), emiss_mean_2n - emiss_std_2n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_2n)], [np.mean(emiss_mean_2n), np.mean(emiss_mean_2n)], '-', c='green' )
ax.plot([1, len(emiss_mean_2n)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7.5, 11.5)
ax.set_title(dir_2 + ', noFS',pad = -15.0)

ax = axs[1, 1]
ax.plot( range(1, len(emiss_mean_2y) + 1), emiss_mean_2y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_2y) + 1), emiss_mean_2y + emiss_std_2y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_2y) + 1), emiss_mean_2y - emiss_std_2y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_2y)], [np.mean(emiss_mean_2y), np.mean(emiss_mean_2y)], '-', c='green' )
ax.plot([1, len(emiss_mean_2y)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7.5, 11.5)
ax.set_title(dir_2 + ', FS',pad = -15.0)

# 3
ax = axs[2, 0]
ax.plot( range(1, len(emiss_mean_3n) + 1), emiss_mean_3n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_3n) + 1), emiss_mean_3n + emiss_std_3n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot([1, len(emiss_mean_3n)], [np.mean(emiss_mean_3n), np.mean(emiss_mean_3n)], '-', c='green' )
ax.plot( range(1, len(emiss_mean_3n) + 1), emiss_mean_3n - emiss_std_3n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_3n)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(6.5, 14.)
ax.set_title(dir_3 + ', noFS',pad = -15.0)

ax = axs[2, 1]
ax.plot( range(1, len(emiss_mean_3y) + 1), emiss_mean_3y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_3y) + 1), emiss_mean_3y + emiss_std_3y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_3y) + 1), emiss_mean_3y - emiss_std_3y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_3y)], [np.mean(emiss_mean_3y), np.mean(emiss_mean_3y)], '-', c='green' )
ax.plot([1, len(emiss_mean_3y)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(6.5, 14.)
ax.set_title(dir_3 + ', FS',pad = -15.0)

# 4
ax = axs[3, 0]
ax.plot( range(1, len(emiss_mean_4n) + 1), emiss_mean_4n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_4n) + 1), emiss_mean_4n + emiss_std_4n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_4n) + 1), emiss_mean_4n - emiss_std_4n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_4n)], [np.mean(emiss_mean_4n), np.mean(emiss_mean_4n)], '-', c='green' )
ax.plot([1, len(emiss_mean_4n)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_xlabel('Scenario index')
ax.set_ylim(7, 100000000000000)
ax.set_title(dir_4 + ', noFS',pad = -15.0)

ax = axs[3, 1]
ax.plot( range(1, len(emiss_mean_4y) + 1), emiss_mean_4y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_4y) + 1), emiss_mean_4y + emiss_std_4y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_4y) + 1), emiss_mean_4y - emiss_std_4y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_4y)], [np.mean(emiss_mean_4y), np.mean(emiss_mean_4y)], '-', c='green' )
ax.plot([1, len(emiss_mean_4y)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_xlabel('Scenario index')
ax.set_ylim(7, 100000000000000)
ax.set_title(dir_4 + ', FS',pad = -15.0)

#fig.suptitle('Predicted Emission (8 Methods)')

plt.show(block=False)

'emis_8'

plt.savefig(dirnam + 'emis_FS_8_1' + imtype, dpi=150, format='jpg')


#==============================================================================
# Plot Emission Prediction for 4 Scenarios, with and without feature skaling
#==============================================================================

fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

# 1
ax = axs[0, 0] 
ax.plot( range(1, len(emiss_mean_5n) + 1), emiss_mean_5n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_5n) + 1), emiss_mean_5n + emiss_std_5n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_5n) + 1), emiss_mean_5n - emiss_std_5n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_5n)], [exp_emiss, exp_emiss], '--', c='red')
ax.plot([1, len(emiss_mean_5n)], [np.mean(emiss_mean_5n), np.mean(emiss_mean_5n)], '-', c='green' )
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7, 11.)
ax.set_title(dir_5 + ', noFS',pad = -15.0)

ax = axs[0, 1]
ax.plot( range(1, len(emiss_mean_5y) + 1), emiss_mean_5y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_5y) + 1), emiss_mean_5y + emiss_std_5y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_5y) + 1), emiss_mean_5y - emiss_std_5y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_5y)], [exp_emiss, exp_emiss], '--', c='red')
ax.plot([1, len(emiss_mean_5y)], [np.mean(emiss_mean_5y), np.mean(emiss_mean_5y)], '-', c='green' )
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7, 11.)
ax.set_title(dir_5 + ', FS',pad = -15.0)

# 2
ax = axs[1, 0]
ax.plot( range(1, len(emiss_mean_6n) + 1), emiss_mean_6n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_6n) + 1), emiss_mean_6n + emiss_std_6n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_6n) + 1), emiss_mean_6n - emiss_std_6n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_6n)], [np.mean(emiss_mean_6n), np.mean(emiss_mean_6n)], '-', c='green' )
ax.plot([1, len(emiss_mean_6n)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7., 30)
ax.set_title(dir_6 + ', noFS',pad = -15.0)

ax = axs[1, 1]
ax.plot( range(1, len(emiss_mean_6y) + 1), emiss_mean_6y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_6y) + 1), emiss_mean_6y + emiss_std_6y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_6y) + 1), emiss_mean_6y - emiss_std_6y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_6y)], [np.mean(emiss_mean_6y), np.mean(emiss_mean_6y)], '-', c='green' )
ax.plot([1, len(emiss_mean_6y)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7, 30)
ax.set_title(dir_6 + ', FS',pad = -15.0)

# 3
ax = axs[2, 0]
ax.plot( range(1, len(emiss_mean_7n) + 1), emiss_mean_7n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_7n) + 1), emiss_mean_7n + emiss_std_7n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot([1, len(emiss_mean_7n)], [np.mean(emiss_mean_7n), np.mean(emiss_mean_7n)], '-', c='green' )
ax.plot( range(1, len(emiss_mean_7n) + 1), emiss_mean_7n - emiss_std_7n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_7n)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7, 11.)
ax.set_title(dir_7 + ', noFS',pad = -15.0)

ax = axs[2, 1]
ax.plot( range(1, len(emiss_mean_7y) + 1), emiss_mean_7y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_7y) + 1), emiss_mean_7y + emiss_std_7y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_7y) + 1), emiss_mean_7y - emiss_std_7y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_7y)], [np.mean(emiss_mean_7y), np.mean(emiss_mean_7y)], '-', c='green' )
ax.plot([1, len(emiss_mean_7y)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_ylim(7, 11.)
ax.set_title(dir_7 + ', FS',pad = -15.0)

# 4
ax = axs[3, 0]
ax.plot( range(1, len(emiss_mean_8n) + 1), emiss_mean_8n, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_8n) + 1), emiss_mean_8n + emiss_std_8n, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_8n) + 1), emiss_mean_8n - emiss_std_8n, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_8n)], [np.mean(emiss_mean_8n), np.mean(emiss_mean_8n)], '-', c='green' )
ax.plot([1, len(emiss_mean_8n)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_xlabel('Scenario index')
ax.set_ylim(7.5, 13)
ax.set_title(dir_8 + ', noFS',pad = -15.0)

ax = axs[3, 1]
ax.plot( range(1, len(emiss_mean_8y) + 1), emiss_mean_8y, 'o-', c='blue', label="emiss_mean" )
ax.plot( range(1, len(emiss_mean_8y) + 1), emiss_mean_8y + emiss_std_8y, '--', c='lightblue', label="emiss_1_sigm" )
ax.plot( range(1, len(emiss_mean_8y) + 1), emiss_mean_8y - emiss_std_8y, '--', c='lightblue' )
ax.plot([1, len(emiss_mean_8y)], [np.mean(emiss_mean_8y), np.mean(emiss_mean_8y)], '-', c='green' )
ax.plot([1, len(emiss_mean_8y)], [exp_emiss, exp_emiss], '--', c='red')
ax.set_xlim(0.5, 27.5)
ax.set_xlabel('Scenario index')
ax.set_ylim(7.5, 13) 
ax.set_title(dir_8 + ', FS', pad = -15.0) #,loc ='right')

#fig.suptitle('Predicted Emission (8 Methods)')

plt.show(block=False)

plt.savefig(dirnam + 'emis_FS_8_2' + imtype, dpi=150, format='jpg')










