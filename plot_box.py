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

outputdir = 'allFigures/'

imtype = '.jpg'

if not os.path.exists(outputdir): 
    os.makedirs(outputdir)


drct[0] = 'GradBoost'
drct[1] = 'RandFor'
drct[2] = 'Ridge'
drct[3] = 'LinReg' 
drct[4] = 'GradBoost'#'ANN'
drct[5] = 'GradBoost'#'fixAnn'
drct[6] = 'GradBoost'#'GaussProc'
drct[7] = 'GradBoost'#'SVM'

fsyn  = '/'  # feature scaling yes/yo

for i in range(8):
    dirnam[i] = drct[i] + fsyn

exp_emiss = 1.4

log_exp_emiss = np.log10(exp_emiss)

for i in range(8):

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



#==============================================================================
# Plot Prediction for 4 Scenarios: 
# GraBoo 1, Ridge 3, ANN 5, GausProc 7
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):
  
    j = 2*i
    mpsig = (emiss_mean[j] + emiss_std[j])
    delta = (emiss_mean[j] - emiss_std[j])
    delta[delta <= 0] = 0.001   # no emissions < 0 !
 
    ax = indx[i] 
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '--', c='red', label='Experiment')
    ax.plot( range(1, len(emiss_mean[j]) + 1), emiss_mean[j],'o-', c='blue', label="emiss_mean" )
    ax.plot( range(1, len(emiss_mean[j]) + 1), mpsig,        '--', c='lightblue', label="emiss_1sigm" )
    ax.plot( range(1, len(emiss_mean[j]) + 1), delta,        '--', c='lightblue' )
    ax.set_xlim(0.5, 27.5)
    ax.set_ylim(6., 12.)  #(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission')
    if( (i == 2) or (i == 3)): 
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[j], pad = -15.0)

ax.legend(loc='lower right')
plt.show(block=False)

#plt.savefig(outputdir+ 'emiss_4' + imtype, dpi=150, format='jpg')


#==============================================================================
# Plot Emission Prediction BOX-Plot for 4 Methods
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):

    j = 2*i
    ax = indx[i] 
    ax.boxplot( np.transpose(emiss[j]) )
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '--', c='red')
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    #ax.set_ylim(3, 17)#(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    #if( i == 2 ):
    #    ax.set_ylim(2, 17)
    ax.set_title(drct[j], pad = -15.0)

plt.show(block=False)

#plt.savefig(outputdir + 'box_emiss_4' + imtype, dpi=150, format='jpg')


#==============================================================================
# Plot Emission Prediction BOX-Plot for 4 Methods re-ordered
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

d_1 = [0, 3, 6,  9, 12, 15, 18, 21, 24]
d_2 = [1, 4, 7, 10, 13, 16, 19, 22, 25]
d_3 = [2, 5, 8, 11, 14, 17, 20, 23, 26]

for i in range(4):

    j = 2*i
    
    em_1 = emiss[j][d_1]
    em_2 = emiss[j][d_2]
    em_3 = emiss[j][d_3]
    
    em_new = np.concatenate((em_1, em_2, em_3), axis=0)
    
    ax = indx[i] 
    ax.boxplot( np.transpose(em_new) )
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '--', c='red')
            
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    #ax.set_ylim(3, 17)#(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    if( i == 2 ):
        ax.set_ylim(2, 17)
    ax.set_title(drct[j], pad = -15.0)

plt.suptitle('RE-ORDERED !!')
plt.show(block=False)

#plt.savefig(outputdir + 'box_emiss_4' + imtype, dpi=150, format='jpg')


#==============================================================================
#==============================================================================

    #bp = ax.boxplot( np.transpose(emiss[j][0]) )
    #plt.setp(bp['boxes'], color='red')

#==============================================================================
# Plot Emission Prediction BOX-Plot, d_1 colored differently
# Intervals with interval_lengs = 1 day marked in blue 
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

d_1 = [0, 3, 6,  9, 12, 15, 18, 21, 24]  #  1 day
d_2 = [1, 4, 7, 10, 13, 16, 19, 22, 25]  #  7 days
d_3 = [2, 5, 8, 11, 14, 17, 20, 23, 26]  # 14 days

for i in range(4):

    j = 2*i
    
    em_1 = emiss[j][d_1]
    em_2 = emiss[j][d_2]
    em_3 = emiss[j][d_3]
    
    em_new = np.concatenate((em_1, em_2, em_3), axis=0)
    
    ax = indx[i] 
    ax.boxplot( np.transpose(emiss[j]) )
    
    mark = ax.boxplot( np.transpose(em_1), positions = d_2 )
    plt.setp( mark['boxes'],    color='m')
    plt.setp( mark['whiskers'], color='m')
    plt.setp( mark['caps'],     color='m')
    plt.setp( mark['fliers'],   color='m')
    
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '-', c='red')
            
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    #ax.set_ylim(3, 17)#(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    if( i == 2 ):
        ax.set_ylim(2, 17)
    ax.set_title(drct[j], pad = -15.0)



plt.show(block=False)

plt.savefig(outputdir + 'box_emiss_1mark_4' + imtype, dpi=150, format='jpg')
