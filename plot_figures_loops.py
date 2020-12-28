#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created April 2020

@author: jadolphs

Fuer matplotlib-Version 3.1.3 !!!

Das alte plot_figures_loops.py war fuer matplotlib 3.0.3 

Einige subtile Änderungen !!!!!!!!!!!
"""
import os
import numpy as np                  
import matplotlib.pyplot as plt        # plotting
import matplotlib.patches as mpatches  # legend manually

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
drct[4] = 'ANN'
drct[5] = 'fixAnn'
drct[6] = 'SVM'
drct[7] = 'GausProc'

fsyn    = '/' 

for i in range(8):
    dirnam[i] = drct[i] + fsyn

exp_emiss = 11.6   # g / (h LU)

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
# Plot Emission Prediction for 8 Scenarios
#==============================================================================

fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]

for i in range(8):
    
    mpsig = (emiss_mean[i] + emiss_std[i])
    delta = (emiss_mean[i] - emiss_std[i])
    delta[delta <= 0] = 0   # no emissions < 0 !
    
    ax = indx[i] 

    if(i == 0):    
        ax.plot([1, len(emiss_mean[i])], [exp_emiss, exp_emiss], '-', c='red', label='observed')
        ax.plot( range(1, len(emiss_mean[i]) + 1), emiss_mean[i],'o-', c='blue', label="predicted" )
        ax.plot( range(1, len(emiss_mean[i]) + 1), mpsig,        '--', c='lightblue', label="1-sigma-interval" )

    ax.plot( range(1, len(emiss_mean[i]) + 1), emiss_mean[i],'o-', c='blue' )
    ax.plot( range(1, len(emiss_mean[i]) + 1), mpsig,        '--', c='lightblue' )
    ax.plot( range(1, len(emiss_mean[i]) + 1), delta,        '--', c='lightblue' )
    ax.plot([1, len(emiss_mean[i])], [exp_emiss, exp_emiss], '--', c='red' )
    ax.set_xlim(0.5, 27.5)
    ax.tick_params(labelsize=11)
    ax.set_ylim(9, 14) 
    # if( (i == 0) or (i == 1)):  
    #    ax.set_ylim(10, 13)      
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission / g h$^{-1}$ LU$^{-1}$', fontsize = 12) 
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index', fontsize = 13)
        
    ax.set_title(drct[i], loc='center', pad = 0.0)

plt.figlegend(loc='upper center', ncol=3, borderaxespad=7, labelspacing=0, prop={'size':10})

plt.savefig(outputdir + 'emiss_8' + imtype, dpi=150, format='jpg') 

plt.show(block=False)


#==============================================================================
# Plot Prediction for 4 Scenarios: 
# GraBoo 1, Ridge 3, ANN 5, SVM 7
#==============================================================================

fig, axs = plt.subplots(2, 2)

fig.subplots_adjust(hspace=0.4)

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):
  
    j = 2*i
    mpsig = (emiss_mean[j] + emiss_std[j])
    delta = (emiss_mean[j] - emiss_std[j])
    delta[delta <= 0] = 0.001   # no emissions < 0 !
 
    ax = indx[i] 
    
    if(i == 0): 
        ax.plot([1, len(emiss_mean[i])], [exp_emiss, exp_emiss], '-',  c='red', label='observed')
        ax.plot( range(1, len(emiss_mean[i]) + 1), emiss_mean[i],'o-', c='blue', label="predicted" )
        ax.plot( range(1, len(emiss_mean[i]) + 1), mpsig,        '--', c='lightblue', label="1-sigma-interval" )
    
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '-', c='red' )
    ax.plot( range(1, len(emiss_mean[j]) + 1), emiss_mean[j],'o-', c='blue' )
    ax.plot( range(1, len(emiss_mean[j]) + 1), mpsig,        '--', c='lightblue' )
    ax.plot( range(1, len(emiss_mean[j]) + 1), delta,        '--', c='lightblue' )
    ax.set_xlim(0.5, 27.5)

    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission  /  g h$^{-1}$ LU$^{-1}$')
    if( (i == 2) or (i == 3)): 
        ax.set_xlabel('Scenario index')
    ax.set_ylim(10, 13)
    ax.set_title(drct[j], loc='center', pad = 0 )
    #ax.set_title(drct[j]+'         ', loc='right', pad = 0 )

plt.figlegend(loc='upper center', ncol=3, borderaxespad=0.1, labelspacing=1, prop={'size': 10})
    
plt.savefig(outputdir+ 'emiss_4' + imtype, dpi=150, format='jpg')

plt.show(block=False)


#==============================================================================
# Plot Train-Test-Pred-Errors for 4 Methods
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

fig.subplots_adjust(hspace=0.4)

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):

    j = 2*i
    ax = indx[i] 

    if (i == 3):
        ax.plot( range(1, len(train_err_mean[j]) + 1), train_err_mean[j], 'o-', c='orange', label="train error" )
        ax.plot( range(1, len(test_err_mean[j]) + 1),  test_err_mean[j],  'o-', c='red',    label="test error" )
        ax.plot( range(1, len(pred_err_mean[j]) + 1),  pred_err_mean[j],  'o-', c='blue',   label="extrapolation error" )
    else:
        ax.plot( range(1, len(train_err_mean[j]) + 1), train_err_mean[j], 'o-', c='orange' )
        ax.plot( range(1, len(test_err_mean[j]) + 1),  test_err_mean[j],  'o-', c='red'  )
        ax.plot( range(1, len(pred_err_mean[j]) + 1),  pred_err_mean[j],  'o-', c='blue' )

    ax.set_xlim(0.5, 27.5)
    '''
    if ((i == 0) or (i == 1)):
        ax.set_ylim(0.0, 3.5)
    if i == 2 or i == 3:
        ax.set_ylim(0, 10)
    '''
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Error  /  g h$^{-1}$ LU$^{-1}$')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
        
    ax.set_title(drct[j], loc='center', pad = 0 )
    
plt.figlegend(loc='upper center', ncol=3, borderaxespad=0.1, labelspacing=1, prop={'size': 10})

plt.savefig(outputdir + 'train_test_pred_err_4' + imtype, dpi=150, format='jpg')

plt.show(block=False)



#==============================================================================
# Plot Emission Prediction BOX-Plot, d_1 colored differently
# Intervals with interval_lengs = 1 day marked in blue 
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

fig.subplots_adjust(hspace=0.4)

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

d_0 = [0, 3, 6,  9, 12, 15, 18, 21, 24]  #  1 day
d_1 = [1, 4, 7, 10, 13, 16, 19, 22, 25]  #  7 days
d_2 = [2, 5, 8, 11, 14, 17, 20, 23, 26]  # 14 days
d_3 = [3, 6, 9, 12, 15, 18, 21, 24, 27]

for i in range(4):

    j = 2*i
    
    em_1 = emiss[j][d_0]
    em_2 = emiss[j][d_1]
    em_3 = emiss[j][d_2]
    
    em_new = np.concatenate((em_1, em_2, em_3), axis=0)
    
    ax = indx[i] 
    ax.boxplot( np.transpose(emiss[j]) )
    
    mark_1 = ax.boxplot( np.transpose(em_1), positions = d_1 )
    plt.setp( mark_1['boxes'],    color='c')
    plt.setp( mark_1['whiskers'], color='c')
    plt.setp( mark_1['caps'],     color='c')
    plt.setp( mark_1['fliers'], markeredgecolor='c')
    
    mark_2 = ax.boxplot( np.transpose(em_2), positions = d_2 )
    plt.setp( mark_2['boxes'],    color='b')
    plt.setp( mark_2['whiskers'], color='b')
    plt.setp( mark_2['caps'],     color='b')
    plt.setp( mark_2['fliers'], markeredgecolor='b')
    
    mark_3 = ax.boxplot( np.transpose(em_3), positions = d_3 )
    plt.setp( mark_3['boxes'],    color='k')
    plt.setp( mark_3['whiskers'], color='k')
    plt.setp( mark_3['caps'],     color='k')
    plt.setp( mark_3['fliers'], markeredgecolor='k')
        
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '-', c='red')
            
    ax.set_xticks([0, 5, 10, 15, 20, 25, 28])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25])
    '''
    if( (i == 0) or (i == 1)):
        ax.set_ylim(9, 14)
    if( (i == 2) or (i == 3)):
        ax.set_ylim(6, 22)
    '''
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission  /  g h$^{-1}$ LU$^{-1}$')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
        
    ax.set_title(drct[j], loc='center', pad = 0 )
    
    patch_1 = mpatches.Patch(color='c', label='1 day')
    patch_2 = mpatches.Patch(color='b', label='7 days')
    patch_3 = mpatches.Patch(color='k', label='14 days')

plt.figlegend(handles = [patch_1, patch_2, patch_3], loc='upper center', ncol=3, 
              borderaxespad=0.1, labelspacing=0, prop={'size': 10})

plt.savefig(outputdir + 'box_emiss_4' + imtype, dpi=150, format='jpg')

plt.show(block=False)

#==============================================================================
# Plot Prediction Error BOX-Plots for 8 Methods
#==============================================================================
'''
fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]

for i in range(8):

    ax = indx[i] 
    ax.boxplot( np.transpose(pred_err[i]) )  
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5,10,15,20,25])
    if( (i % 2 == 0) ):    
        ax.set_ylabel('Error')
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[i], pad = -15.0)

ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'box_error_8' + imtype, dpi=150, format='jpg') 
'''

#==============================================================================
# Plot Prediction Error BOX-Plot for 4 Methods
#==============================================================================
'''
fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):

    j = 2*i
    ax = indx[i] 
    ax.boxplot( np.transpose(pred_err[j]) )
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    #ax.set_ylim(3, 17)#(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Error')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[j], pad = -15.0)

ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'box_error_4' + imtype, dpi=150, format='jpg')
'''

#==============================================================================
# Plot Emission Prediction BOX-Plot for 8 Methods
#==============================================================================
'''
fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]

for i in range(8):

    ax = indx[i] 
    ax.boxplot( np.transpose(emiss[i]) )
    ax.plot([1, len(emiss_mean[i])], [exp_emiss, exp_emiss], '--', c='red')
    ax.set_xticks([5,10,15,20,25])
    ax.set_xticklabels([5,10,15,20,25])
    if( (i % 2 == 0) ):    
        ax.set_ylabel('Emission')
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[i], pad = -15.0)

ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'box_emiss_8' + imtype, dpi=150, format='jpg') 
'''

#==============================================================================
# Plot LOG-Emission Prediction BOX-Plot for 8 Methods
#==============================================================================
'''
fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]

for i in range(8):

    ax = indx[i] 
    ax.boxplot( np.log(np.transpose(emiss[i])) )
    ax.plot([1, len(emiss_mean[i])], [log_exp_emiss, log_exp_emiss], '-', c='red')
    ax.set_xticks([5,10,15,20,25])
    ax.set_xticklabels([5,10,15,20,25])
    if( (i % 2 == 0) ):    
        ax.set_ylabel('Log_Emission')
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[i], pad = -15.0)
    
plt.show(block=False)

plt.savefig(outputdir + 'box_log_emiss_8' + imtype, dpi=150, format='jpg') 
'''

'''
#==============================================================================
# Plot Prediction-Error for 8 Scenarios
#==============================================================================

fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]

for i in range(8):

    ax = indx[i] 
    ax.plot( range(1, len(pred_err_mean[i]) + 1), pred_err_mean[i], 'o-', c='blue', label="pred_err_mean" )
    ax.plot( range(1, len(pred_err_mean[i]) + 1), pred_err_mean[i] + pred_err_std[i], '--', c='lightblue', label="pred_err_1_sigm" )
    ax.plot( range(1, len(pred_err_mean[i]) + 1), pred_err_mean[i] - pred_err_std[i], '--', c='lightblue' )
    ax.set_xlim(0.5, 27.5)
    #ax.set_ylim(0.2, 1)#(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Error')
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[i], pad = -15.0)
ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'error_8' + imtype, dpi=150, format='jpg') 


#==============================================================================
# Plot Prediction-Error (non-logarithmic) for 8 Scenarios
#==============================================================================

fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]



for i in range(8):

    # mn = np.mean(np.array(pred_err),axis=2)[i]
    # sd = np.std(np.array(pred_err),axis=2)[i]

    mn = np.mean(np.array(np.exp(pred_err)), axis=2)[i]
    sd = np.std(np.array(np.exp(pred_err)), axis=2)[i]
        
    ax = indx[i] 
    ax.plot( range(1, len(mn) + 1), mn, 'o-', c='blue', label="pred_err_mean" )
    ax.plot( range(1, len(mn) + 1), mn + sd, '--', c='lightblue', label="pred_err_1_sigm" )
    ax.plot( range(1, len(mn) + 1), mn - sd, '--', c='lightblue' )
    ax.set_xlim(0.5, 27.5)
    #ax.set_ylim(0.2, 1)#(7, 12)

    if( (i % 2 == 0) ): 
        ax.set_ylabel('Error')
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[i], pad = -15.0)
ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'error_8' + imtype, dpi=150, format='jpg') 


#==============================================================================
# Plot Prediction-Error with 1-sigm-interv for 4 Methods
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):

    j = 2*i
    ax = indx[i] 
    ax.plot( range(1, len(pred_err_mean[j]) + 1), pred_err_mean[j], 'o-', c='blue', label="pred_err" )
    ax.plot( range(1, len(pred_err_mean[j]) + 1), pred_err_mean[j] + pred_err_std[j], '--', c='lightblue', label="pred_1sigm" )
    ax.plot( range(1, len(pred_err_mean[j]) + 1), pred_err_mean[j] - pred_err_std[j], '--', c='lightblue' )
    ax.set_xlim(0.5, 27.5)
    #ax.set_ylim(0.25, 0.65)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Prediction error')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[j], pad = -15.0)

ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'pred_error_4' + imtype, dpi=150, format='jpg')


#==============================================================================
# Plot Train-Test-Errors with 1-sigm-interv for 4 Methods
#==============================================================================

fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):

    j = 2*i
    ax = indx[i] 
    ax.plot( range(1, len(train_err_mean[j]) + 1), train_err_mean[j], 'o-', c='blue', label="train_err" )
    ax.plot( range(1, len(train_err_mean[j]) + 1), train_err_mean[j] + train_err_std[j], '--', c='lightblue', label="train_1sigm" )
    ax.plot( range(1, len(train_err_mean[j]) + 1), train_err_mean[j] - train_err_std[j], '--', c='lightblue' )
    ax.plot( range(1, len(test_err_mean[j]) + 1), test_err_mean[j], 'o-', c='red', label="test_err" )
    ax.plot( range(1, len(test_err_mean[j]) + 1), test_err_mean[j] + test_err_std[j], '--', c='orange', label="test_1sigm" )
    ax.plot( range(1, len(test_err_mean[j]) + 1), test_err_mean[j] - test_err_std[j], '--', c='orange' )
    ax.set_xlim(0.5, 27.5)
    #ax.set_ylim(0, 1)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('train_test_error')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    ax.set_title(drct[j], pad = -15.0)

ax.legend(loc='lower right')
plt.show(block=False)

plt.savefig(outputdir + 'train_test_err_4' + imtype, dpi=150, format='jpg')

'''

#==============================================================================
# Plot LOGARITHM of Emission Prediction for 8 Scenarios
#==============================================================================
'''
fig, axs = plt.subplots(4, 2, figsize=(10, 15))   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]]

for i in range(8):
    mpsig = (emiss_mean[i] + emiss_std[i])
    delta = (emiss_mean[i] - emiss_std[i])
    delta[delta <= 0] = 1   # log(0) = -infty
            
    ax = indx[i] 
    
    if(i == 0): 
        ax.plot([1, len(emiss_mean[i])], [log_exp_emiss, log_exp_emiss], '-',  c='red', label='observed')
        ax.plot( range(1, len(emiss_mean[i]) + 1), np.log(emiss_mean[i]),'o-', c='blue', label="predicted" )
        ax.plot( range(1, len(emiss_mean[i]) + 1), np.log(mpsig),        '--', c='lightblue', label="1-sigma-interval" )

    ax.plot([1, len(emiss_mean[i])], [log_exp_emiss, log_exp_emiss], '-',  c='red')
    ax.plot( range(1, len(emiss_mean[i]) + 1), np.log(emiss_mean[i]),'o-', c='blue' )
    ax.plot( range(1, len(emiss_mean[i]) + 1), np.log(mpsig),        '--', c='lightblue' )
    ax.plot( range(1, len(emiss_mean[i]) + 1), np.log(delta),        '--', c='lightblue' )
    ax.set_xlim(0.5, 27.5)
    ax.tick_params(labelsize=11)
    
    if( (i == 0) or (i == 1) or (i == 2) or (i == 3) ): 
        ax.set_ylim(2.3, 2.6)
    if( (i == 4) ): 
        ax.set_ylim(2.25, 2.65)   #(2, 4.2) #1,5, 15
    #if( (i == 5) ): 
    #    ax.set_ylim(2., 3.3) 
    if( (i == 6) ): 
        ax.set_ylim(2.25, 2.65)  
    #if( (i == 7)): 
    #    ax.set_ylim(2.3, 2.6)  
    
        
    if( (i % 2 == 0) ): 
        ax.set_ylabel('ln [Emission / g h$^{-1}$ LU$^{-1}$]', fontsize = 12) #     /     Log [g h$^{-1}$ LU$^{-1}$]', fontsize = 14)
    if( (i == 6) or (i == 7)):    
        ax.set_xlabel('Scenario index', fontsize = 13)

    ax.set_title(drct[i]+'     ', loc='right', pad = -15.0 )

plt.figlegend(loc='upper center', ncol=3, borderaxespad=5.0, labelspacing=0, prop={'size': 10})

plt.savefig(outputdir + 'log_emiss_8' + imtype, dpi=150, format='jpg') 

plt.show(block=False)
'''

#==============================================================================
# Plot Emission Prediction BOX-Plot for 4 Methods
#==============================================================================
'''
fig, axs = plt.subplots(2, 2)   # rows, cols, fig_number

indx = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

for i in range(4):

    j = 2*i
    ax = indx[i] 
    ax.boxplot( np.transpose(emiss[j]) )
    ax.plot([1, len(emiss_mean[j])], [exp_emiss, exp_emiss], '-', c='red')
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    #ax.set_ylim(3, 17)#(7, 12)
    if( (i % 2 == 0) ): 
        ax.set_ylabel('Emission  /  g h$^{-1}$ LU$^{-1}$')
    if( (i == 2) or (i == 3)):    
        ax.set_xlabel('Scenario index')
    #if( i == 2 ):
    #    ax.set_ylim(2, 17)
    ax.set_title(drct[j], pad = -15.0)

plt.show(block=False)
'''
#plt.savefig(outputdir + 'box_emiss_4' + imtype, dpi=150, format='jpg')




#==============================================================================
#==============================================================================


#, bbox_inces='tight', optimize=True, progressive=True)
# plt.tight_layout() # groessere vertikale Abstaende