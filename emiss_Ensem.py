'''
 Ensemble Methods: no hyperparameter tuning neccessary !
 
 LinReg, 
 no ensemble-methods, but no Hyperparams
 
 start with:  "python emiss_Ensemble.py" 
 
 18.11.2019
'''
from prepDataMeth import prepDataMeth

import os
#import sys
import time                       # time
import numpy as np                # numerics   
import pandas as pd               # data frames
import matplotlib.pyplot as plt   # plotting
plt.ioff()                        # turn off display of figures

from sklearn.ensemble import RandomForestRegressor      # Random Forest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import GroupShuffleSplit   # Group Splitter
from sklearn.model_selection import LeaveOneGroupOut 
from sklearn.preprocessing import StandardScaler        # Feature Scaling
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error as mae  # Metric
from sklearn.metrics import mean_squared_error as mse  # Metric
from sklearn.metrics import r2_score as r2  # Metric

from math import sqrt

from toolz import frequencies, valfilter                # tools for 

#------------------------------------------------------------------------------

n_days  = [1, 7, 14]   #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]   # [1, 3, 5, 7, 10, 14]   # [1, 7, 14]   # interval_length, intervals: 3,4,6
n_test  =  1

n_tsw = [ [1,1,1],  [2,1,1], [2,2,0], [3,1,0],  [2,2,2], [3,2,1], [4,1,1], [4,2,0], [5,1,0] ]

n_split  = 1000   #1000 # 5000
n_sample = 30     #100   # number of interval selection fullfilling the n_tsw conditions

name   = 'GB' #'RF' #'GB'    # name of saved figures
dirnam = 'GradBoost/' # 'LinReg/' #'RandFor/' #'LinReg/' #'GradBoost/'    
name   = dirnam + name

if not os.path.exists(dirnam): 
    os.makedirs(dirnam)

plt.close('all')  # close all figures

np.random.seed(1)
scaler = RobustScaler() ## StandardScaler()

#model = LinearRegression(n_jobs = None)
#model = RandomForestRegressor(n_estimators = 100, criterion='mae', random_state = 1, n_jobs = None)
model = GradientBoostingRegressor(loss = 'lad', random_state = 1)

#------------------------------------------------------------------------------

t_start = time.time()

scen_emiss, scen_emiss_mean, scen_emiss_std             = [], [], []

scen_pred_err,  scen_pred_err_mean,  scen_pred_err_std  = [], [], []
scen_pred_r2,   scen_pred_r2_mean,   scen_pred_r2_std   = [], [], []
scen_pred_rmse, scen_pred_rmse_mean, scen_pred_rmse_std = [], [], []

scen_train_err, scen_train_err_mean, scen_train_err_std = [], [], []
scen_test_err, scen_test_err_mean, scen_test_err_std    = [], [], []
scen_test_r2, scen_test_r2_mean, scen_test_r2_std       = [], [], []
scen_test_rmse, scen_test_rmse_mean, scen_test_rmse_std = [], [], []


print("-------------------------------------------------------------------------")

for ntsw in n_tsw:     

    for nday in n_days: 

        n_trans, n_summer, n_winter = ntsw[0], ntsw[1], ntsw[2]
        n_train = n_trans + n_summer + n_winter - n_test
                
        print('n_days, n_intervals  |  n_trans, n_summer, n_winter:    '+str(nday)+', '+str(n_train+n_test)+'  |  '+str(n_trans)+', '+str(n_summer)+', '+str(n_winter) )
        print("-------------------------------------------------------------------------")

        name2 = str(nday) + 'd_' + str(ntsw) + '.png'
        
        prepDataMeth(nday)  # prepare data with intervals of n_days 
        
        df = pd.read_csv("Methan_proc.dat", sep = "\t") 
        
        ## Date Time Temp Wind_dir Wind_spd EF_CH4 LN_EF h_sin h_cos days t_sin t_cos dW_sin dW_cos dt group season

        exp_emiss  = np.mean(df["EF_CH4"])  # emissin in g / (h LU)

        #------------------------------------------------------------------------------
        #Temp date dt vW ef lg_ef t h_sin h_cos t_sin t_cos dW_sin dW_cos group season
        # Transform data-frame => numpy array
        X = df.drop(["Date", "Time", "Wind_dir", "EF_CH4", "LN_EF", "days", "dt", "group", "season"], axis = 1).values  #Features 
        y = df["EF_CH4"].values  # Targets 
        #------------------------------------------------------------------------------
        
        groups = df["group"].values   #  0, 1, 2, ...
        season = df["season"].values  #  S, T, W
        
        # List with season for each interval  
        gr_season = []
        indx = 0
        group = list(groups)
        n_intv = np.max(groups) + 1
        
        # season of each interval
        for i in range(0, n_intv):
            gr_season.append(season[indx])
            indx = indx + group.count(i)
        
        train_error, test_error, test_err_r2, test_err_rms         = [], [], [], []
        cv_train_err, cv_test_err, cv_test_err_r2, cv_test_err_rms = [], [], [], []
        i, j = 0, 0

        # Cross-validation-iterators:
        gss  = GroupShuffleSplit(n_splits = n_split, test_size = n_test, train_size = n_train, random_state = 1)
        logo = LeaveOneGroupOut()
                
                
        # random select n intervals out of all intervals        
        for train, test in gss.split( X, y, groups=groups ):
            
            X_new  = np.concatenate(( X[train], X[test] ))
            y_new  = np.concatenate(( y[train], y[test] ))
            gr_new = np.concatenate(( groups[train], groups[test] ))
            
            # check out interval number and season of training intervals     
            group_list   = list(valfilter(lambda count: count > 1, frequencies(gr_new)).keys())
            season_array = np.array( gr_season )  # list => array, for use of index_list 
            season_list  = list( season_array[group_list] )
        
            nt = season_list.count('T')
            ns = season_list.count('S')
            nw = season_list.count('W')
        
            if( ( j < n_sample) and (nt == n_trans) and (ns == n_summer) and (nw == n_winter) ):
            
                j = j + 1
                
                # do a leave 1 out cross validation for error estimation with the selected intervals:
                for train_new, test_new in logo.split(X_new, y_new, gr_new):
                                        
                    X_train, y_train = X_new[train_new], y_new[train_new]    
                    X_test, y_test   = X_new[test_new], y_new[test_new]     
            
                    groups_train = gr_new[train_new]
                    groups_test  = gr_new[test_new]    
        
                    X_all   = X
                    X_train = scaler.fit_transform(X_train)  # feature scaling
                    X_test  = scaler.transform(X_test)       # feature scaling  
                    X_all   = scaler.transform(X)  
                    
                    model.fit( X_train, y_train, groups_train )  # internal split into train- and val-set
        
                    y_train_pred = model.predict(X_train)
                    y_test_pred  = model.predict(X_test)
                    y_pred_all   = model.predict(X_all)
                    
                    train_error.append( mae(y_train, y_train_pred) )
                    test_error.append(  mae(y_test, y_test_pred) )          
                    test_err_r2.append( r2(y_test, y_test_pred)  ) 
                    test_err_rms.append( sqrt(mse(y_test, y_test_pred)) ) 
                    
                    #print(i, j, group_list, season_list, round(np.mean(test_error), 3), sep="   " )       
                    i = i + 1
        
                #print(i, j, group_list, season_list, round(np.mean(test_error), 3), sep="   " )
            
                cv_train_err.append( np.mean( train_error ))
                cv_test_err.append(  np.mean( test_error ))
                cv_test_err_r2.append(  np.mean( test_err_r2 ))
                cv_test_err_rms.append(  np.mean( test_err_rms ))
                
                train_error, test_error, test_err_r2, test_err_rms = [],[],[],[]
            
                        
        print(" ")
        print("train_error (mean, std):                     ", round(np.mean(cv_train_err), 3), round(np.std(cv_train_err),3), sep="  ")
        print("test_error (mean, std):                      ", round(np.mean(cv_test_err), 3), round(np.std(cv_test_err),3), sep="  ")
        print(" ") 
        print("-------------------------------------------------------------------------")
        
        
        scen_train_err.append( cv_train_err )        
        scen_train_err_mean.append( np.mean(cv_train_err) )
        scen_train_err_std.append( np.std(cv_train_err) )        
        
        scen_test_err.append( cv_test_err )
        scen_test_err_mean.append( np.mean(cv_test_err) )
        scen_test_err_std.append( np.std(cv_test_err) )

        scen_test_r2.append( cv_test_err_r2 )
        scen_test_r2_mean.append( np.mean(cv_test_err_r2) )
        scen_test_r2_std.append( np.std(cv_test_err_r2) )

        scen_test_rmse.append( cv_test_err_rms )
        scen_test_rmse_mean.append( np.mean(cv_test_err_rms) )
        scen_test_rmse_std.append( np.std(cv_test_err_rms) )

        
        fig = plt.figure()
        plt.plot(cv_train_err, 'o-', label="train_error")
        plt.plot(cv_test_err, 'o-', label="test_error")
        plt.xlabel('#')
        plt.ylabel('error')
        plt.title('Error Estimate from Cross Validation')
        plt.legend(loc="best")
        #plt.show(block=False)
        plt.savefig( name + '_test_error_' + name2 )
        plt.close(fig)
                
        fig = plt.figure()
        plt.scatter(y, y_pred_all, s=0.1, c='orange', label='y_pred_all')
        plt.scatter(y_train, y_train_pred, s=1.0, c='blue', label='y_pred_train')
        plt.scatter(y_test, y_test_pred, s=1.0, c='red', label='y_pred_test')
        plt.plot([-1, 3], [-1, 3], 'r')
        plt.xlabel('exp_data')
        plt.ylabel('prediction')
        plt.title('Scatter Plot from Cross Validation')
        plt.legend(loc="best")
        #plt.show(block=False)
        plt.savefig(name + '_cv_scatter_' + name2)
        plt.close(fig)
        
        
        # =============================================================================
        # Re-traning on all intervals without error estimates
        # =============================================================================
                
        pred_error, pred_r2, pred_rmse, pred_emiss_all = [], [], [], []
        j = 0
        
        for train, test in gss.split( X, y, groups=groups ):
            
            X_new     = np.concatenate(( X[train], X[test] ))
            y_new     = np.concatenate(( y[train], y[test] ))
            gr_new    = np.concatenate(( groups[train], groups[test] ))
            train_new = np.concatenate(( train, test ))
        
            # check out interval number and season of training intervals     
            group_list   = list(valfilter(lambda count: count > 1, frequencies(gr_new)).keys())
            season_array = np.array( gr_season )  # list => array, for use of index_list 
            season_list  = list( season_array[group_list] )
            
            nt = season_list.count('T')
            ns = season_list.count('S')
            nw = season_list.count('W')
        
            if( ( j < n_sample) and ( j < n_sample) and (nt == n_trans) and (ns == n_summer) and (nw == n_winter) ):   
                
                j = j + 1
                
                X_train, y_train, groups_train  = X_new, y_new, gr_new    
                
                # all X-values without train (X_disjunct)
                X_dis = np.delete(X, train_new, axis=0)
                y_dis = np.delete(y, train_new)
                
                X_all   = X
                X_train = scaler.fit_transform(X_train)  # feature scaling
                X_all   = scaler.transform(X_all)
                X_dis   = scaler.transform(X_dis)   
        
                model.fit( X_train, y_train, groups_train )  #  internal split into train- and val-set
        
                y_train_pred  = model.predict(X_train)
                y_pred_dis    = model.predict(X_dis)
                y_pred_all    = model.predict(X_all)
        
                pred_err      = mae( y_dis, y_pred_dis )               
                pred_err_r2   = r2(  y_dis, y_pred_dis ) 
                pred_err_rmse = mse( y_dis, y_pred_dis ) 
                
                ef_pred_all  = np.mean(y_pred_all) 
                
                pred_error.append( pred_err)
                pred_r2.append( pred_err_r2)
                pred_rmse.append( pred_err_rmse)
                pred_emiss_all.append( ef_pred_all ) 
        
                #print(j, group_list, season_list, round(pred_err, 3), round(ef_pred_all, 3), sep="   " )                   
                
            
        scen_emiss.append( pred_emiss_all )
        scen_emiss_mean.append( np.mean( pred_emiss_all ) )
        scen_emiss_std.append(  np.std(  pred_emiss_all ) )   
        
        scen_pred_err.append( pred_error )
        scen_pred_err_mean.append( np.mean(pred_error) )
        scen_pred_err_std.append( np.std(pred_error) )

        scen_pred_r2.append( pred_r2 )
        scen_pred_r2_mean.append( np.mean(pred_r2) )
        scen_pred_r2_std.append( np.std(pred_r2) )
            
        scen_pred_rmse.append( pred_rmse )
        scen_pred_rmse_mean.append( np.mean(pred_rmse) )
        scen_pred_rmse_std.append( np.std(pred_rmse) )
            
        print(" ")
        print("y_pred_rmse (mean, std):             ", 
              round(np.sqrt(np.mean(pred_rmse)), 3), round(np.std(pred_rmse), 6), sep="  ")
        print("y_pred_R2 (mean, std):               ", 
              round(np.mean(pred_r2), 3), round(np.std(pred_r2), 6), sep="  ")
            
        print(" ")
        print("y_pred_mae (mean, std):              ", 
              round(np.mean(pred_error), 3), round(np.std(pred_error), 6), sep="  ")
        print(" ") 
        print("TAE                                  ", 
              round(np.mean(pred_emiss_all) - exp_emiss, 6), 
              round( 100*(np.mean(pred_emiss_all) - exp_emiss)/exp_emiss, 6) , sep="  ")
        
        
        print(" ") 
        print("predicted emission all (mean, std):  ", 
              round(np.mean(pred_emiss_all), 3), round(np.std(pred_emiss_all), 3), sep="  ")
        print(" ") 
        print("experimental emission:               ", round(exp_emiss, 3), sep="  ")
        print(" ")       
        print("-------------------------------------------------------------------------")

        
        fig = plt.figure()
        plt.plot(pred_emiss_all, 'o-', label="pred_emiss")
        plt.plot([0, j-1], [exp_emiss, exp_emiss], '-', label="exp_emiss")
        plt.plot([0, j-1], [np.mean(pred_emiss_all), np.mean(pred_emiss_all)], '-', color='0.75',label="pred_mean")
        plt.plot([0, j-1], [np.mean(pred_emiss_all)+np.std(pred_emiss_all), np.mean(pred_emiss_all)+np.std(pred_emiss_all)], ':', color='0.75', label="pred_mean+-std")
        plt.plot([0, j-1], [np.mean(pred_emiss_all)-np.std(pred_emiss_all), np.mean(pred_emiss_all)-np.std(pred_emiss_all)], ':', color='0.75')
        plt.xlabel('#')
        plt.ylabel('NH3-Emission')
        plt.title('Predicted Emission')
        plt.legend(loc="best")
        #plt.show(block=False)
        plt.savefig(name + '_pred_emiss_' + name2)
        plt.close(fig)
                
        fig = plt.figure()
        plt.plot(pred_error, 'o-', label="pred_error")
        plt.xlabel('#')
        plt.ylabel('error')
        plt.title('Prediction Error')
        plt.legend(loc="best")
        # plt.show(block=False)
        plt.savefig(name + '_pred_error_' + name2)
        plt.close(fig)
        
        fig = plt.figure()
        plt.scatter(y, y_pred_all, s=0.1, c='orange', label='y_pred_all')
        plt.scatter(y_train, y_train_pred, s=1.0, c='blue', label='y_train_pred')
        plt.plot([-1, 3], [-1, 3], 'r')
        plt.xlabel('exp_data')
        plt.ylabel('prediction')
        plt.title('Prediction ScatterPlot')
        plt.legend(loc="best")
        #plt.show(block=False)
        plt.savefig(name + '_scatter_' + name2)
        plt.close(fig)
    
    
    
        pred_error, pred_emiss_all = [], []
            
 
    
print(" ")       
print("-------------------------------------------------------------------------")

scen_emiss         = np.array(scen_emiss)
scen_emiss_mean    = np.array(scen_emiss_mean) 
scen_emiss_std     = np.array(scen_emiss_std)     

scen_pred_err      = np.array(scen_pred_err) 
scen_pred_err_mean = np.array(scen_pred_err_mean) 
scen_pred_err_std  = np.array(scen_pred_err_std) 

scen_pred_r2        = np.array(scen_pred_r2) 
scen_pred_r2_mean   = np.array(scen_pred_r2_mean) 
scen_pred_r2_std    = np.array(scen_pred_r2_std) 

scen_pred_rmse      = np.array(scen_pred_rmse) 
scen_pred_rmse_mean = np.array(scen_pred_rmse_mean) 
scen_pred_rmse_std  = np.array(scen_pred_rmse_std) 

scen_train_err      = np.array(scen_train_err)
scen_train_err_mean = np.array(scen_train_err_mean)
scen_train_err_std  = np.array(scen_train_err_std)       

scen_test_err       = np.array(scen_test_err)
scen_test_err_mean  = np.array(scen_test_err_mean)
scen_test_err_std   = np.array(scen_test_err_std)

scen_test_r2        = np.array( scen_test_r2 )
scen_test_r2_mean   = np.array( scen_test_r2_mean ) 
scen_test_r2_std    = np.array( scen_test_r2_std ) 

scen_test_rmse      = np.array( scen_test_rmse )
scen_test_rmse_mean = np.array( scen_test_rmse_mean )
scen_test_rmse_std  = np.array( scen_test_rmse_std )

np.save(dirnam + 'emiss',        scen_emiss)
np.save(dirnam + 'emiss_mean',   scen_emiss_mean)
np.save(dirnam + 'emiss_std',    scen_emiss_std) 

np.save(dirnam + 'pred_err',      scen_pred_err) 
np.save(dirnam + 'pred_err_mean', scen_pred_err_mean) 
np.save(dirnam + 'pred_err_std',  scen_pred_err_std)

np.save(dirnam + 'pred_r2',      scen_pred_r2) 
np.save(dirnam + 'pred_r2_mean', scen_pred_r2_mean) 
np.save(dirnam + 'pred_r2_std',  scen_pred_r2_std)

np.save(dirnam + 'pred_rmse',      scen_pred_rmse) 
np.save(dirnam + 'pred_rmse_mean', scen_pred_rmse_mean) 
np.save(dirnam + 'pred_rmse_std',  scen_pred_rmse_std)

np.save(dirnam + 'train_err',      scen_train_err)
np.save(dirnam + 'train_err_mean', scen_train_err_mean)
np.save(dirnam + 'train_err_std',  scen_train_err_std)       

np.save(dirnam + 'test_err',      scen_test_err)
np.save(dirnam + 'test_err_mean', scen_test_err_mean)
np.save(dirnam + 'test_err_std',  scen_test_err_std)

np.save(dirnam + 'test_r2',      scen_test_r2)
np.save(dirnam + 'test_r2_mean', scen_test_r2_mean)
np.save(dirnam + 'test_r2_std',  scen_test_r2_std)

np.save(dirnam + 'test_rmse',      scen_test_rmse)
np.save(dirnam + 'test_rmse_mean', scen_test_rmse_mean)
np.save(dirnam + 'test_rmse_std',  scen_test_rmse_std)



#------------------------------------------------------------------------------
# TextFile-Output:

exp_new    = np.full(len(scen_emiss_mean), exp_emiss)  # array withconstant  exp-values
mean_emiss = np.full(len(scen_emiss_mean), np.mean(scen_emiss_mean))
mean_error = np.full(len(scen_emiss_mean), np.mean(scen_pred_err_mean))

emiss_new = np.stack((scen_emiss_mean, scen_emiss_std, (scen_emiss_mean - scen_emiss_std), 
                      (scen_emiss_mean + scen_emiss_std), exp_new, mean_emiss), axis=1)

error_new = np.stack((scen_pred_err_mean, scen_pred_err_std, (scen_pred_err_mean - scen_pred_err_std), 
                      (scen_pred_err_mean + scen_pred_err_std), mean_error), axis=1)

df_emiss = pd.DataFrame(emiss_new)
df_error = pd.DataFrame(error_new)

df_emiss.to_csv(name + '_Emiss.dat', header=False, index=True, sep='\t', mode='w')
df_error.to_csv(name + '_Error.dat', header=False, index=True, sep='\t', mode='w')


#------------------------------------------------------------------------------

print(" ")
t_end = time.time()
t_run = (t_end - t_start) / 60
print(" --- %s minutes --- " % t_run )

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
