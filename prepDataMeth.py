import locale
import numpy as np                 # package for numerics
import pandas as pd                # package for data frames  
import matplotlib.pyplot as plt    # package for plotting
from scipy.stats import pearsonr   # pearson correl.coeff.
from datetime import datetime


def prepDataMeth(n_days):
    
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')  # for english months names 
    
    # Import data
    df = pd.read_csv("Methan.csv", sep = "\t")   
    
    # drop NaN-entries raw-wise
    df.dropna(axis=0, inplace=True)
    
    # drop out useless coloums:
    df.drop(["EF_NH3",	"EF_N2O"] , axis = 1, inplace = True)
    
    # restrict to columns with EF_CH4 > 0  (because of logarithm)
    df = df[df["EF_CH4"] >= 0.000000001] 
    
    # group together intevals of interval length intv_leng
    intv_leng = n_days   # interval_length [days]
    
    # new columns: 

    df["LN_EF"] = np.log(df["EF_CH4"])  
    
    df["Tsq"]   = np.square(df["Temp"])  # => besser in die eigentlichen Skripte ?!
    
    df["h_sin"] = df["Time"]  # day_time in hours  
    df["h_cos"] = df["Time"]
        
        
    
    ### Running numbers for the days: 
    date_0 = datetime.strptime(str(int(df["Date"][0])), "%Y%m%d")
    days = []
    
    for i in df.index:
        date = datetime.strptime(str(int(df["Date"][i])), "%Y%m%d")
        time = date - date_0
        time = time.total_seconds()//86400
        days.append(time)
    
    df["days"]  = days
    df["t_sin"] = days
    df["t_cos"] = days 
    
    # new columns, wind direction:
    df["dW_sin"] = df["Wind_dir"]
    df["dW_cos"] = df["Wind_dir"]
    
    ## convert cyclic/periodic variables to sin-cos:
    # day_time [hours]
    df["h_sin"] = np.sin(2*np.pi/24*df["h_sin"])  # Sine
    df["h_cos"] = np.cos(2*np.pi/24*df["h_cos"])  # Cosine
    
    # time [days]
    df["t_sin"] = np.sin(2*np.pi/365.25*df["t_sin"])
    df["t_cos"] = np.cos(2*np.pi/365.25*df["t_cos"])
    
    # wind direction:
    df["dW_sin"] = np.sin(2*np.pi/360*df["dW_sin"])
    df["dW_cos"] = np.cos(2*np.pi/360*df["dW_cos"])
    
    
    #------------------------------------------------------------------------------
    ###  Date  Time  Temp  Wind_dir  Wind_spd  EF_CH4
    '''
    corr, _ = pearsonr(df["Temp"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (T, EF):        ", corr)     
    corr, _ = pearsonr(df["Temp"], np.log(df["EF_CH4"]))  # natural logarithm
    print("Pearson Correl.Coeff. (T, ln_EF):     ", corr)     
    corr, _ = pearsonr(df["Tsq"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (T**2, EF):     ", corr)     
    corr, _ = pearsonr(df["Tsq"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (T**2, ln_EF):  ", corr) 
    corr, _ = pearsonr(df["Wind_spd"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (Wind_spd, EF):        ", corr)     
    corr, _ = pearsonr(df["Wind_spd"], np.log(df["EF_CH4"]))  # natural logarithm
    print("Pearson Correl.Coeff. (Wind_spd, ln_EF):     ", corr) 
    
    corr, _ = pearsonr(df["dW_sin"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (dW_sin, EF):     ", corr) 
    corr, _ = pearsonr(df["dW_sin"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (dW_sin, ln_EF):  ", corr) 
    corr, _ = pearsonr(df["dW_cos"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (dW_cos, EF):     ", corr) 
    corr, _ = pearsonr(df["dW_cos"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (dW_cos, ln_EF):  ", corr) 

    corr, _ = pearsonr(df["h_sin"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (h_sin, EF):     ", corr) 
    corr, _ = pearsonr(df["h_sin"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (h_sin, ln_EF):  ", corr) 
    corr, _ = pearsonr(df["h_cos"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (h_cos, EF):     ", corr) 
    corr, _ = pearsonr(df["h_cos"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (h_cos, ln_EF):  ", corr) 

    corr, _ = pearsonr(df["t_sin"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (t_sin, EF):     ", corr) 
    corr, _ = pearsonr(df["t_sin"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (t_sin, ln_EF):  ", corr) 
    corr, _ = pearsonr(df["t_cos"], df["EF_CH4"]) 
    print("Pearson Correl.Coeff. (t_cos, EF):     ", corr) 
    corr, _ = pearsonr(df["t_cos"], np.log(df["EF_CH4"])) 
    print("Pearson Correl.Coeff. (t_cos, ln_EF):  ", corr) 
    '''
    #------------------------------------------------------------------------------

    
    #df.drop(["dW", "h"], axis = 1, inplace = True)
    
    ### construct column with datetime-format: 07Jan2017:19:00:00
    
    ms = '0000'   # minutesSeconds
    
    timestamp = []
    for i in df.index:
        date = datetime.strptime( str(int(df["Date"][i])) + str(int(df["Time"][i])) + ms, "%Y%m%d%H%M%S")
        time = date.strftime("%d%b%Y:%H:%M:%S")
        timestamp.append(time)
    
    df['dt'] = timestamp
    
    
    #------------------------------------------------------------------------------
    # join together days to intervals of n days
    # date	    dt	                h	t 
    # 13FEB2017	13FEB2017:07:00:00	7	2503	
    # str1 = '14MAY2018:00:00:00'
    # date_1 = datetime.strptime(str1, '%d%b%Y:%H:%M:%S')
    # print(date_2)
    # %d day, %b month as string, %m month as int, %y year as 2 dig, %Y year as 4 dig
    
    n_date = len(df)
    dt_0   = df['dt'].values[0] 
    date_0 = datetime.strptime(dt_0, '%d%b%Y:%H:%M:%S')
    count  = 0
    intv = []
    
    for i in range(n_date):
        dt_i = df['dt'].values[i]     
        date = datetime.strptime(dt_i, '%d%b%Y:%H:%M:%S')
     
        # datetime.tamedelta()-Object with attribute days
        if ((date-date_0).days <= (intv_leng-1)):
            intv.append(count)
            #print(date-date_0)
        else:   
            #print(date-date_0)
            count  = count + 1
            intv.append(count)
            date_0 = date
            #print(date-date_0)
    
    for i in range(n_date):
        dt_i = df['dt'].values[i]     
        date = datetime.strptime(dt_i, '%d%b%Y:%H:%M:%S')
        #print(date, intv[i])
    
    #for i in range(len(set(intv))):
        #print(i, intv.count(i))
    
    df["group"] = intv  #add column "groups"
    
    #-------------------------------------------------------------------------------
    # assign season S, W, T to groups, oriented on months
    # Jan, Feb            <->  W
    # Mar, Apr, May       <->  T
    # Jun, Jul, Aug       <->  S
    # Sep, Oct, Nov, Dec  <->  T
    #
    # in the sample Sep+Oct is missing, so we have 2*W, 5*T, 3*S  
    
    seas = []
    
    for i in range(n_date):
        dt_i = df['dt'].values[i]     
        date = datetime.strptime(dt_i, '%d%b%Y:%H:%M:%S')
        #print(date, intv[i])
    
        if (date.month <= 2):
            seas.append('W')
            #print(date-date_0)
        elif ((date.month >= 3) and (date.month <= 5)):   
            seas.append('T')
        elif ((date.month >= 6) and (date.month <= 8)):   
            seas.append('S')
        elif ((date.month >= 9) and (date.month <= 12)):   
            seas.append('T')
        else:
            print('season_error')
            #print(date-date_0)
    
    df["season"] = seas  #add column "season"
    
    
    #-------------------------------------------------------------------------------
    # if last interval contains less that 80% of amount of expected data, drop last interval
    perc = 0.8
    numb = intv.count(len(set(intv))-1)
    if (numb < intv_leng*24*perc):
        df.drop(df.tail(numb).index, inplace = True)
    
    #-------------------------------------------------------------------------------
    
    # print preprocessed data to a file:
    df.to_csv('Methan_proc.dat', header=True, index=False, sep='\t', mode='w')
    
    ###############################################################################

# if __name__ == "__main__":
#    
#   prepDataMeth(7)