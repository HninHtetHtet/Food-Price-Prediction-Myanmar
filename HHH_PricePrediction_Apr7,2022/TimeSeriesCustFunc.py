# for manipulations
import numpy as np
import pandas as pd

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.style.use('fivethirtyeight')

# For Prediction
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict

# For path finding
import os.path

# SARIMAX_Graph(data, col_to_pred, [pred, pred_ci], [time_range_for_Observation, time_range_for_Prediction], title_of_plot, xlabel, ylabel, [legend_for_Observation, legend_for_Prediction])
def SARIMAX_Graph(data, col, pred, time_range, title, xlabel, ylabel, legend=['Observed', 'Predicted']):
    fig, ax = plt.subplots(figsize=(12,10))
    ax.set(title= title, xlabel= xlabel, ylabel= ylabel)
    ax.set_ylim(bottom=0, top=max(data[col])+1000)
    
    # Plot data points
    data.loc[time_range[0][0]:time_range[0][1], col].plot(ax=ax, style='o', label=legend[0])
    
    # Plot predictions
    pred[0].predicted_mean.loc[time_range[1][0]:time_range[1][1]].plot(ax=ax, style='g', label=legend[1])
    ci = pred[1].loc[time_range[1][0]:time_range[1][1]]
    ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)
    
    legend = ax.legend(loc='lower right')

# Run_SARIMAX(data, (p, d, q), [time_range_for_Observation, time_range_for_Prediction], col_to_pred, (P, D, Q, m), title_of_plot, xlabel, ylabel)
def Run_SARIMAX(data, mod_order, time_range, col, season_order=(0,0,0,0), title='', xlabel='', ylabel=''):
    model=SARIMAX(data, order=mod_order, seasonal_order=season_order).fit()
    getPred = model.get_prediction(start=time_range[1][0], end=time_range[1][1])
    getPred_ci = getPred.conf_int()
    SARIMAX_Graph(data, col, [getPred, getPred_ci], time_range, title, xlabel, ylabel)
    return model

# Compute accuracy with AIC, BIC
def AIC_BIC_withOrderRange(data, p=1, d=1, q=1, P=1, D=1, Q=1, m=1):
    # p, d, q, P, D, Q, m must be greater than 0
    ind = p*d*q*P*D*Q*m
    AIC = np.zeros(ind)
    BIC = np.zeros(ind)
    for i in range(p):
        for j in range(d):
            for k in range(q):
                for I in range(P):
                    for J in range(D):
                        for K in range(Q):
                            for l in range(m):
                                mod=SARIMAX(data, order=(i,j,k))
                                res=mod.fit()
                                AIC[(i*d*q*P*D*Q*m)+(j*q*P*D*Q*m)+(k*P*D*Q*m)+(I*D*Q*m)+(J*Q*m)+(K*m)+l]=res.aic
                                BIC[(i*d*q*P*D*Q*m)+(j*q*P*D*Q*m)+(k*P*D*Q*m)+(I*D*Q*m)+(J*Q*m)+(K*m)+l]=res.bic
    plt.plot(range(1,ind),AIC[1:ind], label="AIC", marker='o')
    plt.plot(range(1,ind),BIC[1:ind], label="BIC", marker='o')
    plt.legend(loc='best',fontsize='small')
    plt.show()
    return AIC, BIC

# Load wfp_food_prices_mmr(Cleaned).csv and clean the data
def LoadClean_wfp_food_prices_mmr():
    current_directory = os.path.dirname('TimeSeriesCustFunc.ipynb')
    current_directory = os.getcwd()
    parent_directory = os.path.split(current_directory)[0] # Repeat as needed
    file_path = parent_directory + '\\Datasets\\wfp_food_prices_mmr(Cleaned).csv'
    data = pd.read_csv(file_path)
    # latitude and longitude are not needed
    data.drop(['latitude', 'longitude'],  axis='columns', inplace=True)
    return data
