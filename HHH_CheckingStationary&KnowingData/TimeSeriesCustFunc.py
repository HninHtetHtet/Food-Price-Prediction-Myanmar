# for manipulations
import numpy as np
import pandas as pd

# For path finding
import os.path

# Load wfp_food_prices_mmr(Cleaned).csv and clean the data
def LoadClean_wfp_food_prices_mmr():
    current_directory = os.path.dirname('TimeSeriesCustFunc.py')
    current_directory = os.getcwd()
    parent_directory = os.path.split(current_directory)[0] # Repeat as needed
    file_path = parent_directory + '\\Datasets\\wfp_food_prices_mmr(Cleaned).csv'
    data = pd.read_csv(file_path)
    # latitude and longitude are not needed
    data.drop(['latitude', 'longitude'],  axis='columns', inplace=True)
    return data
