# for manipulations
import numpy as np
import pandas as pd

# For path finding
import os.path

def get_dataset(ele):
    current_directory = os.path.dirname('custom_function.py')
    current_directory = os.getcwd()
    parent_directory = os.path.split(current_directory)[0] # Repeat as needed
    file_path = parent_directory + ele
    data = pd.read_csv(file_path)
    return data