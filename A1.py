import numpy as np
import pandas as pd

# To calculate the mean
def HEAT_mean(dataframe):
	mean_list = []
	for column_name in dataframe.columns:
		column_np = np.asarray(dataframe[column_name].values)
		column_mean = np.mean()
		mean_list.append(column_mean)
	return mean_list