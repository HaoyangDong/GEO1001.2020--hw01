import pandas as pd
import numpy as np
import math

# To load a xls file
def Load_HEAT(HEAT_file):
	HEAT_df = pd.read_excel(HEAT_file, header=None).drop([0, 1, 2, 3, 4])
	Column_name_list = ['FORMATTED DATE-TIME', 'Wind Direction', 'Wind Speed', 'Crosswind Speed', 'Headwind Speed', 'Temperature', 'Globe Temperature', 'Wind Chill', 'Relative Humidity', 'Heat Stress Index', 'Dew Point', 'Psychro Wet Bulb Temperature', 'Station Pressure', 'Barometric Pressure', 'Altitude', 'Density Altitude', 'NA Wet Bulb Temperature', 'WBGT', 'TWL ', 'Direction Mag']
	HEAT_df.columns=Column_name_list
	# return HEAT_df.drop(columns='FORMATTED DATE-TIME')
	return HEAT_df

# To compute mean, variance, and standard deviation
def HEAT_mean_var_std(dataframe):
    mean_list = []
    var_list = []
    std_list = []
    Column_name_list = ['Wind Direction', 'Wind Speed', 'Crosswind Speed', 'Headwind Speed', 'Temperature', 'Globe Temperature', 'Wind Chill', 'Relative Humidity', 'Heat Stress Index', 'Dew Point', 'Psychro Wet Bulb Temperature', 'Station Pressure', 'Barometric Pressure', 'Altitude', 'Density Altitude', 'NA Wet Bulb Temperature', 'WBGT', 'TWL ', 'Direction Mag']
    for column_name in Column_name_list:
#         if column_name == 'FORMATTED DATE-TIME':
#             pass
        column_np = np.asarray(dataframe[column_name].values).astype(np.float)
        column_mean = np.mean(column_np)
        column_var = np.var(column_np)
        column_std = np.std(column_np)
        mean_list.append(column_mean)
        var_list.append(column_var)
        std_list.append(column_std)
    return mean_list, var_list, std_list

