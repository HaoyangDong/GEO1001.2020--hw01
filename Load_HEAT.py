import pandas as pd

def Load_HEAT(HEAT_file):
	HEAT_df = pd.read_excel(HEAT_file, header=None).drop([0, 1, 2, 3, 4])
	Column_name_list = ['FORMATTED DATE-TIME', 'Direction_True', 'Wind Speed', 'Crosswind Speed', 'Headwind Speed', 'Temperature', 'Globe Temperature', 'Wind Chill', 'Relative Humidity', 'Heat Stress Index', 'Dew Point', 'Psychro Wet Bulb Temperature', 'Station Pressure', 'Barometric Pressure', 'Altitude', 'Density Altitude', 'NA Wet Bulb Temperature', 'WBGT', 'TWL ', 'Direction_Mag']
	HEAT_df.columns=Column_name_list
	return HEAT_df
