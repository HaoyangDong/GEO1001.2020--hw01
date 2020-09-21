#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from Load_HEAT import Load_HEAT
import numpy as np


# In[4]:


file_dir = 'hw01_data'
file_A = file_dir+'/'+'HEAT - A_final.xls'
HEAT_A_df = Load_HEAT(file_A).drop(columns='FORMATTED DATE-TIME')
HEAT_A_df


# In[ ]:


# def get_index(lst = None):
#     return [index for (index,value) in enumerate(lst) if type(value) == str]
# lst = HEAT_A_df['Wind Speed'].values
# get_index(lst)


# In[ ]:


# for column_name in HEAT_A_df.columns:
#     column_np = np.asarray(HEAT_A_df[column_name].values).astype(np.float)
#     print(np.mean(column_np))


# In[ ]:


# Direction_True_np = np.asarray(HEAT_A_df['Direction_True'].values)
# print(np.mean(Direction_True_np))


# In[6]:


def HEAT_mean_var_std(dataframe):
    mean_list = []
    var_list = []
    std_list = []
    for column_name in dataframe.columns:
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


# In[7]:


mean_list_A, var_list_A, std_list_A = HEAT_mean_var_std(HEAT_A_df)
print(mean_list_A,'\n',var_list_A, '\n', std_list_A)


# In[ ]:


file_B = file_dir+'/'+'HEAT - B_final.xls'
HEAT_B_df = Load_HEAT(file_B).drop(columns='FORMATTED DATE-TIME')


# In[5]:


names = locals()
for i in ['A', 'B', 'C', 'D', 'E']:
    names['file_'+i] = file_dir+'/'+'HEAT - '+i+'_final.xls'
    names['HEAT_'+i+'_df'] = Load_HEAT(names.get('file_'+i)).drop(columns='FORMATTED DATE-TIME')
print(HEAT_B_df)


# In[8]:


import matplotlib.pyplot as plt
import matplotlib


# In[12]:


data_temp = []
for ii in ['A', 'B', 'C', 'D', 'E']:
    df = names.get('HEAT_'+ii+'_df')
#     print(df['Temperature'].values.astype(np.float))
    data_temp.extend(df['Temperature'].values.astype(np.float))
print(data_temp)


# In[3]:


names.get('HEAT_A_df')


# In[21]:


y, edges, _ = plt.hist(data_temp, bins=5)
plt.show()
print(edges)


# In[20]:


y, edges, _ = plt.hist(data_temp, bins=50)
print(edges)
print(edges[1:])
print(edges[:-1])
print(y)
plt.show()


# In[14]:


print(max(data_temp),min(data_temp))


# In[15]:


intervals = []
for i in range(6,36):
    intervals.append(i)
print(intervals)


# In[16]:


# plt.xticks(intervals)
y, edges, _ = plt.hist(data_temp, bins=intervals)
# print(egdes)
edges
midpoints = 0.5*(edges[1:]+edges[:-1])
plt.plot(midpoints, y)


# In[22]:


# data_ws = []
# data_wd = []
for iii in ['A', 'B', 'C', 'D', 'E']:
    df = names.get('HEAT_'+ii+'_df')
#     print(df['Temperature'].values.astype(np.float))
#     data_ws.extend(df['Wind Speed'].values.astype(np.float))
#     data_wd.extend(df['Direction_True'].values.astype(np.float))
    names['data_ws_'+iii] = df['Wind Speed'].values.astype(np.float)
    np.asarray(names.get('data_ws_'+iii))
#     np.stack(names.get('data_ws_'+iii), axis=0)
    names['data_wd_'+iii] = df['Direction_True'].values.astype(np.float)
    np.asarray(names.get('data_wd_'+iii))
#     np.stack(names.get('data_wd_'+iii), axis=0)
    names['data_temp_'+iii] = df['Temperature'].values.astype(np.float)
#     np.stack(names.get('data_temp_'+iii), axis=0)
    np.asarray(names.get('data_temp_'+iii))
data_temp_C


# In[23]:


figure, ax = plt.subplots(3, 1)
data_box_ws = np.asarray([data_ws_A, data_ws_B, data_ws_C, data_ws_D, data_ws_E])
data_box_wd = data_wd_A
data_box_temp = data_temp_A
# np.asarray(data_box_ws)
# np.asarray(data_box_wd)
# np.asarray(data_box_temp)
# for iiii in ['B', 'C', 'D', 'E']:
#     np.column_stack((data_box_ws, names.get('data_ws_'+iiii)))
#     np.column_stack((data_box_wd, names.get('data_wd_'+iiii)))
#     np.column_stack((data_box_temp, names.get('data_temp_'+iiii)))
#     data_box_ws = np.asarray([data_box_ws, names.get('data_ws_'+iiii)])
#     data_box_wd = np.asarray([data_box_wd, names.get('data_wd_'+iiii)])
#     data_box_temp = np.asarray([data_box_temp, names.get('data_temp_'+iiii)])
ax[0].boxplot([data_ws_A, data_ws_B, data_ws_C, data_ws_D, data_ws_E])
ax[1].boxplot([data_wd_A, data_wd_B, data_wd_C, data_wd_D, data_wd_E])
ax[2].boxplot([data_temp_A, data_temp_B, data_temp_C, data_temp_D, data_temp_E])
figure.show()
data_box_ws


# In[ ]:


data_box_ws


# In[ ]:


names.get('data_ws_'+'C')
np.append(data_box_temp, names.get('data_temp_'+'C'), axis=0)
data_box_temp


# In[ ]:


plt.boxplot([data_ws_A, data_ws_B, data_ws_C, data_ws_D, data_ws_E])
plt.show()


# In[ ]:


all_data=[np.random.normal(0,std,100) for std in range(1,4)]
all_data

