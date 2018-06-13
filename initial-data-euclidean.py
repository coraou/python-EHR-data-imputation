
# coding: utf-8

# In[ ]:


import pandas as pd
import math
n_df = pd.read_csv('C:\\Users\\Cora\\N.csv')
n_df = n_df.iloc[:30]

def equ(n, m):
    if n == m:
        return 0
    else:
        return 1
    
col_names = ['new_eth','new_rel','new_mar','new_ins']

def judge(n, m):
    lis = []
    for i in col_names:
        lis.append(equ(n_df[i][n],n_df[i][m])) 
    lis.append(n_df['n'][n] - n_df['n'][m])
    a = math.sqrt(lis[0] + lis[1] + lis[2] +lis[3] + lis[4] * lis[4])
    return a 

new_df = pd.DataFrame()
for index,row in n_df.iterrows():
#    m = index + 1
    n_lis = []
    for i in range(0, 1000):
        n_lis.append(judge(i,index))
        
    new_df[index] = n_lis

#new_df.from_dict(new_dict)

new_df.to_csv('C:\\Users\\Cora\\Euclidean.csv') 

