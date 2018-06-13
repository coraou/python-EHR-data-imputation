# -*- coding: utf-8 -*-
"""
Created on Sun May 27 20:04:03 2018

@author: Cora
"""

import pandas as pd

#import numpy as np
patients_df = pd.read_csv('C:\\Users\\Cora\\PATIENTS.csv')
#patients_df = patients_df.iloc[:20]
patients_df['N'] = ""
patients_df['n'] = ""

for index,row in patients_df.iterrows():
    if '/' in row['DOB']:
        string = row['DOB'].split('/')
        d = string[2].split(' ')
        string[2] = d[0]
        string = map(int,string)
        patients_df['N'][index] = (string[0] - 1) * 365 + (string[1] - 1) * 30 + string[2]
        
'''
# to find the maximum value
def maximum1(field_name,l):
    m_d = {}
    for i in l:
        m_d[i] = patients_small[field_name].iloc[i]
    return m_d

y = max(patients_small['YEAR'])    #find the maximum value of year
yl = [index for index,val in enumerate(patients_small['YEAR']) if val == y] #find the indices of the maximum value of year
print yl

patient_m_d = maximum1('MONTH',yl) #to create a dict that has the month values of the maximum year 
print patient_m_d

m = max(patient_m_d.values())
ml = [key for key,value in patient_m_d.iteritems() if value == m] #find the indices of the maximum value of month
print ml

patient_d_d = maximum1('DAY',ml) #to create a dict that has the day values of the maximum month
print patient_d_d

d= max(patient_d_d.values())
dl = [key for key,value in patient_d_d.iteritems() if value == d]
print dl #find the persons who are oldest


#to find the minimum value
def minimum1(field_name,l):
    n_d = {}
    for i in l:
        n_d[i] = patients_small[field_name].iloc[i]
    return n_d

y1 = min(patients_small['YEAR'])    #find the minimum value of year
yll = [index for index,val in enumerate(patients_small['YEAR']) if val == y1] #find the indices of the minimum value of year
print yll

patient_n_d = minimum1('MONTH',yll) #to create a dict that has the month values of the minimum year 
print patient_n_d

m1 = min(patient_n_d.values())
mll = [key for key,value in patient_n_d.iteritems() if value == m1] #find the indices of the minimum value of month
print mll

patient_d_d_n = minimum1('DAY',mll) #to create a dict that has the day values of the minimum month
print patient_d_d_n

d1= min(patient_d_d_n.values())
dll = [key for key,value in patient_d_d_n.iteritems() if value == d1]
print dll #find the persons who are youngest


#for index,row in patients_small.iterrows():
#    patients_small['N'][index] = patients_small['YEAR'][index] + (patients_small['MONTH'][index] - 1) * 30 + patients_small['DAY'][index]
'''
patients_df['N'] = patients_df['N'].convert_objects(convert_numeric=True)
maxvalue = max(patients_df['N'])
minvalue = min(patients_df['N'])
t = 1/float(maxvalue - minvalue)

for index,row in patients_df.iterrows():
    patients_df['N'] = patients_df['N'].convert_objects(convert_numeric=True)
    n = patients_df['N'][index]
#    print type(n)
    patients_df['n'][index] = (n - minvalue) * t
#    print patients_df['n'][index]

print patients_df['n']


patients_df_n = patients_df[['SUBJECT_ID', 'n']]  
patients_df_n.to_csv('C:\\Users\\Cora\\PATIENTS-n.csv')  