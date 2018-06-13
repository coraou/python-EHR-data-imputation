# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:26:15 2018

@author: Cora
"""

import pandas as pd
admission_df = pd.read_csv('C:\\Users\\Cora\\ADMISSIONS.csv')
#admission_small = admission_df.iloc[0:3870] #for testing

admission_df.drop_duplicates(subset = 'SUBJECT_ID') # remove unnecessary info
# to see the unique values of the column
def look(field_name):
    num = admission_df[field_name].nunique() 
    admission_df_elements = admission_df[field_name].unique()
    print num
    print admission_df_elements

# cleaning ethnicity
other = ['PORTUGUESE', 'SOUTH AMERICAN', 'CARIBBEAN ISLAND', 'MIDDLE EASTERN']
searchfor_eth = ['WHITE', 'ASIAN', 'BLACK', 'HISPANIC', 'AMERICAN INDIAN']

for i in range(0,5):
    for index,row in admission_df.iterrows():
        if searchfor_eth[i] in row['ETHNICITY']:
            admission_df.ETHNICITY.iloc[[index]] = searchfor_eth[i]
            print admission_df.ETHNICITY.iloc[[index]]
            
for t in range(0,4):
    for index,row in admission_df.iterrows():
            if other[t] in row['ETHNICITY']:
                admission_df.ETHNICITY.iloc[[index]] = 'OTHER'
                print admission_df.ETHNICITY.iloc[[index]]
                
# cleaning religion
OTHERCHRISTIAN = ["JEHOVAH'S WITNESS", 'GREEK ORTHODOX','EPISCOPALIAN', 'CHRISTIAN SCIENTIST', 'METHODIST',
                   'UNITARIAN-UNIVERSALIST', 'BAPTIST', '7TH DAY ADVENTIST', 'ROMANIAN EAST. ORTH', 'LUTHERAN']
for j in range(0,10):
    for index,row in admission_df.iterrows():
        if row['RELIGION'] == 'HEBREW':
            admission_df.RELIGION.iloc[[index]] = 'JEWISH'
        
        if row['RELIGION'] == OTHERCHRISTIAN[j]:
            print index
            admission_df.RELIGION.iloc[[index]] = 'OTHER-CHRISTIAN'
            

searchfor_new_eth = ['WHITE', 'ASIAN', 'BLACK', 'HISPANIC', 'AMERICAN INDIAN', 'MULTI RACE ETHNICITY','NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','OTHER']
searchfor_new_rel = ['CATHOLIC', 'PROTESTANT QUAKER','JEWISH','BUDDHIST','OTHER','HINDU','MUSLIM']
searchfor_new_ins = ['Private', 'Medicare', 'Medicaid', 'Self Pay', 'Government']
searchfor_new_mar = ['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOWED', 'SEPARATED','LIFE PARTNER']

def normalize(field_name,field_name_2,num,lis):
    admission_df[field_name_2] = ""
    for i in range(0,num):
        for index,row in admission_df.iterrows():
            if row[field_name] == lis[i] :
                admission_df[field_name_2][index] = i
    
normalize('ETHNICITY','new_eth',8,searchfor_new_eth)
normalize('RELIGION','new_rel',7,searchfor_new_rel)
normalize('MARITAL_STATUS','new_mar',6,searchfor_new_mar)
normalize('INSURANCE','new_ins',5,searchfor_new_ins)
df = admission_df[['SUBJECT_ID','new_eth','new_rel','new_mar','new_ins']]
df.to_csv('C:\\Users\\Cora\\ADMISSIONS-N.csv')