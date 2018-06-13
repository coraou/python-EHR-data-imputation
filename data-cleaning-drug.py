# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
drug_df = pd.read_csv('C:\\Users\\Cora\\DRGCODES.csv')
drug_df = drug_df[['SUBJECT_ID','DRG_CODE']]
drug_df.sort_values(by = "SUBJECT_ID",ascending=False)
drug_elements = drug_df['DRG_CODE'].unique()
sub_id = drug_df['SUBJECT_ID'].unique()
drug_elements.sort()
grouped = drug_df.groupby('SUBJECT_ID')
df = {}
for i in sub_id:
    g = grouped.get_group(i)
    lis = g['DRG_CODE']
    s = str(lis).split(' ')
    t= [x for x in s if x]
    t = [x.replace('\n','\\n') for x in t]
    t = [x for x in t if '\\' in x]
    z = []
    a = []
    for x in t:
        s = x.split('\\')
        if s[0] and s[0] not in z:
            z.append(s[0])
    z = map(int,z)
    for b in drug_elements:
        if b in z:
            a.append(1)
            print b
        else:
            a.append(0)
    df[i] = a
#print df
df = pd.DataFrame.from_dict(df,orient='index')
df.to_csv('C:\\Users\\Cora\\1.csv')

