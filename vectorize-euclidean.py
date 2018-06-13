
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
n_df=pd.read_csv("C:\\Users\\Cora\\N.csv")
Vec1=np.asarray(n_df[["new_eth","new_rel","new_mar","new_ins"]])
Vec2=np.asarray(n_df['n'])

def equ(n):
    if n:
        return 0
    else:
        return 1

euq = np.vectorize(equ)

new_df = pd.DataFrame()
for i in range(10000,12000):
    V1 = np.equal(Vec1[i, :], Vec1)
    V1 = euq(V1)
    V1 = np.sum(V1,axis = 1)
    diff = Vec2 - Vec2[i]
    V2 = np.square(diff)
    V = V1 + V2
    V = np.sqrt(V)
    new_df[i] = V

new_df.to_csv('C:\\Users\\Cora\\Euclidean.csv') 


# In[8]:

# for m-distance, but computer breaks down when generating inverse matrix
import pandas as pd
import numpy as np
n_df=pd.read_csv("C:\\Users\\Cora\\N.csv")
Vec=n_df.values
CovMat=np.cov(Vec)
#InverseCovMat=np.linalg.pinv(CovMat)
print CovMat


# In[ ]:


InverseCovMat=np.linalg.inv(CovMat)
print InverseCovMat

