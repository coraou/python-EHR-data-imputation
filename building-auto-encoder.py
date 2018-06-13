# -*- coding: utf-8 -*-

import keras
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model



df = pd.read_csv('C:\\Users\\Cora\\1.csv')
data = np.asarray(df)
#refine work
#splitting the data into training and testing datasets
X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)
X_train, X_valid = train_test_split(X_train, test_size=0.2, random_state=0)

#randomly select 20% of the training data and change data whose value is 1 into 0
train_df = pd.DataFrame(X_train)
sample_train = train_df.sample(frac = 0.2)
sampletr_indices = sample_train.index.values

for index in sampletr_indices:
    for j in range(0,1666):
        if train_df[j][index] == 1:
            train_df[j][index] = 0
            
X_train = np.asarray(train_df)

#randomly select 20% of the testing data and randomly assign 0 or 1 to them
test_df = pd.DataFrame(X_test)
sample_test = test_df.sample(frac = 0.2)
samplete_indices = sample_test.index.values
for index in samplete_indices:
    for j in range(0,1666):
        test_df[j][index] = random.randint(0,1)

X_test = np.asarray(test_df)
    
def autoencoder(X_train,X_test):
    Inputshape = X_train.shape[1]
    # this is our input placeholder
    inputLayer = Input(shape=(Inputshape,))
    # building an encoder
    encoder = Dense(1200, activation='relu')(inputLayer)
    encoder = Dense(1000, activation='relu')(encoder)
    encoder = Dense(500, activation='relu')(encoder)
    # buiding a decoder
    decoder = Dense(1000, activation='relu')(encoder)
    decoder = Dense(1200, activation='relu')(decoder)
    decoder = Dense(Inputshape, activation='sigmoid')(decoder) 

    Autoencoder = Model(input=inputLayer, output=decoder)

    Autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    Autoencoder.summary()

    Autoencoder.fit(X_train, X_train,
                    epochs=3,
                    batch_size=56,
                    shuffle=True,
                    validation_data=(X_valid, X_valid))
    return Autoencoder

# these are for step two:
'''
predictions = Autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_sr = pd.Series(mse)
error_sr.to_csv('C:\\Users\\Cora\\addingnoise-and-evaluatingimpu.csv')
#error_sr.describe()                          

predictions1 = Autoencoder.predict(sample_test)
mse1 = np.mean(np.power(sample_test - predictions1, 2), axis=1)
error_sr1 = pd.Series(mse1)
error_sr1.to_csv('C:\\Users\\Cora\\sample-testing.csv')

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validaate'], loc='upper right');
plt.show()
'''
# these are for step three:
ndf = pd.read_csv('C:\\Users\\Cora\\N.csv')
dataset = np.asarray(ndf)
x_train, x_test = train_test_split(dataset, test_size=0.2, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0).fit(x_train)
cluster_dict = {i: x_train[np.where(kmeans.labels_ == i)] for i in range(kmeans.n_clusters)}
centers = kmeans.cluster_centers_

# training with different clusters:
error = {}
for i in range(kmeans.n_clusters):
    Autoencoder = autoencoder(X_train,X_test)
    x_train, x_valid = train_test_split(cluster_dict[i], test_size=0.2, random_state=0)
    Autoencoder.fit(x_train, x_train,
                    epochs=3,
                    batch_size=56,
                    shuffle=True,
                    validation_data=(x_valid, x_valid))
    c = centers[i] #get the center of that cluster
    Tree = spatial.KDTree(x_test)
    t = x_test[Tree.query(c)[1]] #get the value that is closest to the cluster in a testing dataset
    predict = Autoencoder(t)
    err = (t - predict) ** 2
    error[i] = err
    print(err)
    
print (error)

    

