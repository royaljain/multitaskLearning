from sklearn import linear_model
from sklearn import neural_network
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import pandas as pd
import re


dataPath = "/Users/royal/Desktop/VADModels/"

readFromnpy = True
readFromDf = False

x_train = 0
x_test = 0

x_train = np.load(dataPath + 'deepmoji_features_train_encoding.csv.npy')
x_test = np.load(dataPath + 'deepmoji_features_test_encoding.csv.npy')

print x_train.shape
print x_test.shape



df = pd.read_csv(dataPath+'VADModelData/EmobankTrain.tsv', sep='\t')

y_train_val = df['Valence'].values
y_train_aro = df['Arousal'].values
y_train_dom = df['Dominance'].values


y_train_all = df[['Valence','Arousal','Dominance']].as_matrix()


from keras.models import Model 
from keras.layers import Input, Dense, Activation


model_input  = Input(shape=(2304,))
shared_layer = Dense(2304, input_shape=(2304,))(model_input)
shared_layer_relu = Activation('relu', name='shared_layer')(shared_layer)

layer1 = Dense(2304, activation='relu')(shared_layer_relu)
output1 = Dense(1)(layer1)

layer2 = Dense(2304, activation='relu')(shared_layer_relu)
output2 = Dense(1)(layer2)


#layer3 = Dense(2304, activation='relu')(shared_layer_relu)
#output3 = Dense(1)(layer3)


model = Model(inputs=[model_input], outputs = [output1,output2])

model.compile(optimizer='rmsprop', loss='mean_squared_error',
              loss_weights=[1., 1.])

model.fit([x_train],[y_train_val,y_train_dom], epochs=2, batch_size=32)


intermediate_layer_model = Model(inputs=model.inputs, outputs = [model.get_layer(name='shared_layer').output])


x_train_new = intermediate_layer_model.predict(x_train)
x_test_new = intermediate_layer_model.predict(x_test)


import numpy as np

np.save('multitask_train.csv', x_train_new)
np.save('multitask_test.csv',x_test_new)






