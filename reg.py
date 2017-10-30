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
y_train = 0
y_test = 0

if readFromnpy:
	x_train = np.load('multitask_train.csv.npy')
	x_test = np.load('multitask_test.csv.npy')
	#y_train = np.load(dataPath + '/data/train_y.csv.npy')
	#y_test = np.load(dataPath + '/data/test_y.csv.npy')


if readFromDf:
	x_train = pd.read_csv(dataPath + './data/intermediate_train_output.csv.npy')
	x_test = pd.read_csv(dataPath + './data/intermediate_test_output.csv.npy')
	y_train = pd.read_csv(dataPath + './data/intermediate_train_y.csv.npy')
	y_test = pd.read_csv(dataPath + './data/intermediate_test_y.csv.npy')


print x_train.shape
print x_test.shape



df = pd.read_csv('/Users/royal/Desktop/VADModels/VADModelData/EmobankTrain.tsv', sep='\t')

y_train_val = df['Valence'].tolist() 
y_train_aro = df['Arousal'].tolist() 
y_train_dom = df['Dominance'].tolist() 


y_train_all = df[['Valence','Arousal','Dominance']].as_matrix()


df = pd.read_csv('/Users/royal/Desktop/VADModels/VADModelData/EmobankTest.tsv', sep='\t')

y_val = df['Valence'].tolist() 
y_aro = df['Arousal'].tolist() 
y_dom = df['Dominance'].tolist() 

y_all = df[['Valence','Arousal','Dominance']].as_matrix()

regs = [linear_model.Ridge()]	#, DecisionTreeRegressor(), SVR()]


#regs = [neural_network.MLPRegressor(hidden_layer_sizes=x) for x in [10,100,200,500]]
import scipy.stats

for reg in regs:
  print reg
  reg.fit(x_train,y_train_val)
  y_pred = reg.predict(x_test)
  print 'Valence Corr : ',scipy.stats.pearsonr(y_val,y_pred)


for reg in regs:
  print reg
  reg.fit(x_train,y_train_aro)
  y_pred = reg.predict(x_test)
  print 'Arousal Corr: ',scipy.stats.pearsonr(y_aro,y_pred)


for reg in regs:
  print reg
  reg.fit(x_train,y_train_dom)
  y_pred = reg.predict(x_test)
  print 'Dominance Corr: ',scipy.stats.pearsonr(y_dom,y_pred)

