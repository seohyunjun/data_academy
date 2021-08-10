import os
# Select DIR
#os.getcwd()
from Func_V2 import *
import pandas as pd

import numpy as np

## 
## 자취방

OR_PATH = 'C:\\Users\\admin\\Desktop\\code'
DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'

## 회사
#DATA_PATH = 'D:\\project\\기계시설물 고장 예지 센서\\Training'

os.chdir(DATA_PATH)
type = 'vibration'
kw = '2.2'
machine = 'L-EF-04'
state = '정상'
ab_state = '회전체불평형'

# 정상 : 33552
# 회전체불평형 : 16336
normal_peak, abnormal_peak = total_peak_load(type,kw,machine,state,ab_state,200)

# reshape
np_normal_peak = np.array(normal_peak).reshape(len(normal_peak)//60,60)
np_abnormal_peak = np.array(abnormal_peak).reshape(len(abnormal_peak)//60,60)

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)

X_train = np_normal_peak
X_test = np_abnormal_peak

#reshape
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(np_abnormal_peak.shape[0],1,np_abnormal_peak.shape[1])

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
# set random seed
seed(10)
tf.random.set_seed(10)


# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model


# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()

# fit the model to the data
nb_epochs = 100
batch_size = 10

history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history


os.chdir(OR_PATH)
model.save(f'model/{type}_{kw}_{machine}_{ab_state}.h5')
plt.figure(figsize=(15,10))
plt.plot(history['loss'],label='loss')
plt.legend()
plt.title(f'{type}_{kw}_{machine}_{ab_state} Model Loss Plot')
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{ab_state}_Model_Loss_Plot.jpg',dpi=300)
plt.show()
os.chdir(DATA_PATH)

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred)
#X_pred.index = train.index

scored = pd.DataFrame(index=X_pred.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)

os.chdir(OR_PATH)
plt.figure(figsize=(16,9), dpi=80)
plt.title(f'{type} {kw} {machine} {ab_state} Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue')
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{ab_state}_Loss_Distribution.jpg',dpi=300)
#plt.xlim([0.007,0.008])
plt.show()

plt.figure(figsize=(16,9), dpi=80)
plt.title(f'{type} {kw} {machine} {ab_state} Loss Scatter Plot', fontsize=16)
plt.scatter(range(len(scored)),scored['Loss_mae'])
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{ab_state}_Loss_Scatter.jpg',dpi=300)
plt.show()
os.chdir(DATA_PATH)

# calculate the loss on the test set
test_X_pred = model.predict(X_test)
test_X_pred = test_X_pred.reshape(test_X_pred.shape[0], test_X_pred.shape[2])
test_X_pred = pd.DataFrame(test_X_pred)
#X_pred.index = test.index

test_scored = pd.DataFrame(index=test_X_pred.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
test_scored['Loss_mae'] = np.mean(np.abs(test_X_pred-Xtest), axis = 1)
test_scored['Threshold'] = 0.04
Threshold1 = 0.01
Threshold2 = 0.03

min(test_scored['Loss_mae'])

test_scored['Anomaly'] = [0 if i>Threshold1 and i<Threshold2 else 1 for i in test_scored['Loss_mae']]
test_scored.head()
(len(test_scored)-sum(test_scored['Anomaly']))/len(test_scored)
# 0: Pred_Anomaly 1: Pred_Normal 
os.chdir(OR_PATH)
plt.figure(figsize=(15,9))
plt.title(f'{type}_{kw}_{machine}_{state}_{ab_state}_MAE_Scatter')
plt.scatter(range(len(scored)),scored['Loss_mae'],label=f'{state}')
plt.scatter(range(len(test_scored)),test_scored['Loss_mae'],label=f'{ab_state}')
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{state}_{ab_state}_MAE_Scatter.jpg',dpi=300)
plt.legend()
plt.show()
os.chdir(DATA_PATH)
