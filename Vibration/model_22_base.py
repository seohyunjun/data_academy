import os
from sys import last_value
# Select DIR
#os.getcwd()
from Func_V2 import *
import pandas as pd

import numpy as np

import matplotlib.font_manager as fm
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'NanumSquare_ac'

## 
## �����

#OR_PATH = 'C:\\Users\\admin\\Desktop\\code'
#DATA_PATH = 'D:\\���ü��� ���� ���� ����\\Training'
#VALIDATION_PATH = 'D:\\���ü��� ���� ���� ����\\Validation'
## ȸ��
DATA_PATH = 'D:\\project\\���ü��� ���� ���� ����\\Training'
OR_PATH = 'C:\\Users\\A\\Desktop\\code'
VALIDATION_PATH = 'D:\\project\\���ü��� ���� ���� ����\\Validation'

os.chdir(DATA_PATH)
type = 'vibration'
kw = '7.5'
machine = 'R-PAC-01S'
state = '����'
ab_state = '�����ĺҷ�'


# ���� : 33552
# ȸ��ü������ : 16336
normal_peak, abnormal_peak = total_peak_load(type,kw,machine,state,ab_state,200)

# reshape
np_normal_peak = np.array(normal_peak).reshape(len(normal_peak)//60,60)
np_abnormal_peak = np.array(abnormal_peak).reshape(len(abnormal_peak)//60,60)

# savgol_filter
from scipy.signal import savgol_filter
#plt.plot(np_normal_peak[2],color='b')
#plt.plot(np_normal_peak[-2],color='b')
#plt.plot(savgol_filter(np_normal_peak[2],55,3),color='b')
#plt.plot(savgol_filter(np_normal_peak[-1],55,3),color='b')

#plt.plot(np_abnormal_peak[-2],color='r')
#plt.plot(np_abnormal_peak[2],color='r')
#plt.plot(savgol_filter(np_abnormal_peak[0],55,3),color='r')
#plt.plot(savgol_filter(np_abnormal_peak[-2],55,3),color='r')

#plt.show()
#plt.plot(np_abnormal_peak[0])
#plt.show()
np_normal_peak = savgol_filter(np_normal_peak,55,3)
np_abnormal_peak = savgol_filter(np_abnormal_peak,55,3)

from sklearn.preprocessing import MinMaxScaler
#from sklearn.externals import joblib
import joblib
import seaborn as sns
sns.set(color_codes=True)

X_train = np_normal_peak
X_test = np_abnormal_peak

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)

#reshape
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(np_abnormal_peak.shape[0],1,np_abnormal_peak.shape[1])


X_train = X_train[:8000]
X_test = X_test[:8000]
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
batch_size = 30

history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history


os.chdir(OR_PATH)
model.save(f'model/{type}_{kw}_{machine}_{ab_state}_Scale Min_Max.h5')
plt.figure(figsize=(15,10))
plt.plot(history['loss'],label='loss')
plt.legend()
plt.title(f'{type}_{kw}_{machine}_{ab_state} Model Loss Plot Scale Min Max')
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{ab_state}_Model_Loss_Plot_Scale_Min_Max.jpg',dpi=300)
#plt.show()
plt.close()
os.chdir(DATA_PATH)

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred)
#X_pred.index = train.index

scored = pd.DataFrame(index=X_pred.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)


threshold = np.percentile(scored['Loss_mae'],99)
os.chdir(OR_PATH)
plt.figure(figsize=(16,9), dpi=80)
plt.title(f'{type} {kw} {machine} {ab_state} Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue')
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{ab_state}_Loss_Distribution_ScaleMinMax.jpg',dpi=300)
#plt.xlim([0.007,0.008])
#plt.show()
plt.close()

plt.figure(figsize=(16,9), dpi=80)
plt.title(f'{type} {kw} {machine} {ab_state} Loss Scatter Plot', fontsize=16)
plt.scatter(range(len(scored)),scored['Loss_mae'])
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{ab_state}_Loss_Scatter_ScaleMinMax.jpg',dpi=300)
#plt.show()
plt.close()
os.chdir(DATA_PATH)

## ���� Plot
#line_train_test = X_train[0].reshape(1,1,60)
#line_test = model.predict(line_train_test)
#plt.plot(line_test.reshape(-1))
#plt.show()

# calculate the loss on the test set
test_X_pred = model.predict(X_test)
test_X_pred = test_X_pred.reshape(test_X_pred.shape[0], test_X_pred.shape[2])
test_X_pred = pd.DataFrame(test_X_pred)
#X_pred.index = test.index

test_scored = pd.DataFrame(index=test_X_pred.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
test_scored['Loss_mae'] = np.mean(np.abs(test_X_pred-Xtest), axis = 1)
test_scored['Threshold'] = np.percentile(scored['Loss_mae'],99)

#min(test_scored['Loss_mae'])

# Normal = 1 Anomaly = 0
test_scored['Anomaly'] = test_scored['Loss_mae'].apply(lambda x: 0 if x > np.percentile(scored['Loss_mae'],99) else 1)

training_accuracy = (len(test_scored)-sum(test_scored['Anomaly']))/len(test_scored) # 100%

# 0: Pred_Anomaly 1: Pred_Normal 
os.chdir(OR_PATH)
plt.figure(figsize=(15,9))
plt.title(f'{type}_{kw}_{machine}_{state}_{ab_state}_MAE_Scatter')
plt.scatter(range(len(scored)),scored['Loss_mae'],label=f'{state}')
plt.scatter(range(len(test_scored)),test_scored['Loss_mae'],label=f'{ab_state}')
plt.savefig(f'model_plot/{type}_{kw}_{machine}_{state}_{ab_state}_MAE_Scatter_ScaleMinMax.jpg',dpi=300)
plt.legend()
#plt.show()
plt.close()
os.chdir(DATA_PATH)



############ Validation #####################################################################

os.chdir(VALIDATION_PATH)

## ���� : 4195 VS ������ : 2043
val_normal_peak, val_abnormal_peak = total_peak_load(type,kw,machine,state,ab_state,200)

val_normal = val_normal_peak
val_abnormal = val_abnormal_peak

np_val_normal = np.array(val_normal).reshape(len(val_normal)//60,60)
np_val_abnormal = np.array(val_abnormal).reshape(len(val_abnormal)//60,60)
np_val_normal = savgol_filter(np_val_normal,55,3)
np_val_abnormal = savgol_filter(np_val_abnormal,55,3)

#reshape
val_normal = np_val_normal.reshape(np_val_normal.shape[0],1,np_val_normal.shape[1])
val_abnormal = np_val_abnormal.reshape(np_val_abnormal.shape[0],1,np_val_abnormal.shape[1])

print("Validataion Normal Data shape:", val_normal.shape)
print("Validataion Abnormal Data shape:", val_abnormal.shape)

normal = 1
abnormal = 0

pd_val_normal = pd.DataFrame(np_val_normal)
pd_val_abnormal = pd.DataFrame(np_val_abnormal)

pd_val_normal['label'] = np.ones(len(np_val_normal))
pd_val_abnormal['label'] = np.zeros(len(np_val_abnormal))

Validation = pd.concat([pd_val_normal,pd_val_abnormal])

# Shuffle Data
Validation = Validation.sample(frac=1)  

# Model Load
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
sns.set(color_codes=True)

os.chdir(OR_PATH)
model = tf.keras.models.load_model(f'model/{type}_{kw}_{machine}_{ab_state}_Scale_Min_Max(max).h5')
os.chdir(VALIDATION_PATH)

input_val = np.array(Validation.iloc[:,:-1]).reshape(len(Validation),60)

scaler = MinMaxScaler()
input_val = scaler.fit_transform(input_val)
input_val = input_val.reshape(len(input_val),1,60)

#.03
val_predict = model.predict(input_val)
val_predict = val_predict.reshape(val_predict.shape[0],val_predict.shape[2])
input_val = input_val.reshape(input_val.shape[0],input_val.shape[2])
val_scored = pd.DataFrame()
val_scored['Loss_mae'] = np.mean(np.abs(input_val-val_predict), axis=1)
val_scored['label'] = Validation['label'].values

plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(val_scored['Loss_mae'], bins = 20, kde= True, color = 'blue')
#plt.show()
plt.close()

val_scored['pred'] = val_scored['Loss_mae'].apply(lambda x: 0 if x>threshold else 1)

plt.scatter(range(len(val_scored)),val_scored['label'],label='Actual')
plt.scatter(range(len(val_scored)),val_scored['pred'],label='Pred')
#plt.show()
plt.close()

val_accuracy= sum(val_scored['pred']==val_scored['label'])/len(val_scored) ## 

#Validation confusion matrix
from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_fscore_support
val_scored['label'] = val_scored['label'].apply(lambda x: int(x))
confusion_matrix_ = confusion_matrix(val_scored['label'],val_scored['pred'])
precision_recall_fscore_support_ = precision_recall_fscore_support(val_scored['label'],val_scored['pred'])
os.chdir(OR_PATH)

temp = {
    'type':type,
    'kw':kw,
    'machine':machine,
    'state':state,
    'ab_state':ab_state,
    'training_accuracy':training_accuracy,
    'training_threshold':threshold,
    'validataion_accuracy':val_accuracy,
    'precision':precision_recall_fscore_support_[0][0],'recall':precision_recall_fscore_support_[1][0],'fscore':precision_recall_fscore_support_[2][0],'support':precision_recall_fscore_support_[3][0],
    'confusion_matrix_00':confusion_matrix_[0][0],
    'confusion_matrix_01':confusion_matrix_[0][1],  'confusion_matrix_10':confusion_matrix_[1][0],
    'confusion_matrix_11':confusion_matrix_[1][1]
    }
result = pd.DataFrame(temp,index=[0])

result.to_csv(f'result/{type}_{kw}_{machine}_{ab_state}_result.csv',encoding='cp949',index=False)
