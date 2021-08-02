import os
# Select DIR
#os.getcwd()
from Func_V2 import *
import pandas as pd
## 

## 자취방
DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'

## 회사
#DATA_PATH = 'D:\\project\\기계시설물 고장 예지 센서\\Training'

from sklearn import feature_extraction
### Summary

os.chdir(DATA_PATH)
type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)
#file_name = os.listdir(DATA_PATH)
file = file_names[0]

total = pd.DataFrame()
for file in file_name:
   file_nm = os.path.join(Path,file)
   temp = pd.read_csv(file_nm,encoding='cp949')
   temp['file_name'] = file
   total = pd.concat([total,temp])

#total.to_csv('Validation_Current_By_Edasheet.csv',encoding='cp949',index=False)


import matplotlib.pyplot as plt
#plt.scatter(total['id'],total['x1_std_avg'],color=total['status'])
status_type = total.status.drop_duplicates().values
i=1

plt.figure(figsize=(15,10))
for status in status_type:
       temp = total[total['status']==status]
       i += 1
       plt.scatter(temp['id'],temp['x1_std_avg'],label=f'{status}')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [15, 6]
plt.rcParams['font.family'] = 'NanumSquare_ac'
## 전처리
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
## 기계 전처리
label = LabelEncoder()
total['machine'] = label.fit_transform(total['file_name'])
onehot = OneHotEncoder()
x = onehot.fit_transform(total['machine'].values.reshape(-1,1)).toarray()

machine_onehot_df = pd.DataFrame(x, columns=["machine_"+str(int(i)) for i in range(x.shape[1])])
total.reset_index(inplace=True)
machine_onehot_result = pd.concat([total,machine_onehot_df],axis=1)

## 상태 전처리
label = LabelEncoder()
total['machine_status'] = label.fit_transform(total['status'])
onehot = OneHotEncoder()
x = onehot.fit_transform(total['machine_status'].values.reshape(-1,1)).toarray()

machine_status_onehot_df = pd.DataFrame(x, columns=["status_"+str(int(i)) for i in range(x.shape[1])])
machine_status_onehot_result = pd.concat([machine_onehot_result,machine_status_onehot_df],axis=1)

# machine , machine_status remove
del machine_status_onehot_result['machine']

inputs = machine_status_onehot_result.iloc[:,4:]

inputs.to_csv('inputs.csv',encoding='cp949')

#del inputs['x1_std_avg']
#from sklearn.preprocessing import StandardScaler
#standardScaler = StandardScaler()
#print(standardScaler.fit(inputs))
#train_data_standardScaled = standardScaler.transform(inputs)

### PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(train_data_standardScaled)
#principalDf = pd.DataFrame(data = principalComponents, columns=['principal component1', 'principal component2'])

#### K-Means 비지도학습
from sklearn.cluster import KMeans
#############


##############
#K_means = KMeans()
#K_means.fit(principalDf)
#y_predict = K_means.predict(principalDf)

#total['cluster'] = y_predict

K_means = KMeans(n_clusters=5)
K_means.fit(inputs)
y_predict = K_means.predict(inputs)

total['cluster'] = y_predict


cluster = total.cluster.drop_duplicates().values
i=1
plt.figure(figsize=(15,10))

label0 = total[total['cluster']==0]
plt.scatter(label0['id'],label0['x1_std_avg'],label=f'label0')
label1 = total[total['cluster']==1]
plt.scatter(label1['id'],label1['x1_std_avg'],label=f'label1')
label2 = total[total['cluster']==2]
plt.scatter(label2['id'],label2['x1_std_avg'],label=f'label2')
label3 = total[total['cluster']==3]
plt.scatter(label3['id'],label3['x1_std_avg'],label=f'label3')
label4 = total[total['cluster']==4]
plt.scatter(label4['id'],label4['x1_std_avg'],label=f'label4')
plt.legend()
plt.show()
#plt.savefig('total_vibration_cluster(5).jpg',dpi=300)
label1.to_csv('label1.csv',index=False,encoding='cp949')
#####################################
total.to_csv('total_5.csv',encoding='cp949',index=False)
import pandas as pd
import os
### Summary
file_name = os.listdir('vibration_validation_summary')

file_name

Path = 'vibration_validation_summary'

file = file_name[0]

total = pd.DataFrame()
for file in file_name:
   file_nm = os.path.join(Path,file)
   temp = pd.read_csv(file_nm,encoding='cp949')
   temp['file_name'] = file
   total = pd.concat([total,temp])

total.to_csv('Validation_Current_By_Edasheet_Kmeans5.csv',encoding='cp949',index=False)