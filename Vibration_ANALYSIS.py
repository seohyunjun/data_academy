import os
# Select DIR
#os.getcwd()
from Func_V2 import *
import pandas as pd
## 
## 자취방

OR_PATH = 'C:\\Users\\admin\\Desktop\\code'
DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'

## 회사
#DATA_PATH = 'D:\\project\\기계시설물 고장 예지 센서\\Training'

### Summary

## change directory
os.chdir(DATA_PATH)
#####################################################################
type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)
#file_name = os.listdir(DATA_PATH)
#### Peak Value Extract
org = []
for file in file_names:
   temp = load_vibration_data(path,file)
   t = {
      'num':len(gen_vibration_pre_data(temp)[0]['vibration'])
   }
   org.append(t)
plt.figure(figsize=(15,10))
plt.plot(pd.DataFrame(org),color='green',label='Count')
plt.title(f"{kw}_{machine}_{state}_Peak_Value_Count")
plt.xlabel('x')
plt.ylabel('Peak Count')
plt.legend()
plt.show()
### Abnormal
type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '벨트느슨함'
path,file_names = detect_file_name(type, kw, machine, state)
#file_name = os.listdir(DATA_PATH)
#### Peak Value Extract
org = []
for file in file_names:
   temp = load_vibration_data(path,file)
   t = {
      'num':len(gen_vibration_pre_data(temp)[0]['vibration'])
   }
   org.append(t)

plt.figure(figsize=(15,10))
plt.plot(pd.DataFrame(org),color='green',label='Count')
plt.title(f"{kw}_{machine}_{state}_Peak_Value_Count")
plt.xlabel('x')
plt.ylabel('Peak Count')
plt.legend()
plt.show()
#############################################################       
       
type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)

file_name = file_names[0]
normal_vibration = load_vibration_data(path, file_name)
peak_normal_vibration0 = gen_vibration_pre_data(normal_vibration)

ab_state = '벨트느슨함'
path,file_names = detect_file_name(type, kw, machine, ab_state)
file_name = file_names[0]
abnormal_vibration = load_vibration_data(path, file_name)
peak_abnormal_vibration0 = gen_vibration_pre_data(abnormal_vibration)

plt.scatter(range(len(peak_normal_vibration0)),peak_normal_vibration0)
plt.show()


def gen_vibration_peak_by_len(data,select_len):

       len_range = 12000 // select_len
       peak_max = []
       peak_min = []
       
       for i in range(len_range):
              temp = data['vibration'][0+100*i:100+100*i]
              max_temp = max(temp) 
              min_temp = min(temp) 
              peak_max.append(max_temp)
              peak_min.append(min_temp)
              
       peak = pd.DataFrame()
       peak['max'] = peak_max
       peak['min'] = peak_min
       return peak


       
select_len = 100
len_range = 12000 // select_len
peak_max = []
peak_min = []

for i in range(len_range):
       temp = normal_vibration['vibration'][0+100*i:100+100*i]
       max_temp = max(temp) 
       min_temp = min(temp) 
       peak_max.append(max_temp)
       peak_min.append(min_temp)

peak = pd.DataFrame()
peak['max'] = peak_max
peak['min'] = peak_min

ab_peak = gen_vibration_peak_by_len(abnormal_vibration,100)

plt.figure(figsize=(15,8))
plt.plot(peak['max'],label='normal_peak_max',color='r')
plt.plot(peak['min'],label='normal_peak_min',color='r')
plt.plot(ab_peak['max'],label='abnormal_peak_max',color='b')
plt.plot(ab_peak['min'],label='abnormal_peak_min',color='b')
plt.legend()
plt.title(f'{ab_state} Peak n = {12000//100}')
os.chdir(OR_PATH)
plt.savefig(f'plot\\peak_by_{12000//100}.jpg',dpi=300)
plt.show()
os.chdir(DATA_PATH)

import seaborn as sns

sns.heatmap(peak['max'])