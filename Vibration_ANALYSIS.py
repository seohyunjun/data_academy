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

## change directory
os.chdir(DATA_PATH)

type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)
#file_name = os.listdir(DATA_PATH)
file = file_names[0]

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
       
       
