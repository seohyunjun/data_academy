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

### Summary

## change directory
os.chdir(DATA_PATH)

type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)

np_normal = []
for file in file_names:
    temp = load_vibration_data(path,file)
    value = np.array(temp['vibration'].values)

    np_normal.append(value)

np_normal = np.array(np_normal)

type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
ab_state = '벨트느슨함'
path,file_names = detect_file_name(type, kw, machine, ab_state)

ab_normal = []
for file in file_names:
    temp = load_vibration_data(path,file)
    value = np.array(temp['vibration'].values)

    ab_normal.append(value)

ab_normal = np.array(ab_normal)

np_peak_0 = count_peak(np_normal[0],100)
ab_peak_0 = count_peak(ab_normal[0],100)

plt.figure(figsize=(15,7))
plt.plot(np_peak_0,label='normal')
plt.plot(ab_peak_0,label='abnormal')
plt.legend()
plt.title('peak by 100')
os.chdir(OR_PATH)
plt.savefig('plot/peak_by_100.jpg',dpi=300)
plt.show()
os.chdir(DATA_PATH)
