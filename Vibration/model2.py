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
kw = '37'
machine = 'L-PAHU-02S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)
file = file_names[0]
normal = load_vibration_data(path,file)


state = '축정렬불량'
path,file_names = detect_file_name(type, kw, machine, state)
ab_file = file_names[0]
abnormal = load_vibration_data(path,ab_file)


plt.plot(normal['vibration'])
plt.plot(abnormal['vibration'])
plt.show()

count_peak(normal['vibration'],200)
