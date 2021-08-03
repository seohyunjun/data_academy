import os
os.getcwd()
# Select DIR
from Func_V2 import *
import pandas as pd
import re

## �����
#DATA_PATH = 'D:\\���ü��� ���� ���� ����\\Training'

## ȸ��
DATA_PATH = 'D:\\project\\���ü��� ���� ���� ����\\Training'
ORG_PATH = 'C:\\Users\\A\\code'
### Summary
## change directory
os.chdir(DATA_PATH)

### list_path
kw_input_list = os.listdir('vibration')
kw_list = [re.sub('kW','',x) for x in kw]
type = 'vibration'

for volume in tqdm.tqdm(kw_input_list,total=len(kw_input_list)):
    kw = re.sub('kW','',volume)
    machine_list = os.listdir(f'{type}\{volume}')
    for machine_nm in machine_list:
        machine = machine_nm
        type = 'vibration'
        state_list = os.listdir(f'{type}\{volume}\{machine}')

        if len(state_list)==1:
            for state in state_list:
                    path,file_names = load_vibration_data(type, kw, machine, state)
                    #file_name = os.listdir(DATA_PATH)
                    file = file_names[0]
                    data0 = load_vibration_data(path,file)
                    

        

state = '����'
path,file_names = load_vibration_data(type, kw, machine, state)
#file_name = os.listdir(DATA_PATH)
file = file_names[0]

#####################################################################

type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '����'
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
state = '��Ʈ������'
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