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
ORG_PATH = 'C:\\Users\\A\\Desktop\\code'
### Summary
## change directory
os.chdir(DATA_PATH)

### list_path
kw_input_list = os.listdir('vibration')
kw_list = [re.sub('kW','',x) for x in kw_input_list]
type = 'vibration'

for volume in tqdm.tqdm(kw_input_list,total=len(kw_input_list)):
    kw = re.sub('kW','',volume)
    machine_list = os.listdir(f'{type}\{volume}')
    for machine_nm in machine_list:
        sp_len = 0.13
        machine = machine_nm
        type = 'vibration'
        state_list = os.listdir(f'{type}\{volume}\{machine}')
        if len(state_list)==1:
               if kw=='7.5' and machine=='L-SF-01':
                      sp_len = 0.00008
                      
               for state in state_list:
                      path ,file_names = detect_file_name(type, kw, machine, state)
                      file = file_names[0]
                      data0 = load_vibration_data(path,file)
                      peak_200 = count_peak(data0['vibration'],200)
                      col='red'
                      if state == '����': col='black'
                      plt.figure(figsize=(15,7))
                      plt.ylim([-sp_len,sp_len])                        
                      plt.plot(peak_200['max'],color=col,label=f'{type}_{kw}_{machine}_{state}_max')
                      plt.plot(peak_200['min'],color=col,label=f'{type}_{kw}_{machine}_{state}_min')
                      plt.legend()
                      plt.title(f'{type}_{kw}_{machine}_{state}')
                      os.chdir(ORG_PATH)
                      plt.savefig(f'plot/vibration_peak_plot/{type}_{kw}_{machine}_{state}.jpg',dpi=300)
                      os.chdir(DATA_PATH)
      
        if len(state_list)==2:
               path ,file_names = detect_file_name(type, kw, machine, state_list[0])
               file0 = file_names[0]
               data0 = load_vibration_data(path,file0)
               peak_200_0 = count_peak(data0['vibration'],200) 
               
               path ,file_names = detect_file_name(type, kw, machine, state_list[1])
               file1 = file_names[1]
               data1 = load_vibration_data(path,file1)
               peak_200_1 = count_peak(data1['vibration'],200) 
               plt.figure(figsize=(15,7))
               plt.ylim([-0.13,0.13])                        
               plt.plot(peak_200_0['max'],color='black',label=f'{type}_{kw}_{machine}_{state_list[0]}_max')
               plt.plot(peak_200_0['min'],color='black',label=f'{type}_{kw}_{machine}_{state_list[0]}_min')
               plt.plot(peak_200_1['max'],color='r',label=f'{type}_{kw}_{machine}_{state_list[1]}_max')
               plt.plot(peak_200_1['min'],color='r',label=f'{type}_{kw}_{machine}_{state_list[1]}_min')
               plt.legend()
               plt.title(f'{type}_{kw}_{machine}_{state_list[1]}')
               os.chdir(ORG_PATH)
               plt.savefig(f'plot/vibration_peak_plot/{type}_{kw}_{machine}_{state}.jpg',dpi=300)
               os.chdir(DATA_PATH)


        if len(state_list)==3:
               path ,file_names = detect_file_name(type, kw, machine, state_list[0])
               file0 = file_names[0]
               data0 = load_vibration_data(path,file0)
               peak_200_0 = count_peak(data0['vibration'],200) 
               
               path ,file_names = detect_file_name(type, kw, machine, state_list[1])
               file1 = file_names[1]
               data1 = load_vibration_data(path,file1)
               peak_200_1 = count_peak(data1['vibration'],200) 

               path ,file_names = detect_file_name(type, kw, machine, state_list[2])
               file2 = file_names[2]
               data2 = load_vibration_data(path,file2)
               peak_200_2 = count_peak(data2['vibration'],200) 
                       
               plt.figure(figsize=(15,7))
               plt.ylim([-0.13,0.13])                        
               plt.plot(peak_200_0['max'],color='black',label=f'{type}_{kw}_{machine}_{state_list[0]}_max')
               plt.plot(peak_200_0['min'],color='black',label=f'{type}_{kw}_{machine}_{state_list[0]}_min')
               plt.plot(peak_200_1['max'],color='r',label=f'{type}_{kw}_{machine}_{state_list[1]}_max')
               plt.plot(peak_200_1['min'],color='r',label=f'{type}_{kw}_{machine}_{state_list[1]}_min')
               plt.plot(peak_200_2['max'],color='g',label=f'{type}_{kw}_{machine}_{state_list[2]}_max')
               plt.plot(peak_200_2['min'],color='g',label=f'{type}_{kw}_{machine}_{state_list[2]}_min')
               plt.legend()
               plt.title(f'{type}_{kw}_{machine}_{state_list[1]}')
               os.chdir(ORG_PATH)
               plt.savefig(f'plot/vibration_peak_plot/{type}_{kw}_{machine}_{state}.jpg',dpi=300)
               os.chdir(DATA_PATH)




path ,file_names = detect_file_name('vibration', '7.5', 'L-SF-01', '����')
file1 = file_names[1]
data1 = load_vibration_data(path,file1)
peak_200 = count_peak(data1['vibration'],200) 
plt.figure(figsize=(15,7))
plt.ylim([-0.13,0.13])                        
plt.plot(peak_200['max'],color='black',label=f'{type}_{kw}_{machine}_����_max')
plt.plot(peak_200['min'],color='black',label=f'{type}_{kw}_{machine}_����_min')
plt.legend()
plt.show()
plt.title(f'{type}_{kw}_{machine}_{state_list[1]}')
os.chdir(ORG_PATH)
plt.savefig(f'plot/vibration_peak_plot/{type}_{kw}_{machine}_{state}.jpg',dpi=300)
os.chdir(DATA_PATH)

##################################################################### Peak Num

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