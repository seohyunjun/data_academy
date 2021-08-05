# -*- coding: utf-8 -*-
import os
# Select DIR
from Func_V2 import *
import pandas as pd
import re
import tqdm

## 자취방
#DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'
#ORG_PATH = 'C:\\Users\\admin\\Desktop\\code'
## 회사
DATA_PATH = 'D:\\project\\기계시설물 고장 예지 센서\\Training'
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
                      plt.figure(figsize=(15,7))
                      plt.ylim([-sp_len,sp_len])
                      
                      for i in range(10):     
                        file = file_names[i]
                        data0 = load_vibration_data(path,file)
                        peak_200 = count_peak(data0['vibration'],200)
                        if i==0:
                               plt.plot(range(60),peak_200['max'],color=judge_color(state),label=f'{type}_{kw}_{machine}_{state}_max')
                               plt.plot(range(60),peak_200['min'],color=judge_color(state),label=f'{type}_{kw}_{machine}_{state}_min')                          
                        plt.plot(range(60),peak_200['max'],color=judge_color(state))
                        plt.plot(range(60),peak_200['min'],color=judge_color(state))
                      plt.legend()
                      plt.title(f'{type}_{kw}_{machine}_{state}')
                      os.chdir(ORG_PATH)
                      plt.savefig(f'plot/vibration_peak_plot3/{type}_{kw}_{machine}_{state}.jpg',dpi=300)
                      os.chdir(DATA_PATH)
      
        if len(state_list)==2:
               path ,file_names = detect_file_name(type, kw, machine, state_list[0])
               plt.figure(figsize=(15,7))
               plt.ylim([-0.13,0.13])                        
               for i in range(10):     
                  file = file_names[i]
                  data0 = load_vibration_data(path,file)
                  peak_200_0 = count_peak(data0['vibration'],200)
                  if i==0 :
                         plt.plot(range(60),peak_200_0['max'],color=judge_color(state_list[0]),label=f'{type}_{kw}_{machine}_{state_list[0]}_max')
                         plt.plot(range(60),peak_200_0['min'],color=judge_color(state_list[0]),label=f'{type}_{kw}_{machine}_{state_list[0]}_min')
                         continue
                  plt.plot(range(60),peak_200_0['max'],color=judge_color(state_list[0]))
                  plt.plot(range(60),peak_200_0['min'],color=judge_color(state_list[0]))
               
               path ,file_names = detect_file_name(type, kw, machine, state_list[1])
               for i in range(10):     
                  file1 = file_names[i]
                  data1 = load_vibration_data(path,file1)
                  peak_200_1 = count_peak(data1['vibration'],200) 
                  
                  if i==0 :
                         plt.plot(range(60),peak_200_1['max'],color=judge_color(state_list[1]),label=f'{type}_{kw}_{machine}_{state_list[1]}_max')
                         plt.plot(range(60),peak_200_1['min'],color=judge_color(state_list[1]),label=f'{type}_{kw}_{machine}_{state_list[1]}_min')
                         continue
                  plt.plot(range(60),peak_200_1['max'],color=judge_color(state_list[1]))
                  plt.plot(range(60),peak_200_1['min'],color=judge_color(state_list[1]))
               plt.legend()
               plt.title(f'{type}_{kw}_{machine}_{state_list[1]}')
               os.chdir(ORG_PATH)
               plt.savefig(f'plot/vibration_peak_plot3/{type}_{kw}_{machine}_{state_list[1]}.jpg',dpi=300)
               os.chdir(DATA_PATH)


        if len(state_list)==3:
               path ,file_names = detect_file_name(type, kw, machine, state_list[0])
               plt.figure(figsize=(15,7))
               plt.ylim([-0.13,0.13])                        
               for i in range(10):     
                  file = file_names[i]
                  data0 = load_vibration_data(path,file)
                  peak_200_0 = count_peak(data0['vibration'],200)
                  if i==0 :
                         plt.plot(range(60),peak_200_0['max'],color=judge_color(state_list[0]),label=f'{type}_{kw}_{machine}_{state_list[0]}_max')
                         plt.plot(range(60),peak_200_0['min'],color=judge_color(state_list[0]),label=f'{type}_{kw}_{machine}_{state_list[0]}_min')
                         continue
                  plt.plot(range(60),peak_200_0['max'],color=judge_color(state_list[0]))
                  plt.plot(range(60),peak_200_0['min'],color=judge_color(state_list[0]))
               
               path ,file_names = detect_file_name(type, kw, machine, state_list[1])
               for i in range(10):     
                  file = file_names[i]
                  data1 = load_vibration_data(path,file)
                  peak_200_1 = count_peak(data1['vibration'],200)
                  if i==0 :
                         plt.plot(range(60),peak_200_1['max'],color=judge_color(state_list[1]),label=f'{type}_{kw}_{machine}_{state_list[1]}_max')
                         plt.plot(range(60),peak_200_1['min'],color=judge_color(state_list[1]),label=f'{type}_{kw}_{machine}_{state_list[1]}_min')
                         continue
                  plt.plot(range(60),peak_200_1['max'],color=judge_color(state_list[1]))
                  plt.plot(range(60),peak_200_1['min'],color=judge_color(state_list[1]))
               
               path ,file_names = detect_file_name(type, kw, machine, state_list[2])
               for i in range(10):     
                  file = file_names[i]
                  data2 = load_vibration_data(path,file)
                  peak_200_2 = count_peak(data2['vibration'],200)
                  if i==0 :
                         plt.plot(range(60),peak_200_2['max'],color=judge_color(state_list[2]),label=f'{type}_{kw}_{machine}_{state_list[2]}_max')
                         plt.plot(range(60),peak_200_2['min'],color=judge_color(state_list[2]),label=f'{type}_{kw}_{machine}_{state_list[2]}_min')
                         continue
                  plt.plot(range(60),peak_200_2['max'],color=judge_color(state_list[2]))
                  plt.plot(range(60),peak_200_2['min'],color=judge_color(state_list[2]))
               plt.legend()
               plt.title(f'{type}_{kw}_{machine}_{state_list[1]}')
               os.chdir(ORG_PATH)
               plt.savefig(f'plot/vibration_peak_plot3/{type}_{kw}_{machine}_{state_list[1]}.jpg',dpi=300)
               plt.close()
               os.chdir(DATA_PATH)
               




path ,file_names = detect_file_name('vibration', '7.5', 'L-SF-01', '정상')
file1 = file_names[1]
data1 = load_vibration_data(path,file1)
peak_200 = count_peak(data1['vibration'],200) 
plt.figure(figsize=(15,7))
plt.ylim([-0.13,0.13])                        
plt.plot(peak_200['max'],color='black',label=f'{type}_{kw}_{machine}_정상_max')
plt.plot(peak_200['min'],color='black',label=f'{type}_{kw}_{machine}_정상_min')
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