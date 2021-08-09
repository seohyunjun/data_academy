from Func_V2 import *
import pandas as pd

# 자취방
#OR_PATH = 'C:\\Users\\admin\\Desktop\\code'
#DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'

## 회사
OR_PATH = 'C:\\Users\\A\\Desktop\\code'
DATA_PATH = 'D:\\project\\기계시설물 고장 예지 센서\\Training'
os.chdir(DATA_PATH)
type = 'vibration'
kw = '2.2'
machine = 'L-EF-04'
state = '회전체불평형'


state_list = ['정상','회전체불평형']
temp_path,temp_file_names = detect_file_name(type, kw, machine, state_list[0])
for i in range(10):     
    file = temp_file_names[-i]
    data0 = load_vibration_data(temp_path,file)
    peak_200_0 = count_peak(data0['vibration'],200)
    if i==0 :
            plt.plot(range(60),peak_200_0['max'],color=judge_color(state_list[0]),label=f'{type}_{kw}_{machine}_{state_list[0]}_max')
            plt.plot(range(60),peak_200_0['min'],color=judge_color(state_list[0]),label=f'{type}_{kw}_{machine}_{state_list[0]}_min')
            continue
    plt.plot(range(60),peak_200_0['max'],color=judge_color(state_list[0]))
    plt.plot(range(60),peak_200_0['min'],color=judge_color(state_list[0]))
#plt.show()
path ,file_names = detect_file_name(type, kw, machine, state_list[1])
for i in range(10):     
    file1 = file_names[-i]
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
plt.show()

