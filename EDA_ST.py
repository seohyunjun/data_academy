import pandas as pd

from Func import *
### EDA FORM


type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path_0_0,file_names_0_0 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_0 = file_names_0_0[0]
R_CAHU_03S_15_00 = load_current_data(path_0_0, file_name_0_0)

total = total_current_info(path_0_0,file_names_0_0,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)



state = '벨트느슨함'
path_0_4,file_names_0_4 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_4 = file_names_0_4[0]
R_CAHU_03S_15_04 = load_current_data(path_0_4, file_name_0_4)

total = total_current_info(path_0_4,file_names_0_4,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)

####################################################################################
type = 'current'
kw = '18.5'
machine = 'R-CAHU-02S'
state = '정상'
path_1_0,file_names_1_0 = detect_file_name(type, kw, machine, state)
### 298 : 베어링불량 299 : 정상
file_name_1_0 = file_names_1_0[0]
R_CAHU_02S_18_10 = load_current_data(path_1_0, file_name_1_0)

total = total_current_info(path_1_0,file_names_1_0,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)


state = '베어링불량'
path_1_1,file_names_1_1 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_1_1 = file_names_1_1[0]
R_CAHU_02S_18_11 = load_current_data(path_1_1, file_name_1_1)

total = total_current_info(path_1_1,file_names_1_1,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)


#####################################################################################
type = 'current'
kw = '22'
machine = 'L-CAHU-02S'
state = '정상'
path_2_0,file_names_2_0 = detect_file_name(type, kw, machine, state)
### 298 : 베어링불량 299 : 정상
file_name_2_0 = file_names_2_0[0]
L_CAHU_02S_18_20 = load_current_data(path_2_0, file_name_2_0)

total = total_current_info(path_2_0,file_names_2_0,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)



state = '회전체불평형'
path_2_2,file_names_2_2 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_2_2 = file_names_2_2[0]
L_CAHU_02S_18_22 = load_current_data(path_2_2, file_name_2_2)


total = total_current_info(path_2_2,file_names_2_2,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)

########################################################################################
type = 'current'
kw = '22'
machine = 'L-CAHU-01S'
state = '정상'
path_3_0,file_names_3_0 = detect_file_name(type, kw, machine, state)
### 298 : 베어링불량 299 : 정상
file_name_3_0 = file_names_3_0[0]
L_CAHU_01S_18_30 = load_current_data(path_3_0, file_name_3_0)

total = total_current_info(path_3_0,file_names_3_0,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)



state = '축정렬불량'
path_3_3,file_names_3_3 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_3_3 = file_names_3_3[0]
L_CAHU_01S_18_33 = load_current_data(path_3_3, file_name_3_3)

total = total_current_info(path_3_3,file_names_3_3,type,kw,machine,state)
total_decs = total.describe()
total_decs_mean = total_decs[1:2]
total_decs_mean.to_csv(f'{type}_{kw}_{machine}_{state}_summary.csv',index=False)



