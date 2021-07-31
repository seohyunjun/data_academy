import os
### current 정상 / 벨트느슨함


def detect_file_name(type, kw, machine, state):
    path = type+'/'+kw+'kW'+'/'+machine
    print(os.listdir(path))
    path0 = type+'/'+kw+'kW'+'/'+machine+'/'+state
    file_name = os.listdir(path0)
    return path0,file_name
########################################################
type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(type, kw, machine, state)
###
path
normal = [file for file in file_names if '20201128' in file]


#####################################################
type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '벨트느슨함'
path4,file_names4 = detect_file_name(type, kw, machine, state)
Anomal = [file for file in file_names4 if '20201128' in file]

# 정상 238 / 벨트느슨함 502

# temp 폴더 확인
type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
temp_path,temp_file_names = detect_file_name(type, kw, machine, state)
len(temp_file_names)

# 정상은 00 벨트느슨함은 04

for name in temp_file_names:
    src = os.path.join(temp_path, name)
    if name in normal:
        dst = name[:-4] + '_00.csv'
    if name in Anomal:
        dst = name[:-4] + '_04.csv'
    dst = os.path.join(temp_path, dst)
    os.rename(src, dst)

########################################
type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '벨트느슨함'
path_0,Anomal = detect_file_name(type, kw, machine, state)

state = '정상'
path_0,normal = detect_file_name(type, kw, machine, state)


type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
temp_path,temp_file_names = detect_file_name(type, kw, machine, state)
len(temp_file_names)

# 정상은 00 벨트느슨함은 04

for name in temp_file_names:
    src = os.path.join(temp_path, name)
    if name in normal:
        dst = name[:-4] + '_00.csv'
    if name in Anomal:
        dst = name[:-4] + '_04.csv'
    dst = os.path.join(temp_path, dst)
    os.rename(src, dst)


