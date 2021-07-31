import os
import pandas as pd
import numpy as np
status = '축정렬불량' # 타입
device_id = 'L-CAHU-01S' # 기계명
import tqdm
#Current
# R-CAHU-03S   정상  55_0 벨트느슨함 56_4
# R-CAHU-02S   정상 57_0 베어링불량 58_1
# L-CAHU-02S   정상  61_0 회전체불평형 61_2
# L-CAHU-01S  정상 65_0 축정렬불량 66_33

DATA_IN_PATH = './전류/' + device_id + '/' + status + '/'
file_list = os.listdir(DATA_IN_PATH)

dataColumns = ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_std', 'x1_median', 'x1_per25', 'x1_per75',
               'x2_max', 'x2_min', 'x2_avg', 'x2_std', 'x2_median', 'x2_per25', 'x2_per75', 'x3_max', 'x3_min',
               'x3_avg', 'x3_std', 'x3_median', 'x3_per25', 'x3_per75']
my_df = pd.DataFrame(columns=dataColumns)
i = 0

for fileName in file_list:
    i = i + 1
    if i % 1000 == 0:
        print(i)
    train_data = pd.read_csv(DATA_IN_PATH + fileName, header=None, skiprows=9, delimiter='\t')
    df = pd.DataFrame(train_data[0].str.split(',', 4).tolist(), columns=['id', 'x1', 'x2', 'x3', 'ggg'])
    df = df.astype({'x1': 'float', 'x2': 'float', 'x3': 'float'})

    a = pd.DataFrame(data=[[fileName, device_id, status,
                            np.max(df['x1']), np.min(df['x1']), np.mean(df['x1']), np.std(df['x1']),
                            np.median(df['x1']), np.percentile(df['x1'], 25), np.percentile(df['x1'], 75),
                            np.max(df['x2']), np.min(df['x2']), np.mean(df['x2']), np.std(df['x2']),
                            np.median(df['x2']), np.percentile(df['x2'], 25), np.percentile(df['x2'], 75),
                            np.max(df['x3']), np.min(df['x3']), np.mean(df['x3']), np.std(df['x3']),
                            np.median(df['x3']), np.percentile(df['x3'], 25), np.percentile(df['x3'], 75)
                            ]], columns=dataColumns)
    my_df = my_df.append(a)


summary = my_df.mean()
summary_df = pd.DataFrame(summary)
summary_df = summary_df.transpose()

summary_df['file_name'] = my_df.iloc[0,0]
summary_df['id'] = my_df.iloc[0,1]
summary_df['status'] = my_df.iloc[0,2]

summary_df['x1_avg_std'] = my_df['x1_avg'].std()
summary_df['x2_avg_std'] = my_df['x2_avg'].std()
summary_df['x3_avg_std'] = my_df['x3_avg'].std()

summary_df = summary_df[['file_name','id','status','x1_max', 'x1_min', 'x1_avg', 'x1_avg_std',  'x1_per25', 'x1_median',
       'x1_per75', 'x2_max', 'x2_min', 'x2_avg','x2_avg_std',
       'x2_per25','x2_median', 'x2_per75', 'x3_max', 'x3_min', 'x3_avg','x3_avg_std',
        'x3_per25', 'x3_median','x3_per75']]

#summary_df.columns = ['file_name','id','status','x1_max', 'x1_min', 'x1_avg', 'x1_avg_std','x1_std_avg', 'x1_median', 'x1_per25',
#       'x1_per75', 'x2_max', 'x2_min', 'x2_avg','x2_avg_std', 'x2_std_avg', 'x2_median',
#       'x2_per25', 'x2_per75', 'x3_max', 'x3_min', 'x3_avg','x3_avg_std','x3_std_avg',
#       'x3_median', 'x3_per25', 'x3_per75']

summary_df.to_csv("./" + device_id + "_" + status + "_summary.csv", encoding='cp949',index=False)

###############################################################

import os
import pandas as pd
import numpy as np
status = '정상' # 타입
device_id = 'R-CAHU-03S' # 기계명
st = '53_0'
index = f'vibration_{st}'
# Vibration

# R-CAHU-03S   정상  53_0 벨트느슨함 54_4
# R-CAHU-02S   정상 55_0 베어링불량 56_1
# L-CAHU-02S   정상  59_0 회전체불평형 60_2
# L-CAHU-01S  정상 63_0 축정렬불량 64_3
info = pd.read_excel('vibration_info.xlsx')
for i in range(len(info)):
    device_id = info['machine'][i]
    status = info['state'][i]  # 타입
    st = info['index'][i]
    index = f'vibration_{st}'

    DATA_IN_PATH = './진동/' + device_id + '/' + status + '/'
    file_list = os.listdir(DATA_IN_PATH)

    dataColumns = ['file_name', 'id', 'status',
                   'x1_max', 'x1_min', 'x1_avg', 'x1_std', 'x1_median', 'x1_per25',  'x1_per75',
                   'x2_max', 'x2_min', 'x2_avg', 'x2_std', 'x2_median', 'x2_per25', 'x2_per75',
                   'x3_max', 'x3_min',  'x3_avg', 'x3_std', 'x3_median', 'x3_per25', 'x3_per75']
    my_df = pd.DataFrame(columns=dataColumns)
    i = 0
    for fileName in file_list:
        i = i + 1
        if i % 100 == 0:
            print(i)
        train_data = pd.read_csv(DATA_IN_PATH + fileName, header=None, skiprows=9, delimiter='\t')
        df = pd.DataFrame(train_data[0].str.split(',', 4).tolist(), columns=['id', 'x1', 'ggg'])
        df = df.astype({'x1': 'float'})

        a = pd.DataFrame(data=[[fileName, device_id, status,
                                np.max(df['x1']), np.min(df['x1']), np.mean(df['x1']), np.std(df['x1']),
                                np.median(df['x1']), np.percentile(df['x1'], 25), np.percentile(df['x1'], 75)]],
                         columns=dataColumns)
        my_df = my_df.append(a)
    my_df.to_csv(f"{index}.csv", encoding='cp949', index=False)

    summary = my_df.mean()
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.transpose()

    summary_df['file_name'] = my_df.iloc[0, 0]
    summary_df['id'] = my_df.iloc[0, 1]
    summary_df['status'] = my_df.iloc[0, 2]

    summary_df['x1_avg_std'] = my_df['x1_avg'].std()

    summary_df = summary_df[
        ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_avg_std', 'x1_per25','x1_median'
         'x1_per75']]

    #summary_df.columns = ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_avg_std', 'x1_std_avg',
     #                     'x1_median', 'x1_per25', 'x1_per75']

    # summary_df.to_csv("./" + device_id + "_" + status +f'_vibration'+ "_summary.csv", encoding='cp949',index=False)
    summary_df.to_csv(f"{index}_summary.csv", encoding='cp949', index=False)

#####################
import re
import pandas as pd
import re
import numpy as np
import os
current_info = pd.read_csv('current_info.csv',encoding='cp949')

for i in tqdm.tqdm(range(50,len(current_info)),total=len(range(50,len(current_info)))):
    device_id = current_info['machine'][i]
    status = current_info['state'][i]  # 타입
    st = str(current_info['index'][i]).zfill(2)
    kW = float(current_info['current'][i])
    state_num = current_info['state_num'][i]
    if kW % 1 == 0:
         kW = int(kW)
    kW = f'{kW}kW'
    index = f'current_{st}_{state_num}'

    DATA_IN_PATH = './기계시설물 고장 예지 센서/' + 'Training' + '/' + 'current' + '/' + kW + '/' + device_id + '/' + status + '/'
    file_list = os.listdir(DATA_IN_PATH)

    dataColumns = ['file_name', 'id', 'status',
                   'x1_max', 'x1_min', 'x1_avg', 'x1_std', 'x1_median', 'x1_per25','x1_per75',
                   'x2_max', 'x2_min', 'x2_avg', 'x2_std', 'x2_median', 'x2_per25', 'x2_per75',
                   'x3_max', 'x3_min', 'x3_avg', 'x3_std', 'x3_median', 'x3_per25', 'x3_per75']
    my_df = pd.DataFrame(columns=dataColumns)
    i = 0
    for fileName in file_list:
        i = i + 1
        if i % 1000 == 0:
            print(i)
        train_data = pd.read_csv(DATA_IN_PATH + fileName, header=None, skiprows=9, delimiter='\t')
        df = pd.DataFrame(train_data[0].str.split(',', 4).tolist(), columns=['id', 'x1', 'x2', 'x3', 'ggg'])
        df = df.astype({'x1': 'float', 'x2': 'float', 'x3': 'float'})

        a = pd.DataFrame(data=[[fileName, device_id, status,
                                np.max(df['x1']), np.min(df['x1']), np.mean(df['x1']), np.std(df['x1']),
                                np.median(df['x1']), np.percentile(df['x1'], 25), np.percentile(df['x1'], 75),
                                np.max(df['x2']), np.min(df['x2']), np.mean(df['x2']), np.std(df['x2']),
                                np.median(df['x2']), np.percentile(df['x2'], 25), np.percentile(df['x2'], 75),
                                np.max(df['x3']), np.min(df['x3']), np.mean(df['x3']), np.std(df['x3']),
                                np.median(df['x3']), np.percentile(df['x3'], 25), np.percentile(df['x3'], 75)
                                ]], columns=dataColumns)
        my_df = my_df.append(a)
    my_df.to_csv(f"{index}.csv", encoding='cp949', index=False)
    summary = my_df.mean()
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.transpose()

    summary_df['file_name'] = my_df.iloc[0, 0]
    summary_df['id'] = my_df.iloc[0, 1]
    summary_df['status'] = my_df.iloc[0, 2]

    summary_df['x1_avg_std'] = my_df['x1_avg'].std()
    summary_df['x2_avg_std'] = my_df['x2_avg'].std()
    summary_df['x3_avg_std'] = my_df['x3_avg'].std()

    summary_df = summary_df[['file_name', 'id', 'status',
                             'x1_max', 'x1_min', 'x1_avg', 'x1_avg_std','x1_std', 'x1_per25', 'x1_median','x1_per75',
                             'x2_max', 'x2_min', 'x2_avg', 'x2_avg_std','x2_std','x2_per25', 'x2_median', 'x2_per75',
                             'x3_max', 'x3_min', 'x3_avg', 'x3_avg_std','x3_std','x3_per25', 'x3_median', 'x3_per75']]

    summary_df.columns = ['file_name','id','status',
                          'x1_max', 'x1_min', 'x1_avg', 'x1_avg_std','x1_std_avg', 'x1_median', 'x1_per25','x1_per75',
                          'x2_max', 'x2_min', 'x2_avg','x2_avg_std', 'x2_std_avg', 'x2_median','x2_per25', 'x2_per75',
                          'x3_max', 'x3_min', 'x3_avg','x3_avg_std','x3_std_avg', 'x3_median', 'x3_per25', 'x3_per75']
    summary_df.to_csv(f"{index}_summary.csv", encoding='cp949', index=False)

################################### vibration
import re
import pandas as pd
import numpy as np
import os
import tqdm
#vibration_info = pd.read_csv('vibration_info1.csv',encoding='cp949')
#vibration_info['machine'] = vibration_info['machine'].apply(lambda x: re.split('_',x)[-1:][0])


#vibration_info['index'] = vibration_info['index'].apply(lambda x: str(x).zfill(2))

#vibration_info.to_csv('vibration_info1.csv',index=False,encoding='cp949')
vibration_info = pd.read_csv('vibration_info1.csv',encoding='cp949')
for i in tqdm.tqdm(range(len(vibration_info)),total=len(vibration_info)):
    device_id = vibration_info['machine'][i]
    status = vibration_info['state'][i]  # 타입
    st = str(vibration_info['index'][i]).zfill(2)
    kW = float(vibration_info['current'][i])
    state_index = vibration_info['state_index'][i]
    if kW % 1 == 0:
         kW = int(kW)
    kW = f'{kW}kW'
    index = f'vibration_{st}_{state_index}'

    DATA_IN_PATH = './기계시설물 고장 예지 센서/' + 'Training' + '/' + 'vibration' + '/' + kW + '/' + device_id + '/' + status + '/'
    file_list = os.listdir(DATA_IN_PATH)
    dataColumns = ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_std', 'x1_median', 'x1_per25','x1_per75']


    my_df = pd.DataFrame(columns=dataColumns)
    i = 0
    for fileName in file_list:
        i = i + 1
        if i % 1000 == 0:
            print(i)
        train_data = pd.read_csv(DATA_IN_PATH + fileName, header=None, skiprows=9, delimiter='\t')
        df = pd.DataFrame(train_data[0].str.split(',', 4).tolist(), columns=['id', 'x1', 'ggg'])
        df = df.astype({'x1': 'float'})

        a = pd.DataFrame(data=[[fileName, device_id, status,
                                np.max(df['x1']), np.min(df['x1']), np.mean(df['x1']), np.std(df['x1']),
                                np.median(df['x1']), np.percentile(df['x1'], 25), np.percentile(df['x1'], 75)]], columns=dataColumns)
        my_df = my_df.append(a)
    my_df.to_csv(f"{index}.csv", encoding='cp949', index=False)
    summary = my_df.mean()
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.transpose()

    summary_df['file_name'] = my_df.iloc[0, 0]
    summary_df['id'] = my_df.iloc[0, 1]
    summary_df['status'] = my_df.iloc[0, 2]

    summary_df['x1_avg_std'] = my_df['x1_avg'].std()

    summary_df = summary_df[
        ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_avg_std','x1_std', 'x1_per25', 'x1_median','x1_per75']]
    summary_df.columns = ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_avg_std','x1_std_avg', 'x1_per25', 'x1_median','x1_per75']
    summary_df.to_csv(f"{index}_summary.csv", encoding='cp949', index=False)