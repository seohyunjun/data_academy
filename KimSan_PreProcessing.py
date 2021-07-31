import os
import pandas as pd

status = '베어링불량'
device_id = 'R-CAHU-02R'
# L-CAHU-01R   정상   축정렬불량   회전체불평형
# L-CAHU-01S   정상   회전체불평형
# R-CAHU-01R  정상 벨트느슨함
# R-CAHU-02R  정상 베어링불량

DATA_IN_PATH = './data-ac/c/' + device_id + '/' + status + '/'
file_list = os.listdir(DATA_IN_PATH)

dataColumns = ['file_name', 'id', 'status', 'x1_max', 'x1_min', 'x1_avg', 'x1_std', 'x1_median', 'x1_per25', 'x1_per75',
               'x2_max', 'x2_min', 'x2_avg', 'x2_std', 'x2_median', 'x2_per25', 'x2_per75', 'x3_max', 'x3_min',
               'x3_avg', 'x3_std', 'x3_median', 'x3_per25', 'x3_per75']
my_df = pd.DataFrame(columns=dataColumns)
i = 0

for fileName in file_list:
    i = i + 1
    if i % 100 == 0:
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
my_df.to_csv("./" + device_id + "_" + status + ".csv")