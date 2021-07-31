import os

import pandas as pd

path = 'summary'
file_names = os.listdir('summary')

data = pd.DataFrame()
for file in file_names:
    file_path = os.path.join(path, file)
    temp = pd.read_csv(file_path, encoding='cp949')
    data = pd.concat([data,temp])

p = data.iloc[:,1:]
p