from Func import *
import pandas as pd


type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
temp_path,temp_file_names = detect_file_name(type, kw, machine, state)
len(temp_file_names)

vibration10 = load_vibration_data(temp_path,temp_file_names[10])
vibration10['vibration'][:500].plot()
vibration10['vibration'][:500].rolling(window=10).mean().plot()
plt.plot(vibration10['vibration'])

pre_data = gen_vibration_pre_data(vibration10)

plt.scatter(range(len(pre_data[0]['vibration'][:500])),pre_data[0]['vibration'][:500],color='red')
plt.plot(range(len(pre_data[0]['vibration'][:500])),pre_data[0]['vibration'][:500])
