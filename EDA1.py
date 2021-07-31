import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm


import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Func import *


type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
path_0,file_names_0 = detect_file_name(type, kw, machine, state)

### 298 : 벨트느슨함 299 : 정상
file_name = file_names_0[299]
normal = load_data(path_0, file_names_0[299])

anomaly = load_data(path_0,file_names_0[298])

normal_gen = gen_pre_data(normal)

anomaly_gen = gen_pre_data(anomaly)

fig, ax = plt.subplots(3,1)
ax[0].set_ylim([-100, 100])
ax[1].set_ylim([-100, 100])
ax[2].set_ylim([-100, 100])

ax[0].scatter(range(len(normal_gen[0]['x'])),normal_gen[0]['x'],color='b')
ax[1].scatter(range(len(normal_gen[1]['y'])),normal_gen[1]['y'],color='b',label='normal')
ax[2].scatter(range(len(normal_gen[2]['z'])),normal_gen[2]['z'],color='b')


ax[0].scatter(range(len(anomaly_gen[0]['x'])),anomaly_gen[0]['x'],color='r')
ax[1].scatter(range(len(anomaly_gen[1]['y'])),anomaly_gen[1]['y'],color='r',label='anomal')
ax[2].scatter(range(len(anomaly_gen[2]['z'])),anomaly_gen[2]['z'],color='r')

fig.legend()