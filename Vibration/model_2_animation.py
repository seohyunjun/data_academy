import os
# Select DIR
#os.getcwd()
from Func_V2 import *
import pandas as pd

import numpy as np

## 
## 자취방

#OR_PATH = 'C:\\Users\\admin\\Desktop\\code'
#DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'
#VALIDATION_PATH = 'D:\\기계시설물 고장 예지 센서\\Validation'

## 회사
DATA_PATH = 'D:\\project\\기계시설물 고장 예지 센서\\Training'
VALIDATION_PATH = 'D:\\기계시설물 고장 예지 센서\\Validation'
OR_PATH = 'C:\\Users\\A\\Desktop\\code'

os.chdir(DATA_PATH)
type = 'vibration'
kw = '2.2'
machine = 'L-DSF-01'
state = '정상'
ab_state = '축정렬불량'

path,file_names = detect_file_name(type, kw, machine, state)

normal_file = file_names[0]
normal_data = load_vibration_data(path,normal_file)
normal_data_peak = count_peak(normal_data['vibration'],200)['max']

path,file_names = detect_file_name(type, kw, machine, ab_state)
abnormal_file = file_names[0]
abnormal_data = load_vibration_data(path,abnormal_file)
abnormal_data_peak = count_peak(abnormal_data['vibration'],200)['max']


def plot_vibration_animation(normal,ab_normal,label1=state,label2=ab_state,len_bid=20,bid=5,save=None,save_name=None):
    fig, ax = plt.subplots(1,1)
    t = range(len(normal))
    x = normal.values
    y = ab_normal.values
    line_x, = ax.plot(range(0,len_bid),color='r',label=f'{label1}')
    line_y, = ax.plot(range(0,len_bid),color='g',label=f'{label2}')
    ax.set_ylim(0,0.03)
    ax.legend()
    def animate(i):
        line_x.set_ydata(x[0+bid*i:len_bid+bid*i])
        line_y.set_ydata(y[0+bid*i:len_bid+bid*i])
        print(i)
        return [line_x,line_y]

    t = (len(x)-len_bid) // bid
    ani_x= animation.FuncAnimation(fig, animate,t, interval=100, blit=True, save_count=1)
    #ani_y = animation.FuncAnimation(fig, animate_y, interval=200, blit=True, save_count=1)
    #ani_z = animation.FuncAnimation(fig, animate_z, interval=200, blit=True, save_count=1)
    if save==True:
        writer = animation.FFMpegWriter(fps=5,) #metadata=dict(artist='Me'), bitrate=1800)
        #ani_x.save("movie.mp4")
        ani_x.save(f'{save_name}.gif', writer='imagemagick', fps=10, dpi=100)
    plt.show()

os.chdir(OR_PATH)
plot_vibration_animation(normal_data_peak,abnormal_data_peak,state,ab_state,save=True,save_name='model_2')
os.chdir(DATA_PATH)
