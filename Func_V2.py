import os
import matplotlib
from numpy.core import multiarray
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm

## Font
import matplotlib.font_manager as fm
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'NanumSquare_ac'
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
Nanum_list = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

from dateutil.parser import parse

import re

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def draw_current_plot(normal,anomal):
    t = normal['time'].values
    plt.plot(t,normal['x'], color='r',label='Normal')
    plt.plot(t,normal['y'], color='r')
    plt.plot(t,normal['z'], color='r')
    plt.plot(t,anomal['x'], color='b',label='Anomal')
    plt.plot(t,anomal['y'], color='b')
    plt.plot(t,anomal['z'], color='b')
    plt.legend()
    plt.show()
    #plt.savefig('Normal Anomal.jpg',dpi=300)
def draw_vibration_plot(normal,anomal):
    t = normal['time'].values
    plt.plot(t,normal['vibration'], color='r',label='Normal')
    plt.plot(t,anomal['vibration'], color='b',label='Anomal')
    plt.legend()
    plt.show()
    #plt.savefig('Normal Anomal.jpg',dpi=300)
def detect_file_name(type, kw, machine, state):
    path = type+'/'+kw+'kW'+'/'+machine
    print(os.listdir(path))
    path0 = type+'/'+kw+'kW'+'/'+machine+'/'+state
    file_name = os.listdir(path0)
    return path0,file_name
def load_current_data(path,file_name):
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',', skiprows = [0,1,2,3,4,5,6,7,8])
    first = first.iloc[:,:-1]
    first.columns = ['time','x','y','z']
    return first
def load_vibration_data(path,file_name):
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',', skiprows = [0,1,2,3,4,5,6,7,8])
    first = first.iloc[:,:-1]
    first.columns = ['time','vibration']
    return first
def print_date(file_name):
    return file_name[-23:-10]

def gen_current_pre_data(normal):
    data_temp = []
    for type in ['x','y','z']:
        temp = []
        temp_idx = []
        bound = []
        for idx,i in enumerate(normal[type]):
            #idx first
            if idx==0 and i>=0:
                pre_state=1
            if idx==0 and i<0:
                pre_state=0

            if i >= 0:
                state=1
            else:
                state=0

            change_state=state+pre_state

            if change_state==1:
                if state==1:
                    temp.append(min(bound))
                    temp_idx.append(idx)
                if state==0:
                    temp.append(max(bound))
                    temp_idx.append(idx)
                    bound=[]
            if state==1:
                bound.append(i)
            else:
                bound.append(i)
            pre_state=state

        type_temp = {
            type:temp,
            idx:temp_idx
        }
        data_temp.append(type_temp)
    return data_temp
def gen_vibration_pre_data(normal):
    data_temp = []
    for type in ['vibration']:
        temp = []
        temp_idx = []
        bound = []
        for idx,i in enumerate(normal[type]):
            #idx first
            if idx==0 and i>=0:
                pre_state=1
            if idx==0 and i<0:
                pre_state=0

            if i >= 0:
                state=1
            else:
                state=0

            change_state=state+pre_state

            if change_state==1:
                if state==1:
                    temp.append(min(bound))
                    temp_idx.append(idx)
                if state==0:
                    temp.append(max(bound))
                    temp_idx.append(idx)
                    bound=[]
            if state==1:
                bound.append(i)
            else:
                bound.append(i)
            pre_state=state

        type_temp = {
            type:temp,
            idx:temp_idx
        }
        data_temp.append(type_temp)

    data_temp = pd.DataFrame(data_temp[0]['vibration'],columns=['peak_vibration'])
    return data_temp
def load_current_rms_data(path,file_name):
    type = pd.read_csv(f'{path}/{file_name}', header=None, sep=',', nrows=1, skiprows=[0, 1, 2])
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2,3,4,5,6])
    first.drop(columns=[0,4],axis=1,inplace=True)
    first.columns = ['x','y','z']
    first['type'] = int(type[1])
    first['time'] = pd.to_datetime(parse(re.sub('_', '', print_date(file_name))))
    return first

def pre_load_current_rms_data(path,file_name):
    type = pd.read_csv(f'{path}/{file_name}', header=None, sep=',', nrows=1, skiprows=[0, 1, 2])
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2,3,4,5,6])
    first.drop(columns=[0,4],axis=1,inplace=True)
    first.columns = ['x','y','z']
    first['type'] = int(type[1])
    first['time'] = pd.to_datetime(parse(re.sub('_', '', file_name[-26:-11])))
    return first
def load_vibration_rms_data(path,file_name):
    type = pd.read_csv(f'{path}/{file_name}', header=None, sep=',', nrows=1, skiprows=[0, 1, 2])
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2,3,4,5,6])
    first.drop(columns=[0,4],axis=1,inplace=True)
    first.columns = ['vibration']
    first['type'] = int(type[1])
    first['time'] = pd.to_datetime(parse(re.sub('_','',print_date(file_name))))
    return first


def current_3d_plot(normal,anomaly=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normal['x'], normal['y'], normal['z'], cmap='Greens', marker='o', s=10, label='normal')
    if anomaly != None:
        ax.scatter(anomaly['x'], anomaly['y'], anomaly['z'], marker='o', s=10, label='')

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    plt.legend()
    plt.show()



def current_total_rms(path_0,file_names_0):
    rms_data = pd.DataFrame()
    for file in file_names_0:
        temp = load_current_rms_data(path_0, file)
        rms_data = pd.concat([rms_data, temp])
    return rms_data
############################################
def total_current_info(path_0_0,file_names_0_0,type,kw,machine,state):
    total = []
    for file in file_names_0_0:
        info = current_info(path_0_0,file)
        total.append(info)
    total = pd.DataFrame(total)
    total.to_csv(f'{type}_{kw}_{machine}_{state}.csv')
    return total


def current_info(path, file_name):
    temp = load_current_data(path, file_name)

    x_max = temp.describe()['x']['max']
    x_min = temp.describe()['x']['min']
    x_avg = temp.describe()['x']['mean']
    x_std = temp.describe()['x']['std']
    x_per25 = temp.describe()['x']['25%']
    x_median = temp.describe()['x']['50%']
    x_per75 = temp.describe()['x']['75%']

    z_max = temp.describe()['z']['max']
    z_min = temp.describe()['z']['min']
    z_avg = temp.describe()['z']['mean']
    z_std = temp.describe()['z']['std']
    z_per25 = temp.describe()['z']['25%']
    z_median = temp.describe()['z']['50%']
    z_per75 = temp.describe()['z']['75%']

    y_max = temp.describe()['y']['max']
    y_min = temp.describe()['y']['min']
    y_avg = temp.describe()['y']['mean']
    y_std = temp.describe()['y']['std']
    y_per25 = temp.describe()['y']['25%']
    y_median = temp.describe()['y']['50%']
    y_per75 = temp.describe()['y']['75%']

    temp = {
        'x_max':x_max,
        'x_min':x_min,
        'x_avg':x_avg,
        'x_std':x_std,
        'x_per25':x_per25,
        'x_median':x_median,
        'x_per75':x_per75,

        'y_max': y_max,
        'y_min': y_min,
        'y_avg': y_avg,
        'y_std': y_std,
        'y_per25': y_per25,
        'y_median': y_median,
        'y_per75': y_per75,

        'z_max': z_max,
        'z_min': z_min,
        'z_avg': z_avg,
        'z_std': z_std,
        'z_per25': z_per25,
        'z_median': z_median,
        'z_per75': z_per75,
    }

    return temp

def plot_animation(current_data,len_bid=100,bid=50,save=None,save_name=None):
    fig, ax = plt.subplots(1,1)

    t = current_data['time']
    x = current_data['x']
    y = current_data['y']
    z = current_data['z']

    line_x, = ax.plot(range(0,len_bid),color='r',label='x')
    line_y, = ax.plot(range(0,len_bid),color='g',label='y')
    line_z, = ax.plot(range(0,len_bid),color='b',label='z')
    ax.set_ylim(-50,50)
    ax.legend()
    def animate(i):
        line_x.set_ydata(x[0+bid*i:len_bid+bid*i])
        line_y.set_ydata(y[0+bid*i:len_bid+bid*i])
        line_z.set_ydata(z[0+bid*i:len_bid+bid*i])
        print(i)
        return [line_x,line_y, line_z]

    t = (len(x)-len_bid) // bid
    ani_x= animation.FuncAnimation(fig, animate,t, interval=199, blit=True, save_count=1)
    #ani_y = animation.FuncAnimation(fig, animate_y, interval=200, blit=True, save_count=1)
    #ani_z = animation.FuncAnimation(fig, animate_z, interval=200, blit=True, save_count=1)
    if save==True:
        writer = animation.FFMpegWriter(fps=5,) #metadata=dict(artist='Me'), bitrate=1800)
        #ani_x.save("movie.mp4")
        ani_x.save(f'{save_name}.gif', writer='imagemagick', fps=30, dpi=100)
    plt.show()


def count_peak(value,num):
    peak_len = len(value) // num
    peak_max = []
    peak_min = []
    for i in range(peak_len):
        temp = value[0+num*i:num+num*i]
        max_temp = max(temp)
        min_temp = min(temp)

        peak_max.append(max_temp)
        peak_min.append(min_temp)

    peak = pd.DataFrame()
    peak['max'] = peak_max
    peak['min'] = peak_min

    return peak


def judge_color(state):
    if state=='정상':
        col = 'black'
    if state=='베어링불량':
        col = 'red'
    if state=='회전체불평형':
        col = 'blue'
    if state=='축정렬불량':
        col = 'green'
    if state=='벨트느슨함':
        col = 'orange'
    return col