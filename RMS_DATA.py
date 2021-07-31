import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm


from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from dateutil.parser import parse
import datetime
import time

import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Func import *


type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
path_0,file_names_0 = detect_file_name(type, kw, machine, state)

def load_current_rms_data(path,file_name):
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2,3,4,5,6])
    first.drop(columns=[0,4],axis=1,inplace=True)
    first.columns = ['x','y','z']
    first['type'] = int(file_name[-5:-4])
    first['time'] = pd.to_datetime(parse(re.sub('_','',print_date(file_name)[0])))
    return first
def load_vibration_rms_data(path,file_name):
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2,3,4,5,6])
    first.drop(columns=[0,4],axis=1,inplace=True)
    first.columns = ['x','y','z']
    first['type'] = int(file_name[-5:-4])
    first['time'] = pd.to_datetime(parse(re.sub('_','',print_date(file_name)[0])))
    return first



load_current_rms_data(path_0,file_names_0[20])

rms_data = pd.DataFrame()
for file in file_names_0:
    temp = load_current_rms_data(path_0,file)
    rms_data = pd.concat([rms_data,temp])


n = 100
xmin, xmax, ymin, ymax, zmin, zmax = -100, 100, -100, 100, -100, 100

normal = rms_data[rms_data['type']==0]
belt = rms_data[rms_data['type']==4]

plt.rcParams["figure.figsize"] = (6, 6)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(normal['x'], normal['y'], normal['z'],cmap='Greens', marker='o', s=10,label='normal')
ax.scatter(belt['x'], belt['y'], belt['z'], c=belt['type'], marker='o', s=10,label='l_belt')

ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()
plt.show()

current_3d_plot(normal)



#################################

type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'

path_0,file_names_0 = detect_file_name(type, kw, machine, state)
def current_total_rms(path_0,file_names_0):
    rms_data = pd.DataFrame()
    for file in file_names_0:
        temp = load_current_rms_data(path_0, file)
        rms_data = pd.concat([rms_data, temp])
    return rms_data


normal = rms_data

type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '벨트느슨함'
path_0,file_names_0 = detect_file_name(type, kw, machine, state)
anomal = current_total_rms(path_0,file_names_0)
#belt = anomal
n = 100
xmin, xmax, ymin, ymax, zmin, zmax = -100, 100, -100, 100, -100, 100

normal = rms_data[rms_data['type']==0]
belt = rms_data[rms_data['type']==4]


### belt(11072) 74.23% vs normal(3842) 25.7%
#11072/(11072+3842)
plt.rcParams["figure.figsize"] = (6, 6)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(normal['x'], normal['y'], normal['z'],cmap='Greens', marker='o', s=10,label='normal')
ax.scatter(belt['x'], belt['y'], belt['z'], c=belt['type'], marker='o', s=10,label='loose_belt')

ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()
plt.show()

current_3d_plot(normal)

path = path_0
file_name = file
type = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2])
first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',',nrows=1,skiprows=[0,1,2,3,4,5,6])

first.drop(columns=[0,4],axis=1,inplace=True)
first.columns = ['x','y','z']
first['type'] = int(type[1])
first['time'] = pd.to_datetime(parse(re.sub('_','',print_date(file_name)[0])))
return first