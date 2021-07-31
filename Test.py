import matplotlib.pyplot as plt
import pandas as pd

from Func import load_current_data

type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
path_0,file_names_0 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name = file_names_0[299]
normal = load_data(path_0, file_names_0[299])

############## animation

def animate_x(i):
    line_x.set_ydata(x[0+i:700+i])  # update the data.
    print(i)
    return line_x,

def animate_y(i):
    line_y.set_ydata(y[0+i:700+i])  # update the data.
    return line_y,

def animate_z(i):
    line_z.set_ydata(z[0+i:700+i])  # update the data.
    return line_z,

fig, ax = plt.subplots(3,1)
t = normal['time']
x = normal['x']
y = normal['y']
z = normal['z']

plt.title(f'time: {print_date(file_name)[0]} state: {print_date(file_name)[1]}')
line_x, = ax[0].plot(range(0, 700), x[0:700], color='r')
line_y, = ax[1].plot(range(0, 700), y[0:700], color='g')
line_z, = ax[2].plot(range(0, 700), z[0:700], color='b')

ani_x = animation.FuncAnimation(fig, animate_x,frames= 1301, interval=25, blit=True, save_count=1,repeat=True)
ani_y = animation.FuncAnimation(fig, animate_y, frames= 1301,interval=25, blit=True, save_count=1,repeat=True)
ani_z = animation.FuncAnimation(fig, animate_z, frames= 1301,interval=25, blit=True, save_count=1,repeat=True)
plt.show()

draw_plot(normal,anomaly)
##############
def gen_pre_data(normal):
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

pd.DataFrame(data_temp)
plt.scatter(range(len(data_temp[0]['x'])),data_temp[0]['x'])
len(data_temp[0]['x'][0:])

#==============================================================
current_3d_plot(normal)


plt.rcParams["figure.figsize"] = (6, 6)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(normal['x'][:500], normal['y'][:500], normal['z'][:500],cmap='Greens', marker='o', s=10,label='normal')
#ax.scatter(belt['x'], belt['y'], belt['z'], c=belt['type'], marker='o', s=10,label='l_belt')

ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()
plt.show()
###########################################################
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

def update_graph(num):
    data = normal
    graph._offsets3d = (data['x'][0:50+num*50], data['y'][0:50+num*50], data['z'][0:50+num*50])
    title.set_text('3D Test, time={}'.format(num))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

graph = ax.scatter(normal.x, normal.y, normal.z)
plt.show()

ani = animation.FuncAnimation(fig, update_graph, 40,interval=400, blit=False)
####################################################################################
type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path_0_0,file_names_0_0 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_0 = file_names_0_0[0]
R_CAHU_03S_15_00 = load_current_data(path_0_0, file_name_0_0)

state = '벨트느슨함'
path_0_4,file_names_0_4 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_4 = file_names_0_4[0]
R_CAHU_03S_15_04 = load_current_data(path_0_4, file_name_0_4)

####################################################################################
type = 'current'
kw = '18.5'
machine = 'R-CAHU-02S'
state = '정상'
path_1_0,file_names_1_0 = detect_file_name(type, kw, machine, state)
### 298 : 베어링불량 299 : 정상
file_name_1_0 = file_names_1_0[0]
R_CAHU_02S_18_10 = load_current_data(path_1_0, file_name_1_0)

state = '베어링불량'
path_1_1,file_names_1_1 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_1_1 = file_names_1_1[0]
R_CAHU_02S_18_11 = load_current_data(path_1_1, file_name_1_1)

#####################################################################################
type = 'current'
kw = '22'
machine = 'L-CAHU-02S'
state = '정상'
path_2_0,file_names_2_0 = detect_file_name(type, kw, machine, state)
### 298 : 베어링불량 299 : 정상
file_name_2_0 = file_names_2_0[0]
L_CAHU_02S_18_20 = load_current_data(path_2_0, file_name_2_0)

state = '회전체불평형'
path_2_2,file_names_2_2 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_2_2 = file_names_2_2[0]
L_CAHU_02S_18_22 = load_current_data(path_2_2, file_name_2_2)
########################################################################################
type = 'current'
kw = '22'
machine = 'L-CAHU-01S'
state = '정상'
path_3_0,file_names_3_0 = detect_file_name(type, kw, machine, state)
### 298 : 베어링불량 299 : 정상
file_name_3_0 = file_names_3_0[0]
L_CAHU_01S_18_30 = load_current_data(path_3_0, file_name_3_0)

state = '축정렬불량'
path_3_3,file_names_3_3 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_3_3 = file_names_3_3[0]
L_CAHU_01S_18_33 = load_current_data(path_3_3, file_name_3_3)
#####################################################################################
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

file_name = file_names_0_0[50]
data00 = load_current_data(path_0_0, file_name)
file_name = file_names_0_4[50]
data04 = load_current_data(path_0_4, file_name)

file_name = file_names_1_0[50]
data10 = load_current_data(path_1_0, file_name)
file_name = file_names_1_1[50]
data11 = load_current_data(path_1_1, file_name)

file_name = file_names_2_0[50]
data20 = load_current_data(path_2_0, file_name)
file_name = file_names_2_2[50]
data22 = load_current_data(path_2_2, file_name)


file_name = file_names_3_0[50]
data30 = load_current_data(path_3_0, file_name)
file_name = file_names_3_3[50]
data33 = load_current_data(path_3_3, file_name)

##################################################################################


normal = data00
anomal = data04
def update_graph(num):
    normal = normal
    anomal = anomal
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    graph._offsets3d = (normal['x'][0:50+num*50], normal['y'][0:50+num*50], normal['z'][0:50+num*50])
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    graph._offsets3d = (anomal['x'][0:50 + num * 50], anomal['y'][0:50 + num * 50], anomal['z'][0:50 + num * 50])
    title.set_text('3D Test, time={}'.format(num))

title = ax.set_title('3D Test')

fig = plt.figure(figsize= plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection='3d')
graph1 = ax.scatter(normal.x, normal.y, normal.z,color='blue',label='Normal')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()


ax = fig.add_subplot(1,2,2, projection='3d')
graph2 = ax.scatter(anomal.x, anomal.y, anomal.z,color='orange',label='Error')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

plt.show()

ani = animation.FuncAnimation(fig, update_graph, 40,interval=400, blit=False)


########### 벨트 느슨함 정상 비교
type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path_0_0,file_names_0_0 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_0 = file_names_0_0[0]
R_CAHU_03S_15_00 = load_current_data(path_0_0, file_name_0_0)

state = '벨트느슨함'
path_0_4,file_names_0_4 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_4 = file_names_0_4[0]
R_CAHU_03S_15_04 = load_current_data(path_0_4, file_name_0_4)
########################################

file_name_0_0 = file_names_0_0[0]
file_name_0_4 = file_names_0_4[0]
R_CAHU_03S_15_00 = load_current_data(path_0_0, file_name_0_0)
R_CAHU_03S_15_04 = load_current_data(path_0_4, file_name_0_4)


normal = R_CAHU_03S_15_00
anomal = R_CAHU_03S_15_04

title = ax.set_title('3D Test')

fig = plt.figure(figsize= plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection='3d')
graph1 = ax.scatter(normal.x, normal.y, normal.z,color='blue',label='Normal')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()


ax = fig.add_subplot(1,2,2, projection='3d')
graph2 = ax.scatter(anomal.x, anomal.y, anomal.z,color='orange',label='Error')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

plt.show()