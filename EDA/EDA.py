import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm


import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def draw_plot(normal,anomal):
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
def detect_file_name(type, kw, machine, state):
    path = type+'/'+kw+'kW'+'/'+machine
    print(os.listdir(path))
    path0 = type+'/'+kw+'kW'+'/'+machine+'/'+state
    file_name = os.listdir(path0)
    return path0,file_name
def load_currnet_data(path,file_name):
    first = pd.read_csv(f'{path}/{file_name}', header=None,sep=',', skiprows = [0,1,2,3,4,5,6,7,8])
    first = first.iloc[:,:-1]
    first.columns = ['time','x','y','z']
    return first
def print_date(file_name):
    state = file_name[-6:-4]
    return file_name[-26:-11],state
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
############################################
##0: 정상  벨트느슨함 : 4
########################

## 15kW DJ_Station B_CH15_R-CAHU-03S

type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = 'temp'
temp_path_0,temp_file_names_0 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name = file_names_0[299]
normal = load_currnet_data(path_0, file_names_0[299])

############## animation

def animate_x(i):
    line_x.set_ydata(x[0+i:100+i])  # update the data.
    #print(i)
    return line_x,

def animate_y(i):
    line_y.set_ydata(y[0+i:100+i])  # update the data.
    return line_y,

def animate_z(i):
    line_z.set_ydata(z[0+i:100+i])  # update the data.
    return line_z,

fig, ax = plt.subplots(3,1)
t = normal['time']
x = normal['x']
y = normal['y']
z = normal['z']

plt.title(f'time: {print_date(file_name)[0]} state: {print_date(file_name)[1]}')
line_x, = ax[0].plot(range(0, 100), x[0:100], color='r')
line_y, = ax[1].plot(range(0, 100), y[0:100], color='g')
line_z, = ax[2].plot(range(0, 100), z[0:100], color='b')

ani_x = animation.FuncAnimation(fig, animate_x,frames= 1901, interval=25, blit=True, save_count=1,repeat=True)
ani_y = animation.FuncAnimation(fig, animate_y, frames= 1901,interval=25, blit=True, save_count=1,repeat=True)
ani_z = animation.FuncAnimation(fig, animate_z, frames= 1901,interval=25, blit=True, save_count=1,repeat=True)
plt.show()

##############################


file_name = file_names_0[298]
anomaly = load_currnet_data(path_0, file_names_0[298])
def animate_x1(i):
    line_x1.set_ydata(x1[0+i:100+i])  # update the data.
    #print(i)
    return line_x1,

def animate_y1(i):
    line_y1.set_ydata(y1[0+i:100+i])  # update the data.
    return line_y1,

def animate_z1(i):
    line_z1.set_ydata(z1[0+i:100+i])
    print(i)
    # update the data.
    return line_z1,

####################################################

fig1, ax1 = plt.subplots(3,1)
plt.title(f'time: {print_date(file_name)[0]} state: {print_date(file_name)[1]}')
t1 = anomaly['time']
x1 = anomaly['x']
y1 = anomaly['y']
z1 = anomaly['z']

line_x1, = ax1[0].plot(range(0, 100), x1[0:100], color='r')
line_y1, = ax1[1].plot(range(0, 100), y1[0:100], color='g')
line_z1, = ax1[2].plot(range(0, 100), z1[0:100], color='b')

ani_x1 = animation.FuncAnimation(fig1, animate_x1,frames= 1901, interval=25, blit=True, save_count=1)#,repeat=True)
ani_y1 = animation.FuncAnimation(fig1, animate_y1, frames= 1901,interval=25, blit=True, save_count=1)#,repeat=True)
ani_z1 = animation.FuncAnimation(fig1, animate_z1, frames= 1901,interval=25, blit=True, save_count=1)#,repeat=True)
plt.show()
####################################################
########################################

file = temp_file_names_0[0]
des_data = []
for file in tqdm.tqdm(temp_file_names_0, total=len(temp_file_names_0)):
    data = pre_load_current_rms_data(temp_path_0,file)
    temp = {
        'time': data['time'].values[0],
        'type': data['type'].values[0],
        'x_rms': data['x'].values[0],
        'y_rms':data['y'].values[0],
        'z_rms':data['z'].values[0]
    }
    des_data.append(temp)
df_des_data = pd.DataFrame(des_data)

normal = df_des_data[df_des_data['type']==0]
anomal = df_des_data[df_des_data['type']==4]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,10))

plt.subplot(figsize=(15,10))
ax[0].scatter(normal['time'],normal['x_rms'],color='b',label='n_x_rms')
ax[0].scatter(normal['time'],normal['x_rms'],color='b',label='n_x_rms')
ax[0].scatter(anomal['time'],anomal['x_rms'],color='r',label='a_x_rms')
ax[0].scatter(anomal['time'],anomal['x_rms'],color='r',label='a_x_rms')

plt.scatter(normal['time'],normal['y_rms'],color='b',label='n_y_rms')
plt.scatter(normal['time'],normal['y_rms'],color='b',label='n_y_rms')
plt.scatter(anomal['time'],anomal['y_rms'],color='r',label='a_y_rms')
plt.scatter(anomal['time'],anomal['y_rms'],color='r',label='a_y_rms')
plt.legend()
plt.show()
plt.scatter(normal['time'],normal['z_rms'],color='b',label='n_z_rms')
plt.scatter(normal['time'],normal['z_rms'],color='b',label='n_z_rms')
plt.scatter(anomal['time'],anomal['z_rms'],color='r',label='a_z_rms')
plt.scatter(anomal['time'],anomal['z_rms'],color='r',label='a_z_rms')
plt.legend()
plt.show()

plt.title('time per 3sec min, max (20201128)')
plt.savefig('time_per_3sec_min,max(20201128)N/A.jpg',dpi=300)
##########################################################################
state = '벨트느슨함'
path_4,file_names_4 = detect_file_name(type, kw, machine, state)

#####################################################################
####################


total_data = pd.DataFrame()
for name in file_names_0:
    temp = load_currnet_data(path_0,name)
    total_data = pd.concat([total_data,temp])


fig_t, ax_t = plt.subplots(figsize=(21,5))
xdata, ydata0, ydata1, ydata2 = [], [], [],[]

#plt.title(f'time: {print_date(file_name)[0]} state: {print_date(file_name)[1]}')
t_t = total_data['time']
t_x = total_data['x']
t_y = total_data['y']
t_z = total_data['z']
ln_x, = ax_t.plot([], [], color='r',animated=True)
ln_y, = ax_t.plot([],[], color='g',animated=True)
ln_z, = ax_t.plot([], [], color='b',animated=True)



#line_x_t, = ax_t.plot(range(0, 2000), t_x[0:2000], color='r')
#line_y_t, = ax_t.plot(range(0, 2000), t_y[0:2000], color='g')
#line_z_t, = ax_t.plot(range(0, 2000), t_z[0:2000], color='b')

def init():
    ax_t.set_ylim([-100, 100])
    ax_t.set_xlim([0, 3])
    ln_x.set_data(xdata,ydata0)
    ln_y.set_data(xdata,ydata1)
    ln_z.set_data(xdata,ydata2)
    #ln_x.set_data(range(0, 2000), t_x[0+2000*i:2000+2000*i])
    #ln_y.set_data(range(0, 2000), t_y[0 + 2000 * i:2000 + 2000 * i])
    #ln_z.set_data(range(0, 2000), t_z[0 + 2000 * i:2000 + 2000 * i])
    return ln_x,ln_x,ln_x

def animate_x_total_data(i):
    xdata = t_t[0:2000]
    ydata0.append(t_x[0 + 2000 * i:2000 + 2000 * i])
    ydata1.append(t_y[0 + 2000 * i:2000 + 2000 * i])
    ydata2.append(t_z[0 + 2000 * i:2000 + 2000 * i])
    ln_x.set_data(xdata, ydata0)
    ln_y.set_data(xdata, ydata1)
    ln_z.set_data(xdata, ydata2)

    #line_x_t = ax_t.plot(range(2000), t_x[0 + 2000 * i:2000 + 2000 * i], color='r', label='x')  # update the data.
    #line_y_t = ax_t.plot(range(2000), t_y[0 + 2000 * i:2000 + 2000 * i], color='g', label='y')  # update the data.
    #line_z_t = ax_t.plot(range(2000), t_z[0 + 2000 * i:2000 + 2000 * i], color='b', label='z')
    #ax_t.legend()
    # update the data.
    #print(i)
    return ln_x,ln_y,ln_z
ani_t = animation.FuncAnimation(fig_t, animate_x_total_data,frames= 740,
                                init_func=init,interval=180, blit=True)#,repeat=True)
#anim.save('sine_wave_interval_100ms.gif', writer='imagemagick')
#plt.show()

##############################################################################################
### 정상 벨트느슨함 데이터 3D Plot

# n : 299 a : 298
file_name = file_names_0[500]
normal = load_currnet_data(path_0, file_names_0[299])

file_name = file_names_0[501]
anomaly = load_currnet_data(path_0, file_names_0[298])

plt.rcParams["figure.figsize"] = (6, 6)

