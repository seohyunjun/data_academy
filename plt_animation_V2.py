import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
type = 'current'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path_0_0,file_names_0_0 = detect_file_name(type, kw, machine, state)
### 298 : 벨트느슨함 299 : 정상
file_name_0_0 = file_names_0_0[0]
R_CAHU_03S_15_00 = load_current_data(path_0_0, file_name_0_0)

first = R_CAHU_03S_15_00

fig, ax = plt.subplots(1,1)

t = first['time']
x = first['x']
y = first['y']
z = first['z']

line_x, = ax.plot(range(0,100), x[0:100],color='r')
line_y, = ax.plot(range(0,100), y[0:100],color='g')
line_z, = ax.plot(range(0,100), z[0:100],color='b')

def animate(i):
    line_x.set_ydata(x[0+i:100+i])
    line_y.set_ydata(y[0+i:100+i])
    line_z.set_ydata(z[0+i:100+i])
    return [line_x,line_y, line_z]

ani_x= animation.FuncAnimation(fig, animate, interval=200, blit=True, save_count=1)
#ani_y = animation.FuncAnimation(fig, animate_y, interval=200, blit=True, save_count=1)
#ani_z = animation.FuncAnimation(fig, animate_z, interval=200, blit=True, save_count=1)
writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800, ani_x.save("movie.mp4"))
plt.show()


def animate_x(i):
    line_x.set_ydata(x[0 + i:100 + i])  # update the data.
    return line_x,
def animate_y(i):
    line_y.set_ydata(y[0 + i:100 + i])  # update the data.
    return line_y,
def animate_z(i):
    line_z.set_ydata(z[0 + i:100 + i])  # update the data.
    return line_z,

def plot_animation(first,save=None):
    fig, ax = plt.subplots(3, 1)
    t = first['time']
    x = first['x']
    y = first['y']
    z = first['z']
    line_x, = ax[0].plot(range(0, 100), x[0:100], color='r')
    line_y, = ax[1].plot(range(0, 100), y[0:100], color='g')
    line_z, = ax[2].plot(range(0, 100), z[0:100], color='b')

    ani_x = animation.FuncAnimation(fig, animate_x, interval=100, blit=True, save_count=1)
    ani_y = animation.FuncAnimation(fig, animate_y, interval=100, blit=True, save_count=1)
    ani_z = animation.FuncAnimation(fig, animate_z, interval=100, blit=True, save_count=1)
    plt.show()

    if save==True:
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800),animate_x.save("movie.mp4")


for i in range(0,10):
    if i == 0:
        fig, ax = plt.subplots()
        line_z, = ax.plot(range(0, 100), z[0:100], color='b')
    line_z.set_ydata(z[0 + i:100 + i])
    plt.show()
    #time.sleep(1)

###############################