



total_data = pd.DataFrame()
for name in file_names_0:
    temp = load_data(path_0,name)
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
