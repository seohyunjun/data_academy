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
graph2 = ax.scatter(anomal.x, anomal.y, anomal.z,color='orange',label='Error4')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

plt.show()

#======================
normal = data10
anomal = data11
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
graph2 = ax.scatter(anomal.x, anomal.y, anomal.z,color='orange',label='Error1')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

plt.show()
#===========================
normal = data20
anomal = data22
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
graph2 = ax.scatter(anomal.x, anomal.y, anomal.z,color='orange',label='Error2')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

plt.show()
##################################
normal = data30
anomal = data33
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
graph1 = ax.scatter(normal.x, normal.y, normal.z,color='blue',label='Normal')#,alpha=0.2)
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()


ax = fig.add_subplot(1,2,2, projection='3d')
graph2 = ax.scatter(anomal.x, anomal.y, anomal.z,color='orange',label='Error3')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

plt.show()


###################################################################

normal0 = data00
normal1 = data10
normal2 = data20
normal3 = data30
##########################################################################################
title = ax.set_title('3D Test')

fig = plt.figure(figsize = plt.figaspect(0.5))
ax = fig.add_subplot(1,4,1, projection='3d')
graph1 = ax.scatter(normal0.x, normal0.y, normal0.z,color='blue',label='Normal0')#,alpha=0.2)
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()


ax = fig.add_subplot(1,4,2, projection='3d')
graph2 = ax.scatter(normal1.x, normal1.y, normal1.z,color='orange',label='Normal1')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

ax = fig.add_subplot(1,4,3, projection='3d')
graph3 = ax.scatter(normal2.x, normal2.y, normal2.z,color='orange',label='Normal2')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()

ax = fig.add_subplot(1,4,4, projection='3d')
graph4 = ax.scatter(normal3.x, normal3.y, normal3.z,color='orange',label='Normal3')
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
plt.legend()
plt.show()