#os.getcwd()
os.chdir('Training')

from Func import *
import os

type = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path, file_names = detect_file_name(type, kw, machine, state)
file_name = file_names[0]
vibration_1_0 = load_vibration_data(path,file_name)


#Furier Transform
from scipy.fft import fft, ifft
x = np.array(vibration_1_0['vibration'].values)
yinv = ifft(x)

yinv_df = pd.DataFrame(yinv)
yinv_df.plot()
plt.plot(yinv_df)
plt.show()



