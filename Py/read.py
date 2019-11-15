# -*- coding: utf-8 -*-
"""
@author: Ueli5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from serial import Serial
from datetime import datetime

np.set_printoptions(linewidth=160)

arduino = Serial('COM4', 9600)
now = datetime.now().strftime('%H-%M_%d-%m')
print(now)
data = np.empty((1, 4))
i = 0
plt.ion()
while i <= 2900:
    out = arduino.readline().decode('utf-8')[:-2]
    out = out.split(';')
    out = [float(i) for i in out]
    data = np.append(data, [out], axis=0)

    i += 1
    print(out)
    plt.clf()
    plt.scatter(data[1:, 0], data[1:, 1])
    plt.pause(0.05)
    plt.draw()

data = np.delete(data, 0, 0)

print(data)
np.savetxt('Cool_Step_' + now + '.txt', data, delimiter=r";")
plt.show()
