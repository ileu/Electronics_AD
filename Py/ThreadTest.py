# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:46:03 2018

@author: Ueli5
"""

import numpy as np

import threading
import pandas as pd
import time

np.set_printoptions(linewidth=160)

data = np.array([['', 'Col1', 'Col2'], ['Row1', 1, 2], ['Row2', 3, 4]])

dataframe = pd.DataFrame(data=data[1:, 1:],  # values
                         index=data[1:, 0],  # 1st column as index
                         columns=data[0, 1:])  # 1st row as the column names
print(dataframe)
print(data[1:, 1:])
print(data[1:, 0])
print(data[0, 1:])
x = 3


class PID(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.stop = False

    def run(self):
        global x
        while not self.stop:
            print("dadadada")
            x += 1
            time.sleep(0.5)
        print("yolo")


test = PID()

print("ready")
input()
print("started")
test.start()
print("yala")
input()
print("st")
test.stop = True
test.join()
print("stopped")

print(x)
