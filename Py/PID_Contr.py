# -*- coding: utf-8 -*-
"""
@author: Ueli5
"""

import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
import threading
from serial import Serial
from datetime import datetime
import time

np.set_printoptions(linewidth=160)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# parameter for pid
abc = [[10, 90, 0.3]]
# transformation of error to pmw value
max_correction = [-300, 0]
pmw_band = [0, 160]
param = np.polyfit(max_correction, pmw_band, 1)
# target temperature
set_temp = 26
# constants of the thermistor
r0 = 1e5
t0 = 25
b_ther = 4396.64
kelvin = 273.15
set_temp += kelvin


def fan_speed(correct, m, q):
    """
    converting the pid output to a PWM signal
    :param correct: pid error output
    :param m: slope for liner fit
    :param q: intercept of the fit
    :return: PWM signal
    """
    rpm = int(round(m * correct + q))

    if rpm > 255:
        return 255
    elif rpm < 0:
        return 0
    else:
        return rpm


def pid_contr(err, perr, it, kp, ki, kd):
    """
    PID controller
    :param err: error of the measurement
    :param perr: previous error
    :param it: integrated error
    :param kp: const for prop error
    :param ki: const for integr error
    :param kd: const for deriv error
    :return: pid error value
    """
    p = kp * err
    it += ki * err
    d = kd * (err - perr)
    return p + d + it, it


def calc_temp(vol):
    """
    calculates the temperature from the voltage
    :param vol: measured voltage
    :return: temperature in kelvin
    """
    r = (1023.0 / vol - 1) * r0
    t = np.log(r / r0) / b_ther + 1.0 / (t0 + kelvin)
    return 1 / t


class PID(threading.Thread):
    """
    Main part of the data scanning done with a thread
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.stop = False

    def run(self):
        # use global constants
        global data
        global integral
        global a
        global b
        global c
        # initial measurement read out and converting to list
        out = arduino.readline().decode('utf-8')[:-2]
        out = out.split(';')
        out = [float(i) for i in out]
        temp = calc_temp(out[1])
        # append everything to data list
        data = np.append(data, [[*out, temp - set_temp]], axis=0)
        # tell arduino to not run fan
        arduino.write(struct.pack("!B", 0))
        while not self.stop:
            # processing of the arduino data
            out = arduino.readline().decode('utf-8')[:-2]
            out = out.split(';')
            out = [float(i) for i in out]
            temp = calc_temp(out[1])
            data = np.append(data, [[*out, temp - set_temp]], axis=0)
            # running the pid and calculate new fan speed
            pid, integral = pid_contr(data[-1, 3], data[-2, 3], integral, a, b, c)
            speed = fan_speed(pid, *param)
            # send new fan speed to arduino
            arduino.write(struct.pack("!B", speed))
            # plot and print the data while reading
            print(out, speed, pid)
            plt.clf()
            plt.scatter(data[1:, 0] * 1e-3, calc_temp(data[1:, 1]) - kelvin)
            plt.pause(0.02)
            plt.draw()


# start dynamic plotting
plt.ion()
for a, b, c in abc:
    # start arduino
    arduino = Serial('COM4', 9600)
    time.sleep(1)
    now = datetime.now().strftime('%H-%M_%d-%m')
    print(now)
    print(a, b, c)
    # prepare data and PID
    data = np.empty((1, 4))
    data = np.delete(data, 0, 0)
    controller = PID()
    integral = 0
    time.sleep(1)
    # look if arduino is ready and than start
    print(arduino.readline().decode('utf-8')[:-2])
    controller.start()
    # cancel if enter is pressed in console
    input()
    controller.stop = True
    controller.join()
    print("Measurement finished")
    # save data and close arduino
    labels = ["time", "resistance", "FanCount", "dTemp"]
    data = pd.DataFrame(data, columns=labels)
    data.to_pickle("./pickles/PID_a" + str(a) + "b" + str(b) + "c" + str(c) + "_" + now + ".p")
    print("done")
    arduino.close()
