# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:46:03 2018

@author: Ueli5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import *
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import glob
import pandas as pd

np.set_printoptions(linewidth=160)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

r0 = 1e5
r0_abs = 1e3
t0 = 25
b_ther = 4715
kelvin = 273.15

b_sens = 4275
b_sens_abs = 25

v_abs = 2

t0 += kelvin


def demin_regression(x, y, xerr, yerr, alpha=0.05, error=False):
    cov = np.cov(x, y)
    n = len(x)
    xbar = np.mean(x)
    ybar = np.mean(y)
    sx = cov[0, 0]
    sy = cov[1, 1]
    sxy = cov[0, 1]
    sr = np.var(xerr) / np.var(yerr)
    b1 = ((sr * sy - sx) + np.sqrt((sx - sr * sy) ** 2 + 4.0 * sr * sxy ** 2)) / (2.0 * sr * sxy)
    b0 = ybar - b1 * xbar

    if error:
        tna = stats.t.ppf(1.0 - alpha / 2.0, n - 2)

        x_the = np.array([np.delete(x, i) for i in range(n)])
        xerr_the = np.array([np.delete(xerr, i) for i in range(n)])
        y_the = np.array([np.delete(y, i) for i in range(n)])
        yerr_the = np.array([np.delete(yerr, i) for i in range(n)])
        cov_the = np.array([np.cov(*param) for param in zip(x_the, y_the)])
        sx_the = cov_the[:, 0, 0]
        sy_the = cov_the[:, 1, 1]
        sxy_the = cov_the[:, 0, 1]
        sr_the = np.var(xerr_the) / np.var(yerr_the)
        b1_the = ((sr_the * sy_the - sx_the) + np.sqrt(
            (sx_the - sr_the * sy_the) ** 2 + 4.0 * sr_the * sxy_the ** 2)) / (
                         2 * sr_the * sxy_the)
        b0_the = np.mean(y_the) - b1 * np.mean(x_the)

        pseudo_b1 = n * b1 - (n - 1) * b1_the
        pseudo_b0 = n * b0 - (n - 1) * b0_the
        jack_b1 = np.sum(pseudo_b1) / n
        jack_b0 = np.sum(pseudo_b0) / n
        var_b1 = np.sum((pseudo_b1 - jack_b1) ** 2) / (n - 1.0)
        var_b0 = np.sum((pseudo_b0 - jack_b0) ** 2) / (n - 1.0)
        std_b1 = np.sqrt(var_b1 / n)
        std_b0 = np.sqrt(var_b0 / n)

        return np.array([b1, b0]), np.array([std_b1, std_b0]) * tna, tna
    return np.array([b1, b0])


def temp2res(temp, b, r):
    return r * np.exp((1.0 / (temp + kelvin) - 1.0 / t0) * b)


def res2temp(res, b, r):
    return (np.log(res / r) / b + 1.0 / t0) ** (-1)


def temp2vol(temp, b):
    return np.rint(1023.0 / (np.exp((1.0 / (temp + kelvin) - 1.0 / t0) * b) + 1.0))


def res2vol(res, r):
    return np.rint(1023.0 / (res / r + 1.0))


def vol2temp(vol, b):
    temp = np.log(1023.0 / vol - 1.0) / b + 1.0 / t0
    return 1.0 / temp


def vol2temp_abs(vol, verr, b, berr):
    err = (1023.0 / (vol * b * (1023.0 - vol)) * verr) ** 2
    err += (np.log(1023.0 / vol - 1.0) / (b ** 2) * berr) ** 2
    err = np.sqrt(err)
    return err * vol2temp(vol, b) ** 2


def vol2res(vol, r):
    return (1023.0 / vol - 1.0) * r


def vol2res_abs(vol, verr, r, rerr):
    err = (1023.0 / (vol ** 2) * r * verr) ** 2
    err += ((1023.0 / vol - 1.0) * rerr) ** 2
    return np.sqrt(err)


data1 = pd.read_pickle("pickles/mess_calib_u.p")

"""
u = 150
plt.figure()
plt.plot(data1[:u, 0], vol2res(data1[:u, 1], r0), 'x', label='resistance')
plt.plot(data1[u:, 0], vol2res(data1[u:, 1], r0), 'x', label='resistance')
plt.title('Calib res')
plt.xlabel('time')
plt.ylabel('resistance')
plt.legend()

cov1 = np.cov(data1[u:, 1], data1[u:, 2])

print(cov1)
print(cov1[0, 1] / np.sqrt(cov1[1, 1] * cov1[0, 0]))

plt.figure()
plt.title('Calibration')

x = np.log(vol2res(data1[u:, 1], r0) / r0)

x_abs = (vol2res_abs(data1[u:, 1], v_abs, r0, r0_abs) / vol2res(data1[u:, 1], r0)) ** 2
x_abs += (r0_abs / r0) ** 2
x_abs = np.sqrt(x_abs)
y = 1.0 / vol2temp(data1[u:, 2], b_sens)
y_abs = 1.0 / (vol2temp(data1[u:, 2], b_sens) ** 2) * vol2temp_abs(data1[u:, 1], v_abs, b_sens, b_sens_abs)

test1 = unc.ufloat(b_sens, b_sens_abs)
test2 = unp.uarray(data1[u:, 1], v_abs)
fit, fit_err, _ = demin_regression(x, y, x_abs, y_abs, error=True)
plt.errorbar(x, y, y_abs, x * 0.02, fmt='x', label='data', ecolor="green", elinewidth=7, zorder=3, alpha=0.3)
plt.plot(x, y, 'bx', zorder=4)
fit1 = np.polyfit(x, y, 1, w=1.0 / y_abs)
plt.xlabel('log(r/r0)')
plt.ylabel('1/T')
print("lst sq", 1 / fit1)
print("deming", 1 / fit)
print(1 / fit ** 2 * fit_err)
plt.plot(x, fit1[0] * x + fit1[1], label='fit', color='r', zorder=10)
plt.plot(x, fit[0] * x + fit[1], label='deming', color='m', zorder=10)


def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0] * x + B[1]


linear = Model(f)
mydata = Data(x, y, wd=np.var(x_abs), we=np.var(y_abs))
myodr = ODR(mydata, linear, beta0=[1., 2.])
myoutput = myodr.run()
# myoutput.pprint()
plt.plot(x, f(myoutput.beta, x), ':', label="odr", zorder=10, color="orange")
print("ODR", 1 / myoutput.beta)
print(1 / myoutput.beta ** 2 * myoutput.sd_beta)

plt.legend()


import re

filenames = glob.glob(r"pickles" + "/PID*.p")
print(filenames)


def model_osc(x, a1, a2, b, k):
    return np.exp(-b * x) * (a1 * np.sin(k * x) + a2 * np.cos(-k * x))


print(model_osc(2, 3, 4, 5, 6))

for file in filenames:
    plt.figure()
    datad = pd.read_pickle(file)
    data = datad.values
    time = data[:, 0]
    temp = vol2temp(data[:, 1], b_ther) - kelvin
    file = file.split("_")[1]
    numbers = [float(i) for i in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", file)]
    ampl = np.abs(temp[0] - 26)
    p0 = [-0.5 * numbers[0] / numbers[2], -0.5 * numbers[0] / numbers[2], -0.5 * numbers[0] / numbers[2]]
    try:
        param = curve_fit(model_osc, time, temp, [ampl, ampl, *p0])
        print(param)
        plt.plot(time, model_osc(time, *param), label="model")
    except:
        print("NONONONONOOO")

    plt.plot(time, temp, label="data")
    plt.title(file)
    plt.legend()
print(datad.get(["time", "resistance"]).to_clipboard(index=False))
plt.show()
"""
