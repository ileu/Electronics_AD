# -*- coding: utf-8 -*-
"""
@author: Ueli5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal as signal
from scipy.fftpack import fft
from scipy.optimize import curve_fit
import matplotlib as mpl
import glob
import pandas as pd

np.set_printoptions(linewidth=160)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

r0 = 1e5
r0_abs = 1e3
t0 = 25
b_ther = 4396.644
b_ther_abs = 33.831
kelvin = 273.15

b_sens = 4275
b_sens_abs = 25

v_abs = 2

t0 += kelvin

grosse = 16


def demin_regression(x, y, xerr, yerr, alpha=0.05, error=False, conf=False):
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

        if conf:
            x_conf = sr * b1 * (y - (b0 + b1 * x)) / (1.0 + sr * b1 ** 2)
            y_conf = -1 * (y - (b0 + b1 * x)) / (1.0 + sr * b1 ** 2)
            return np.array([b1, b0]), np.array([std_b1, std_b0]) * tna, tna, np.array([x_conf, y_conf])

        return np.array([b1, b0]), np.array([std_b1, std_b0]) * tna, tna
    return np.array([b1, b0])


def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.

    INPUTS:
        - Single prediction,
        - y_test
        - All test set predictions,
        - Prediction interval threshold (default = .95)
    OUTPUT:
        - Prediction interval for single prediction
    '''

    # get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions) ** 2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
    # get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    # generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper


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


def model_osci(t, a, tau, omega, phi, c):
    """
    für omega=phi=0 exp abfall
    :param t:
    :param a:
    :param tau:
    :param omega:
    :param phi:
    :param c:
    :return:
    """
    return a * np.exp(-t / tau) * np.cos(omega / (2 * np.pi) * t + phi) + c


def low_pass(data, alpha):
    out = data
    for i in range(1, len(out)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]

    return out


filenames = glob.glob(r"pickles" + "/*.p")
datas = {}
names = []
print(filenames)
for file in filenames:
    name = "_".join(file.split("\\")[-1].split("_")[:2])
    names.append(name)
    datas[name] = pd.read_pickle(file)
print(names)
print("data loaded")

cmap = plt.cm.get_cmap('jet_r')
data = datas["Cool_Step"].values

fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.1, 0.1, .8, .8])
spliter = [i for i in range(len(data)) if data[i - 1, -1] != data[i, -1]]
k_values = np.take(data, spliter, 0)[:, -1]
color = []
for split in np.split(data, spliter)[1:]:
    temp = vol2temp(split[:, 1], b_ther)
    bins = np.rint(temp * 10) / 10.0
    bins = np.arange(bins.min() - 0.1, temp.max() + 0.1, 0.1)
    color.append(cmap(split[0, -1] / 255))
    print(cmap(split[0, -1] / 255),split[0, -1])
    ax.hist(temp, bins, alpha=1, edgecolor='black', linewidth=1.2, color=color[-1], label=split[0, -1])

ax2 = fig.add_axes([0.91, 0.1, 0.03, 0.8])
cmap = mpl.colors.ListedColormap(color)
norm = mpl.colors.BoundaryNorm(np.arange(0, cmap.N + 1, 1), cmap.N)
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, ticks=np.arange(0, cmap.N, 1) + 0.5)
cb.set_ticklabels([str(int(k * 100 // 255)) + "%" for k in k_values])
ax2.set_title("Fan speed")

plt.savefig('plots/cool_step.png')
plt.savefig('plots/cool_step.pdf')

u = 150
cov1 = np.cov(data[u:, 1], data[u:, 2])
print(cov1)
print(cov1[0, 1] / np.sqrt(cov1[1, 1] * cov1[0, 0]))

plt.figure()
plt.title('Calibration')
data = datas["mess_calib"].values
x = np.log(vol2res(data[u:, 1], r0) / r0)
x_sample = np.linspace(x.min(), x.max(), 1000)
x_abs = (vol2res_abs(data[u:, 1], v_abs, r0, r0_abs) / vol2res(data[u:, 1], r0)) ** 2
x_abs += (r0_abs / r0) ** 2
x_abs = np.sqrt(x_abs)

y = 1.0 / vol2temp(data[u:, 2], b_sens)
y_abs = 1.0 / (vol2temp(data[u:, 2], b_sens) ** 2) * vol2temp_abs(data[u:, 1], v_abs, b_sens, b_sens_abs)

fit, fit_err, tna, [x_conf, y_conf] = demin_regression(x, y, x_abs, y_abs, error=True, conf=True)
print("deming", 1 / fit)
print(1 / fit ** 2 * fit_err)
print(1 / np.polyfit(x, y, 1))
y_fit = fit[0] * x + fit[1]
y_sample = fit[0] * x_sample + fit[1]
prediction = get_prediction_interval(y_sample, y, y_fit)

# plt.errorbar(x, y, y_abs, x * 0.02, fmt='x', label='data', ecolor="green", elinewidth=0, zorder=3, alpha=1)
plt.plot(x, y, 'x', zorder=4, label='data')
plt.fill_between(x_sample, prediction[0], prediction[2], label="95% prediction band", facecolor='r', alpha=0.3)

plt.plot(x_sample, y_sample, label='deming', color='k', zorder=10)

plt.xlabel('log(r/r0)', fontsize=grosse)
plt.ylabel('1/T', fontsize=grosse)
plt.legend()
plt.tight_layout()
plt.savefig('plots/mess_calib.png')
plt.savefig('plots/mess_calib.pdf')
"""
plt.figure()
data = datas["max_temp"].values
temp = vol2temp(data[:, 1], b_ther)
temp_abs = vol2temp_abs(data[:, 1], v_abs, b_ther, b_ther_abs)
plt.errorbar(data[:, 0], temp, yerr=temp_abs, fmt='x', ecolor="k", elinewidth=0.75, capsize=3, capthick=0.75,
             label='temperature')
plt.xlabel('time')
plt.ylabel('temperature')
plt.title('Equilibrium temp')
plt.legend()
"""
plt.figure()
data = datas["two_point"].values
temp = vol2temp(data[:, 1], b_ther)
temp_abs = vol2temp_abs(data[:, 1], v_abs, b_ther, b_ther_abs)
plt.errorbar(data[:, 0], temp, yerr=temp_abs, fmt='x', ecolor="k", elinewidth=0.75, capsize=3, capthick=0.75,
             label='temperature')
plt.xlabel('time', fontsize=grosse)
plt.ylabel('temperature', fontsize=grosse)
plt.tight_layout()
plt.savefig('plots/two_point.png')
plt.savefig('plots/two_point.pdf')
plt.legend()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for ax, name in zip(np.reshape(ax, (1, 4))[0], [s for s in names if s.startswith("PID")]):
    data = datas[name]
    data = data.values
    temp = vol2temp(data[:, 1], b_ther)
    temp_abs = vol2temp_abs(data[:, 1], v_abs, b_ther, b_ther_abs)
    time = data[:, 0] * 1e-3
    time_edges = np.arange(0, np.rint(time[-1]) + 3, 2)[1:]
    time_bins = np.digitize(time, time_edges)
    n, bin_edges = np.histogram(time, time_edges)
    test = [i for i in range(len(data)) if time_bins[i - 1] != time_bins[i]]
    param = [1,1,1,1,1] # todo
    # print(name)
    # print(time)
    # print(time_edges)
    q, v, w = stats.binned_statistic(time, temp, statistic='mean', bins=time_edges)
    v_center = (v[1:] + v[:-1]) * 0.5
    ax.plot(time, temp, 'x', zorder=1)
    ax.plot(v_center, q, 'x', color='red', zorder=3)
    ax.fill_between(v_center, q - 0.2, q + 0.2, color="green", alpha=0.3, zorder=2)
    ax.set_xlabel('time', fontsize=grosse)
    ax.set_ylabel('temperature', fontsize=grosse)
    ax.set_title(name)
    fig.tight_layout()

plt.savefig('plots/pid.png')
plt.savefig('plots/pid.pdf')

plt.figure()
data = datas["dutty_rpm"].values
plt.plot(data[:, 0] / 255 * 100, data[:, 1], 'x', label='data')
plt.xlabel('Duty cycle in %', fontsize=grosse)
plt.ylabel('RPM', fontsize=grosse)
plt.legend()
plt.tight_layout()
plt.savefig('plots/rpm.png')
plt.savefig('plots/rpm.pdf')
"""
print(names)
plt.figure()
bins = np.linspace(24, 28, 50)
for name in [s for s in names if s.startswith("Equi")]:
    print("hier")
    data = datas[name].values
    temp = vol2temp(data[:, 1], b_ther)
    bins = np.rint(temp * 10) / 10.0
    bins = np.arange(bins.min() - 0.1, temp.max() + 0.1, 0.1)
    plt.hist(temp, bins, alpha=0.5, label=name)

plt.legend()

plt.figure()
for name in [s for s in names if s.startswith("Equi")]:
    data = datas[name]
    data = data.values
    temp = vol2temp(data[:, 1], b_ther)
    temp_abs = vol2temp_abs(data[:, 1], v_abs, b_ther, b_ther_abs)
    time = data[:, 0]
    plt.plot(time, temp, 'x', label=name)

plt.xlabel('time')
plt.ylabel('temp')
plt.title('Equilibirum temp for different fan speed')
plt.legend()

plt.figure()
data = datas["Cool_Step"].values
temp = vol2temp(data[:, 1], b_ther)
plt.plot(data[:, 0], temp, 'x', label='temp')
plt.xlabel('time')
plt.ylabel('temp')
plt.title('Temperature cooling with different speeds6 ')

plt.figure()
plt.plot(data6[:, 2], data6[:, 1], 'x', label="data")
plt.xlabel('duty cylce')
plt.ylabel('temp')
plt.title('Temperature cooling with different speeds')
plt.legend()
"""
plt.show()
