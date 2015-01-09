import logging

import numpy as np
import matplotlib.pyplot as plt

from periodogram import acf

logging.basicConfig(level=logging.WARNING)

def make_sine(xmin=1,xmax=100,cadence=0.02043423,shifts=[0],amplitudes=[1.]):
    x = arange(xmin,xmax,cadence)
    y = np.ones(len(x))
    for i,shift in enumerate(shifts):
        y *= np.sin(x + shift)*amplitudes[i]
    y += np.random.random(len(x))*0.5
    return x,y

def plot_test(x,y,input_period,fit_period):
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x,y,"k.")
    for per in np.arange(1,ax.get_xlim()[1],input_period):
        ax.plot([per,per],ax.get_ylim(),"g--")
    for per in np.arange(1,ax.get_xlim()[1],fit_period):
        ax.plot([per,per],ax.get_ylim(),"r-")

def test_single_sine():
    x,y = make_sine(xmax=60)
    a = acf.acf()
    a.fit(x,y)
    period,period_err = a.period_search()
    print "Input = 2*pi = {}".format(2*np.pi)
    print period,period_err
    plot_test(x,y,2*np.pi,period)

def test_double_sine():
    x,y = make_sine(xmax=60,shifts=[0.,np.pi/5.],amplitudes=[1.,0.9])
    a = acf.acf()
    a.fit(x,y)
    period,period_err = a.period_search()
    print "Input = 2*pi = {}".format(2*np.pi)
    print period,period_err
    plot_test(x,y,2*np.pi,period)


plt.close("all")
test_single_sine()
test_double_sine()
