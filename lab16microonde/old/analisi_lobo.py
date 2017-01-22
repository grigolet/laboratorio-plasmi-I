# %%
import matplotlib.pyplot as plt # per i plot
import numpy as np # per manipolazione vettori
from scipy.odr import * # per i fit
import os # per leggere e listare i file
import json # per i parametri di configurazione
from math import exp
import cmath
%matplotlib
%config InlineBackend.figure_format = 'png'

x, y = np.genfromtxt('data_lobo_verticale.txt', skip_header=1, unpack=True, usecols=(0,1))
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
ax.plot(x, y, 'ro-')
