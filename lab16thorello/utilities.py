import numpy as np
from scipy.odr import * # per i fit
import matplotlib.pyplot as plt

'''
Definizione della funzione di fit
'''
def resistenza_fitting_function(B, x):
    return B[0] + B[1]*x


'''
Preso il nome del file con la carrateristica del filamento, restituisce la resistenza stimata dal fit.
'''
def resistenza_filamento(file_name, x_error=1, y_error=0.02):
    x, y = np.genfromtxt(file_name, unpack=True, skip_header=2)
    x_err = np.full((1, np.size(x)), 1 / x_error ** 2)
    y_err = np.full((1, np.size(y)), 1 / y_error ** 2)
    linear = Model(resistenza_fitting_function)
    data = Data(x, y, wd=x_err, we=y_err)
    odr = ODR(data, linear, beta0=[1.,1.])
    output = odr.run()
    return {
        'q': output.beta[0],
        'R': output.beta[1],
        'sigma_q': output.sd_beta[0],
        'sigma_R': output.sd_beta[1],
        'chi2': output.sum_square,
        'dof': np.size(x) - 2
    }

def plot_data(file_name, skip_header=2, title='', label='data', x_label='', y_label=''):
    x, y = np.genfromtxt(file_name, unpack=True, skip_header=skip_header)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label=label)
    ax.grid()
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plot_caratteristica_plasma(file_name, file_name_2 = None, title='', x_label=None, y_label=None, notes=None):
    if not x_label:
        x_label = '$V_{bobina} (V)$'
    if not y_label:
        y_label = '$I_{bobina}$ (A)'

    x, y = np.genfromtxt(file_name, unpack=True)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    label = file_name.split('_')[1][:-4]
    ax.plot(x, y, 'o', label=label)

    if file_name_2:
        x_2, y_2 = np.genfromtxt(file_name_2, unpack=True)
        label_2 = file_name.split('_')[1][:-4]
        ax.plot(x_2, y_2, 'o', label=label_2)

    if notes:
        ax.annotate(notes, xy=(.75, .75), xycoords='axes fraction', fontsize=14,
                    horizontalalignment='left',
                    verticalalignment='top')

    ax.legend()

    return fig