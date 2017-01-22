import numpy as np
from scipy.odr import * # per i fit
import matplotlib.pyplot as plt
from math import sin, cos

# funzione per calcolare la lunghezza d'onda in guida
def calc_lambda_guida(nu, a, c):
    first = (nu/c)**2
    second = (1/(2*a))**2
    return (first - second)**(-1/2)

#  definisco la funzione di fit
def fit_function(params, x):
    return params[4]*(params[0]**2 + params[1]**2 + 2*params[0]*params[1]*np.cos(2*params[2]*x + params[3])) + params[5]

# Calcolo del rapporto d'onda stazionaria
def rapporto_onda_stazionaria(U_F, U_B):
    return abs(U_F + U_B) / abs(U_F - U_B)

# Calcolo impedenza del carico
def impedenza_guida_onda(lambda_guida, lambda_vuoto, a, b):
    return 377*(lambda_guida/lambda_vuoto)*(2*b/a)

# Calcolo dell'impedenza del carico
def impedenza_carico(impedenza_guida_onda, U_F, U_B, delta_phi):
    cos_phi = float(cos(delta_phi))
    sin_phi = float(sin(delta_phi))
    real = impedenza_guida_onda * (U_F**2 - U_B**2) / (U_F**2 + U_B**2 - 2*U_F*U_B*cos_phi)
    imaginary = impedenza_guida_onda * (2*U_F*U_B*sin_phi) / (U_F**2 + U_B**2 - 2*U_F*U_B*cos_phi)
    return [real, imaginary]

'''
Preso data che e' il nome del file intestato come x, y, xerr, yerr e params che è un array dei parametri
iniziali, restituisce i valori dei parametri stimati. Se non sono presenti gli errori nel file bisogna
indicare use_custom_errors e x_error, y_error
I parametri stimati comprendono:
chi_2
dof
U_F
sigma_U_F
U_B
sigma_U_B
k
sigma_k
delta_phi
sigma_delta_phi
amplitude
sigma_amplitude
parameters'''
def execute_fit(data, params, use_custom_errors=True, x_error=0.05, y_error=0.5, ifixb=(1,1,1,1,1,1)):
    xerr = np.array([])
    yerr = np.array([])

    if use_custom_errors:
        x_data, y_data = np.genfromtxt(data, unpack=True)
        xerr = np.full( (1, np.size(x_data)), 1/x_error**2)
        yerr = np.full( (1, np.size(y_data)), 1/y_error**2)
    else:
        x_data, y_data, xerr, yerr = np.genfromtxt(data, unpack=True)

    fit_model = Model(fit_function)
    fit_data = Data(x_data, y_data, wd=xerr, we=yerr)
    fit_odr = ODR(fit_data, fit_model, beta0=params, ifixb=ifixb)
    fit_output = fit_odr.run()

    return_params = {'chi_2': fit_output.sum_square, 'dof': np.size(x_data) - 1, 'U_F': fit_output.beta[0],
                     'sigma_U_F': fit_output.sd_beta[0], 'U_B': fit_output.beta[1], 'sigma_U_B': fit_output.sd_beta[1],
                     'k': fit_output.beta[2], 'sigma_k': fit_output.sd_beta[2], 'delta_phi': fit_output.beta[3],
                     'sigma_delta_phi': fit_output.sd_beta[3], 'amplitude': fit_output.beta[4],
                     'sigma_amplitude': fit_output.sd_beta[4], 'q': fit_output.beta[5],
                     'sigma_q': fit_output.sd_beta[5]}

    return return_params

def plot_data(file_name, params, title='', use_custom_errors=True, x_error=0.05, y_error=0.5):
    xerr = np.array([])
    yerr = np.array([])

    # definisco la funzione di fit
    if use_custom_errors:
        x_data, y_data = np.genfromtxt(file_name, unpack=True)
        xerr = np.linspace(x_error, x_error, np.size(x_data))
        yerr = np.linspace(y_error, y_error, np.size(y_data))
    else:
        x_data, y_data, xerr, yerr = np.genfromtxt(file_name, unpack=True)

    x_fit = np.arange(x_data[0], x_data[-1], (x_data[-1] - x_data[0])/1000.)
    amplitude = params['amplitude']
    U_F = params['U_F']
    U_B = params['U_B']
    k = params['k']
    q = params['q']
    sigma_U_F = params['sigma_U_F']
    sigma_U_B = params['sigma_U_B']
    sigma_k = params['sigma_k']
    delta_phi = params['delta_phi']
    sigma_delta_phi = params['sigma_delta_phi']
    sigma_amplitude = params['sigma_amplitude']
    sigma_q = params['sigma_amplitude']
    chi_2_ridotto = params['chi_2']/params['dof']

    parameters_text = """
        $U_f$  = {} $\pm$ {}
        $U_b$  = {} $\pm$ {}
        $k$    = {} $\pm$ {}
        $\Delta \phi$ = {} $\pm$ {}
        $A$    = {} $\pm$ {}
        $q$    = {} $\pm$ {}
        $\chi_2/dof$    = {}
    """.format(round(U_F, 2), round(sigma_U_F, 2), round(U_B, 2), round(sigma_U_B, 2), round(k, 2), round(sigma_k, 2),
               round(delta_phi, 2), round(sigma_delta_phi, 2), round(amplitude, 2), round(sigma_amplitude, 2),
               round(q, 2), round(sigma_q, 2), round(chi_2_ridotto, 2))

    y_fit = fit_function([U_F, U_B, k, delta_phi, amplitude, q], x_fit)
    fig, ax = plt.subplots()
    ax.errorbar(x_data, y_data, xerr=xerr, yerr=yerr,  fmt='+', label="Data")
    ax.annotate(parameters_text, xy=(.75,.75), xycoords='axes fraction', fontsize=14,
            horizontalalignment='left',
            verticalalignment='top')
    ax.plot(x_fit, y_fit, label="Fit")
    ax.grid()
    ax.legend()
    ax.set_title(title);
    ax.set_ylabel('mV')
    ax.set_xlabel('cm')
    return fig











