# %%
import matplotlib.pyplot as plt # per i plot
import numpy as np # per manipolazione vettori
from scipy.odr import * # per i fit

# %matplotlib
# %config InlineBackend.figure_format = 'svg'

# %%
#x_aperto, y_aperto, sigma_x, sigma_y = np.genfromtxt('data_aperto.txt', skip_header=1, unpack=True)
x_adattato, y_adattato = np.genfromtxt('data_adattato.txt', skip_header=1, unpack=True)
x_riflettente, y_riflettente = np.genfromtxt('data_riflettente.txt', skip_header=1, unpack=True)
x_horn, y_horn = np.genfromtxt('data_horn.txt', skip_header=1, unpack=True)

plt.grid(True)
#plt.plot(x_aperto, y_aperto, 'ro-', label='aperto')
plt.plot(x_adattato, y_adattato, 'bo-', label='adattato')
plt.plot(x_riflettente, y_riflettente, 'go-', label='riflettente')
plt.plot(x_horn, y_horn, 'yo-', label='horn')
plt.legend()
plt.ylim(0, 45)

# %%
# Variabili fissate
lambda_guida = 3.65
lambda_vuoto = 2.85
a            = 2.29
b            = 1.01

# definisco la funzione di fit
def fit_function(params, x):
    return params[4]*(params[0]**2 + params[1]**2 + 2*params[0]*params[1]*np.cos(2*params[2]*x + params[3]))

# Calcolo del rapporto d'onda stazionaria
def rapporto_onda_stazionaria(U_F, U_B):
    return (U_F + U_B) / (U_F - U_B)

# Calcolo impedenza del carico
def impedenza_guida_onda(lambda_guida, lambda_vuoto, a, b):
    return 377*(lambda_guida/lambda_vuoto)*(2*b/a)

# Calcolo dell'impedenza del carico
def impedenza_carico(impedenza_guida_onda, U_F, U_B, delta_phi):
    cos_phi = float(cos(delta_phi))
    sin_phi = float(sin(delta_phi))
    real = impedenza_guida_onda * (U_F**2 - U_B**2) / (U_F**2 + U_B**2 - 2*U_F*U_B*cos_phi)
    imaginary = impedenza_guida_onda * (2*U_F*U_B*sin_phi) / (U_F**2 + U_B**2 - 2*U_F*U_B*cos_phi)
    return real, imaginary


sigma_x = 0.05 # cm
sigma_u = 0.1 # mV
waves = Model(fit_function)
data = Data(x_aperto, y_aperto, wd=np.full( (1, np.size(x_aperto)), 1/sigma_x**2),\
        we=np.full((1, np.size(y_aperto)), 1/sigma_u**2))
myodr = ODR(data, waves, beta0=[5.4, 1.12, 1.72, 3., .5], ifixb=[1,1,1,1,0])
myoutput = myodr.run()
myoutput.pprint()
chi_2     = myoutput.sum_square
dof       = np.size(x_aperto) - 1;
U_F       = myoutput.beta[0]
sigma_U_F = myoutput.sd_beta[0]
U_B        = myoutput.beta[1]
sigma_U_B   = myoutput.sd_beta[1]
k        = myoutput.beta[2]
sigma_k   = myoutput.sd_beta[2]
delta_phi        = myoutput.beta[3]
sigma_delta_phi = myoutput.sd_beta[3]
amplitude        = myoutput.beta[4]
sigma_amplitude = myoutput.sd_beta[4]
parameters = [U_F, U_B, k, delta_phi, amplitude]

x = np.linspace(8, 13, 200)
y = fit_function(parameters, x)
plt.plot(x, y, label='Fit')
plt.plot(x_aperto, y_aperto, label='Dati')
plt.grid()
plt.legend()

swr = rapporto_onda_stazionaria(U_F, U_B)
impedenza_onda = impedenza_guida_onda(lambda_guida, lambda_vuoto, a, b)
impedenza_totale_reale, impedenza_totale_immaginaria = impedenza_carico(impedenza_onda, U_F, U_B, delta_phi)

print(swr)
print(impedenza_onda)
print('{} + i {}'.format(impedenza_totale_reale, impedenza_totale_immaginaria))
