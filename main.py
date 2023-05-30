import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import statistics
import scipy

# plt.style.use('extensys')

def roundup(x):
    """Returns the input value rounded up to one significant figure."""
    if int(np.log10(x)) == np.log10(x):
        y = np.log10(x)
    elif np.log10(x) < 0:
        y = int(np.log10(x)) - 1
    else:
        y = int(np.log10(x))

    if int(x * (10 ** (-y))) * 10 ** y != x:
        return int(x * (10 ** (-y)) + 1) * 10 ** y
    else:
        return x


def voltamp_characteristic(V,sig):
    I = V * sig
    return I

def hall_voltage(I, s):
    V = I * s
    return V

def temp_dep(beta, a, b):
    lVh = a * beta + b
    return lVh

# Determining the conductivity of the first plate

sigma1arr = []
sigma1errarr = []

for file in os.listdir('Measurements/Hallvoltage_vs_current_first_plate'):
    if '0A' in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_current_first_plate/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1"}, inplace=True)
        data.dropna(inplace=True)
        popt, pcov = curve_fit(voltamp_characteristic, data.VB1, data.IA1, maxfev=500000, sigma=0.00003*np.ones(len(data.IA1)))
        sigma1arr.append(2*popt[0])
        sigma1errarr.append(2*np.sqrt(np.diag(pcov))[0])
        plt.errorbar(data.VB1, data.IA1, xerr=0.003, yerr=0.00003, label="Data")
        plt.plot(data.VB1, voltamp_characteristic(data.VB1, *popt), label="Fit")
        plt.xlabel("Voltage[V]")
        plt.ylabel("Current[A]")
        plt.legend()
        plt.savefig("Figures/" + file + ".png", dpi=500)
        plt.close()
        # plt.show()
weights = 1 / np.power(sigma1errarr,2)
sigma1 = np.sum(np.multiply(weights, sigma1arr)) / np.sum(weights)
sigma1err = np.sqrt(1/np.sum(weights))

# Determiningn the conductivity of the second plate

sigma2arr = []
sigma2errarr = []

for file in os.listdir('Measurements/Hallvoltage_vs_current_second_plate'):
    if '0A' in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_current_second_plate/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1"}, inplace=True)
        data.dropna(inplace=True)
        popt, pcov = curve_fit(voltamp_characteristic, data.VB1, data.IA1, maxfev=500000, sigma=0.00003*np.ones(len(data.IA1)))
        sigma2arr.append(2*popt[0])
        sigma2errarr.append(2*np.sqrt(np.diag(pcov))[0])
        plt.errorbar(data.VB1, data.IA1, xerr=0.003, yerr=0.00003, label="Data")
        plt.plot(data.VB1, voltamp_characteristic(data.VB1, *popt), label="Fit")
        plt.xlabel("Voltage[V]")
        plt.ylabel("Current[A]")
        plt.legend()
        plt.savefig("Figures/" + file + ".png", dpi=500)
        plt.close()
        # print(file + str(popt))
        # plt.show()

weights = 1 / np.power(sigma2errarr,2)
sigma2 = np.sum(np.multiply(weights, sigma2arr)) / np.sum(weights)
sigma2err = np.sqrt(1/np.sum(weights))

# Determining the slope of the Hall voltage for the first plate for different values of magnetic field

mu1array = []
mu1errarray = []

for file in os.listdir('Measurements/Hallvoltage_vs_current_first_plate'):
    if '1A' in file or '2A' in file or '3A' in file or '4A' in file or '5A' in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_current_first_plate/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1"}, inplace=True)
        data.dropna(inplace=True)
        data.Mag = data.Mag * 1e-3
        popt, pcov = curve_fit(hall_voltage, data.IA1, data.VB2, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)))
        mu1 = -popt[0] * sigma1 * 1e-3 / np.average(data.Mag)
        mu1array.append(mu1)
        mu1errarray.append(np.abs(mu1) * np.sqrt((np.sqrt(np.diag(pcov))[0]/popt[0])**2 + (sigma1/sigma1err)**2 + (0.5e-3/np.average(data.Mag))**2))
        plt.errorbar(data.IA1, data.VB2, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(data.IA1, hall_voltage(data.IA1, *popt), label="Fit")
        plt.ylabel("Voltage[V]")
        plt.xlabel("Current[A]")
        plt.legend()
        plt.title(str(np.average(data.Mag)))
        plt.savefig("Figures/" + file + ".png", dpi=500)
        plt.close()
        # print(file + str(popt))
        # plt.show()

weights = np.divide(1, np.power(mu1errarray,2))
mu1 = np.sum(np.multiply(weights, mu1array)) / np.sum(weights)
mu1err = np.sqrt(1/np.sum(weights))
n1 = 1 / mu1 * sigma1 / scipy.constants.e
n1err = np.abs(n1) * np.sqrt((mu1/mu1err)**2 + (sigma1/sigma1err)**2)

# Determining the slope of the Hall voltage the second plate for different values of magnetic field

mu2array = []
mu2errarray = []
for file in os.listdir('Measurements/Hallvoltage_vs_current_second_plate'):
    if '1A' in file or '2A' in file or '3A' in file or '4A' in file or '5A' in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_current_second_plate/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1"}, inplace=True)
        data.dropna(inplace=True)
        data.Mag = data.Mag * 1e-3
        popt, pcov = curve_fit(hall_voltage, data.IA1, data.VB2, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)))
        mu2 = -popt[0] * sigma2 * 1e-3 / np.average(data.Mag)
        mu2array.append(mu2)
        mu2errarray.append(np.abs(mu2) * np.sqrt((np.sqrt(np.diag(pcov))[0]/popt[0])**2 + (sigma2/sigma2err)**2 + (0.5e-3/np.average(data.Mag))**2))
        plt.errorbar(data.IA1, data.VB2, xerr=0.00003, yerr=0.003, label="Data")
        plt.errorbar(data.IA1, data.VB2, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(data.IA1, hall_voltage(data.IA1, *popt), label="Fit")
        plt.ylabel("Voltage[V]")
        plt.xlabel("Current[A]")
        plt.legend()
        plt.title(str(np.average(data.Mag)))
        plt.savefig("Figures/" + file + ".png", dpi=500)
        plt.close()
        # print(file + str(popt))
        #plt.show()

weights = np.divide(1, np.power(mu2errarray,2))
mu2 = np.sum(np.multiply(weights, mu2array)) / np.sum(weights)
mu2err = np.sqrt(1/np.sum(weights))
n2 = 1 / mu2 * sigma2 / scipy.constants.e
n2err = np.abs(n2) * np.sqrt((mu2/mu2err)**2 + (sigma2/sigma2err)**2)

# Determining the band gap based on the temperature measurements for the first plate

for file in os.listdir('Measurements/Hallvoltage_vs_temperature/mag_curr_5A'):
    if "csv" in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_temperature/mag_curr_5A/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1", "Voltage U_A2 / V": "tvolt"}, inplace=True)
        data.drop(data[data.VB2==0].index, inplace=True)
        data.dropna(inplace=True)
        lVh = np.log(np.abs(data.VB2))
        beta = 1 / (scipy.constants.k * (data.tvolt * 100 + 273.15))
        popt, pcov = curve_fit(temp_dep, lVh, beta, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)), p0=[1e20, 1e20])
        # print(data)
        plt.errorbar(lVh, beta, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(lVh, temp_dep(lVh, *popt), label="Fit")
        plt.ylabel("Logarithnm of Voltage")
        plt.xlabel("Beta")
        plt.legend()
        plt.savefig("Figures/" + file + ".png", dpi=500)
        plt.close()
        # plt.title(str(np.average(data.Mag)))
        # print(file + str(popt))
        #plt.show()

for file in os.listdir('Measurements/Hallvoltage_vs_temperature/mag_curr_5A'):
    if "csv" in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_temperature/mag_curr_4A/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1", "Voltage U_A2 / V": "tvolt"}, inplace=True)
        data.dropna(inplace=True)
        data.drop(data[data.VB2==0].index, inplace=True)
        lVh = np.log(np.abs(data.VB2))
        beta = 1 / (scipy.constants.k * (data.tvolt * 100 + 273.15))
        popt, pcov = curve_fit(temp_dep, lVh, beta, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)), p0=[1e20, 1e20])
        plt.errorbar(lVh, beta, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(lVh, temp_dep(lVh, *popt), label="Fit")
        plt.ylabel("Logarithnm of Voltage")
        plt.xlabel("Beta")
        plt.legend()
        # plt.title(str(np.average(data.Mag)))
        # print(file + str(popt))
        #plt.show()

# Determining the band gap based on the temperature measurements for the second plate

output = 'Conductivity of the first plate: ' + str(sigma1) + "±" + str(sigma1err) + "\n"
output += 'Conductivity of the second plate: ' + str(sigma2) + "±" + str(sigma2err) + "\n"
output += 'Charge mobility of the first plate: ' + str(mu1) + "±" + str(mu1err) + "\n"
output += 'Dopant density of the first plate: ' + str(n1) + "±" + str(n1err) + "\n"
output += 'Charge mobility of the second plate: ' + str(mu2) + "±" + str(mu2err) + "\n"
output += 'Dopant density of the second plate: ' + str(n2) + "±" + str(n2err) + "\n"


with open('output.txt', 'w') as f:
    f.write(output)