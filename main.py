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
        sigma1arr.append(4*popt[0])
        sigma1errarr.append(4*np.sqrt(np.diag(pcov))[0])
        plt.errorbar(data.VB1, data.IA1, xerr=0.003, yerr=0.00003, label="Data")
        plt.plot(data.VB1, voltamp_characteristic(data.VB1, *popt), label="Fit")
        plt.xlabel("Voltage[V]")
        plt.ylabel("Current[A]")
        plt.legend()
        plt.savefig("Figures/" + file + ".png", dpi=500)
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
        sigma2arr.append(4*popt[0])
        sigma2errarr.append(4*np.sqrt(np.diag(pcov))[0])
        plt.errorbar(data.VB1, data.IA1, xerr=0.003, yerr=0.00003, label="Data")
        plt.plot(data.VB1, voltamp_characteristic(data.VB1, *popt), label="Fit")
        plt.xlabel("Voltage[V]")
        plt.ylabel("Current[A]")
        plt.legend()
        # print(file + str(popt))
        # plt.show()

weights = 1 / np.power(sigma2errarr,2)
sigma2 = np.sum(np.multiply(weights, sigma2arr)) / np.sum(weights)
sigma2err = np.sqrt(1/np.sum(weights))

# Determining the slope of the Hall voltage for the first plate for different values of magnetic field

for file in os.listdir('Measurements/Hallvoltage_vs_current_first_plate'):
    if '1A' in file or '2A' in file or '3A' in file or '4A' in file or '5A' in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_current_first_plate/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1"}, inplace=True)
        data.dropna(inplace=True)
        popt, pcov = curve_fit(hall_voltage, data.IA1, data.VB2, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)))
        plt.errorbar(data.IA1, data.VB2, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(data.IA1, hall_voltage(data.IA1, *popt), label="Fit")
        plt.ylabel("Voltage[V]")
        plt.xlabel("Current[A]")
        plt.legend()
        plt.title(str(np.average(data.Mag)))
        # print(file + str(popt))
        # plt.show()

# Determining the slope of the Hall voltage the second plate for different values of magnetic field

for file in os.listdir('Measurements/Hallvoltage_vs_current_second_plate'):
    if '1A' in file or '2A' in file or '3A' in file or '4A' in file or '5A' in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_current_second_plate/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1"}, inplace=True)
        data.dropna(inplace=True)
        popt, pcov = curve_fit(hall_voltage, data.IA1, data.VB2, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)))
        plt.errorbar(data.IA1, data.VB2, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(data.IA1, hall_voltage(data.IA1, *popt), label="Fit")
        plt.ylabel("Voltage[V]")
        plt.xlabel("Current[A]")
        plt.legend()
        plt.title(str(np.average(data.Mag)))
        # print(file + str(popt))
        #plt.show()
    
plt.show()
# Determining the band gap based on the temperature measurements for the first plate

for file in os.listdir('Measurements/Hallvoltage_vs_temperature/mag_curr_5A'):
    if "csv" in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_temperature/mag_curr_5A/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1", "Voltage U_A2 / V": "tvolt"}, inplace=True)
        data.drop(data[data.VB2==0].index, inplace=True)
        data.dropna(inplace=True)
        lVh = np.log(np.abs(data.VB2))
        beta = 1 / (scipy.constants.k * (data.tvolt * 100 + 273.15))
        popt, pcov = curve_fit(hall_voltage, lVh, beta, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)))
        # print(data)
        plt.errorbar(lVh, beta, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(lVh, hall_voltage(lVh, *popt), label="Fit")
        plt.ylabel("Logarithnm of Voltage")
        plt.xlabel("Beta")
        plt.legend()
        # plt.title(str(np.average(data.Mag)))
        # print(file + str(popt))
        #plt.show()

for file in os.listdir('Measurements/Hallvoltage_vs_temperature/mag_curr_5A'):
    if "csv" in file:
        data = pd.read_csv('Measurements/Hallvoltage_vs_temperature/mag_curr_4A/' + file)
        data.rename(columns={"Magn. flux density B_A1 / mT": "Mag", "Voltage U_B2 / V": "VB2", "Voltage U_B1 / V": "VB1", "Current I_A1 / A": "IA1", "Voltage U_A2 / V": "tvolt"}, inplace=True)
        data.dropna(inplace=True)
        lVh = np.log(data.VB2)
        beta = 1 / (scipy.constants.k * (data.tvolt * 100 + 273.15))
        popt, pcov = curve_fit(hall_voltage, lVh, beta, maxfev=500000, sigma=0.003*np.ones(len(data.VB2)))
        plt.errorbar(lVh, beta, xerr=0.00003, yerr=0.003, label="Data")
        plt.plot(lVh, hall_voltage(lVh, *popt), label="Fit")
        plt.ylabel("Logarithnm of Voltage")
        plt.xlabel("Beta")
        plt.legend()
        # plt.title(str(np.average(data.Mag)))
        # print(file + str(popt))
        #plt.show()

# Determining the band gap based on the temperature measurements for the second plate

output = 'Conductivity of the first plate: ' + str(sigma1) + "±" + str(sigma1err) + "\n"
output += 'Conductivity of the second plate: ' + str(sigma2) + "±" + str(sigma2err) + "\n"
# output += 

with open('output.txt', 'w') as f:
    f.write(output)