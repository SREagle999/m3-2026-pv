# example with a logistic function predicting house category growth in Manchester (2024 given data)
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# data
manchester = pd.read_excel(".../datasets/M3_2024_data.xlsx", sheet_name="Manchester",skiprows=1,nrows=30)
data = manchester.values.T
x = np.array(data[0], dtype=float)
y = np.array(data[2:], dtype=float)

# Logistic Function
def logistic(x, l, k, x0):
    return l / (1 + np.exp(-k * (x - x0)))

# plot function - in a normal logistic function, the carrying capacity would be found
def plot_logistic(y, initial_guess,col):
    y = np.array(y, dtype=float)
    popt, pcov = curve_fit(logistic, x, y, p0=initial_guess, maxfev=1000) # optimal values and covariance matrix
    a,b,c = popt
    while a <= y.max():   # arbitrary carrying capacity fix for lack of actual online research
      a += 1000
    y_pred = logistic(x,a,b,c)

    plt.figure(dpi=500,figsize=(10,5))
    plt.scatter(x,y,color="purple")
    plt.plot(x,y_pred,color=col)

    x_new = np.array(np.linspace(2022,2084,62))
    y_new_pred = logistic(x_new,a,b,c)
    plt.plot(x_new,y_new_pred,color=col,linestyle="dashed")

plot_logistic(data[2], initial_guess=[3000,1,1992], col="blue")
plot_logistic(data[3], initial_guess=[95000,1,1992], col="red")
plot_logistic(data[4], initial_guess=[95000,1,1992], col="green")
plot_logistic(data[5], initial_guess=[70000,1,1992], col="green")
plt.show()