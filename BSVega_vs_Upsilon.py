# -*- coding: utf-8 -*-
"""
Spyder Editor

Code written by Samuel Drapeau with modifications by Johannes Wiesel and Jan Obloj

This file produces plots comparing our first order sensitivity with BS vega.
"""

# %%


# To run the stuff, you need the package plotly in your anaconda "conda install plotly"

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
init_notebook_mode()
pio.renderers.default='svg'

import numpy as np
import numpy.random
import pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize

import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time.time())

def toc(fmt="Elapsed: %s s"):
    print(fmt % (time.time() - _tstart_stack.pop()))


# %%
# We first provide the computation of a call option according to BS (we assume Log normal distribution)
# definition of the dplus and minus functions
# and the BS formula. 

def dplus(S, K, T, sigma):
    sigmaT = sigma * T ** 0.5
    return np.log(S/K)/sigmaT + sigmaT/2

def dminus(S, K, T, sigma):
    sigmaT = sigma * T ** 0.5
    return np.log(S/K)/sigmaT - sigmaT/2

def BS(S, K, T, sigma, Type = 1):
    factor1 = S * norm.cdf(Type * dplus(S, K, T, sigma))
    factor2 = K * norm.cdf(Type * dminus(S, K, T, sigma))
    return Type * (factor1 - factor2)

# Now we provide the computation for the exact call according to the computations in BDT
# We take p = 2

def Robust_Call_Exact_fun(S, K, T, sigma, delta):
    def fun(v): #v[0] = a, v[1] = lambda
        price = BS(S,max(K - (2 * v[0] + 1)/ (2 * v[1]),0.000001), T, sigma)
        return price + v[0] ** 2 / (2 * v[1]) + 0.5 * v[1] * delta ** 2
    def cons_fun(v): # the value of v[0] should be constrained to keep strike positive
        tmp = K - (2 * v[0] + 1)/ (2 * v[1])
        return tmp
    
    cons = ({'type': 'ineq', 'fun' : cons_fun})
    guess = np.array([0, 1])
    bounds = ((-np.Inf, np.Inf), (0, np.Inf))
    res = minimize(fun, guess,
                        constraints=cons,
                        method='SLSQP',
                        bounds=bounds
                  )
    return res.fun
Robust_Call_Exact = np.vectorize(Robust_Call_Exact_fun)

# Now we provide the computation for the first order model uncertainty sensitivity (Upsilon)
# and the resulting BS robust price approximation
# We take p = 2

def Robust_Call_Upsilon(S, K, T, sigma, delta):
    muK = norm.cdf(dminus(S, K, T, sigma))
    correction = np.sqrt(muK * (1-muK))
    return correction

def Robust_Call_Approximation(S, K, T, sigma, delta):
    price = BS(S, K, T, sigma)
    correction = Robust_Call_Upsilon(S, K, T, sigma, delta)
    return price + correction * delta


# %%
# Ploting the robust call and FO appriximation for a given strike and increasing uncertainty radius

S = 1
K = 1.2
T = 1
sigma = 0.2
Delta = np.linspace(0, 0.2, 50)


Y0 = BS(S, K, T, sigma)
Y1 = Robust_Call_Approximation(S, K, T, sigma, Delta)
Y2 = Robust_Call_Exact(S, K, T, sigma, Delta)

fig = go.Figure()
fig.add_scatter(x = Delta, y = Y1, name = 'FO')
fig.add_scatter(x = Delta, y = Y2, name = 'RBS')
#fig.layout.title = "Exact Robust Call vs First Order Approx: Strike K="+str(K)+", BS Price="+str(np.round(Y0,4))
fig.layout.xaxis.title = "delta"
fig.layout.yaxis.title = "Price"

iplot(fig)

# %%
# Ploting the robust call and FO appriximation for a given radius of uncertainty and a range of strikes

S = 1
K = np.linspace(0.6, 1.4, 100)
T = 1
sigma = 0.2
delta = 0.05


Y0 = Robust_Call_Approximation(S, K, T, sigma, delta)
Y1 = Robust_Call_Exact(S, K, T, sigma, delta)
Y2 = BS(S, K, T, sigma)

fig = go.Figure()
fig.add_scatter(x = K, y = Y0, name = 'FO')
fig.add_scatter(x = K, y = Y1, name = 'Exact')
fig.add_scatter(x = K, y = Y2, name = 'BS')
fig.layout.title = "Call Price vs Exact Robust Call and First Order Approx : delta ="+str(delta)
fig.layout.xaxis.title = "Strike"
fig.layout.yaxis.title = "Price"

iplot(fig)

# %%
# Run a plot to comapre BS Vega and BS Upsilon (Uncertainty Sensitivity)
# Plots show the sensitivities

S = 1
K = np.linspace(0.4 * S, 2 * S, 100)
T = 1
sigma = 0.2
delta = 0.02 #is irrelevant here

Y1 = S * (norm.pdf(dplus(S, K , T, sigma)))
Y0 = S * (Robust_Call_Upsilon(S, K, T, sigma, delta))

fig = go.Figure()
fig.add_scatter(x = K, y = Y0, name = 'BS Upsilon')
fig.add_scatter(x = K, y = Y1, name = 'BS Vega')
#fig.layout.title = "Call Price Sensitivity: Vega vs Upsilon, sigma= "+str(sigma)
fig.layout.xaxis.title = "Strike"
fig.layout.yaxis.title = "Price"



iplot(fig)

# %%
# Run a ploting to comapre BS Vega and BS Upsilon (Uncertainty Sensitivity)
# Plots show the sensitivities

S = 1
K = np.linspace(0.6 * S, 1.4 * S, 100)
T = 1
sigma = 0.2
delta = 0.02 #is irrelevant here

Y0 = S * (norm.pdf(dplus(S, K * np.exp(T * sigma ** 2), T, sigma)) + 1/2-1/np.sqrt(2 * np.pi))
Y1 = S * (Robust_Call_Upsilon(S, K, T, sigma, delta))

fig = go.Figure()
fig.add_scatter(x = K, y = Y0, name = 'BS Vega (shifted) + const')
fig.add_scatter(x = K, y = Y1, name = 'BS Upsilon')
fig.layout.title = "Call Price Sensitivity: Vega vs Upsilon, sigma= "+str(sigma)
fig.layout.xaxis.title = "Strike"
fig.layout.yaxis.title = "Price"



iplot(fig)





