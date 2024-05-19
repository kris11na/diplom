# -*- coding: utf-8 -*-
"""Kendall.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QpnihI2TKE3G52grKCVEernlYyRgiDbr
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data_raw_germ_DAX = pd.read_html('https://en.wikipedia.org/wiki/DAX', match='Ticker')

data_raw_germ_DAX = data_raw_germ_DAX[0]
data_raw_germ_DAX.head()

data_raw_germ_DAX = data_raw_germ_DAX.sort_values('Index weighting (%)1', ascending=False).iloc[:30]

stocks_DAX = {}
for index, row in data_raw_germ_DAX.iterrows():
    ticker = str(row['Ticker'])
    ticker_dax_data = yf.download(ticker, start="2023-01-01", end="2023-12-31", progress=False)
    if ticker_dax_data.shape[0] != 0:
        stocks_DAX[ticker] = ticker_dax_data

data_raw_GB_FTSE100 = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index', match='Ticker')

data_raw_GB_FTSE100 = data_raw_GB_FTSE100[0]
data_raw_GB_FTSE100['Ticker'] = data_raw_GB_FTSE100['Ticker'] + '.L'
data_raw_GB_FTSE100.head()

stocks_FTSE100 = {}
for index, row in data_raw_GB_FTSE100.iterrows():
    ticker = str(row['Ticker'])
    ticker_ftse_data = yf.download(ticker, start="2023-01-01", end="2023-12-31", progress=False)
    if ticker_ftse_data.shape[0] != 0:
        stocks_FTSE100[ticker] = ticker_ftse_data

data_raw_Fr_CAC40 = pd.read_html('https://en.wikipedia.org/wiki/CAC_40', match='Ticker' )

data_raw_Fr_CAC40 = data_raw_Fr_CAC40[0]
data_raw_Fr_CAC40.head()

stocks_CAC40 = {}
for index, row in data_raw_Fr_CAC40.iterrows():
    ticker = str(row['Ticker'])
    ticker_cac_data = yf.download(ticker, start="2023-01-01", end="2023-12-31", progress=False)
    if ticker_cac_data.shape[0] != 0:
        stocks_CAC40[ticker] = ticker_cac_data

"""Скачали данные об акциях из индексов за 2023 год"""

#Германия

DAX_returns_dict=dict()

for ticker in stocks_DAX.keys():
    DAX_returns_dict[ticker] = np.log(stocks_DAX[ticker]['Close'] / stocks_DAX[ticker]['Close'].shift(1))
dax_DF_returns = pd.DataFrame(DAX_returns_dict).dropna()

#Франция

CAC_returns_dict=dict()

for ticker in stocks_CAC40.keys():
    CAC_returns_dict[ticker] = np.log(stocks_CAC40[ticker]['Close'] / stocks_CAC40[ticker]['Close'].shift(1))
CAC_DF_returns = pd.DataFrame(CAC_returns_dict).dropna()

#Англия

FTSE_returns_dict=dict()

for ticker in stocks_FTSE100.keys():
    FTSE_returns_dict[ticker] = np.log(stocks_FTSE100[ticker]['Close'] / stocks_FTSE100[ticker]['Close'].shift(1))
FTSE_DF_returns = pd.DataFrame(FTSE_returns_dict).dropna()

dax_DF = dax_DF_returns.corr(method='kendall')

dax_DF

lst_DAX_corr = []

for i in range(len(dax_DF_returns.corr(method='kendall'))):
    for j in range(i+1, len(dax_DF_returns.corr(method='kendall'))):
        lst_DAX_corr.append(dax_DF_returns.corr(method='kendall').values[i][j])
plt.hist(lst_DAX_corr, bins=50)
plt.show()

FTSE_DF = FTSE_DF_returns.corr(method='kendall')

FTSE_DF

lst_FTSE_corr = []

for i in range(len(FTSE_DF)):
    for j in range(i+1, len(FTSE_DF)):
        lst_FTSE_corr.append(FTSE_DF.values[i][j])
plt.hist(lst_FTSE_corr, bins=50)
plt.show()

CAC_DF = CAC_DF_returns.corr(method='kendall')

CAC_DF

lst_CAC_corr = []

for i in range(len(CAC_DF)):
    for j in range(i+1, len(CAC_DF)):
        lst_CAC_corr.append(CAC_DF.values[i][j])
plt.hist(lst_CAC_corr, bins=50)
plt.show()

from scipy.stats import norm

from src.test_statistics import kendall_statistics

import numpy as np

def compute_bounds(correlation_matrix, alpha, gamma_0):
    upper_bounds = []
    lower_bounds = []
    ce = norm.ppf(0.05)
    cn = norm.ppf(1-(0.05))

      # Вычисление верхних и нижних границ для каждой пары (i, j)
    for i in range(N):
        for j in range(N):
            if i < j:


                  # Проверка гипотезы для верхней границы
                if correlation_matrix[i,j] >= ce:

                    upper_bounds.append(correlation_matrix[i, j])

                  # Проверка гипотезы для нижней границы
                if correlation_matrix[i,j] > cn:
                    lower_bounds.append(correlation_matrix[i, j])

    return upper_bounds, lower_bounds

N = CAC_DF_returns.corr().shape[0]
M=((N*(N-1))/2)
alpha = 0.1
gamma_0_values = np.arange(-0.1, 1.0, 0.05)
#gamma_0_values = [0.05, 0.1, 0.4, 0.5]
matrix = np.array(CAC_DF_returns.corr())

upper_bounds_counts = []
lower_bounds_counts = []
uncertainty_fr = []
K1f = []
K2f = []
K1 = 0
K2 = 0
for gamma_0 in gamma_0_values:
  correlation_matrix = []
  correlation_matrix = kendall_statistics(CAC_DF_returns, gamma_0)
  upper_bounds, lower_bounds = compute_bounds(correlation_matrix,alpha, gamma_0)
  upper_bounds_counts.append(len(upper_bounds))
  lower_bounds_counts.append(len(lower_bounds))
  uncertainty_fr.append(len(upper_bounds)-len(lower_bounds))
  K1f.append(abs((len(upper_bounds)-len(lower_bounds))/M))
  K2f.append(abs((len(upper_bounds)-len(lower_bounds)))/abs((len(lower_bounds)+(M-len(upper_bounds)))))
  K1 = K1 + ((1/22)*(abs((len(upper_bounds)-len(lower_bounds))/M)))
  K2 = K2 + ((1/22)* (abs((len(upper_bounds)-len(lower_bounds)))/abs((len(lower_bounds)+(M-len(upper_bounds))))))

print(K1,K2)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, upper_bounds_counts, label='Upper bounds')
plt.plot(gamma_0_values, lower_bounds_counts, label='Lower bounds')
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('France')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values,K1)
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('France K1')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, K2)
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('France K2')
plt.legend()
plt.grid(True)
plt.show()

N = FTSE_DF_returns.corr().shape[0]
M=((N*(N-1))/2)
alpha = 0.1
gamma_0_values = np.arange(-0.1, 1.0, 0.05)
#gamma_0_values = [0.05, 0.1, 0.4, 0.5]
matrix = np.array(FTSE_DF_returns.corr())

upper_bounds_counts = []
lower_bounds_counts = []
uncertainty_eng = []
K1e = []
K2e = []
K1=0
K2=0
for gamma_0 in gamma_0_values:
  correlation_matrix = []
  correlation_matrix = kendall_statistics(FTSE_DF_returns, gamma_0)
  upper_bounds, lower_bounds = compute_bounds(correlation_matrix,alpha, gamma_0)
  upper_bounds_counts.append(len(upper_bounds))
  lower_bounds_counts.append(len(lower_bounds))
  uncertainty_eng.append(len(upper_bounds)-len(lower_bounds))
  K1e.append(abs((len(upper_bounds)-len(lower_bounds))/M))
  K2e.append(abs((len(upper_bounds)-len(lower_bounds)))/abs((len(lower_bounds)+(M-len(upper_bounds)))))
  K1 = K1 + ((1/22)*(abs((len(upper_bounds)-len(lower_bounds))/M)))
  K2 = K2 + ((1/22)* (abs((len(upper_bounds)-len(lower_bounds)))/abs((len(lower_bounds)+(M-len(upper_bounds))))))

print(K1,K2)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, upper_bounds_counts, label='Upper bounds')
plt.plot(gamma_0_values, lower_bounds_counts, label='Lower bounds')
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('England')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, K1)
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('England K1')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, K2)
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('England K2')
plt.legend()
plt.grid(True)
plt.show()

N = dax_DF_returns.corr().shape[0]
M=((N*(N-1))/2)
alpha = 0.1
gamma_0_values = np.arange(-0.1, 1.0, 0.05)
#gamma_0_values = [0.05, 0.1, 0.4, 0.5]
matrix = np.array(dax_DF_returns.corr())

upper_bounds_counts = []
lower_bounds_counts = []
uncertainty_germ = []
K1g = []
K2g = []
K1=0
K2=0
for gamma_0 in gamma_0_values:
  correlation_matrix = []
  correlation_matrix = kendall_statistics(dax_DF_returns, gamma_0)
  upper_bounds, lower_bounds = compute_bounds(correlation_matrix,alpha, gamma_0)
  upper_bounds_counts.append(len(upper_bounds))
  lower_bounds_counts.append(len(lower_bounds))
  uncertainty_germ.append(len(upper_bounds)-len(lower_bounds))
  K1g.append(abs((len(upper_bounds)-len(lower_bounds))/M))
  K2g.append(abs((len(upper_bounds)-len(lower_bounds)))/abs((len(lower_bounds)+(M-len(upper_bounds)))))
  K1 = K1 + ((1/22)*(abs((len(upper_bounds)-len(lower_bounds))/M)))
  K2 = K2 + ((1/22)* (abs((len(upper_bounds)-len(lower_bounds)))/abs((len(lower_bounds)+(M-len(upper_bounds))))))

print(K1,K2)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, upper_bounds_counts, label='Upper bounds')
plt.plot(gamma_0_values, lower_bounds_counts, label='Lower bounds')
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('German')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, K1)
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('German K1')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, K2)
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('German K2')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values, uncertainty_germ, label='German')
plt.plot(gamma_0_values, uncertainty_fr, label='France')
plt.plot(gamma_0_values, uncertainty_eng, label='England')
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('Uncertainty')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values,K1f, label='France')
plt.plot(gamma_0_values,K1e, label='England')
plt.plot(gamma_0_values,K1g, label='German')
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('France K1')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(gamma_0_values,K2f, label='France')
plt.plot(gamma_0_values,K2e, label='England')
plt.plot(gamma_0_values,K2g, label='German')
plt.xlabel('Gamma_0')
plt.ylabel('Number of bounds')
plt.title('France K1')
plt.legend()
plt.grid(True)
plt.show()