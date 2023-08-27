from scipy.stats import norm, lognorm, gaussian_kde, kstest, chisquare
from pandas import pandas as pd
import numpy as np
import os
import sqlalchemy as db
import pyodbc
import warnings
import re
from matplotlib import pyplot as plt
import matplotlib as mpl
import mpi4py
from mpi4py import MPI
import seaborn as sns
from scipy import stats as st
from math import sqrt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

''' prevents numpy scientific/exponential notation on print, default is False '''
''' force-suppressing the scientific notation, it is rounded and justified '''
np.set_printoptions(suppress=False, formatter={'float_kind': '{:0.2f}'.format})
''' ignoring warnings '''
warnings.filterwarnings(action='ignore')


''' importing transactional dataset from the SQL Server database, named 'TransRepository' '''
''' the SQL Server dialect uses pyodbc as the default DBAPI: '''
server = 'DESKTOP-3REJ4PS'
database = 'TransRepository'
driver = 'SQL Server'
username = 'dathscom'

password = '987654321'
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
engine = db.create_engine(connection_string)
connection = engine.connect()


''' selecting dataset from the database '''
df_dates = pd.DataFrame(pd.read_sql_query('''select Date
                                                 , isnull([Survival],0.0) as Survival
                                                 , isnull([Socialization],0.0) as Socialization
                                                 , isnull([Self-Realization],0.0) as [Self-Realization]
                                                 , isnull([Money],0.0) as Money
                                            from (
                                            select Date
                                                 , TopCategory
                                                 , Amount
                                            from t_funds_transfer 
                                            ) t
                                            pivot (sum(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                            order by [Date] ''', connection))

'''-------------------------------------------------------------------------------------------'''
'''                              Survival Category                                            '''
'''-------------------------------------------------------------------------------------------'''

''' converting from date format to integer and preprocessing using min max scaler '''
working_df = df_dates[['Date', 'Survival']].copy()

''' converting from date format to integer '''
working_df['Date'] = working_df['Date'].apply(lambda x: x.value)

scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
scaler.fit(working_df[['Date']])
working_df['Date'] = scaler.transform(working_df[['Date']])

scaler.fit(working_df[['Survival']])
working_df['Survival'] = scaler.transform(working_df[['Survival']])

'''
How to determine correct number of clusters (k)?
We need to specify Sum of Squared Errors (SSE) and then obeying Elbow Technique, so plotting Elbow 
'''
def optimise_k_means(data, number_of_clusters):
    means = []
    sse = []

    for _ in range(1, number_of_clusters):
        km = KMeans(n_clusters=_)
        km.fit(data)
        means.append(_)
        sse.append(km.inertia_)

    fig = plt.subplots(figsize=(10, 5))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia (Sum of Squared Error)")
    plt.grid(True)
    plt.plot(means, sse, 'o-')
    plt.show()

#optimise_k_means(working_df[['Date', 'Survival']], 10)

''' fitting the clustering models by creating multiple clusters '''
for _ in range(1,5):
    km = KMeans(n_clusters = _ )
    y_predicted = km.fit_predict(working_df[['Date', 'Survival']])
    working_df[f'Cluster_{_}'] = y_predicted

# print(f'\n{working_df}')

''' plotting the graph '''
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,15))
for _, ax in enumerate(fig.axes, start=1):
    ax.scatter(x=working_df['Date'], y=working_df['Survival'], c=working_df[f'Cluster_{_}'])
    if _ == 1:
        ax.set_title(f'Daily Expenses with {_} Cluster', fontsize=14)
    else:
        ax.set_title(f'Daily Expenses with {_} Clusters', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax.set_xlabel('Whole Time Period', fontsize=12)
    ax.set_ylabel('Survival Category', fontsize=12)
    fig.subplots_adjust(hspace=.50)

plt.show()

'''-------------------------------------------------------------------------------------------'''
'''                              Socialization Category                                       '''
'''-------------------------------------------------------------------------------------------'''

''' converting from date format to integer and preprocessing using min max scaler '''
working_df = df_dates[['Date', 'Socialization']].copy()

''' converting from date format to integer '''
working_df['Date'] = working_df['Date'].apply(lambda x: x.value)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(working_df[['Date']])
working_df['Date'] = scaler.transform(working_df[['Date']])

scaler.fit(working_df[['Socialization']])
working_df['Socialization'] = scaler.transform(working_df[['Socialization']])


''' fitting the clustering models by creating multiple clusters '''
for _ in range(1, 5):
    km = KMeans(n_clusters=_)
    y_predicted = km.fit_predict(working_df[['Date', 'Socialization']])
    working_df[f'Cluster_{_}'] = y_predicted

# print(f'\n{working_df}')

''' plotting the graph '''
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
for _, ax in enumerate(fig.axes, start=1):
    ax.scatter(x=working_df['Date'], y=working_df['Socialization'], c=working_df[f'Cluster_{_}'])
    if _ == 1:
        ax.set_title(f'Daily Expenses with {_} Cluster', fontsize=14)
    else:
        ax.set_title(f'Daily Expenses with {_} Clusters', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax.set_xlabel('Whole Time Period', fontsize=12)
    ax.set_ylabel('Socialization Category', fontsize=12)
    fig.subplots_adjust(hspace=.50)

plt.show()

'''-------------------------------------------------------------------------------------------'''
'''                              Self-Realization Category                                    '''
'''-------------------------------------------------------------------------------------------'''

''' converting from date format to integer and preprocessing using min max scaler '''
working_df = df_dates[['Date', 'Self-Realization']].copy()

''' converting from date format to integer '''
working_df['Date'] = working_df['Date'].apply(lambda x: x.value)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(working_df[['Date']])
working_df['Date'] = scaler.transform(working_df[['Date']])

scaler.fit(working_df[['Self-Realization']])
working_df['Self-Realization'] = scaler.transform(working_df[['Self-Realization']])


''' fitting the clustering models by creating multiple clusters '''
for _ in range(1, 5):
    km = KMeans(n_clusters=_)
    y_predicted = km.fit_predict(working_df[['Date', 'Self-Realization']])
    working_df[f'Cluster_{_}'] = y_predicted

# print(f'\n{working_df}')

''' plotting the graph '''
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
for _, ax in enumerate(fig.axes, start=1):
    ax.scatter(x=working_df['Date'], y=working_df['Self-Realization'], c=working_df[f'Cluster_{_}'])
    if _ == 1:
        ax.set_title(f'Daily Expenses with {_} Cluster', fontsize=14)
    else:
        ax.set_title(f'Daily Expenses with {_} Clusters', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax.set_xlabel('Whole Time Period', fontsize=12)
    ax.set_ylabel('Self-Realization Category', fontsize=12)
    fig.subplots_adjust(hspace=.50)

plt.show()

'''-------------------------------------------------------------------------------------------'''
'''                                  Money Category                                           '''
'''-------------------------------------------------------------------------------------------'''

''' converting from date format to integer and preprocessing using min max scaler '''
working_df = df_dates[['Date', 'Money']].copy()

''' converting from date format to integer '''
working_df['Date'] = working_df['Date'].apply(lambda x: x.value)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(working_df[['Date']])
working_df['Date'] = scaler.transform(working_df[['Date']])

scaler.fit(working_df[['Money']])
working_df['Money'] = scaler.transform(working_df[['Money']])


''' fitting the clustering models by creating multiple clusters '''
for _ in range(1, 5):
    km = KMeans(n_clusters=_)
    y_predicted = km.fit_predict(working_df[['Date', 'Money']])
    working_df[f'Cluster_{_}'] = y_predicted

# print(f'\n{working_df}')

''' plotting the graph '''
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
for _, ax in enumerate(fig.axes, start=1):
    ax.scatter(x=working_df['Date'], y=working_df['Money'], c=working_df[f'Cluster_{_}'])
    if _ == 1:
        ax.set_title(f'Daily Expenses with {_} Cluster', fontsize=14)
    else:
        ax.set_title(f'Daily Expenses with {_} Clusters', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax.set_xlabel('Whole Time Period', fontsize=12)
    ax.set_ylabel('Money Category', fontsize=12)
    fig.subplots_adjust(hspace=.50)

plt.show()