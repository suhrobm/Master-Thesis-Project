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

# prevents numpy scientific/exponential notation on print, default is False
# force-suppressing the scientific notation, it is rounded and justified
np.set_printoptions(suppress=False, formatter={'float_kind': '{:0.2f}'.format})
# ignoring warnings
warnings.filterwarnings(action='ignore')




# Importing transactional dataset from the SQL Server database, named 'TransRepository'
# The SQL Server dialect uses pyodbc as the default DBAPI:
server = 'DESKTOP-3REJ4PS'
database = 'TransRepository'
driver = 'SQL Server'
username = 'dathscom'

password = '987654321'
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
engine = db.create_engine(connection_string)
connection = engine.connect()

'''-------------------------------------------------------------------------------------------'''
'''                              Monthly Average Expenses of All Categories                   '''
'''-------------------------------------------------------------------------------------------'''
df_monthly_mean_all = pd.DataFrame(pd.read_sql_query('''    select
                                                                t.Category
                                                              , t.MonthNumbers
                                                              , avg(t.Amount) as MonthlyAverage
                                                            from
                                                            (
                                                               select
                                                                   t.Category
                                                                 , case when ('2020-01-01' <= t.Date and t.Date <= '2020-01-31') then '2020-01-30'
                                                                        when ('2020-02-01' <= t.Date and t.Date <= '2020-02-29') then '2020-02-30'
                                                                        when ('2020-03-01' <= t.Date and t.Date <= '2020-03-31') then '2020-03-30'
                                                                        when ('2020-04-01' <= t.Date and t.Date <= '2020-04-30') then '2020-04-30'
                                                                        when ('2020-05-01' <= t.Date and t.Date <= '2020-05-31') then '2020-05-30'
                                                                        when ('2020-06-01' <= t.Date and t.Date <= '2020-06-30') then '2020-06-30'
                                                                        when ('2020-07-01' <= t.Date and t.Date <= '2020-07-31') then '2020-07-30'
                                                                        when ('2020-08-01' <= t.Date and t.Date <= '2020-08-31') then '2020-08-30'
                                                                        when ('2020-09-01' <= t.Date and t.Date <= '2020-09-30') then '2020-09-30'
                                                                        when ('2020-10-01' <= t.Date and t.Date <= '2020-10-31') then '2020-10-30'
                                                                        when ('2020-11-01' <= t.Date and t.Date <= '2020-11-30') then '2020-11-30'
                                                                        when ('2020-12-01' <= t.Date and t.Date <= '2020-12-31') then '2020-12-30'
                                                                        when ('2021-01-01' <= t.Date and t.Date <= '2021-01-31') then '2021-01-30'
                                                                        when ('2021-02-01' <= t.Date and t.Date <= '2021-02-28') then '2021-02-30'
                                                                        when ('2021-03-01' <= t.Date and t.Date <= '2021-03-31') then '2021-03-30'
                                                                        when ('2021-04-01' <= t.Date and t.Date <= '2021-04-30') then '2021-04-30'
                                                                        when ('2021-05-01' <= t.Date and t.Date <= '2021-05-31') then '2021-05-30'
                                                                        when ('2021-06-01' <= t.Date and t.Date <= '2021-06-30') then '2021-06-30'
                                                                        else null end as MonthNumbers
                                                                , t.Amount
                                                               from T_FUNDS_TRANSFER t
                                                            ) t
                                                            group by t.Category, t.MonthNumbers
                                                            order by t.Category, t.MonthNumbers ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_monthly_mean_all, y='MonthlyAverage', x='MonthNumbers', hue='Category', style='Category', palette="nipy_spectral", linewidth=1, ax=ax)
ax.grid(True)
plt.ticklabel_format(style='plain', axis='y')
ax.set_ylim(0, 10000)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Monthly Average Expenses of All Categories', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()

'''-------------------------------------------------------------------------------------------'''
'''                          Monthly Average Expenses of Food Categories                      '''
'''-------------------------------------------------------------------------------------------'''
df_monthly_mean_food = pd.DataFrame(pd.read_sql_query('''   select
                                                                t.Category
                                                              , t.MonthNumbers
                                                              , avg(t.Amount) as MonthlyAverage
                                                            from
                                                            (
                                                               select
                                                                   t.Category
                                                                 , case when ('2020-01-01' <= t.Date and t.Date <= '2020-01-31') then '2020-01-30'
                                                                        when ('2020-02-01' <= t.Date and t.Date <= '2020-02-29') then '2020-02-30'
                                                                        when ('2020-03-01' <= t.Date and t.Date <= '2020-03-31') then '2020-03-30'
                                                                        when ('2020-04-01' <= t.Date and t.Date <= '2020-04-30') then '2020-04-30'
                                                                        when ('2020-05-01' <= t.Date and t.Date <= '2020-05-31') then '2020-05-30'
                                                                        when ('2020-06-01' <= t.Date and t.Date <= '2020-06-30') then '2020-06-30'
                                                                        when ('2020-07-01' <= t.Date and t.Date <= '2020-07-31') then '2020-07-30'
                                                                        when ('2020-08-01' <= t.Date and t.Date <= '2020-08-31') then '2020-08-30'
                                                                        when ('2020-09-01' <= t.Date and t.Date <= '2020-09-30') then '2020-09-30'
                                                                        when ('2020-10-01' <= t.Date and t.Date <= '2020-10-31') then '2020-10-30'
                                                                        when ('2020-11-01' <= t.Date and t.Date <= '2020-11-30') then '2020-11-30'
                                                                        when ('2020-12-01' <= t.Date and t.Date <= '2020-12-31') then '2020-12-30'
                                                                        when ('2021-01-01' <= t.Date and t.Date <= '2021-01-31') then '2021-01-30'
                                                                        when ('2021-02-01' <= t.Date and t.Date <= '2021-02-28') then '2021-02-30'
                                                                        when ('2021-03-01' <= t.Date and t.Date <= '2021-03-31') then '2021-03-30'
                                                                        when ('2021-04-01' <= t.Date and t.Date <= '2021-04-30') then '2021-04-30'
                                                                        when ('2021-05-01' <= t.Date and t.Date <= '2021-05-31') then '2021-05-30'
                                                                        when ('2021-06-01' <= t.Date and t.Date <= '2021-06-30') then '2021-06-30'
                                                                        else null end as MonthNumbers
                                                                , t.Amount
                                                               from T_FUNDS_TRANSFER t where t.Category = 'food'
                                                            ) t
                                                            group by t.Category, t.MonthNumbers
                                                            order by t.Category, t.MonthNumbers ''', connection))

#df_food = df_monthly_mean[df_monthly_mean['Category'] == 'food']
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_monthly_mean_food, y='MonthlyAverage', x='MonthNumbers', hue='Category', style='Category', palette="nipy_spectral", linewidth=1, ax=ax)
ax.grid(True)
plt.ticklabel_format(style='plain', axis='y')
ax.set_ylim(0, 1000)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Monthly Average Expenses of Food Category', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''                          Daily Total Expenses of All Top Categories                       '''
'''-------------------------------------------------------------------------------------------'''
df_daily_total_expenses = pd.DataFrame(pd.read_sql_query('''  select [Date]
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
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_daily_total_expenses, y=df_daily_total_expenses['Survival'], x=df_daily_total_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_daily_total_expenses, y=df_daily_total_expenses['Socialization'], x=df_daily_total_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_daily_total_expenses, y=df_daily_total_expenses['Self-Realization'], x=df_daily_total_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_daily_total_expenses, y=df_daily_total_expenses['Money'], x=df_daily_total_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Daily Total Expenses of All Top Categories', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''               Daily Total Expenses of All Top Categories (zooming-in)                     '''
'''-------------------------------------------------------------------------------------------'''
df_daily_expenses = pd.DataFrame(pd.read_sql_query('''  select [Date]
                                                             , isnull([Survival],0.0) as Survival
                                                             , isnull([Socialization],0.0) as Socialization
                                                             , isnull([Self-Realization],0.0) as [Self-Realization]
                                                             , isnull([Money],0.0) as Money
                                                        from (
                                                        select Date
                                                             , TopCategory
                                                             , Amount
                                                        from t_funds_transfer
                                                        --where Regnum = '' -- here, we can specify an agent/customer/consumer/client identifier
                                                        ) t
                                                        pivot (sum(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                        order by [Date] ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_daily_expenses, y=df_daily_expenses['Survival'], x=df_daily_expenses['Date'].iloc[:200], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_daily_expenses, y=df_daily_expenses['Socialization'], x=df_daily_expenses['Date'].iloc[:200], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_daily_expenses, y=df_daily_expenses['Self-Realization'], x=df_daily_expenses['Date'].iloc[:200], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_daily_expenses, y=df_daily_expenses['Money'], x=df_daily_expenses['Date'].iloc[:200], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Daily Total Expenses of All Top Categories (zooming-in)', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''               Daily Total Expenses of All Top Categories by an Agent(3671470)             '''
'''-------------------------------------------------------------------------------------------'''
df_daily_expenses_agent = pd.DataFrame(pd.read_sql_query('''   select [Date]
                                                                     , isnull([Survival],0.0) as Survival
                                                                     , isnull([Socialization],0.0) as Socialization
                                                                     , isnull([Self-Realization],0.0) as [Self-Realization]
                                                                     , isnull([Money],0.0) as Money
                                                                from (
                                                                select Date
                                                                     , TopCategory
                                                                     , Amount
                                                                from t_funds_transfer
                                                                where Regnum = '3671470'
                                                                ) t
                                                                pivot (sum(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                                order by [Date] ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Survival'], x=df_daily_expenses_agent['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Socialization'], x=df_daily_expenses_agent['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Self-Realization'], x=df_daily_expenses_agent['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Money'], x=df_daily_expenses_agent['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Daily Total Expenses by an Agent ', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)

ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''      Daily Total Expenses of All Top Categories by an Agent(3671470) (zooming-in)         '''
'''-------------------------------------------------------------------------------------------'''
df_daily_expenses_agent = pd.DataFrame(pd.read_sql_query('''   select [Date]
                                                                     , isnull([Survival],0.0) as Survival
                                                                     , isnull([Socialization],0.0) as Socialization
                                                                     , isnull([Self-Realization],0.0) as [Self-Realization]
                                                                     , isnull([Money],0.0) as Money
                                                                from (
                                                                        select Date
                                                                             , TopCategory
                                                                             , Amount
                                                                        from t_funds_transfer
                                                                        where Regnum = '3671470'
                                                                      ) t
                                                                pivot (sum(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                                order by [Date] ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Survival'], x=df_daily_expenses_agent['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Socialization'], x=df_daily_expenses_agent['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Self-Realization'], x=df_daily_expenses_agent['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_daily_expenses_agent, y=df_daily_expenses_agent['Money'], x=df_daily_expenses_agent['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Daily Total Expenses by an Agent (zooming-in)', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''                      Monthly Total Expenses of All Top Categories                         '''
'''-------------------------------------------------------------------------------------------'''
df_monthly_total_expenses = pd.DataFrame(pd.read_sql_query('''  select
                                                                        Monthly
                                                                      , sum(t.Survival) as Survival
                                                                      , sum(t.Socialization) as Socialization
                                                                      , sum(t.[Self-Realization]) as [Self-Realization]
                                                                      , sum(t.Money) as Money
                                                                from (
                                                                        select Monthly
                                                                             , isnull([Survival],0.0) as Survival
                                                                             , isnull([Socialization],0.0) as Socialization
                                                                             , isnull([Self-Realization],0.0) as [Self-Realization]
                                                                             , isnull([Money],0.0) as Money
                                                                        from (
                                                                                select case when ('2020-01-01' <= t.Date and t.Date <= '2020-01-31') then '2020-01-30'
                                                                                            when ('2020-02-01' <= t.Date and t.Date <= '2020-02-29') then '2020-02-30'
                                                                                            when ('2020-03-01' <= t.Date and t.Date <= '2020-03-31') then '2020-03-30'
                                                                                            when ('2020-04-01' <= t.Date and t.Date <= '2020-04-30') then '2020-04-30'
                                                                                            when ('2020-05-01' <= t.Date and t.Date <= '2020-05-31') then '2020-05-30'
                                                                                            when ('2020-06-01' <= t.Date and t.Date <= '2020-06-30') then '2020-06-30'
                                                                                            when ('2020-07-01' <= t.Date and t.Date <= '2020-07-31') then '2020-07-30'
                                                                                            when ('2020-08-01' <= t.Date and t.Date <= '2020-08-31') then '2020-08-30'
                                                                                            when ('2020-09-01' <= t.Date and t.Date <= '2020-09-30') then '2020-09-30'
                                                                                            when ('2020-10-01' <= t.Date and t.Date <= '2020-10-31') then '2020-10-30'
                                                                                            when ('2020-11-01' <= t.Date and t.Date <= '2020-11-30') then '2020-11-30'
                                                                                            when ('2020-12-01' <= t.Date and t.Date <= '2020-12-31') then '2020-12-30'
                                                                                            when ('2021-01-01' <= t.Date and t.Date <= '2021-01-31') then '2021-01-30'
                                                                                            when ('2021-02-01' <= t.Date and t.Date <= '2021-02-28') then '2021-02-30'
                                                                                            when ('2021-03-01' <= t.Date and t.Date <= '2021-03-31') then '2021-03-30'
                                                                                            when ('2021-04-01' <= t.Date and t.Date <= '2021-04-30') then '2021-04-30'
                                                                                            when ('2021-05-01' <= t.Date and t.Date <= '2021-05-31') then '2021-05-30'
                                                                                            when ('2021-06-01' <= t.Date and t.Date <= '2021-06-30') then '2021-06-30'
                                                                                            else null end as Monthly
                                                                                     , TopCategory
                                                                                     , Amount
                                                                                from t_funds_transfer t
                                                                             ) t
                                                                        pivot (sum(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                                ) t
                                                                group by Monthly
                                                                order by Monthly ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_monthly_total_expenses, y=df_monthly_total_expenses['Survival'], x=df_monthly_total_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_monthly_total_expenses, y=df_monthly_total_expenses['Socialization'], x=df_monthly_total_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_monthly_total_expenses, y=df_monthly_total_expenses['Self-Realization'], x=df_monthly_total_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_monthly_total_expenses, y=df_monthly_total_expenses['Money'], x=df_monthly_total_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Monthly Total Expenses of All Top Categories', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''                        Daily Average Expenses of All Top Categories                          '''
'''-------------------------------------------------------------------------------------------'''
df_daily_mean_expenses = pd.DataFrame(pd.read_sql_query(''' select [Date]
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
                                                            pivot (avg(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                            order by [Date] ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Survival'], x=df_daily_mean_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Socialization'], x=df_daily_mean_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Self-Realization'], x=df_daily_mean_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Money'], x=df_daily_mean_expenses['Date'], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Daily Average Expenses of All Top Categories', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''                Daily Average Expenses of All Top Categories (zooming-in)                     '''
'''-------------------------------------------------------------------------------------------'''
df_daily_mean_expenses = pd.DataFrame(pd.read_sql_query('''    select [Date]
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
                                                        pivot (avg(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                        order by [Date] ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Survival'], x=df_daily_mean_expenses['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Socialization'], x=df_daily_mean_expenses['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Self-Realization'], x=df_daily_mean_expenses['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_daily_mean_expenses, y=df_daily_mean_expenses['Money'], x=df_daily_mean_expenses['Date'].iloc[:120], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Daily Average Expenses of All Top Categories (zooming-in)  ', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()
'''-------------------------------------------------------------------------------------------'''
'''                      Monthly Average Expenses of All Top Categories                       '''
'''-------------------------------------------------------------------------------------------'''
df_monthly_mean_expenses = pd.DataFrame(pd.read_sql_query('''  select 
                                                                        Monthly
                                                                      , avg(t.Survival) as Survival
                                                                      , avg(t.Socialization) as Socialization
                                                                      , avg(t.[Self-Realization]) as [Self-Realization]
                                                                      , avg(t.Money) as Money
                                                                from (
                                                                        select Monthly
                                                                             , isnull([Survival],0.0) as Survival
                                                                             , isnull([Socialization],0.0) as Socialization
                                                                             , isnull([Self-Realization],0.0) as [Self-Realization]
                                                                             , isnull([Money],0.0) as Money
                                                                        from (
                                                                                select case when ('2020-01-01' <= t.Date and t.Date <= '2020-01-31') then '2020-01-30'
                                                                                            when ('2020-02-01' <= t.Date and t.Date <= '2020-02-29') then '2020-02-30'
                                                                                            when ('2020-03-01' <= t.Date and t.Date <= '2020-03-31') then '2020-03-30'
                                                                                            when ('2020-04-01' <= t.Date and t.Date <= '2020-04-30') then '2020-04-30'
                                                                                            when ('2020-05-01' <= t.Date and t.Date <= '2020-05-31') then '2020-05-30'
                                                                                            when ('2020-06-01' <= t.Date and t.Date <= '2020-06-30') then '2020-06-30'
                                                                                            when ('2020-07-01' <= t.Date and t.Date <= '2020-07-31') then '2020-07-30'
                                                                                            when ('2020-08-01' <= t.Date and t.Date <= '2020-08-31') then '2020-08-30'
                                                                                            when ('2020-09-01' <= t.Date and t.Date <= '2020-09-30') then '2020-09-30'
                                                                                            when ('2020-10-01' <= t.Date and t.Date <= '2020-10-31') then '2020-10-30'
                                                                                            when ('2020-11-01' <= t.Date and t.Date <= '2020-11-30') then '2020-11-30'
                                                                                            when ('2020-12-01' <= t.Date and t.Date <= '2020-12-31') then '2020-12-30'
                                                                                            when ('2021-01-01' <= t.Date and t.Date <= '2021-01-31') then '2021-01-30'
                                                                                            when ('2021-02-01' <= t.Date and t.Date <= '2021-02-28') then '2021-02-30'
                                                                                            when ('2021-03-01' <= t.Date and t.Date <= '2021-03-31') then '2021-03-30'
                                                                                            when ('2021-04-01' <= t.Date and t.Date <= '2021-04-30') then '2021-04-30'
                                                                                            when ('2021-05-01' <= t.Date and t.Date <= '2021-05-31') then '2021-05-30'
                                                                                            when ('2021-06-01' <= t.Date and t.Date <= '2021-06-30') then '2021-06-30'
                                                                                            else null end as Monthly
                                                                                     , TopCategory
                                                                                     , Amount
                                                                                from t_funds_transfer t 				
                                                                             ) t
                                                                        pivot (avg(Amount) for TopCategory in (Survival, Socialization, [Self-Realization], Money)) as pvt
                                                                ) t
                                                                group by Monthly
                                                                order by Monthly ''', connection))
# plotting the graph
fig, ax = plt.subplots()
sns.lineplot(data=df_monthly_mean_expenses, y=df_monthly_mean_expenses['Survival'], x=df_monthly_mean_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Survival')
sns.lineplot(data=df_monthly_mean_expenses, y=df_monthly_mean_expenses['Socialization'], x=df_monthly_mean_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Socialization')
sns.lineplot(data=df_monthly_mean_expenses, y=df_monthly_mean_expenses['Self-Realization'], x=df_monthly_mean_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Self-Realization')
sns.lineplot(data=df_monthly_mean_expenses, y=df_monthly_mean_expenses['Money'], x=df_monthly_mean_expenses['Monthly'], palette="nipy_spectral", linewidth=1, ax=ax, label='Money')
plt.ticklabel_format(style='plain', axis='y')
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Monthly Average Expenses of All Top Categories', fontsize=14)
ax.set_xlabel('Whole Time Period', fontsize=12)
ax.set_ylabel('Money / Payment (RUR)', fontsize=12)
plt.show()























#sns.scatterplot(data=df_monthly_mean, y='MonthlyAverage', x='MonthNumbers', hue="Category")
#ax.set_xticklabels(['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06'])