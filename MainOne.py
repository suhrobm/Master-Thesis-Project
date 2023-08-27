from scipy.stats import norm, lognorm, gaussian_kde, kstest, chisquare
from pandas import pandas as pd
import numpy as np
import os
import sqlalchemy as db
import pyodbc
import warnings
import re
from matplotlib import pyplot as plt
import mpi4py
from mpi4py import MPI
import seaborn as sns
from scipy import stats as st
from math import sqrt

# prevents numpy scientific/exponential notation on print, default is False
# force-suppressing the scientific notation, it is rounded and justified
np.set_printoptions(suppress=False, formatter={'float_kind':'{:0.2f}'.format})
# ignoring warnings
warnings.filterwarnings(action='ignore')

# runtime configuration options that become effective at import time of the MPI module,
# by default, runtime configuration is set to TRUE
mpi4py.rc.initialize = False # do not initialize MPI automatically
mpi4py.rc.finalize = False # do not finalize MPI automatically


'''
In this project the banking transactional dataset has been taken into account.
Author: Suhrob Munavarov

Run with:

   mpiexec -n 4 python MainOne.py

to run on 4 processors.

This illustrates BLOCKING point-to-point communication

'''
'''
To be able to accomplish this task, we need to be familiar with 
                 NumPy
                 Pandas
                 Scipy
                 Matplotlib
                 Seaborn libraries
                 SQLAlchemy
                 
                 
SQLAlchemy is the Python SQL toolkit and Object Relational Mapper that 
gives application developers the full power and flexibility of SQL.                 
Then you should really look into sqlalchemy package. 
It provides you with full power of SQL from inside the python in a form of pythonic objects. 
You can use it without actually interfacing with the database, however if you want to, 
it provides you with tools for that as well.
SQLAlchemy is a popular SQL toolkit and Object Relational Mapper.

'''
''' Vocabulary 

analysis ~ decomposition into components in order to study (a complex thing (bulk dataset), concept, theory etc.)
           the process of breaking down a substance into its constituent parts, or the result of this process.
normal density distribution ~ bell curve, symmetrical distribution  (i.e. mean = median = mode)

A random variable can be either discrete (having specific values) or continuous (any value in a continuous range).
A continuous variable is defined as a variable which can take an uncountable set of values or infinite set of values. 
Ex., if a variable over a non-empty range of the real numbers is continuous, then it can take on any value in that range.


Probability Density Functions (PDF) ~ Cumulative Distribution Functions (CDF)
                                 
                                 In parametric PDF
                                   
The distribution can be determined by parameters (statistical moments) such as mean, median, standard deviation or logs.
The histogram is not complex (i.e. just one spike, sivri uç) as it is unimodal distribution (having one mode).
Several different types of 'parametric PDF methods/estimators' are:
                                                                Normal Distribution ~ bell curve shape
                                                                Exponential Distribution 
                                                                Uniform Distribution ~ a six-sided die
                                                                Binomial Distribution 
                                                                Lognormal Distribution 
                                                                Poisson Distribution 
                                                                Beta Distribution 
                                                                                   
i.e. there are several different types of parametric PDF methods.

                                 Non parametric PDF
                                 
Non-Parametric is used when there is multimodal distribution and when we cannot figure out the distribution type.                               
Histogram is getting complex as it is multimodal distribution (with more then one spikes), in that scenario 
we need to define the kernel density estimator that estimates the plot.
This is often the case when the data has two peaks (bimodal distribution) or many peaks (multimodal distribution).
Where data clustering/binning is more, the height of the curve is high and increasing where density is compacting 
and decreasing for vice versa.

Kernel Density Estimation is a technique that lets us create a smooth curve, 
i.e. this is a technique that makes reference to mapping data points to a curve, smoothing processing. 
The aim is try to be able to fit the gaussian curve in case of non-parametric approach scenarios. 

In probability and statistics, 
density estimation is the construction of 
                                            an estimate, 
                                            based on observed data, 
                                            of an unobservable underlying probability density function. 
In laymen terms, density estimation refers to mapping data points to a curve or function that would be best 
representation of the data.

Kernel Density Estimation is ONE of the techniques of the density estimation functions.
Density estimation function ~ estimated density function ~ constructor function using observed data.
Probability density function is real graphical representation of the density
which is often not realistic, absurd ==> we need to use a constructor function, ex., KDE

Statistical tests as the Kolmogorov–Smirnov test the Shapiro–Wilk test and Chi-Square test are for 
determining the normality/uniformity of the distribution (of skewness of data).  

'''

''' Coding notes
more specifically if we want to convert each element on a column to a floating point number then
the lambda operator will take the values on that column (as x) and return them back as a float value
replacing commas with dots

.loc operator:  is primarily label based ~ human readable labels
.iloc operator: is primarily integer position based ~ computer logical indexing
 in the iloc[row_number, column_number], we need to pass the row and column index positions, 
 and it fetches the cell value based on that indexes

'''
'''                                       Phase I
------------------------------------------------------------------------------------------------------------------------
For the 1. phase, I would like to make some study with reference to the uni-variate random variables.
'''

''' 1.1
I am distinguishing a smaller portion of the original dataset (subset) with main variables for my further study, 
so that for each of them we could implement several manipulations.
'''

# Calculating confidence intervals for 25%, 50% and 75% quantiles
def conf_intervals(data, qn):
    # 95% quantile of Gaussian distribution
    norm_q95 = norm.ppf(0.95)
    kernel = gaussian_kde(data)

    p25 = len(data[data < qn[5]]) / len(data)
    sigma25 = (sqrt((p25 * (1 - p25)) / len(data))) / kernel(qn[5])
    p50 = len(data[data < qn[10]]) / len(data)
    sigma50 = (sqrt((p50 * (1 - p50)) / len(data))) / kernel(qn[10])
    p75 = len(data[data < qn[15]]) / len(data)
    sigma75 = (sqrt((p75 * (1 - p75)) / len(data))) / kernel(qn[15])

    conf_q25 = norm_q95 * sigma25
    conf_q50 = norm_q95 * sigma50
    conf_q75 = norm_q95 * sigma75

    return [conf_q25, conf_q50, conf_q75]


def main():

    # Point to Point blocking communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # if we want to use a file in the program then it should be located in the current working directory,
    # for checking the current working directory, file paths are case sensitive
    # print(os.getcwd())

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

    # print()
    # print(engine.table_names())

    # print() # Constructing DataFrame test
    # df_test = pd.DataFrame(pd.read_sql_query('''select top 5 *
    #                                             from t_funds_transfer t
    #                                             order by t.regnum''', connection))
    # print(df_test)
    #
    # print() # Constructing DataFrame category_subcategory
    # df_category_subcategory = pd.DataFrame(pd.read_sql_query('''select distinct
    #                                                                    t.subcategory
    #                                                                  , t.category
    #                                                              from t_funds_transfer t
    #                                                              order by t.category''', connection))
    # print(df_category_subcategory)
    #
    # print() # Constructing DataFrame which shows a quick statistic summary of the obtained data

    # df_general_statistics = pd.DataFrame(pd.read_sql_query('''select
    #                                                                 t.amount
    #                                                           from t_funds_transfer t''', connection))

    # df_gs = df_general_statistics.describe()['amount'].apply(lambda x: re.sub(',', '.', str(x)))
    # print(df_gs)
    # print()
    # print('The mean is : ' + df_gs.loc['mean']) # single label for row and column

    if rank == 0:
        '''   CREATING 2 DISCRETE AND 2 CONTINUOUS RANDOM VARIABLES FROM THE TRANSACTIONAL DATASET  '''
        '''-----------------------------------------------------------------------------------------'''
        # creating discrete random variable and passing them into the DataFrame
        df_charity = pd.DataFrame(pd.read_sql_query('''select 
                                                              t.Amount 
                                                       from T_FUNDS_TRANSFER t
                                                       where t.Category = 'charity'
                                                       and t.Date between '2020-01-01' and '2020-01-30' ''', connection))
        print(f'Number of transactions for charity category: {len(df_charity)}')
        comm.send(df_charity, dest=1, tag=11)

        # creating discrete random variable and passing them into the DataFrame
        df_beauty = pd.DataFrame(pd.read_sql_query('''select 
                                                                t.Amount
                                                      from T_FUNDS_TRANSFER t
                                                      where t.Category = 'beauty'                           
                                                      and t.Date between '2020-01-01' and '2020-01-30' ''', connection))
        print(f'Number of transactions for beauty category: {len(df_beauty)}')
        comm.send(df_beauty, dest=2, tag=22)

        # creating continuous random variable and passing them into the DataFrame
        df_money = pd.DataFrame(pd.read_sql_query('''select 
                                                                t.Amount 
                                                     from T_FUNDS_TRANSFER t
                                                     where t.Category = 'money' ''', connection))
        print(f'Number of transactions for money category: {len(df_money)}')
        comm.send(df_money, dest=3, tag=33)

        # creating continuous random variable and passing them into the DataFrame
        df_food = pd.DataFrame(pd.read_sql_query('''select 
                                                                t.Amount 
                                                    from T_FUNDS_TRANSFER t
                                                    where t.Category = 'food' ''', connection))
        print(f'Number of transactions for food category: {len(df_food)}')
        comm.send(df_food, dest=4, tag=44)


        # creating continuous random variable and passing them into the DataFrame
        df_allcat = pd.DataFrame(pd.read_sql_query('''select 
                                                               t.Category
                                                             , t.Amount
                                                             , t.TransactionDate
                                                    from T_FUNDS_TRANSFER t ''', connection))
        print(f'Number of transactions for all category: {len(df_allcat)}')
        comm.send(df_allcat, dest=5, tag=55)



    elif rank == 1:
        '''                    CHARITY ASSESSMENT                    '''  # has countable data set
        '''----------------------------------------------------------'''
        '''
        The first step in 'Density Estimation' is to create a histogram of the observations in the given data set
        '''

        df_charity = comm.recv(source=MPI.ANY_SOURCE, tag=11)

        charity_list = df_charity['Amount'].to_list()

        '''
         The choice of the number of bins is important as it controls the coarseness(roughness, rudeness) of the
         distribution (number of bars) and, in turn, how well the density of the observations is plotted. 
         It is a good idea to experiment with different bin sizes for a give data sample to get multiple 
         perspectives or views on the same data. Here i took 100 bins, which might better capture the density.
         Bins ~ disjoint categories
        '''
        ''' histogram is the simplest non-parametric density estimation technique'''
        y = np.arange(0,5000,50)
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

        plt.subplot(221)
        plt.ticklabel_format(axis='both', style='plain')
        plt.hist(x=charity_list,bins=y,ec='pink')
        #plt.xticks(y)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Count/Frequencies of Observations') # frequencies of observations, counting the number of events that fall into each bin
        plt.title('Charity')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.50)


        # Plotting a histogram and kernel density estimation
        plt.subplot(222)
        plt.ticklabel_format(axis='both', style='plain')
        # histograms and KDE can be combined using distplot function
        sns.distplot(charity_list, kde=True, norm_hist=True)  # , fit=lognorm
        plt.ylabel('Density')
        plt.xlabel('Range of Amounts')
        plt.title('Charity')
        plt.legend('', ncol=1, loc='upper right')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.50)


        plt.subplot(223)
        # we can get a smooth distribution estimate using the kernel density estimate that Seaborn does with kdeplot function
        ''' sns.kdeplot(data, shade=True) '''
        sns.kdeplot(charity_list, fill=True)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Density')
        plt.title('Charity')
        fig.subplots_adjust(wspace=.25)


        # calculating dataset quantiles & modeling quantiles
        # The distribution parameters are determined using the fit function based on the maximum likelihood method
        # modelin daha uygun (fit) hale gelmesini yardimci olan
        # KDE ~ smoothing out process ~ fitting process ~ balancing process
        # quantile are sometimes called percentile
        percentiles = np.linspace(0, 100, 21)
        qauntiles_charity = np.percentile(df_charity['Amount'], percentiles)
        model = lognorm.fit(df_charity['Amount'])
        qauntiles_lognorm = lognorm.ppf(percentiles / 100.0, *model)
        # the acronym ppf stands for percent point function, which is another name for the quantile function.
        # determination of the parameters of the lognormal distribution
        x = np.linspace(np.min(df_charity['Amount']), np.max(df_charity['Amount']))

        # Plotting a quantile biplot for empirical and theoretical (lognormal) distribution
        # Building a quantile biplot
        plt.subplot(224)
        plt.ticklabel_format(axis='both', style='plain')
        plt.plot(qauntiles_lognorm, qauntiles_charity, ls='', marker="o", markersize=4)
        plt.plot(x, x, color="k", ls="--")
        plt.xlim(-50, 800)
        plt.ylim(-50, 800)
        plt.grid(True)
        plt.xlabel('Theoretical (lognormal) distribution')
        plt.ylabel('Empirical distribution')
        plt.title('Q-Q Plot')
        plt.show()


        df_charity_sum = pd.DataFrame(pd.read_sql_query('''select 
                                                                  sum(t.Amount) as Amount
                                                           from T_FUNDS_TRANSFER t
                                                           where t.Category = 'charity'
                                                           and t.Date between '2020-01-01' and '2020-01-30' ''', connection))
        amount_total = df_charity_sum.iloc[0,df_charity_sum.columns.get_loc('Amount')]
        print(f'Total volume of charity is: {amount_total}')

        # Calculating confidence intervals for 25%, 50% and 75% quantiles
        confidence_charity = conf_intervals(df_charity['Amount'], qauntiles_charity)
        # Estimating correctness of fitted distributions using 2 statistical tests (Kolmogorov-Smirnov & Chi-Square)
        # Calculation of the Kolmogorov-Smirnov test and chi-square
        ks_charity = kstest(df_charity['Amount'], 'lognorm', model, N=100)
        chi2_charity = chisquare(df_charity['Amount'])

        print(f'25%, 50%, 75% - confidence intervals for charity category:\n {confidence_charity}')
        print(f'Using K-S, determining the normality of the charity (of skewness of data): {ks_charity}')
        print(f'Using C-S, determining the normality of the charity (of skewness of data): {chi2_charity}')

    elif rank == 2:
        '''                     BEAUTY ASSESSMENT                    '''  # has countable data set
        '''----------------------------------------------------------'''
        df_beauty = comm.recv(source=MPI.ANY_SOURCE, tag=22)

        beauty_list = df_beauty['Amount'].to_list()

        y = np.arange(0, 10000, 100)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        plt.subplot(221)
        plt.ticklabel_format(axis='both', style='plain')
        plt.hist(beauty_list,y,ec='pink')
        # plt.xticks(y)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Count/Frequencies of Observations')
        plt.title('Beauty')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.50)


        # Plotting a histogram and kernel density estimation
        plt.subplot(222)
        plt.ticklabel_format(axis='both', style='plain')
        # histograms and KDE can be combined using distplot function
        sns.distplot(beauty_list, kde=True, norm_hist=True)
        plt.ylabel('Density')
        plt.xlabel('Range of Amounts')
        plt.title('Beauty')
        plt.legend('', ncol=1, loc='upper right')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.50)


        plt.subplot(223)
        # we can get a smooth distribution estimate using the kernel density estimate that Seaborn does with kdeplot function
        ''' sns.kdeplot(data, shade=True) '''
        sns.kdeplot(beauty_list, shade=True)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Density')
        plt.title('Beauty')
        fig.subplots_adjust(wspace=.25)

        # Calculating dataset quantiles & modeling quantiles
        # The distribution parameters are determined using the fit function based on the maximum likelihood method
        percentiles = np.linspace(0, 100, 21)
        qauntiles_beauty = np.percentile(df_beauty['Amount'], percentiles)
        model = lognorm.fit(df_beauty['Amount'])
        qauntiles_lognorm = lognorm.ppf(percentiles / 100.0, *model)
        # Determination of the parameters of the lognormal distribution
        x = np.linspace(np.min(df_beauty['Amount']), np.max(df_beauty['Amount']))

        # Plotting a quantile biplot for empirical and theoretical (lognormal) distribution
        # Building a quantile biplot
        plt.subplot(224)
        plt.ticklabel_format(axis='both', style='plain')
        plt.plot(qauntiles_lognorm, qauntiles_beauty, ls='', marker="o", markersize=4)
        plt.plot(x, x, color="k", ls="--")
        plt.xlim(-50, 800)
        plt.ylim(-50, 800)
        plt.grid(True)
        plt.xlabel('Theoretical (lognormal) distribution')
        plt.ylabel('Empirical distribution')
        plt.title('Q-Q Plot')
        plt.show()

        df_beauty_sum = pd.DataFrame(pd.read_sql_query('''select 
                                                                 sum(t.Amount) as Amount
                                                            from T_FUNDS_TRANSFER t
                                                            where t.Category = 'beauty'
                                                            and t.Date between '2020-01-01' and '2020-01-30' ''', connection))
        amount_total = df_beauty_sum.iloc[0, df_beauty_sum.columns.get_loc('Amount')]
        print(f'Total volume of beauty is: {amount_total}')

        # Calculating confidence intervals for 25%, 50% and 75% quantiles
        confidence_beauty = conf_intervals(df_beauty['Amount'], qauntiles_beauty)
        # Estimating correctness of fitted distributions using 2 statistical tests (Kolmogorov-Smirnov & Chi-Square)
        # Calculation of the Kolmogorov-Smirnov test and chi-square
        ks_beauty = kstest(df_beauty['Amount'], 'lognorm', model, N=100)
        chi2_beauty = chisquare(df_beauty['Amount'])

        print(f'25%, 50%, 75% - confidence intervals for beauty category:\n {confidence_beauty}')
        print(f'Using K-S, determining the normality of the beauty (of skewness of data): {ks_beauty}')
        print(f'Using C-S, determining the normality of the beauty (of skewness of data): {chi2_beauty}')

    elif rank == 3:
        '''                      MONEY ASSESSMENT                    '''  # has uncountable data set
        '''----------------------------------------------------------'''
        df_money = comm.recv(source=MPI.ANY_SOURCE, tag=33)

        money_list = df_money['Amount'].to_list()

        y = np.arange(0, 50000, 500)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        plt.ticklabel_format(axis='both', style='plain') # we can disable both through this, to prevent "scientific notation" refers to a multiplier for the numbers show, while the "offset" on the x-axis is a separate term that is added.
        plt.subplot(221)
        plt.ticklabel_format(axis='y', style='plain')
        plt.hist(money_list,y,ec='pink')
        # plt.xticks(y)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Count/Frequencies of Observations')
        plt.title('Money')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.50)


        # Plotting a histogram and kernel density estimation
        plt.subplot(222)
        plt.ticklabel_format(axis='both', style='plain')
        # histograms and KDE can be combined using distplot function
        sns.distplot(money_list, kde=True, norm_hist=True)
        plt.ylabel('Density')
        plt.xlabel('Range of Amounts')
        plt.title('Money')
        plt.legend('', ncol=1, loc='upper right')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.50)


        plt.subplot(223)
        # we can get a smooth distribution estimate using the kernel density estimate that Seaborn does with kdeplot function
        ''' sns.kdeplot(data, shade=True) '''
        sns.kdeplot(money_list, shade=True)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Density')
        plt.title('Money')
        fig.subplots_adjust(wspace=.25)

        # Calculating dataset quantiles & modeling quantiles
        # The distribution parameters are determined using the fit function based on the maximum likelihood method
        percentiles = np.linspace(0, 100, 21)
        qauntiles_money = np.percentile(df_money['Amount'], percentiles)
        model = lognorm.fit(df_money['Amount'])
        qauntiles_lognorm = lognorm.ppf(percentiles / 100.0, *model)
        # Determination of the parameters of the lognormal distribution
        x = np.linspace(np.min(df_money['Amount']), np.max(df_money['Amount']))

        # Plotting a quantile biplot for empirical and theoretical (lognormal) distribution
        # Building a quantile biplot
        plt.subplot(224)
        plt.ticklabel_format(axis='both', style='plain')
        plt.plot(qauntiles_lognorm, qauntiles_money, ls='', marker="o", markersize=4)
        plt.plot(x, x, color="k", ls="--")
        plt.xlim(-50, 800)
        plt.ylim(-50, 800)
        plt.grid(True)
        plt.xlabel('Theoretical (lognormal) distribution')
        plt.ylabel('Empirical distribution')
        plt.title('Q-Q Plot')
        plt.show()

        df_money_sum = pd.DataFrame(pd.read_sql_query('''select 
                                                                sum(t.Amount) as Amount
                                                         from T_FUNDS_TRANSFER t
                                                         where t.Category = 'money' ''', connection))
        # In the iloc[row_number, column_number], we need to pass the row and column index positions, and it fetches the cell value based on that
        amount_total = df_money_sum.iloc[0, df_money_sum.columns.get_loc('Amount')]
        print(f'Total volume of money is: {amount_total}')

        # Calculating confidence intervals for 25%, 50% and 75% quantiles
        confidence_money = conf_intervals(df_money['Amount'], qauntiles_money)
        # Estimating correctness of fitted distributions using 2 statistical tests (Kolmogorov-Smirnov & Chi-Square)
        # Calculation of the Kolmogorov-Smirnov test and chi-square
        ks_money = kstest(df_money['Amount'], 'lognorm', model, N=100)
        chi2_money = chisquare(df_money['Amount'])

        print(f'25%, 50%, 75% - confidence intervals for money category:\n {confidence_money}')
        print(f'Using K-S, determining the normality of the money (of skewness of data): {ks_money}')
        print(f'Using C-S, determining the normality of the money (of skewness of data): {chi2_money}')

    elif rank == 4:
        '''                      FOOD ASSESSMENT                     '''  # has uncountable data set
        '''----------------------------------------------------------'''
        df_food = comm.recv(source=MPI.ANY_SOURCE, tag=44)

        food_list = df_food['Amount'].to_list()

        y = np.arange(0, 6000, 60)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        plt.subplot(221)
        plt.ticklabel_format(axis='both', style='plain')
        plt.hist(food_list,y,ec='pink')
        # plt.xticks(y)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Count/Frequencies of Observations')
        plt.title('Food')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.59)


        # Plotting a histogram and kernel density estimation
        plt.subplot(222)
        plt.ticklabel_format(axis='both', style='plain')
        # histograms and KDE can be combined using distplot function
        g = sns.distplot(food_list, kde=True, norm_hist=True)
        g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.ylabel('Density')
        plt.xlabel('Range of Amounts')
        plt.title('Food')
        plt.legend('', ncol=1, loc='upper right')
        fig.subplots_adjust(wspace=.25)
        fig.subplots_adjust(hspace=.59)


        plt.subplot(223)
        # we can get a smooth distribution estimate using the kernel density estimate that Seaborn does with kdeplot function
        ''' sns.kdeplot(data, shade=True) '''
        sns.kdeplot(food_list, shade=True)
        plt.xlabel('Range of Amounts')
        plt.ylabel('Density')
        plt.title('Food')
        fig.subplots_adjust(wspace=.25)


        # Calculating dataset quantiles & modeling quantiles
        # The distribution parameters are determined using the fit function based on the maximum likelihood method
        percentiles = np.linspace(0, 100, 21)
        qauntiles_food = np.percentile(df_food['Amount'], percentiles)
        model = lognorm.fit(df_food['Amount'])
        qauntiles_lognorm = lognorm.ppf(percentiles / 100.0, *model)
        # Determination of the parameters of the lognormal distribution
        x = np.linspace(np.min(df_food['Amount']), np.max(df_food['Amount']))

        # Plotting a quantile biplot for empirical and theoretical (lognormal) distribution
        # Building a quantile biplot
        plt.subplot(224)
        plt.ticklabel_format(axis='both', style='plain')
        plt.plot(qauntiles_lognorm, qauntiles_food, ls='', marker="o", markersize=4)
        plt.plot(x, x, color="k", ls="--")
        plt.xlim(-50, 800)
        plt.ylim(-50, 800)
        plt.grid(True)
        plt.xlabel('Theoretical (lognormal) distribution')
        plt.ylabel('Empirical distribution')
        plt.title('QQ-plot')
        plt.show()

        df_food_sum = pd.DataFrame(pd.read_sql_query('''select 
                                                                sum(t.Amount) as Amount
                                                         from T_FUNDS_TRANSFER t
                                                         where t.Category = 'food' ''', connection))
        # In the iloc[row_number, column_number], we need to pass the row and column index positions, and it fetches the cell value based on that
        amount_total = df_food_sum.iloc[0, df_food_sum.columns.get_loc('Amount')]
        print(f'Total volume of money is: {amount_total}')

        # Calculating confidence intervals for 25%, 50% and 75% quantiles
        confidence_food = conf_intervals(df_food['Amount'], qauntiles_food)
        # Estimating correctness of fitted distributions using 2 statistical tests (Kolmogorov-Smirnov & Chi-Square)
        # Calculation of the Kolmogorov-Smirnov test and chi-square
        ks_food = kstest(df_food['Amount'], 'lognorm', model, N=100)
        chi2_food = chisquare(df_food['Amount'])

        print(f'25%, 50%, 75% - confidence intervals for food category:\n {confidence_food}')
        print(f'Using K-S, determining the normality of the food (of skewness of data): {ks_food}')
        print(f'Using C-S, determining the normality of the food (of skewness of data): {chi2_food}')


    elif rank == 5:
        '''                      All CATEGORY ASSESSMENT                     '''  # has uncountable data set
        '''------------------------------------------------------------------'''
        df_allcat = comm.recv(source=MPI.ANY_SOURCE, tag=55)

        sns.set_theme(style="whitegrid")

        y = df_allcat[df_allcat['Category'] == 'food']



        sns.lineplot(y=y, x=x, palette="tab10", linewidth=2.5)











    connection.close()




''' Starting program '''
if __name__ == "__main__":
    # MPI.Init()  # manual initialization of the MPI environment
    main()
    MPI.Finalize()  # manual finalization of the MPI environment

