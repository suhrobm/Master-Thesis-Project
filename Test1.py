# # example of a bimodal data sample
# from matplotlib import pyplot
# from numpy.random import normal
# from numpy import hstack
# # generate a sample
# sample1 = normal(loc=20, scale=5, size=300)
# sample2 = normal(loc=40, scale=5, size=700)
# sample = hstack((sample1, sample2))
# # plot the histogram
# pyplot.hist(sample, bins=50)
# pyplot.show()
#
#

import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde, kstest, chisquare
import seaborn as sns
from matplotlib import pyplot as plt

numargs = lognorm.numargs
a, b = 4.32, 3.18
rv = lognorm(a, b)
print ("RV : \n", rv)

quantile = np.arange (0.01, 1, 0.1)
# Random Variates
R = lognorm.rvs(a, b)
print ("Random Variates : \n", R)

# PDF
R = lognorm.pdf(a, b, quantile)
print ("\nProbability Distribution : \n", R)


sns.kdeplot(R, fill=True)
plt.show()

