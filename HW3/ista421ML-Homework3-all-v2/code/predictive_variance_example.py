## Extension of predictive_variance_example.py
# Port of predictive_variance_example.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Predictive variance example

import numpy as np
import matplotlib.pyplot as plt
##plt.ion()

def true_function(x):
    """ true function $t = 5x^3-x^2+x$ """
    return 5*x**3 - x**2 + x

def sample_from_function(N=100, noise_var=1000, xmin=-5., xmax=5.):
    """ Sample data from the true function.
        N: Number of samples
        Returns a noisy sample t_sample from the function
        and the true function t. """
    x_range = xmax - xmin
    x_mid = x_range/2.
    x = np.sort(x_range*np.random.rand(N) - x_mid)
    t = true_function(x)
    # add standard normal noise using np.random.randn
    # (standard normal is a Gaussian N(0, 1.0),
    #  so multiplying by np.sqrt(noise_var) make it N(0,noise_ver))
    t = t + np.random.randn(x.shape[0])*np.sqrt(noise_var)
    return x,t

## sample 100 points from function
x,t = sample_from_function(100)


# Chop out some x data
# the following line expresses a boolean function over the values in x;
# this produces a list of the indices of list x for which the test
# was not met; these indices are then deleted from x and t.
##pos = ((x>=-0.5) & (x<=2.5)).nonzero()
##x = np.delete(x, pos, 0)
##t = np.delete(t, pos, 0)


# reshape the x and t to be column vectors in an np.matrix form
# so that we can perform matrix operations.
x = np.asmatrix(np.reshape(x,(x.shape[0],1)))
t = np.asmatrix(np.reshape(t,(t.shape[0],1)))

## Plot just the sampled data
plt.figure(0)
plt.scatter(np.asarray(x), np.asarray(t), color = 'k', edgecolor = 'k')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Sampled data from $t=5x^3-x^2+x$, $x \in [-5.0,5.0]$')

## Fit models of various orders
orders = [1,3,5,9]

## Make a set of 100 evenly-spaced x values between -5 and 5
testx = np.asmatrix(np.linspace(-5, 5, 100)).conj().transpose()

## Generate plots of predicted variance (error bars) for various model orders
for i in orders:
    # create input representation for given model polynomial order
    X = np.asmatrix(np.zeros(shape = (x.shape[0], i + 1)))
    testX = np.asmatrix(np.zeros(shape = (testx.shape[0], i + 1)))
    for k in range(i + 1):
        X[:, k] = np.power(x, k)
        testX[:, k] = np.power(testx, k)
    N = X.shape[0]

    # fit model parameters
    w = np.linalg.inv(X.T*X)*X.T*t
    ss = (1./N)*(t.T*t - t.T*X*w)

    # calculate predictions
    testmean = testX*w
    testvar = ss * np.diag(testX*np.linalg.inv(X.T*X)*testX.T)

    # Plot the data and predictions
    plt.figure()
    plt.scatter(np.asarray(x), np.asarray(t), color = 'k', edgecolor = 'k')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.errorbar(np.asarray(testx.T)[0], np.asarray(testmean.T)[0], np.asarray(testvar)[0])
    ti = 'Plot of predicted variance for model with polynomial order {:g}'.format(i)
    plt.ylim(-1000,1000)
    plt.title(ti)

## Generate plots of funcions whose parameters are sampled based on cov(\hat{w})
num_function_samples = 20
for i in orders:
    # create input representation for given model polynomial order
    X = np.asmatrix(np.zeros(shape = (x.shape[0], i + 1)))
    testX = np.asmatrix(np.zeros(shape = (testx.shape[0], i + 1)))
    for k in range(i + 1):
        X[:, k] = np.power(x, k)
        testX[:, k] = np.power(testx, k)

    # fit model parameters
    w = np.linalg.inv(X.T*X)*X.T*t
    ss = (1./N)*(t.T*t - t.T*X*w)
    
    # Sample functions with parameters w sampled from a Gaussian with
    # $\mu = \hat{\mathbf{w}}$
    # $\Sigma = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$
    # determine cov(w)
    covw = np.asarray(ss)[0][0]*np.linalg.inv(X.T*X)
    # The following samples num_function_samples of w from Gaussian based on covw
    wsamp = np.random.multivariate_normal(np.asarray(w.T)[0], covw, num_function_samples)
    # Calculate means for each function
    testmean = testX*wsamp.T
    
    # Plot the data and functions
    plt.figure()
    plt.scatter(np.asarray(x), np.asarray(t), color = 'k', edgecolor = 'k')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.plot(np.asarray(testx), np.asarray(testmean), color = 'b')
    plt.xlim(-2,4) # (-3, 3)
    plt.ylim(-400,400)
    ti = 'Plot of {0} functions where parameters w were sampled from\n' + \
         'cov(w) of model with polynomial order {1}'.format(num_function_samples, i)
    plt.title(ti)
    plt.show()

raw_input('Press <ENTER> to continue...')

