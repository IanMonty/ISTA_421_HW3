## approx_expected_value.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling
import numpy as np
import matplotlib.pyplot as plt
##plt.ion()  ## removed because is crashes the plot on my computer
## We are trying to compute the expected value of
# $y^2$
##
# Where
# $p(y)=U(0,1)$
## 
# Which is given by:
# $\int y^2 p(y) dy$
##
# The analytic result is:
# $\frac{1}{3}$
## Generate samples
ys = np.random.rand(3000, 1)
# compute the expectation
ey2 = np.mean(np.sin(ys))
print '\nSample-based approximation: {:f}'.format(ey2)
## Look at the evolution of the approximation
posns = np.arange(1, ys.shape[0], 10)
ey2_evol = np.zeros((posns.shape[0]))
for i in range(posns.shape[0]):
    ey2_evol[i] = np.mean(np.sin(ys[0:posns[i]]))
plt.figure(1)
plt.plot(posns, ey2_evol)
plt.plot(np.array([posns[0], posns[-1]]), np.array([np.cos(0)-np.cos(1), np.cos(0)-np.cos(1)]), color='r')
plt.show()
plt.xlabel('Samples')
plt.ylabel('Approximation')

##raw_input('Press <ENTER> to continue...')

