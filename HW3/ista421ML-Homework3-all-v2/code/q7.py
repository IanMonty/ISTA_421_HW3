import numpy as np
import matplotlib.pyplot as plt

def true_function(x):
    return 5*x**3 - x**2 + x

def sample_from_function(N=100, noise_var=1000, xmin=-5., xmax=5.):
    x_range = xmax - xmin
    x_mid = x_range/2.
    x = np.sort(x_range*np.random.rand(N) - x_mid)
    t = true_function(x)
    t = t + np.random.randn(x.shape[0])*np.sqrt(noise_var)
    return x,t

x,t = sample_from_function(100)

plt.scatter(x,t)
plt.show()
act_t = t

t = np.sort(t)
orders = [1,3,5,9,11,13,15,17,19,21]
samp = []
for l in range(0,20):
    samp.append(np.random.choice(100,25))

samp = np.array(samp)
##print(samp)


for k in orders:
    X = np.zeros([25,k+1])
    Y = np.zeros([25,1])
    for j in range(0,20):
        samp_pos = samp[j,:]
        for i in range(0,25):
            for r in range(0,k+1):
                X[i,r] = x[samp_pos[i]] ** r
            Y[i,:] = t[samp_pos[i]]
            X = np.array(X)
            Y = np.array(Y)
        AtA = np.linalg.inv(np.dot(np.transpose(X),X))
        Atb = np.dot(np.transpose(X),Y)
        fit = np.dot(AtA,Atb)

        ##print(fit)

        
        fit_x = np.arange(-5.,5.,.1)
        fit_y = np.zeros(fit_x.size)
        fit_act = np.zeros(fit_x.size)
        it = 0
        for n in fit_x:
            pos_y = 0
            for m in range(0,k+1):
                spot = fit[m]* (n ** m)
                pos_y = pos_y + spot

            act_y = 5*n**3 - n**2 + n
            fit_act[it] = act_y
            fit_y[it] = pos_y
            it = it + 1
        ##print(fit_y)

        plt.plot(fit_x,fit_y, color = 'b')
        plt.plot(fit_x,fit_act, color = 'r', linewidth = 3.0)
        plt.scatter(x,act_t)
        ti = ('Plot of best fit of 20 sample regressions having 25 samples each for\n' + \
             'regression of polynomial order {0}' .format(k, i))
        plt.title(ti)
    plt.xlim(-5.5,5.5)
    plt.ylim(-800,800)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
        
    
