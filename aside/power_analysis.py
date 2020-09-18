import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.ion()
​
​
def example():
    mean = 1
    stds = .25
    n = 10
    a1 = np.random.randn(n)*stds+mean
    a2 = np.random.randn(n)*stds
    plt.figure(figsize=(2,4))
    plt.plot([1]*n,a1,'bo')
    plt.plot([2]*n,a2,'ro')
    plt.xlim(0.5,2.5)
    plt.ylim(-1,2)
    plt.tight_layout()
    ttest = stats.ttest_ind(a1,a2)
    p = ttest[1]/2 # one sided
​
​
def sample(mean, stds, n):
    a1 = np.random.randn(n)*stds+mean
    a2 = np.random.randn(n)*stds
    ttest = stats.ttest_ind(a1,a2)
    p = ttest[1]/2 # one sided
    return p
​
def experiment(mean, std,n,num_reps=20):
    return np.mean([sample(mean,std,n) for i in np.arange(1,num_reps)])
​
def study():
    plt.figure()
    plt.axhline(0.05,linestyle='--', color='k',alpha=0.25)
    n = np.arange(1,25,1)
    p = [experiment(1,0.25,x) for x in n]
    plt.plot(n,p,'ro',label='(1,0.25)')
    p = [experiment(1,0.5,x) for x in n]
    plt.plot(n,p,'bo',label='(1,0.5)')
    p = [experiment(1,1,x) for x in n]
    plt.plot(n,p,'ko',label='(1,1)')  
    plt.legend()
    plt.ylabel('Avg. P value')
