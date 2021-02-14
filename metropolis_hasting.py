import numpy as np
from scipy import stats
from tqdm import tqdm
#import warnings
#import time    

def transition_model(x,std):
    ts = []
    for i in range(len(x)):
        ts.append(np.random.normal(x[i],std[1],(1,)))
    return ts


def prior(x,lim_pri):
    sw =0
    i =0
    while sw == 0:
        if lim_pri[i][0]<=x[i]<=lim_pri[i][1]:
            if i < len(x)-1:
                sw = 0
                i = i+1
            else:
                return 0
        else:
            sw = 1
    return 1

def acceptance(x, x_new):
    if x_new>x:      
        return True
    else:
        accept= 0.001
    return (accept < (np.exp(x_new-x)))
    
def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,iterations,data,acceptance_rule):

    x = param_init
    accepted = []
    rejected = []  
    prob =[]
    for i in tqdm(range(iterations)):
        x_new =  transition_model(x)    
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data) 
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
            prob.append(x_new_lik)  
            
    return np.array(accepted), np.array(rejected),prob
   

            
                
            
            
