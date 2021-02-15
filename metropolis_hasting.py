import numpy as np
from scipy import stats
from tqdm import tqdm
#import warnings
#import time    

def transition_model(x,std):
    return np.random.normal(x,std,(len(x),))


def prior(x,lim_pri):
    sw =0
    i =0
    while sw == 0:
        if lim_pri[i][0]<=x[i]<=lim_pri[i][1]:
            if i < len(x)-1:
                sw = 0
                i = i+1
            else:
                return 1
        else:
            sw = 1
    return 0

def acceptance(x, x_new):
    if x_new>x:      
        return True
    else:
        accept= 0.1
    return (accept < (np.exp(x_new-x)))

def model(x,z):
    y = x[0]*z**2+x[1]
    return y
 
def likelyhood(x,y,model):

    yhat = model(x[0:len(x)-1])
    err = np.linalg.norm(yhat-y)
    lik = -err/(2*x[-1]**2)
    return lik
    
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
            accepted.append(x_new[:])
        else:
            rejected.append(x_new)            
            prob.append(x_new_lik)  
            
    return np.array(accepted), np.array(rejected),prob

zo = np.linspace(0,10,100)
error =np.random.normal(0,1,100)
y = 2.23*zo**2+4.56 + error

#------- Transition Modes ---- #
std_tr = [0.5,0.5,0.01]
tr_model = lambda x: transition_model(x,std_tr)
#------- Prior limits ---- #
pri_lim = [[0,10],[0,10],[0,10]]
pri_model = lambda x: prior(x,pri_lim)
#------- model ---- #
m_model = lambda x: model(x,zo)
#------- lik ---- #
lik= lambda x,y: likelyhood(x,y,m_model)

xoo = [2,4,0.1]
accepted,rejected,prob = metropolis_hastings(lik,pri_model, tr_model,xoo,10000,y,acceptance)



            
                
            
            
