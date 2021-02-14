#import matplotlib.pyplot as plt
import numpy as np
import Op_Pasco_Bridge as OP
import scipy.stats as ss
import Model_Geometry as MG 
import openseespy.opensees as ops
import warnings
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
plt.close('all')
warnings.filterwarnings("ignore")
ops.wipe()
Lo, Ho, zo, z2o = 1200,200,240,240*2
# Beam Properties [mm]
A, Iy, Iz, J, Eo = 51.7, 306.7, 692.1, 76.1,2339
# Geometry
s_Lo, s_Ho, s_zo, s_z2o,s_Eo = 5,20,20,20,5

f1,f2 = 11.47 , 28.5
s1o,s2o,s_s2o,s_s1o = 1,0.5,1,0.1
ymed = [f1,f2]

transition_model = lambda x: [np.random.normal(x[0],s_Lo,(1,)),\
                              np.random.normal(x[1],s_Ho,(1,)),\
                              np.random.normal(x[2],s_zo,(1,)),\
                              np.random.normal(x[3],s_z2o,(1,)),\
                              np.random.normal(x[4],s_s1o,(1,)),\
                              np.random.normal(x[5],s_Eo,(1,)),]

def likelihood(x,ymed):
    ops.wipe()
    
    
    Nodes,conect,Supp = MG.Model_Geometry(float(x[0]),float(x[1]),float(x[2]),float(x[3]))
    freq = OP.Op_Pasco_Bridge(Nodes,conect,Supp,A, Iy, Iz, J, float(x[5]))
    like1 = np.sum(-np.log(x[4] * np.sqrt(2* np.pi) )-((np.array(ymed[0])-np.array(freq[0]))**2) / (2*x[4]**2))
    like2 = np.sum(-np.log(x[4] * np.sqrt(2* np.pi) )-((np.array(ymed[1])-np.array(freq[1]))**2) / (2*x[4]**2))
    
# =============================================================================
#     print('freq',freq)
#   
#     print('frequecias',ymed)
#     print('resta',((np.array(ymed[0])-np.array(freq[0]))))
    like = like1+like2
# =============================================================================
    return like

def prior(x):
    mupri1, sd1 = 1200,200*2
    mupri2, sd2 = 200,100
    mupri3, sd3 = 175,200*2
    mupri4, sd4 = 175*2,66.66*2
    mupri5 = 2
    mupri6, sd6 = 1500,1000
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    #It makes the new sigma infinitp}ely unlikely.
# =============================================================================
#     if(x[0]<=0 or x[1] <=0 or x[2]<=0 or x[3]<=0 or x[4] <=0 or x[5] <=2000):
#        
#         return 0
#     else:
# 
#         return 1
# =============================================================================
    return ss.uniform(mupri1,sd1).pdf(x[0])*ss.norm(mupri2,sd2).pdf(x[1])*\
           ss.norm(mupri3,sd3).pdf(x[2])*ss.norm(mupri4,sd4).pdf(x[3])*\
           mupri5*ss.beta(2, 3).pdf(x[4])*ss.uniform(mupri6,sd6).pdf(x[5])
           
def acceptance(x, x_new):
    
# =============================================================================
#     print('prior i-1:::::',prior(x))
#     print('like i-1::::' ,x)
#     print('like i::::' ,x_new)
# =============================================================================
    if x_new>x:
        return True
    else:
        accept= 0.001
# =============================================================================
#         print('accepp:::::::' , accept)
#         # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
#         # less likely x_new are less likely to be accepted
#         print('Termino raro',accept < (np.exp(x_new-x)))
#         print('termino raro x2',np.exp(x_new-x))
# =============================================================================
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

param_init = [Lo, Ho, zo, z2o,s1o,Eo]

accepted, rejected,prob = metropolis_hastings(likelihood,prior, transition_model,param_init ,100,[f1,f2],acceptance)


for i in range(len(accepted[1,:])):
    plt.figure(1)
    plt.plot(accepted[:,i])
    plt.xlabel('Accepted Samples')
    plt.ylabel('Markov Chain')    
    

for i in range(len(accepted[-5,:])):
    Nodes,conect,Supp = MG.Model_Geometry(float(accepted[i,0]),float(accepted[i,1]),float(accepted[i,2]),float(accepted[i,3]))
    freq = OP.Op_Pasco_Bridge(Nodes,conect,Supp,A, Iy, Iz, J, float(accepted[i,5]))
    print('L  [mm]  ',np.round(accepted[i,0],3))
    print('H  [mm]  ',np.round(accepted[i,1],3))
    print('z  [mm]  ',np.round(accepted[i,2],3))
    print('z2 [mm]  ',np.round(accepted[i,3],3))
    print('E  [Mpa] ',np.round(accepted[i,5],3))
    print('###### Updated modal frequencies#######')
    print('OMA Mode 1 ',f1,' [Hz] ')
    print('OMA Mode 2 ',f2,' [Hz]')  
    print('Mode 1 Opensees',np.round(freq[0], 2),' [Hz]')
    print('Mode 2 Opensees',np.round(freq[1], 2),' [Hz]')
    
L = accepted[:,0]
H = accepted[:,1]
z = accepted[:,2]
z2 = accepted[:,3]
E = accepted[:,5]
plt.figure()
for i in range(0,len(accepted[1,:])):
     plt.plot(accepted[:,i])

# =============================================================================
# trace = np.asmatrix(accepted)
# plt.close('all')
# df2 = pd.DataFrame(trace[1000:-1,0:4], columns = ['L','H','z','z2'])
# scatter_matrix(df2, alpha = 0.04)
# 
# plt.figure
# df2.plot.hist(stacked=True, bins=100,alpha=0.5)
# =============================================================================
# =============================================================================
# scatterplotmatrix(np.asmatrix(accepted), figsize=(10, 8))
# plt.tight_layout()
# plt.show()
# =============================================================================
# =============================================================================
# x = rejected[-1,:]a
# Nodes,conect,Supp = MG.Model_Geometry(float(x[0]),float(x[1]),float(x[2]),float(x[3])+float(x[2]))
# freq = OP.Op_Pasco_Bridge(Nodes,conect,Supp,A, Iy, Iz, J, float(x[5]))
# print(freq)
# =============================================================================


 


#]: plt.pyplot.close('all') like = likelihood(L,z,z2,E,s1,s2)
# =============================================================================
# like = lambda xo: likelihood(xo[0],xo[1],xo[2],xo[3],xo[4],xo[5])
# 
# opt = scipy.optimize.fmin(func=like ,x0=[L,H,z,z2,s1,s2,E],xtol=0.01,maxiter=1000)
# =============================================================================