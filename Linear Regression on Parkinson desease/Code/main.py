# -*- coding: utf-8 -*-
"""
PEP 8 -- Style Guide for Python Code
https://www.python.org/dev/peps/pep-0008/

@author: visintin
without classes and methods
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sub.algs as myAlgs

plt.close('all') # close all the figures that might still be open from previous runs
x=pd.read_csv("data/parkinsons_updrs.csv") # read the dataset; xx is a dataframe
x.describe().T # gives the statistical description of the content of each column
x.info()
features=list(x.columns)
print(features)
#features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
#       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
#       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
#       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
X=x.drop(['subject#','test_time'], axis=1)# drop unwanted features
Np,Nc=X.shape# Np = number of rows/ptients Nf=number of features+1 (total UPDRS is included)
features=list(X.columns)
#%% correlation
Xnorm=(X-X.mean())/X.std()# normalized data
c=Xnorm.cov()# note: xx.cov() gives the wrong result

plt.figure()
plt.matshow(np.abs(c.values),fignum=0)
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)
plt.colorbar()
plt.title('Covariance matrix of the features')
plt.tight_layout()
plt.savefig('./corr_coeff.png') # save the figure
plt.show()

plt.figure()
c.motor_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation=90)#, **kwargs)
plt.title('corr. coeff. among motor UPDRS and the other features')
plt.tight_layout()
plt.show()


plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.title('corr. coeff. among total UPDRS and the other features')
plt.tight_layout()
plt.show()


#%% Generate the shuffled data
np.random.seed(267331) # set the seed for random shuffling
indexsh=np.arange(Np)
np.random.shuffle(indexsh)
Xsh=X.copy(deep=True)
Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False)
Xsh=Xsh.sort_index(axis=0)
#%% Generate training, validation and test matrices
Ntr=int(Np*0.75)  # number of training points
Nte=Np-Ntr   # number of test points
#%% evaluate mean and st.dev. for the training data only
X_tr=Xsh[0:Ntr]# dataframe that contains only the training data
mm=X_tr.mean()# mean (series)
ss=X_tr.std()# standard deviation (series)
my=mm['total_UPDRS']# mean of motor UPDRS
sy=ss['total_UPDRS']# st.dev of motor UPDRS
#%% Normalize the three subsets
Xsh_norm=(Xsh-mm)/ss# normalized data
ysh_norm=Xsh_norm['total_UPDRS']# regressand only
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)# regressors only

X_tr_norm=Xsh_norm[0:Ntr]
X_te_norm=Xsh_norm[Ntr:]
y_tr_norm=ysh_norm[0:Ntr]
y_te_norm=ysh_norm[Ntr:]

#%% Linear Least Squares
w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)
y_hat_te_norm=X_te_norm@w_hat
#y_hat_te=sy*h_hat_te_norm+my
#MSE=np.mean((y_hat_te-y_te)**2)
MSE_norm=np.mean((y_hat_te_norm-y_te_norm)**2)
MSE=sy**2*MSE_norm
#%% plots
# plot the optimum weight vector
regressors=list(X_tr_norm.columns)
Nf=len(w_hat)
nn=np.arange(Nf)
plt.figure(figsize=(6,4))
plt.plot(nn,w_hat,'-o')
ticks=nn
plt.xticks(ticks, regressors, rotation=90)#, **kwargs)
plt.ylabel(r'$\^w(n)$')
plt.title('LLS-Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig('./LLS-what.png')
plt.show()
# plot the error histogram
E_tr=(y_tr_norm-X_tr_norm@w_hat)*sy# training
E_te=(y_te_norm-X_te_norm@w_hat)*sy# test
e=[E_tr,E_te]
plt.figure(figsize=(6,4))
plt.hist(e,bins=50,density=True, histtype='bar',
         label=['training','test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS-Error histograms')
plt.tight_layout()
plt.savefig('./LLS-hist.png')
plt.show()
# plot the regression line
y_hat_te=(X_te_norm@w_hat)*sy+my
y_te=y_te_norm*sy+my
plt.figure(figsize=(6,4))
plt.plot(y_te,y_hat_te,'.')
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.xlabel(r'$y$')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./LLS-yhat_vs_y.png')
plt.show()
#%% statistics of the errors
E_tr_mu=E_tr.mean()
E_tr_sig=E_tr.std()
E_tr_MSE=np.mean(E_tr**2)
y_tr=y_tr_norm*sy+my
R2_tr=1-E_tr_sig**2/np.mean(y_tr**2)
E_te_mu=E_te.mean()
E_te_sig=E_te.std()
E_te_MSE=np.mean(E_te**2)
y_te=y_te_norm*sy+my
R2_te=1-E_te_sig**2/np.mean(y_te**2)
rows=['Training','test']
cols=['mean','std','MSE','R^2']
p=np.array([[E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr],
            [E_te_mu,E_te_sig,E_te_MSE,R2_te]])
results=pd.DataFrame(p,columns=cols,index=rows)
print(results)


print("Stoch:")
# # Solve Stochastic gradient algorithm with Adam optimization
sga_with_Adam = myAlgs.SolveStochasticGradientWithADAM(y_tr_norm, X_tr_norm, Nf, Ntr)
w_hat, err = sga_with_Adam.get_W_hat()
title = "Stochastic Graident with Adam optimizer"

myAlgs.plotDiagrams(y_tr_norm, y_te_norm, X_tr_norm, X_te_norm, w_hat, sy, my, title, err)





# for n_batches in range(2, 50, 3):
n_batches = 2
print("Minibatches:")
    # # Solve Minibatch
minibatch = myAlgs.SolveMinibatch(y_tr_norm, X_tr_norm, Nf, Ntr)
w_hat, err = minibatch.get_W_hat(n_batches)
title = "Minibatch with " + str(n_batches) + " batches: "

myAlgs.plotDiagrams(y_tr_norm, y_te_norm, X_tr_norm, X_te_norm, w_hat, sy, my, title, err)








