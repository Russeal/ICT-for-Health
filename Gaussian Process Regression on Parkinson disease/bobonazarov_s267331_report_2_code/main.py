# -*- coding: utf-8 -*-
"""
PEP 8 -- Style Guide for Python Code
https://www.python.org/dev/peps/pep-0008/

@author: visintin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sub.algorithms as myAlgs


def GPR(X_train,y_train,X_val,r2,s2):
    """ Estimates the output y_val given the input X_val, using the training data
    and  hyperparameters r2 and s2"""
    Nva=X_val.shape[0]
    yhat_val=np.zeros((Nva,))
    sigmahat_val=np.zeros((Nva,))
    for k in range(Nva):
        x=X_val[k,:]# k-th point in the validation dataset
        A=X_train-np.ones((Ntr,1))*x
        dist2=np.sum(A**2,axis=1)
        ii=np.argsort(dist2)
        ii=ii[0:N-1];
        refX=X_train[ii,:]
        Z=np.vstack((refX,x))
        sc=np.dot(Z,Z.T)# dot products
        e=np.diagonal(sc).reshape(N,1)# square norms
        D=e+e.T-2*sc# matrix with the square distances
        R_N=np.exp(-D/2/r2)+s2*np.identity(N)#covariance matrix
        R_Nm1=R_N[0:N-1,0:N-1]#(N-1)x(N-1) submatrix
        K=R_N[0:N-1,N-1]# (N-1)x1 column
        d=R_N[N-1,N-1]# scalar value
        C=np.linalg.inv(R_Nm1)
        refY=y_train[ii]
        mu=K.T@C@refY# estimation of y_val for X_val[k,:]
        sigma2=d-K.T@C@K
        sigmahat_val[k]=np.sqrt(sigma2)
        yhat_val[k]=mu
    return yhat_val,sigmahat_val


plt.close('all')
xx=pd.read_csv("data/parkinsons_updrs.csv") # read the dataset
z=xx.describe().T # gives the statistical description of the content of each column
#xx.info()
# features=list(xx.columns)
features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
          'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
          'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
          'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
#%% scatter plots
todrop=['subject#', 'sex', 'test_time',
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']
x1=xx.copy(deep=True)
X=x1.drop(todrop,axis=1)
#%% Generate the shuffled dataframe
np.random.seed(267331)
Xsh = X.sample(frac=1).reset_index(drop=True)
[Np,Nc]=Xsh.shape
F=Nc-1
#%% Generate training, validation and testing matrices
Ntr=int(Np*0.5)  # number of training points
Nva=int(Np*0.25) # number of validation points
Nte=Np-Ntr-Nva   # number of testing points
X_tr=Xsh[0:Ntr] # training dataset
# find mean and standard deviations for the features in the training dataset
mm=X_tr.mean()
ss=X_tr.std()
my=mm['total_UPDRS']# get mean for the regressand
sy=ss['total_UPDRS']# get std for the regressand
# normalize data
Xsh_norm=(Xsh-mm)/ss
ysh_norm=Xsh_norm['total_UPDRS']
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)
Xsh_norm=Xsh_norm.values
ysh_norm=ysh_norm.values
# get the training, validation, test normalized data
X_train_norm=Xsh_norm[0:Ntr]
X_val_norm=Xsh_norm[Ntr:Ntr+Nva]
X_test_norm=Xsh_norm[Ntr+Nva:]
y_train_norm=ysh_norm[0:Ntr]
y_val_norm=ysh_norm[Ntr:Ntr+Nva]
y_test_norm=ysh_norm[Ntr+Nva:]
y_train=y_train_norm*sy+my
y_val=y_val_norm*sy+my
y_test=y_test_norm*sy+my

#%% Apply Gaussian Process Regression
N=10
err_val_2 = np.zeros((300, 3), dtype=float)
i = 0
for s2 in [2*1e-5, 5*1e-5, 1e-4, 2*1e-4, 5*1e-4]:
    for r2 in np.arange(1, 10, 0.25):
        yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2,s2)
        mse = np.mean((y_val_norm-yhat_val_norm)**2)
        err_val_2[i,0] = r2
        err_val_2[i,1] = s2
        err_val_2[i,2] = mse
        i += 1

yhat_train_norm,sigmahat_test=GPR(X_train_norm,y_train_norm,X_train_norm,r2,s2)
yhat_train=yhat_train_norm*sy+my
yhat_test_norm,sigmahat_test=GPR(X_train_norm,y_train_norm,X_test_norm,r2,s2)
yhat_test=yhat_test_norm*sy+my
yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2,s2)
yhat_val=yhat_val_norm*sy+my
err_train=y_train-yhat_train
err_test=y_test-yhat_test
err_val=y_val-yhat_val

#%% plots
plt.figure()
plt.plot(err_val_2[0:35,0],err_val_2[0:35,2],'r', label="s2 = 2*10^(-5)")
plt.plot(err_val_2[36:71,0],err_val_2[36:71,2],'b', label="s2 = 5*10^(-5)")
plt.plot(err_val_2[72:107,0],err_val_2[72:107,2],'y', label="s2 = 10^(-4)")
plt.plot(err_val_2[108:143,0],err_val_2[108:143,2],'k', label="s2 = 2*10^(-4)")
plt.plot(err_val_2[144:179,0],err_val_2[144:179,2],'m', label="s2 = 5*10^(-4)")
plt.legend(loc="upper right")
plt.grid()
plt.xlabel('r2')
plt.ylabel('MSE')
plt.title('MSE -r2-s2')
v=plt.axis()

err_val_3 = err_val_2[0:180,:]
err_val_3=err_val_3[np.argsort(err_val_3[:, 2])]
r2 = err_val_3[0,0]
s2 = err_val_3[0,1]
best_mse = err_val_3[0,2]
plt.plot(r2, best_mse, 'go')
plt.tight_layout()
plt.savefig('./optimization.png')
print("Ideal parameter of r2: {}, s2: {}".format(r2, s2))






#%% plots
plt.figure()
plt.plot(y_test,yhat_test,'.b')
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression')
plt.tight_layout()
plt.savefig('./Gaussian Process Regression.png')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5


plt.figure()
plt.errorbar(y_test,yhat_test,yerr=3*sigmahat_test*sy,fmt='o',ms=2)
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression - with errorbars')
plt.tight_layout()
plt.savefig('./Gaussian Process Regression - with errorbars.png')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5

e=[err_train,err_val,err_test]
plt.figure()
plt.hist(e,bins=50,density=True,range=[-8,17], histtype='bar',label=['Train.','Val.','Test'])
plt.xlabel('error')
plt.ylabel('P(error in bin)')
plt.legend()
plt.grid()
plt.title('Error histogram')
plt.tight_layout()
plt.savefig('./GP_error_hist.png')
plt.show()
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5

print('MSE train',round(np.mean((err_train)**2),3))
print('MSE test',round(np.mean((err_test)**2),3))
print('MSE valid',round(np.mean((err_val)**2),3))
print('Mean error train',round(np.mean(err_train),4))
print('Mean error test',round(np.mean(err_test),4))
print('Mean error valid',round(np.mean(err_val),4))
print('St dev error train',round(np.std(err_train),3))
print('St dev error test',round(np.std(err_test),3))
print('St dev error valid',round(np.std(err_val),3))
print('R^2 train',round(1-np.mean((err_train)**2)/np.std(y_train**2),4))
print('R^2 test',round(1-np.mean((err_test)**2)/np.std(y_test**2),4))
print('R^2 val',round(1-np.mean((err_val)**2)/np.std(y_val**2),4))






#Apply LLS

LLS = myAlgs.SolveLLS(y_train_norm, X_train_norm, y_test_norm, X_test_norm, sy, my)
LLS.LLS_With_Diagrams()








