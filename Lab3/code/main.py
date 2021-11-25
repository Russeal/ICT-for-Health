import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def findROC(x,y):#
    """ findROC(x,y) generates data to plot the ROC curve.
    x and y are two 1D vectors each with length N
    x[k] is the scalar value measured in the test
    y[k] is either 0 (healthy person) or 1 (ill person)
    The output data is a 2D array N rows and three columns
    data[:,0] is the set of thresholds
    data[:,1] is the corresponding false alarm
    data[:,2] is the corresponding sensitivity"""

    if x.min()>0:# add a couple of zeros, in order to have the zero threshold
        x=np.insert(x,0,0)# add a zero as the first element of xs
        y=np.insert(y,0,0)# also add a zero in y

    ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
    ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
    x0=x[ii0]# test values for healthy patients
    x1=x[ii1]# test values for ill patients
    xs=np.sort(x)# sort test values: they represent all the possible  thresholds
    # if x> thresh -> test is positive
    # if x <= thresh -> test is negative
    # number of cases for which x0> thresh represent false positives
    # number of cases for which x0<= thresh represent true negatives
    # number of cases for which x1> thresh represent true positives
    # number of cases for which x1<= thresh represent false negatives
    # sensitivity = P(x>thresh|the patient is ill)=
    #             = P(x>thresh, the patient is ill)/P(the patient is ill)
    #             = number of positives in x1/number of positives in y
    # false alarm = P(x>thresh|the patient is healthy)
    #             = number of positives in x0/number of negatives in y
    Np=ii1.size# number of positive cases
    Nn=ii0.size# number of negative cases
    data=np.zeros((Np+Nn,3),dtype=float)
    i=0
    ROCarea=0
    for thresh in xs:
        n1=np.sum(x1>thresh)#true positives
        sens=n1/Np
        n2=np.sum(x0>thresh)#false positives
        falsealarm=n2/Nn
        data[i,0]=thresh
        data[i,1]=falsealarm
        data[i,2]=sens
        if i>0:
            ROCarea=ROCarea+sens*(data[i-1,1]-data[i,1])
        i=i+1
    return data,ROCarea
#%%
plt.close('all')
xx=pd.read_csv("data/covid_serological_results.csv")
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1 = unclear, 2=illness
Test1=xx.IgG_Test1_titre.values
Test2=xx.IgG_Test2_titre.values
ii=np.argwhere(swab==1).flatten()
#%%
swab=np.delete(swab,ii)
swab=swab//2
Test1=np.delete(Test1,ii)
Test2=np.delete(Test2,ii)
#%%
ii0=np.argwhere(swab==0)
ii1=np.argwhere(swab==1)





test1_V = np.reshape(Test1, (862,1))
db_default = DBSCAN(eps = 3, min_samples = 3).fit(test1_V)
labels = db_default.labels_
ii_labels_1=np.argwhere(labels==-1).flatten()
Test1 = np.delete(Test1,ii_labels_1)
swab1=np.delete(swab,ii_labels_1)
print(ii_labels_1)
print('test')


# Test 2
test2_V = np.reshape(Test2, (862,1))
db_default = DBSCAN(eps = 0.1, min_samples = 3).fit(test2_V)
labels = db_default.labels_
ii_labels_2=np.argwhere(labels==-1).flatten()
Test2 = np.delete(Test2,ii_labels_2)
swab2=np.delete(swab,ii_labels_2)
ii0=np.argwhere(swab2==0)
ii1=np.argwhere(swab2==1)

print(ii_labels_2)
print('test')




plt.figure()
plt.hist(Test2[ii0],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test2[ii1],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()
plt.text(0., 0., 'ICT for Health',
         fontsize=40, color='gray', alpha=0.5,
         ha='left', va='bottom', rotation='30')
plt.title('Test2')


#%%
data_Test2, area=findROC(Test2,swab2)

plt.figure()
plt.plot(data_Test2[:,1],data_Test2[:,2],'-',label='Test2')
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend()
plt.title('ROC - ')
plt.figure()
plt.plot(data_Test2[:,0],data_Test2[:,1],'.',label='False alarm')
plt.plot(data_Test2[:,0],data_Test2[:,2],'.',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()

plt.figure()
plt.plot(data_Test2[:,0],1-data_Test2[:,1],'-',label='Specificity')
plt.plot(data_Test2[:,0],data_Test2[:,2],'-',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()


