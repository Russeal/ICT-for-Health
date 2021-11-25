import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SolveLLS():

    def __init__(self, y_tr_norm, X_tr_norm, y_te_norm, X_te_norm, sy, my):
        self.y_tr_norm = y_tr_norm
        self.X_tr_norm = X_tr_norm
        self.y_te_norm = y_te_norm
        self.X_te_norm = X_te_norm
        self.sy = sy
        self.my = my


    def LLS_With_Diagrams(self):
        w_hat = np.linalg.inv(self.X_tr_norm.T@self.X_tr_norm)@(self.X_tr_norm.T@self.y_tr_norm)
        y_hat_te_norm = self.X_te_norm@w_hat
        MSE_norm = np.mean((y_hat_te_norm - self.y_te_norm)**2)
        MSE=self.sy**2*MSE_norm



        # plot the regression line
        y_hat_te=(self.X_te_norm@w_hat)*self.sy+self.my
        y_te=self.y_te_norm*self.sy+self.my
        plt.figure(figsize=(6,4))
        plt.plot(y_te,y_hat_te,'.')
        v=plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        plt.xlabel(r'$y$')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title('Linear Least Squares: test')
        plt.tight_layout()
        plt.savefig('./LLS-yhat_vs_y.png')
        plt.show()

        # plot the error histogram
        E_tr = (self.y_tr_norm-self.X_tr_norm@w_hat)*self.sy# training
        E_te = (self.y_te_norm-self.X_te_norm@w_hat)*self.sy# test
        e = [E_tr,E_te]
        plt.figure(figsize=(6,4))
        plt.hist(e,bins=50,density=True, histtype='bar',
                 label=['training','test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel('P(error in bin)')
        plt.legend()
        plt.grid()
        plt.title('LLS-Error histograms')
        plt.tight_layout()
        plt.savefig('./LLS-hist.png')
        plt.show()
        
        
        E_tr_mu = E_tr.mean()
        E_tr_sig = E_tr.std()
        E_tr_MSE = np.mean(E_tr**2)
        y_tr = self.y_tr_norm * self.sy + self.my
        R2_tr = 1 - E_tr_sig**2 / np.mean(y_tr**2)
        E_te_mu = E_te.mean()
        E_te_sig = E_te.std()
        E_te_MSE = np.mean(E_te**2)
        y_te = self.y_te_norm * self.sy + self.my
        R2_te = 1 - E_te_sig**2 / np.mean(y_te**2)
        rows = ['Training', 'test']
        cols = ['mean', 'std', 'MSE', 'R^2']
        p = np.array([[E_tr_mu, E_tr_sig, E_tr_MSE, R2_tr],
                      [E_te_mu, E_te_sig, E_te_MSE, R2_te]])
        results = pd.DataFrame(p, columns =cols, index=rows)
        print(results)
