import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SolveStochasticGradientWithADAM():

    def __init__(self, y_tr_norm, X_tr_norm, Nf, Ntr):
        self.y_tr_norm = y_tr_norm
        self.X_tr_norm = X_tr_norm
        self.Nf = Nf
        self.Ntr = Ntr


    def get_W_hat(self):
        epochs = 30
        gamma = 1e-3
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-6
        w_hat = np.random.rand(self.Nf, )
        err = np.zeros((epochs, 2))

        mean = np.zeros((self.Nf, ))
        mean_sv = np.zeros((self.Nf, ))

        for n in range(epochs):
            # error = (y_tr_norm - X_tr_norm @ w_hat)**2

            for i in range(self.Ntr):
                error_grad = 2 * (self.X_tr_norm.iloc[i, :]@w_hat - self.y_tr_norm[i]) * self.X_tr_norm.T.iloc[:, i]
                # for simple stoch grad  ==>  {{  w_hat = w_hat - gamma*error_grad  }}

                # for adam bro
                mean = beta_1 * mean + (1 - beta_1) * error_grad
                mean_sv = beta_2 * mean_sv + (1 - beta_2) * error_grad**2

                mean_hat = mean / (1 - beta_1**(n + 1))
                mean_sv_hat = mean_sv / (1 - beta_2**(n + 1))

                w_hat = w_hat - (gamma * mean_hat)/(np.sqrt(mean_sv_hat) + epsilon)

            err[n, 0] = n
            err[n, 1] = np.mean((self.X_tr_norm @ w_hat - self.y_tr_norm)**2)

        return w_hat, err




class SolveMinibatch():

    def __init__(self, y_tr_norm, X_tr_norm, Nf, Ntr):
        self.y_tr_norm = y_tr_norm
        self.X_tr_norm = X_tr_norm
        self.Nf = Nf
        self.Ntr = Ntr

    def get_W_hat(self, n_batches):

        epochs = 30
        gamma = 1e-4
        w_hat = np.random.rand(self.Nf, )
        err = np.zeros((epochs, 2))
        
        for n in range(epochs):

            for i in range(0, self.Ntr, n_batches):

                if i+n_batches <= self.Ntr:
                    mini_X_tr_norm = self.X_tr_norm.iloc[i:i+n_batches, :]
                    mini_y_tr_norm = self.y_tr_norm[i:i+n_batches]
                else:
                    mini_X_tr_norm = self.X_tr_norm.iloc[i:, :]
                    mini_y_tr_norm = self.y_tr_norm[i:]

                error_grad = 2 * (mini_X_tr_norm.T @ mini_X_tr_norm @ w_hat - mini_X_tr_norm.T @ mini_y_tr_norm)
                w_hat = w_hat - gamma * error_grad
            err[n, 0] = n
            err[n, 1] = np.mean((self.X_tr_norm @ w_hat - self.y_tr_norm)**2)
            # print("batch {} epoch{} error {}".format(n_batches,n,np.mean((self.X_tr_norm @ w_hat - self.y_tr_norm)**2)))

        return w_hat, err





class plotDiagrams():

    def __init__(self, y_tr_norm, y_te_norm, X_tr_norm, X_te_norm, w_hat, sy, my, title, err):

        y_hat_te_norm = X_te_norm @ w_hat
        MSE_norm = np.mean((y_hat_te_norm - y_te_norm)**2)

        # plot the optimum weight vector
        regressors = list(X_tr_norm.columns)
        nn=np.arange(len(w_hat))
        plt.figure(figsize=(6,4))
        plt.plot(nn,w_hat,'-o')
        plt.xticks(nn, regressors, rotation=90)#, **kwargs)
        plt.ylabel(r'$\^w(n)$')
        plt.title(title + '-Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig('./' + title + '-w_hat.png')
        plt.show()

        #################################################
        #plot error on epochs
        plt.plot(err[:, 0] + 1, err[:, 1], 'r')
        plt.xticks(np.arange(0, 31, step=2))
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.title(title + '-Mean Square Error')
        plt.grid()
        plt.savefig('./' + title + '-err.png')
        plt.show()

        # plot the error histogram

        E_tr = (y_tr_norm - X_tr_norm @ w_hat) * sy         # training
        E_te = (y_te_norm - X_te_norm @ w_hat) * sy         # test
        e = [E_tr, E_te]

        plt.figure(figsize=(6,4))
        plt.hist(e,bins=50,density=True, histtype='bar',
                 label=['training','test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(title + '-Error histograms')
        plt.tight_layout()
        plt.savefig('./' + title + '-hist.png')
        plt.show()

        #################################################

        # plot the regression line

        y_te = y_te_norm * sy + my
        y_hat_te=y_hat_te_norm * sy + my

        plt.figure(figsize=(6,4))
        plt.plot(y_te,y_hat_te,'.')
        v = plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        # plt.ylim(0, 60)
        plt.xlabel(r'$y$')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title(title + '-test')
        plt.tight_layout()
        plt.savefig('./' + title + '-yhat_vs_y.png')
        plt.show()

        #################################################

        #%% statistics of the errors
        E_tr_mu = E_tr.mean()
        E_tr_sig = E_tr.std()
        E_tr_MSE = np.mean(E_tr**2)
        y_tr = y_tr_norm * sy + my
        R2_tr = 1 - E_tr_sig**2 / np.mean(y_tr**2)
        E_te_mu = E_te.mean()
        E_te_sig = E_te.std()
        E_te_MSE = np.mean(E_te**2)
        y_te = y_te_norm * sy + my
        R2_te = 1 - E_te_sig**2 / np.mean(y_te**2)
        rows = ['Training', 'test']
        cols = ['mean', 'std', 'MSE', 'R^2']
        p = np.array([[E_tr_mu, E_tr_sig, E_tr_MSE, R2_tr],
                      [E_te_mu, E_te_sig, E_te_MSE, R2_te]])
        results = pd.DataFrame(p, columns =cols, index=rows)
        print(results)


