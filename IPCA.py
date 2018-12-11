import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.decomposition import PCA

path = 'transformed-data/factor-model/combined-projection.csv'

mm = np.matmul

class IPCA():
    def __init__(self, cproj):
        self.cproj = cproj
        self.managed_portfolios = self._managed_portfolios(cproj)
        self.mport_rets = self._mport_rets(self.managed_portfolios)
        self.Z = cproj.drop("RET", axis=1)
        self.R = cproj[['RET']]
        self.gamma_b = None
        self.F = None
        
    def init_gamma_b(self, K):
        self.gamma_b = PCA(n_components=K).fit(self.mport_rets).components_.T
        return self.gamma_b
        
    @staticmethod
    def _managed_portfolios(cproj):
        res = cproj.drop("RET", axis=1).apply(lambda col: col * cproj['RET'])
        #res = cproj.apply(lambda row: row.drop("RET") * row['RET'], axis=1)
        return res
    
    @staticmethod
    def _mport_rets(managed_portfolios):
        res = managed_portfolios.groupby(level=1).mean()
        return res
    
    def get_rZt(self, dat):
        lev1 = self.Z.index.get_level_values(1)
        Zt = self.Z.loc[lev1==dat]
        return Zt.values
    
    def get_ipca_data(self):
        data = self.cproj.groupby(level=1).apply(lambda grp: (grp.drop("RET",axis=1).values, grp[['RET']].values))
        return data
    
    def update_ft(self, gamma_b, data):
        ft = []
        for (Zt, rt) in data:
            res = np.linalg.lstsq(mm(Zt,self.gamma_b), rt, rcond=-1)[0]
            ft.append(res.reshape(-1))
        return np.array(ft)
    
    def update_gammab(self, F, data, lmbda = 0.):
        lhs = None
        rhs = None
        for (Zt, rt), ft in zip(data, F):
            Zft = np.kron(Zt, ft)
            lhprod = (np.matmul(Zft.T, Zft))
            mprod = mm(Zft.T, rt)
            if lhs is None:
                lhs = lhprod
                rhs = mprod
            else:
                lhs += lhprod
                rhs += mprod

        darr = self.text_crit()
        darr = [[v]*self.F.shape[1] for v in darr]
        darr = np.array(darr)
        darr = darr.flatten()
        
        lhs += lmbda*np.diag(darr)*1.

        res = mm(np.linalg.inv(lhs),rhs)
        res = res.reshape(-1, ft.shape[0])
        return res
    
    def fit(self, K, n_steps, l2_lmbda=0.0):
        self.init_gamma_b(K)
        data = self.get_ipca_data()
        for i in range(n_steps): # Alternating Least Squares
            print("ALS Step")
            self.F = self.update_ft(self.gamma_b, data)
            self.gamma_b = self.update_gammab(self.F, data, lmbda=l2_lmbda)
            #print(self.r2())
            
    def get_Fhat(self, dat, predictive=False):
        i = np.where(self.cproj.index.levels[1]==dat)[0][0]
        if predictive:
            ft = self.F.mean(axis=0)
        else:
            ft = self.F[i,:]
        return ft
    
    def get_it_data(self, sid, dat, predictive=False):
        rit = self.R.loc[sid, dat]
        zit = self.Z.loc[sid, dat]
        ft = self.get_Fhat(dat, predictive)
        return (rit, zit, ft)
    
    def error(self, sid, dat, predictive=False):
        rit, zit, ft = self.get_it_data(sid, dat, predictive)
        rhat = mm(zit.T, np.matmul(self.gamma_b, ft))
        eps = rit.values[0] - rhat
        return eps
    
    def r2(self):
        dct = {}
        for pname, predictive in {"Predictive": True, "Total": False}.items():
            sq_error = 0.
            for idx in self.Z.index:
                eps = self.error(*idx, predictive=predictive)
                sq_error += (eps*eps)
            denom = (self.R**2).sum().values[0]
            tot_r2 = 1. - sq_error / denom
            dct[pname] = tot_r2

        ser = pd.Series(dct)
        return ser

    def text_crit(self):
        is_text = [col.startswith('text') for col in self.Z.columns]
        return is_text
    
    def text_gamma(self):
        is_text = self.text_crit()
        subs_gamma = self.gamma_b[is_text,:]
        return subs_gamma
    
    def W_stat(self, gamma):
        W = mm(gamma.reshape(-1), gamma.reshape(-1))
        return W
    
    def get_DP(self):
        lev_vals = self.Z.index.get_level_values(1)

        D = []
        P = []
        for dat in self.Z.index.levels[1]:
            Zt = self.Z.loc[lev_vals==dat].values
            xt = self.mport_rets[self.Z.index.levels[1]==dat]
            xt = np.array(xt)
            predt = mm(mm(mm(Zt.T, Zt), self.gamma_b), self.get_Fhat(dat))
            dt = xt - predt
            D.append(dt)
            P.append(predt)
        D = np.array(D)
        P = np.array(P)

        D = D.reshape(D.shape[0], D.shape[2])
        return D,P
    
    def wild_bootstrap(self, B=10):
        D,P = self.get_DP()

        T = np.random.standard_t(5, D.shape[0]) * np.sqrt((5-2.)/5.)

        newxs = []
        Ws = []
        for (the_i, b) in enumerate(range(B)):
            choices = np.random.choice(D.shape[0], D.shape[0])
            resid = (D[choices,:].T * T).T
            newx = P[choices,:] + resid
            newxs.append(newx)
        return newxs
    
    def test_features(self, feat_func=lambda s: s.text_gamma(), K=3, B=10, n_steps=3):
        text_gamma = feat_func(self)
        true_W = self.W_stat(text_gamma)
        newxs = self.wild_bootstrap(B)
        Ws = []
        for the_i, newx in enumerate(newxs):
            new_ipca = IPCA(self.cproj)
            new_ipca.mport_rets = newx
            new_ipca.fit(K=K, n_steps=n_steps)
            o_text_gamma = feat_func(new_ipca)
            a_W = new_ipca.W_stat(o_text_gamma)
            Ws.append(a_W)
            print(the_i, "/", B)

        pval = (np.array([true_W > w for w in Ws])).mean()
        return pval
    
    @staticmethod
    def cons_ipca_filter_columns(cproj, cond):
        res = cproj[[col for col in cproj.columns if cond(col)]]
        res = IPCA(res)
        return res

