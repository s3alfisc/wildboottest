import numpy as np
import pandas as pd
from numba import jit
from itertools import product
from wildboottest.weights import draw_weights
import warnings

class WildDrawFunctionException(Exception):
    pass

class TestMatrixNonConformabilityException(Exception):
  pass

class Wildboottest: 
  
  '''
  Create an object of Wildboottest and get p-value by successively applying
  methods in the following way: 
    
  wb = Wildboottest(X = X, Y = y, cluster = cluster, R = R, B = B)
  wb.get_scores(bootstrap_type = "11", impose_null = True)
  wb.get_numer()
  wb.get_denom()
  wb.get_tboot()
  wb.get_vcov()
  wb.get_tstat()
  wb.get_pvalue()  
  
  Later we can replace X, Y, cluster, R with an object estimated via 
  statsmodels or linearmodels and a "param" values (as in fwildclusterboot::boottest())
  
  '''
  
  def __init__(self, X, Y, cluster, R, B, bootcluster = None, seed = None):
      
      "Initialize the wildboottest class"
      #assert bootstrap_type in ['11', '13', '31', '33']
      #assert impose_null in [True, False]
      
      if bootcluster is None: 
        bootcluster = cluster

      for i in [X, Y, cluster, bootcluster]:
        if isinstance(i, list):
          raise TypeError(f"{i} cannot be a list")

      if isinstance(X, (pd.DataFrame, pd.Series)):
        self.X = X.values
      else:
        self.X = X
        
      if isinstance(Y, (pd.DataFrame, pd.Series)):
        self.Y = Y.values
      else:
        self.Y = Y
        
      if isinstance(cluster, pd.DataFrame):
        self.clustid = cluster.unique()
        self.cluster = cluster.values
      if isinstance(bootcluster, pd.DataFrame):
        self.bootclustid = bootcluster.unique()
        self.bootcluster = bootcluster.values
      else:
        self.clustid = np.unique(cluster)
        self.bootclustid = np.unique(bootcluster)
        self.bootcluster = bootcluster
        
      if isinstance(seed, int):
        np.random.seed(seed)

      self.N_G_bootcluster = len(self.bootclustid)
      self.G  = len(self.clustid)

      self.N = X.shape[0]
      self.k = X.shape[1]
      self.B = B
      self.R = R
      
      if self.X.shape[1] != self.R.shape[0]:
        raise TestMatrixNonConformabilityException("The number of rows in the test matrix R, does not ")
      
      X_list = []
      y_list = []
      tXgXg_list = []
      tXgyg_list = []
      tXX = np.zeros((self.k, self.k))
      tXy = np.zeros(self.k)
      
      #all_cluster = np.unique(bootcluster)
      
      for g in self.bootclustid:
        
        # split X and Y by (boot)cluster
        X_g = self.X[np.where(self.bootcluster == g)]
        Y_g = self.Y[np.where(self.bootcluster == g)]
        tXgXg = np.transpose(X_g) @ X_g
        tXgyg = np.transpose(X_g) @ Y_g
        X_list.append(X_g)
        y_list.append(Y_g)
        tXgXg_list.append(tXgXg)
        tXgyg_list.append(tXgyg)
        tXX = tXX + tXgXg
        tXy = tXy + tXgyg
      
      self.X_list = X_list
      self.Y_list = y_list
      self.tXgXg_list = tXgXg_list
      self.tXgyg_list = tXgyg_list
      self.tXX = tXX
      self.tXy = tXy
        
      self.tXXinv = np.linalg.inv(tXX)
      self.RtXXinv = np.matmul(R, self.tXXinv)
      
  def get_weights(self, weights_type):
    
    self.weights_type = weights_type 
    
    if 2**self.N_G_bootcluster < self.B and weights_type=='rademacher':
      self.full_enumeration = True
      warnings.warn("2^G < the number of boot iterations, setting full_enumeration to True.")
    else: 
      self.full_enumeration = False
      
    self.v, self.B = draw_weights(
      t = self.weights_type, 
      full_enumeration = self.full_enumeration, 
      N_G_bootcluster = self.N_G_bootcluster,
      boot_iter = self.B
    )  
    
  def get_scores(self, bootstrap_type, impose_null, adj = True, cluster_adj = True):
    
      if bootstrap_type[1:2] == '1':
        self.crv_type = "crv1"
        self.ssc = 1
        if(adj == True):
          self.ssc = self.ssc * (self.N - 1) / (self.N - self.k)
        if(cluster_adj == True):
          self.ssc = self.ssc * self.G / (self.G - 1)
      elif bootstrap_type[1:2] == '3':
        self.crv_type = "crv3"
        self.ssc = (self.G - 1) / self.G

      bootstrap_type_x = bootstrap_type[0:1] + 'x'

      if impose_null == True:
        self.bootstrap_type = "WCR" + bootstrap_type_x
      else:
        self.bootstrap_type = "WCU" + bootstrap_type_x
    
      # not needed for all types, but compute anyways
      self.beta_hat = self.tXXinv @ self.tXy

      # precompute required objects for computing scores & vcov's
      if self.bootstrap_type in ["WCR3x"]: 
        
        X = self.X
        X1 = X[:,self.R == 0]
        X1_list = []
        tX1gX1g_list = []
        tX1gyg_list = []
        tXgX1g_list = []
        tX1X1 = np.zeros((self.k-1, self.k-1))
        tX1y = np.zeros(self.k-1)
          
        for ix, g in enumerate(self.bootclustid):
          #ix = g = 1
          X1_list.append(X1[np.where(self.bootcluster == g)])
          tX1gX1g_list.append(np.transpose(X1_list[ix]) @ X1_list[ix])
          tX1gyg_list.append(np.transpose(X1_list[ix]) @ self.Y_list[ix])
          tXgX1g_list.append(np.transpose(self.X_list[ix]) @  X1_list[ix])
          tX1X1 = tX1X1 + tX1gX1g_list[ix]
          tX1y = tX1y + tX1gyg_list[ix]
          
        beta_1g_tilde = []
        
        for ix, g in enumerate(self.bootclustid):
          beta_1g_tilde.append(np.linalg.pinv(tX1X1 - tX1gX1g_list[ix]) @ (tX1y - tX1gyg_list[ix]))

        beta = beta_1g_tilde
        M = tXgX1g_list

      elif self.bootstrap_type in ["WCU3x"]: 
            
        beta_g_hat = []
        for ix, g in enumerate(self.bootclustid):
          beta_g_hat.append(np.linalg.pinv(self.tXX - self.tXgXg_list[ix]) @ (self.tXy - self.tXgyg_list[ix]))
  
        beta = beta_g_hat
        M = self.tXgXg_list
          
      elif self.bootstrap_type in ["WCR1x"]: 
            
        A = 1 / (np.transpose(self.R) @ self.tXXinv @ self.R)
        beta_tilde = self.beta_hat - self.tXXinv @ self.R * A * (self.R @ self.beta_hat - 0)
        beta = beta_tilde
        M = self.tXgXg_list
          
      elif self.bootstrap_type in ["WCU1x"]: 
            
        beta = self.beta_hat 
        M = self.tXgXg_list

      # compute scores based on tXgyg, M, beta
      scores_list = []
      
      if(self.bootstrap_type in ["WCR1x", "WCU1x"]):
        
        for ix, g in enumerate(self.bootclustid):
        
          scores_list.append(self.tXgyg_list[ix] - M[ix] @ beta)
      
      elif(self.bootstrap_type in ["WCR3x", "WCU3x"]):
        
        for ix, g in enumerate(self.bootclustid):
        
          scores_list.append(self.tXgyg_list[ix] - M[ix] @ beta[ix])
        
      self.scores_mat = np.transpose(np.array(scores_list)) # k x G 
      
  def get_numer(self):
    
      # Calculate the bootstrap numerator
      self.Cg = self.R @ self.tXXinv @ self.scores_mat 
      self.numer = self.Cg @ self.v
    
  def get_denom(self):
    
      if self.crv_type == "crv1":
    
        H = np.zeros((self.G, self.G))
    
        # numba optimization possible? 
        for ixg, g in enumerate(self.bootclustid):
          for ixh, h in enumerate(self.bootclustid):
            # can be improved by replacing list calls with matrices; 
            H[ixg,ixh] = self.R @ self.tXXinv @ self.tXgXg_list[ixg] @ self.tXXinv @ self.scores_mat[:,ixh]
  
        # now compute denominator
        # numba / cython / c++ optimization possible? Porting this part from 
        # R to c++ gives good speed improvements
        @jit
        def compute_denom(Cg, H, bootclustid, B, G, v, ssc):
          
          denom = np.zeros(B + 1)
      
          for b in range(0, B+1):
            Zg = np.zeros(G)
            for ixg, g in enumerate(bootclustid):
              vH = 0
              for ixh, h in enumerate(bootclustid):
                vH += v[ixh,b] * H[ixg,ixh]
              Zg[ixg] = Cg[ixg] * v[ixg,b] - vH
            
            # todo: ssc
            denom[b] = ssc * np.sum(np.power(Zg,2))
            
          return denom
          
        self.denom = compute_denom(self.Cg, H, self.bootclustid, self.B, self.G, self.v, self.ssc)
      
      elif self.crv_type == "crv3":
        
        self.inv_tXX_tXgXg = []
        for ix, g in enumerate(self.bootclustid):
          self.inv_tXX_tXgXg.append(np.linalg.pinv(self.tXX - self.tXgXg_list[ix]))
      
        self.denom = np.zeros(self.B + 1)
      
        for b in range(0, self.B + 1):
        
          scores_g_boot = np.zeros((self.G, self.k))
          v_ = self.v[:,b]
      
          for ixg, g in enumerate(self.bootclustid):
        
            scores_g_boot[ixg,:] = self.scores_mat[:,ixg] * v_[ixg]
      
          scores_boot = np.sum(scores_g_boot, axis = 0)
          delta_b_star = self.tXXinv @ scores_boot
      
          delta_diff = np.zeros((self.G, self.k))
      
          for ixg, g in enumerate(self.bootclustid):
        
            score_diff = scores_boot - scores_g_boot[ixg,:]
            delta_diff[ixg,:] = (
        
              (self.inv_tXX_tXgXg[ixg] @ score_diff - delta_b_star)**2
        
              )
          # se's
          self.denom[b] = self.ssc * np.sum(delta_diff, axis = 0)[np.where(self.R == 1)]
      
      
  def get_tboot(self):
    
      t_boot = self.numer / np.sqrt(self.denom)
      self.t_boot = t_boot[1:(self.B+1)] # drop first element - might be useful for comp. of

  def get_vcov(self):
    
    if self.crv_type == "crv1":
          
      meat = np.zeros((self.k,self.k))
      for ixg, g in enumerate(self.bootclustid):
        score = np.transpose(self.X_list[ixg]) @ (self.Y_list[ixg] - self.X_list[ixg] @ self.beta_hat)
        meat += np.outer(score, score)
      
      self.vcov = self.tXXinv @ meat @ self.tXXinv
      
    elif self.crv_type == "crv3": 
  
      # calculate leave-one out beta hat
      beta_jack = np.zeros((self.G, self.k))
      for ixg, g in enumerate(self.bootclustid):
        beta_jack[ixg,:] = (
          np.linalg.pinv(self.tXX - self.tXgXg_list[ixg]) @ (self.tXy - np.transpose(self.X_list[ixg]) @ self.Y_list[ixg])
        )
    
      if not hasattr(self, "beta_hat"):
        beta_hat = self.tXXinv @ self.tXy
      
      beta_center = self.beta_hat
      
      vcov3 = np.zeros((self.k, self.k))
      for ixg, g in enumerate(self.bootclustid):
        beta_centered = beta_jack[ixg,:] - beta_center
        vcov3 += np.outer(beta_centered, beta_centered)
      
      self.vcov =  vcov3
    
        
  def get_tstat(self):
        
    se = np.sqrt(self.ssc * self.R @ self.vcov @ np.transpose(self.R))
    t_stats = self.beta_hat / se
    self.t_stat = t_stats[np.where(self.R == 1)]

  def get_pvalue(self, pval_type = "two-tailed"):
    
    if pval_type == "two-tailed":
      self.pvalue = np.mean(np.abs(self.t_stat) < abs(self.t_boot))
    elif pval_type == "equal-tailed":
      pl = np.mean(self.t_stat < self.t_boot)
      ph = np.mean(self.t_stat > self.t_boot)
      self.pvalue = 2 * min(pl, ph)
    elif pval_type == ">":
      self.pvalue = np.mean(self.t_stat < self.t_boot)
    else: 
      self.pvalue = np.mean(self.t_stat > self.t_boot)


def wildboottest(model, cluster, B, param = None, weights_type = 'rademacher',impose_null = True, bootstrap_type = '11', seed = None):

  
  '''
  Run a wild cluster bootstrap based on an object of class 'statsmodels.regression.linear_model.OLS'
  
  Args: 
    model(statsmodels.regression.linear_model.OLS'): A statsmodels regression object
    param(str): A string of length one, containing the test parameter of interest
    cluster(np.array): A numpy array of dimension one, containing the clustering variable.
    B(int): The number of bootstrap iterations to run
    weights_type(str): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb' or 'normal'. 
                       'rademacher' by default.
    impose_null(logical): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                          True by default. 
    bootstrap_type(str). A string of length one. Allows to choose the bootstrap type 
                         to be run. Either '11', '31', '13' or '33'. '11' by default.
    seed(int). Option to provide a random seed. 
    
  Returns: 
    A wild cluster bootstrapped p-value. 
    
  Example: 
    
    from wildboottest.wildboottest import wildboottest
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    
    np.random.seed(12312312)
    N = 1000
    k = 10
    G = 10
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    X = pd.DataFrame(X)
    X.rename(columns = {0:"X1"}, inplace = True)
    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)
    model = sm.OLS(Y, X)
    wildboottest(model, param = "X1", cluster = cluster, B = 9999)
    wildboottest(model, cluster = cluster, B = 9999)
  '''

  # does model.exog already exclude missing values?
  X = model.exog
  # interestingly, the dependent variable is called 'endogeneous'
  Y = model.endog
  # weights not yet used, only as a placeholder
  weights = model.weights
  
  xnames = model.data.xnames
  ynames = model.data.ynames
  
  pvalues = []
  tstats = []
  
  def generate_stats(param):

    R = np.zeros(len(xnames))
    R[xnames.index(param)] = 1
    # Just test for beta=0
    
    # is it possible to fetch the clustering variables from the pre-processed data 
    # frame, e.g. with 'excluding' observations with missings etc
    # cluster = ...
        
    boot = Wildboottest(X = X, Y = Y, cluster = cluster, 
                        R = R, B = B, seed = seed)
    boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
    boot.get_weights(weights_type = weights_type)
    boot.get_numer()
    boot.get_denom()
    boot.get_tboot()
    boot.get_vcov()
    boot.get_tstat()
    boot.get_pvalue(pval_type = "two-tailed")
    
    pvalues.append(boot.pvalue)
    tstats.append(boot.t_stat)
  
    return pvalues, tstats
    
  if param is None:
    for x in xnames:
      pvalues, tstats = generate_stats(x)
    param = xnames
  elif isinstance(param, str):
    pvalues, tstats = generate_stats(param)
  else:
    raise Exception("`param` not correctly specified")
  
  res = {
    'param': param,
    'statistic': tstats,
    'p-value': pvalues
  }
  
  return pd.DataFrame(res)
  
if __name__ == '__main__':
    import statsmodels.api as sm
    import numpy as np

    np.random.seed(12312312)
    N = 1000
    k = 10
    G= 10
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)
    
    X_df = pd.DataFrame(data=X, columns = [f"col_{i}" for i in range(k)])
    
    Y_df = pd.DataFrame(data=Y, columns = ['outcome'])

    model = sm.OLS(Y_df, X_df)
    
    print("--- WCB ---")
    print(model.fit().summary())
    
    print("--- NonRobust ---")
    print(model.fit(cov_type='wildclusterbootstrap',
              cov_kwds = {'cluster' : cluster,
                          'B' : 9999,
                          'weights_type' : 'rademacher',
                          'impose_null' : True, 
                          'bootstrap_type' : '11', 
                          'seed' : None}).summary())
