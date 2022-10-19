import pytest
import numpy as np

def WCR11_not_WCU11():
  
  N = 100
  k = 2
  G= 20
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  beta = np.random.normal(0,1,k)
  beta[0] = 0.1
  u = np.random.normal(0,1,N)
  y = 1 + X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  bootcluster = cluster
  R = np.zeros(k)
  R[0] = 1
  B = 999

  boot = Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = 99999, seed = 12341)
  boot.get_scores(bootstrap_type = "11", impose_null = True)
  boot.get_weights(weights_type = "rademacher")
  boot.get_numer()
  boot.get_denom()
  boot.get_tboot()
  boot.get_vcov()
  boot.get_tstat()
  boot.get_pvalue(pval_type = "two-tailed")
  
  wcu = wb.Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = 99999, seed = 12341)
  wcu.get_scores(bootstrap_type = "11", impose_null = False)
  wcu.get_weights(weights_type = "rademacher")
  wcu.get_numer()
  wcu.get_denom()
  wcu.get_tboot()
  wcu.get_vcov()
  wcu.get_tstat()
  wcu.get_pvalue(pval_type = "two-tailed")

  # score matrices of WCR11 and WCU11 should be different - currently not the case
  assert not np.array_equal(boot.scores_mat, wcu.scores_mat)
  assert not np.array_equal(boot.t_boot, wcu.t_boot)
  assert np.array_equal(boot.t_stat, wcu.t_stat)
  assert not np.array_equal(boot.pvalue, wcu.pvalue) # unless both pvals are zero or 1...


def test_r_vs_py():
  
  # based on data created via the development_notebook.Rmd
  # with B = 99999 bootstrap iterations, WCR11
  # automate this via rpy2 (?) or add reproducible R and Python scripts
  # after wildboottest.py has nice interface for statsmodels/linearmodels
  # to reproduce: search for the commit, run dev notebookm run WCR11 in python
  # etc ...
  
  from wildboottest.wildboottest import wildboottest, Wildboottest
  import statsmodels.api as sm
  import numpy as np
  import pandas as pd
  import statsmodels.api as sm
  
  np.random.seed(7567)
  N = 1000
  k = 2
  # small sample size -> full enumeration
  G= 3
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  beta = np.random.normal(0,1,k)
  beta[0] = 0.005
  u = np.random.normal(0,1,N)
  Y = 1 + X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  B = 99999
  X_df = pd.DataFrame(X)
  Y_df = pd.DataFrame(Y)
  cluster_df = pd.DataFrame(cluster)
  df = pd.concat([X_df, Y_df, cluster_df], axis = 1)  
  df.columns = ['X1', 'X2','Y', 'cluster']
  df.to_csv("~/wildboottest/data/test_df.csv")
  model = sm.OLS(Y_df, X_df)
  
  bootstrap_types = ['11', '31']
  impose_null = [True, False]
  weights_type = "rademacher"
  bootcluster = cluster
  R = np.array([1,0])
  
  boot_tstats = []
  for bootstrap_type in ['11', '31']: 
    for impose_null in [True, False]:
      boot = Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = 12341)
      boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
      boot.get_weights(weights_type = "rademacher")
      boot.get_numer()
      boot.get_denom()
      boot.get_tboot()
      boot.get_vcov()
      boot.get_tstat()
      boot.get_pvalue(pval_type = "two-tailed")
      boot_tstats.append(boot.t_boot)

  # now get all bootstrapped t-stats from fwildclusterboot
  # R and Python need to return *identical results* (due to 
  # deterministic, fully enumerated weights matrix)
  
  df = pd.DataFrame(np.transpose(np.array(boot_tstats)))
  df.columns = ['WCR11', 'WCR31', 'WCU11', 'WCU31']
  
  r_df = pd.read_csv("~/wildboottest/data/test_df_fwc_res.csv")
  
  #asser df['WCR11'].sort_values().equals(r_df['WCU11'].sort_values()) 
  
  

  
def full_enum_works():
  
  N = 1000
  k = 10
  G= 4
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  beta = np.random.normal(0,1,k)
  beta[0] = 0.005
  u = np.random.normal(0,1,N)
  Y = 1 + X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  bootcluster = cluster
  R = np.zeros(k)
  R[0] = 1
  B = 99999
  
  wcr = Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = 12341)
  boot.get_scores(bootstrap_type = "11", impose_null = False)
  boot.get_weights(weights_type = "rademacher")
  boot.get_numer()
  boot.get_denom()
  boot.get_tboot()
  
  assert len(boot.t_boot) == 2**G
  assert boot.full_enumeration == True
  




