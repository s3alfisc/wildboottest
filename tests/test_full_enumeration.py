import pytest
import numpy as np

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

fwildclusterboot = importr("fwildclusterboot")
stats = importr('stats')

def test_r_vs_py_deterministic():
  
  '''
  test compares bootstrapped t-statistics for R and Python 
  versions in the full enumeration case. Under full enum, 
  the weights matrices are identical (up to the ordering 
  of columns), and therefore bootstrap t-statistics need to be
  *exactly* identical (if the same small sample correction 
  is applied). 
  '''
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

  np.random.seed(12312)
  N = 10000
  k = 3
  # small sample size -> full enumeration
  G= 5
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  X[:,0] = 1
  beta = np.random.normal(0,1,k)
  beta[1] = 0.005
  u = np.random.normal(0,1,N)
  Y = X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  B = 99999
  X_df = pd.DataFrame(X)
  Y_df = pd.DataFrame(Y)
  cluster_df = pd.DataFrame(cluster)
  df = pd.concat([X_df, Y_df, cluster_df], axis = 1)  
  df.columns = ['intercept','X1','X2','Y', 'cluster']
  #df.to_csv("data/test_df.csv")
  
  # convert df to an R dataframe
  with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

  r_model = stats.lm("Y ~ X1 + X2", data=r_df)
  R = np.array([0,1,0])
  
  boot_tstats = []
  fwildclusterboot_boot_tstats = []
  
  for bootstrap_type in ['11', '31']: 
    for impose_null in [True, False]:
      # python implementation
      boot = Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = cluster, R = R, B = B, seed = 12341)
      boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
      boot.get_weights(weights_type = "rademacher")
      boot.get_numer()
      boot.get_denom()
      boot.get_tboot()
      boot.get_vcov()
      boot.get_tstat()
      boot.get_pvalue(pval_type = "two-tailed")
      boot_tstats.append(boot.t_boot)
      
      # R implementation
      r_t_boot = fwildclusterboot.boottest(
        r_model,
        param = "X1",
        clustid = ro.Formula("~cluster"),
        B=99999,
        bootstrap_type=bootstrap_type,
        impose_null=impose_null,
        ssc=fwildclusterboot.boot_ssc(adj=False, cluster_adj=False)
      )
      
      fwildclusterboot_boot_tstats.append(list(r_t_boot.rx2("t_boot")))
      
  df = pd.DataFrame(np.transpose(np.array(boot_tstats)))
  df.columns = ['WCR11', 'WCR31', 'WCU11', 'WCU31']
  
  # r_df = pd.read_csv("data/test_df_fwc_res.csv")[['WCR11', "WCR31", "WCU11", "WCU31"]]
  r_df = pd.DataFrame(np.transpose(np.array(fwildclusterboot_boot_tstats)))
  r_df.columns = ['WCR11', 'WCR31', 'WCU11', 'WCU31']
  
  # all values need to be sorted
  print("Python")
  print(df.sort_values(by=list(df.columns),axis=0).head())
  print("\n")
  print("R")
  print(r_df.sort_values(by=list(r_df.columns),axis=0).head())  
  
  def mse(x, y):
    return np.mean(np.power(x - y, 2))
  
  assert mse(df['WCR11'].sort_values(), r_df['WCR11'].sort_values()) < 1e-15
  assert mse(df['WCU11'].sort_values(), r_df['WCU11'].sort_values()) < 1e-15
  assert mse(df['WCR31'].sort_values(), r_df['WCR31'].sort_values()) < 1e-15
  assert mse(df['WCU31'].sort_values(), r_df['WCU31'].sort_values()) < 1e-15

  
  
