import pytest
import numpy as np

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

fwildclusterboot = importr("fwildclusterboot")
stats = importr('stats')


def test_r_vs_py_stochastic():
  
  '''
  test compares bootstrapped inference for R and Python 
  versions for large B. 
  p-values, confidence intervals (tba) should be identical
  (difference converges to 0 in probability).
  Non-bootstrapped test statistics should be exactly equal 
  given the same small sample adjustments are applied
  '''

  from wildboottest.wildboottest import wildboottest, Wildboottest
  import statsmodels.api as sm
  import numpy as np
  import pandas as pd

  np.random.seed(7512367)
  N = 1000
  k = 3
  # small sample size -> full enumeration
  G= 25
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
  
  # convert df to an R dataframe
  with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

  r_model = stats.lm("Y ~ X1 + X2", data=r_df)
  R = np.array([0,1,0])


  boot_pvals = []
  
  for bootstrap_type in ['11', '31']: 
    for impose_null in [True, False]:
      for weights_type in ['rademacher','mammen', 'webb','norm']:
        for pval_type in ['two-tailed', 'equal-tailed', '>', '<']:
          
          # python implementation
          boot = Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = cluster, R = R, B = B, seed = 12341)
          boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
          boot.get_weights(weights_type = weights_type)
          boot.get_numer()
          boot.get_denom()
          boot.get_tboot()
          boot.get_vcov()
          boot.get_tstat()
          boot.get_pvalue(pval_type = pval_type)
          boot_pvals.append(boot.pvalue)
          
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

  
  # test condition ... 
