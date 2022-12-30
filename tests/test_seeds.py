import pytest
from wildboottest.wildboottest import wildboottest
import statsmodels.api as sm
import numpy as np
import pandas as pd


def data(G):
  np.random.seed(12312)
  N = 1000
  k = 3
  # small sample size -> full enumeration
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  X[:,0] = 1
  beta = np.random.normal(0,1,k)
  beta[1] = 0.005
  u = np.random.normal(0,1,N)
  Y = X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  X_df = pd.DataFrame(X)
  Y_df = pd.DataFrame(Y)
  cluster_df = pd.DataFrame(cluster)
  df = pd.concat([X_df, Y_df, cluster_df], axis = 1)  
  df.columns = ['intercept','X1','X2','Y', 'cluster']
  B = 99999

  return df, B


def test_seeds_CL():

    df, B = data(50)
    N = df.shape[0]
    X = df[['intercept', 'X1', 'X2']]
    Y = df['Y']
    R = np.array([0,1,0])

    model = sm.OLS(Y, X)
    
    # for CL
    run1 = wildboottest(model, param = "X1", cluster = df.cluster, B= 999, seed = 12)
    run2 = wildboottest(model, param = "X1", cluster = df.cluster, B= 999, seed = 12)
  
    assert run1.iloc[0][0] == run2.iloc[0][0]
    assert run1.iloc[0][1] == run2.iloc[0][1]
    
    np.random.seed(123)
    run1 = wildboottest(model, param = "X1", cluster = df.cluster, B= 999)
    np.random.seed(123)
    run2 = wildboottest(model, param = "X1", cluster = df.cluster, B= 999)

    assert run1.iloc[0][0] == run2.iloc[0][0]
    assert run1.iloc[0][1] == run2.iloc[0][1]

    run1 = wildboottest(model, param = "X1", cluster = df.cluster, B= 999)
    run2 = wildboottest(model, param = "X1", cluster = df.cluster, B= 999)

    assert run1.iloc[0][0] == run2.iloc[0][0]
    assert run1.iloc[0][1] != run2.iloc[0][1]


def test_seeds_HC():

    df, B = data(50)
    df = df[0:200]
    N = df.shape[0]
    X = df[['intercept', 'X1', 'X2']]
    Y = df['Y']
    R = np.array([0,1,0])

    model = sm.OLS(Y, X)
    
    # for CL
    run1 = wildboottest(model, param = "X1", B= 999, seed = 12)
    run2 = wildboottest(model, param = "X1", B= 999, seed = 12)
  
    assert run1.iloc[0][0] == run2.iloc[0][0]
    assert run1.iloc[0][1] == run2.iloc[0][1]
    
    np.random.seed(123)
    run1 = wildboottest(model, param = "X1", B= 9999)
    np.random.seed(123)
    run2 = wildboottest(model, param = "X1", B= 9999)

    assert run1.iloc[0][0] == run2.iloc[0][0]
    assert run1.iloc[0][1] == run2.iloc[0][1]

    run1 = wildboottest(model, param = "X1", B= 999)
    run2 = wildboottest(model, param = "X1", B= 999)

    assert run1.iloc[0][0] == run2.iloc[0][0]
    assert run1.iloc[0][1] != run2.iloc[0][1]
