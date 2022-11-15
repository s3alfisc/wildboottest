def test_error_warnings():
  
  '''
  test that errors and warnings are thrown when appropriate for 
  both the statsmodels interface and the Wildboottest method, e.g.
  - that an error is thrown when regression weights are used
  - ... other things supported by statsmodels.OLS but not wildboottest
    are tried
  - know edge cases that lead to errors provide valuable info (e.g. WCR with 
    one param regressions)
  '''
  assert 1 == 1
