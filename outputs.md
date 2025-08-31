python regression.py 
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.686
Model:                            OLS   Adj. R-squared:                  0.634
Method:                 Least Squares   F-statistic:                     13.12
Date:                Sun, 31 Aug 2025   Prob (F-statistic):             0.0111
Time:                        12:38:34   Log-Likelihood:                -8.2953
No. Observations:                   8   AIC:                             20.59
Df Residuals:                       6   BIC:                             20.75
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.6429      0.614      4.304      0.005       1.140       4.145
x1             0.4405      0.122      3.622      0.011       0.143       0.738
==============================================================================
Omnibus:                        0.528   Durbin-Watson:                   2.103
Prob(Omnibus):                  0.768   Jarque-Bera (JB):                0.474
Skew:                          -0.054   Prob(JB):                        0.789
Kurtosis:                       1.813   Cond. No.                         11.5
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Predictions: [3.08333333 3.52380952 3.96428571 4.4047619  4.8452381  5.28571429
 5.72619048 6.16666667]