import pandas as pd
import numpy as np

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm

def table_summary(df):
  summary = df.describe()
  summary['Variables'] = summary.index
  summary = summary[['Variables'] + [col for col in summary.columns if col != 'Variables']]
  return summary

def table_box(dict_reg):
  model = dict_reg['OLS']
  acorr_result = acorr_ljungbox(model.resid)
  test_df = pd.DataFrame({
    'Lag': range(1,11),
    'Statistique de test': acorr_result.lb_stat,
    'p-valeur': acorr_result.lb_pvalue
  })
  return test_df
    
def table_vif(dict_reg):
  model = dict_reg['OLS']
  X = model.model.exog
  vif = pd.DataFrame(
    {
      'Variable': model.model.exog_names,
      'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])],
    }
  )
  return vif.iloc[1:]

def table_odds(dict_reg):
  model = list(dict_reg.values())[0]
  odds_ratios = pd.DataFrame(
    {
      'Variable': model.model.exog_names,
      "OR": np.exp(model.params),
      "Lower CI": np.exp(model.conf_int()[0]),
      "Upper CI": np.exp(model.conf_int()[1]),
    }
  )
  return odds_ratios.iloc[1:]

def table_seuils(dict_reg):
  model = dict_reg['Ordered']
  threshold_params = model.params[model.model.k_vars:]
  threshold_values = model.model.transform_threshold_params(threshold_params)
  threshold_values_rounded = np.round(threshold_values, 4)[1:-1]

  threshold_df = pd.DataFrame({
      'Intercept': threshold_params.index,
      'Vraie coeff': threshold_values_rounded
  })
  return threshold_df