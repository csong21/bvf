import pystan
import pandas as pd
from bayesian_analysis import BayesianAnalysis, BayesianRevenue, BayesianConversion
from bayesian_testing import BayesianTesting

sm_revenue = pystan.StanModel(file='revenue_model.stan')

column_for_data_analysis = 'data'
analysis_type = 'Bayesian-Revenue'
stan_model = sm_revenue
prior_alpha = 1
prior_beta = 1

bt = BayesianTesting(column_for_data_analysis, analysis_type, 
                     stan_model, prior_alpha, prior_beta)

posteriors = bt._generate_bucket_posteriors(subxp)
stats = bt._generate_test_statistics(posteriors)

