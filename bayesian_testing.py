import pystan
import pandas as pd
import numpy as np
from bayesian_analysis import BayesianAnalysis, BayesianRevenue, BayesianConversion


class BayesianTesting(object):
	"""
	Performs Bayesian testing on experiment buckets
	"""
	def __init__(
		self,
		column_for_data_analysis: str,
		analysis_type: str,
		stan_model:  pystan.model.StanModel,
	):
		self.column_for_data_analysis = column_for_data_analysis
		self.analysis_type = analysis_type
		self.stan_model = stan_model

	def _generate_bucket_posteriors(
		self, 
		df: pd.DataFrame, 
		prior_alpha: int,
		prior_beta: int
	) -> list:

		if self.analysis_type == 'Bayesian-Conversion':
			bp = BayesianConversion(self.column_for_data_analysis, prior_alpha, prior_beta, self.stan_model)
		if self.analysis_type == 'Bayesian-Revenue':
			bp = BayesianRevenue(self.column_for_data_analysis, prior_alpha, prior_beta, self.stan_model)
		
		unique_buckets = df.bucket.unique()
		posteriors = []
		for bucket in unique_buckets:
			bucket_df = df.loc[df["bucket"] == bucket]
			posteri = bp._generate_posterior_distribution(bucket_df)
			posteriors.append(posteri)

		return posteriors

	def _generate_test_statistics(
		self, 
		posteriors: list, 
	) -> list:

		

		return 
			

	def _calculate_expected_loss(
        self, res_lists: list, num_groups: int
    ) -> list:
        
        if num_groups == 2: #ATM Only support 2 groups, need to expand
            a = res_lists[0]
            b = res_lists[1]
          
            loss_stats = []  
            loss_a = np.mean(np.maximum(b - a, 0))
            loss_b = np.mean(np.maximum(a - b, 0))
            loss_stats.append(loss_a)
            loss_stats.append(loss_b)
        raise ValueError(f'{variant} is misspecified') 

        return loss_stats
