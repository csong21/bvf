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
		num_groups: int 
	) -> list:
		for bucket, posteri in posteriors:

		

		return 
	
	@staticmethod
	def _calculate_expected_loss(
        res_lists: list, num_groups: int
    ) -> list:
        loss_stats = []
        if num_groups == 2:
            a = res_lists[0]
            b = res_lists[1]
            loss_a = np.mean(np.maximum(b - a, 0))
            loss_b = np.mean(np.maximum(a - b, 0))
            loss_stats.append(loss_a)
            loss_stats.append(loss_b)

        if num_groups == 3:
        	a = res_lists[0]
            b = res_lists[1]
            c = res_lists[2]
            loss_a = np.mean(np.maximum(b - a, 0))
            loss_a = np.mean(np.maximum(loss_a, c - a))
            loss_b = np.mean(np.maximum(a - b, 0))
            loss_b = np.mean(np.maximum(loss_b, c - b))
            loss_c = np.mean(np.maximum(a - c, 0))
            loss_b = np.mean(np.maximum(loss_c, b - c))
            loss_stats.append(loss_a)
            loss_stats.append(loss_b)
            loss_stats.append(loss_c)

        if num_groups == 4:
        	a = res_lists[0]
            b = res_lists[1]
            c = res_lists[2]
            d = res_lists[3]

            loss_a = np.mean(np.maximum(b - a, 0))
            loss_a = np.mean(np.maximum(loss_a, c - a))
            loss_a = np.mean(np.maximum(loss_a, d - a))

            loss_b = np.mean(np.maximum(a - b, 0))
            loss_b = np.mean(np.maximum(loss_b, c - b))
            loss_b = np.mean(np.maximum(loss_b, d - b))

            loss_c = np.mean(np.maximum(a - c, 0))
            loss_c = np.mean(np.maximum(loss_c, b - c))
            loss_c = np.mean(np.maximum(loss_c, d - c))

            loss_d = np.mean(np.maximum(a - d, 0))
            loss_d = np.mean(np.maximum(loss_d, b - d))
            loss_d = np.mean(np.maximum(loss_d, c - d))
            
            loss_stats.append(loss_a)
            loss_stats.append(loss_b)
            loss_stats.append(loss_c)
            loss_stats.append(loss_d)

        return loss_stats
