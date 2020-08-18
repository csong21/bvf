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
			posteriors.append((bucket, posteri))

		return posteriors

	def _generate_test_statistics(
		self, 
		posteriors: list,
		num_groups: int 
	) -> list:
		for bucket, posteri in posteriors:

		

		return 
	
	@staticmethod
	def _calculate_chance_to_beat_all(
		posteriors: list, num_groups: int
	) -> list:
		buckets, res_lists = zip(*posteriors)
		prob_stats = []
		if num_groups == 2:
			bucket_a = buckets[0]
        	bucket_b = buckets[1]
			a = res_lists[0]
			b = res_lists[1]
			diff = a - b
			prob_a = np.sum(diff>0)/len(diff)
			prob_b = 1 - prob_a
			prob_stats.append((bucket_a, prob_a))
			prob_stats.append((bucket_b, prob_b))

		if num_groups == 3:
			bucket_a = buckets[0]
        	bucket_b = buckets[1]
        	bucket_c = buckets[2]
			a = res_lists[0]
			b = res_lists[1]
			c = res_lists[2]
			diff_ab = a - b
			diff_ac = a - c
			diff_bc = b - c
			samples = len(diff_ab)

			prob_a = np.sum((diff_ac > 0) & (diff_ab > 0))/samples
			prob_b = np.sum((diff_bc > 0) & (diff_ab < 0))/samples
			prob_c = np.sum((diff_ac < 0) & (diff_bc < 0))/samples
			prob_stats.append((bucket_a, prob_a))
			prob_stats.append((bucket_b, prob_b))
			prob_stats.append((bucket_c, prob_c))
		return prob_stats


	@staticmethod
	def _calculate_expected_loss(
        posteriors: list, num_groups: int
    ) -> list:
    	buckets, res_lists = zip(*posteriors)
        loss_stats = []
        if num_groups == 2:
        	bucket_a = buckets[0]
        	bucket_b = buckets[1]
            a = res_lists[0]
            b = res_lists[1]
            loss_a = np.mean(np.maximum(b - a, 0))
            loss_b = np.mean(np.maximum(a - b, 0))
            loss_stats.append((bucket_a, loss_a))
            loss_stats.append((bucket_b, loss_b))

        if num_groups == 3:
        	bucket_a = buckets[0]
        	bucket_b = buckets[1]
        	bucket_c = buckets[2]
        	a = res_lists[0]
            b = res_lists[1]
            c = res_lists[2]
            loss_a = np.mean(np.maximum(b - a, 0))
            loss_a = np.mean(np.maximum(loss_a, c - a))
            loss_b = np.mean(np.maximum(a - b, 0))
            loss_b = np.mean(np.maximum(loss_b, c - b))
            loss_c = np.mean(np.maximum(a - c, 0))
            loss_b = np.mean(np.maximum(loss_c, b - c))
            loss_stats.append((bucket_a, loss_a))
            loss_stats.append((bucket_b, loss_b))
            loss_stats.append((bucket_c, loss_c))

        if num_groups == 4:
        	bucket_a = buckets[0]
        	bucket_b = buckets[1]
        	bucket_c = buckets[2]
        	bucket_d = buckets[3]
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

            loss_stats.append((bucket_a, loss_a))
            loss_stats.append((bucket_b, loss_b))
            loss_stats.append((bucket_c, loss_c))
            loss_stats.append((bucket_d, loss_d))

        return loss_stats
