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
        prior_alpha: int, 
        prior_beta: int, 
    ):
        self.column_for_data_analysis = column_for_data_analysis
        self.analysis_type = analysis_type
        self.stan_model = stan_model
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def _generate_bucket_posteriors(
        self, 
        df: pd.DataFrame, 
    ) -> list:

        if self.analysis_type == 'Bayesian-Conversion':
            bp = BayesianConversion(self.column_for_data_analysis, self.prior_alpha, self.prior_beta, self.stan_model)
        if self.analysis_type == 'Bayesian-Revenue':
            bp = BayesianRevenue(self.column_for_data_analysis, self.prior_alpha, self.prior_beta, self.stan_model)
        
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
    ) -> list:
        num_groups = len(posteriors)
        stats = []
        stats_qrts = []
        for bucket, posteri in posteriors:
            qrts = np.percentile(posteri, [5,25,50,75,95])
            stats_qrts.append((bucket, qrts))

        prob_stats = self._calculate_chance_to_beat_all(posteriors, num_groups)
        loss_stats = self._calculate_expected_loss(posteriors, num_groups)
        stats.append((stats_qrts, prob_stats, loss_stats))

        return stats

    
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
        for i in range(num_groups):
            j = 0
            loss_i = 0
            while(j < num_groups):
                if j == i:
                    j += 1
                    continue
                else:
                    diff = res_lists[j] - res_lists[i]
                    loss_i = np.mean(np.maximum(diff, loss_i))
                    j += 1
            loss_stats.append((buckets[i], loss_i))

        return loss_stats
