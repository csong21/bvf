import pandas as pd 
import numpy as np 
import pystan

class BayesianAnalysis(object):
    """
    Perform Bayesian analysis on a dateset

    """
    def __init__(
        self, 
        column_for_data_analysis: str,
        prior_alpha: int, 
        prior_beta: int, 
        stan_model: pystan.model.StanModel
    ):
        self.column_for_data_analysis = column_for_data_analysis
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.stan_model = stan_model

    def _generate_posterior_distribution(
        self, df: pd.DataFrame, 
    ) -> np.array:
        pass 

class BayesianRevenue(BayesianAnalysis):
    def __init__(
        self, 
        column_for_data_analysis: str,
        prior_alpha: int, 
        prior_beta: int, 
        stan_model: pystan.model.StanModel

    ):
        self.column_for_data_analysis = column_for_data_analysis
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.stan_model = stan_model

    def _generate_posterior_distribution(
        self, df: pd.DataFrame, 
    ) -> np.array:

        """
        Performs MCMC sampling to generate posteriror distributions on revenue.
        """
        # For converted users:
        idx = df[self.column_for_data_analysis] > 0
        subdf = df.loc[idx]
        # Generate a conversion metric based on revenue 
        y = idx.values.astype(np.int64)
        r = subdf[self.column_for_data_analysis].values

        dat = {
            'N': len(y),
            'C': y.sum(),
            'prior_alpha' : self.prior_alpha,
            'prior_beta': self.prior_beta,
            'r': r
        }

        fit= self.stan_model.sampling(data=dat, iter=1000, chains=1)

        res = fit.extract(permuted=True)
        # Samples from the posterior distribution of avg. revenue
        res = res['gtv_per_participant'] # HARD CODED IN THE STAN MODEL NOW
        return res


class BayesianConversion(BayesianAnalysis):
    def __init__(
        self, 
        column_for_data_analysis: str,
        prior_alpha: int, 
        prior_beta: int, 
        stan_model: pystan.model.StanModel

    ):
        self.column_for_data_analysis = column_for_data_analysis
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.stan_model = stan_model

    def _generate_posterior_distribution(
        self, 
        df: pd.DataFrame, 
    ) -> np.array:

        """
        Performs MCMC sampling to generate posteriror distributions on conversion rates.
        """

        y = df[self.column_for_data_analysis].values.astype(np.int64)

        dat = {
            'observations': len(y),
            'observation_conversion_count': y.sum(),
            'prior_alpha' : self.prior_alpha,
            'prior_beta': self.prior_beta
        }

        fit= self.stan_model.sampling(data=dat, iter=1000, chains=1)

        res = fit.extract(permuted=True)
        # Samples from the posterior distribution of conversion rate
        res = res['theta'] # HARD CODED IN THE STAN MODEL NOW
        return res
