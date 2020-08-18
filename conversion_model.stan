data {
  int<lower=0> observations;            
  int<lower=0> observation_conversion_count;
  real<lower=0> prior_alpha;
  real<lower=0> prior_beta;
}

parameters {
  real<lower=0, upper=0.8> theta;
}

model {
  theta ~ beta(prior_alpha, prior_beta);
  observation_conversion_count ~ binomial(observations, theta);
}

