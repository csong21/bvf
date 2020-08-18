data {
  int<lower=0> N;    // number of samples
  real prior_mu;     // prior logit mean
  real prior_sigma;  // prior logit sigma 
  int cc;          // converted control
  int cb;          // converted bucket
  int nc;          // not converted control
  int nb;          // not converted bucket
}

parameters {
  real beta0;        // intercept (estimate for control)
  real beta1;        // difference between classes
}

model {
  beta0 ~ normal(prior_mu, prior_sigma);
  beta1 ~ normal(prior_mu, prior_sigma);
  binomial
  for (n in 1:N){
    y[n] ~ bernoulli(beta0 + beta1 * x[n])
  }

} 
