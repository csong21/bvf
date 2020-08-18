data {
  int<lower=0> N;   //observations         
  int<lower=0> C;   //conversion_count;
  real<lower=0> prior_alpha;
  real<lower=0> prior_beta;
  real r[C];         //observed revenue
}

parameters {
  real<lower=0, upper=0.8> lambda;
  real<lower=0> theta;
}

model {
  lambda ~ beta(prior_alpha, prior_beta);
  C ~ binomial(N, lambda);
  for (n in 1:C){
   r[n] ~ exponential(theta);
  }
}

generated quantities {
  real<lower=0> gtv_per_participant;
  real<lower=0> aov;
  gtv_per_participant = lambda * 1 / theta;
  aov = 1/ theta;

}
  