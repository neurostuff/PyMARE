// Bayesian hierarchical meta-regression.
//
//   y_i     ~ normal(x_i' beta + theta_{g(i)}, sigma_i)   i = 1..N
//   theta_g ~ normal(0, tau)                              g = 1..K
//
// The Stan User's Guide random-effects meta-analysis model (Measurement Error
// and Meta-Analysis) with that guide's extension to observation-level
// predictors.
//
// sigma_i is the *known* sampling standard deviation sqrt(v_i): Stan's normal()
// takes a scale, not a variance. tau is the between-group standard deviation;
// tau2 = tau^2 is the variance every other PyMARE estimator reports.
data {
  int<lower=1> N;                            // observations
  int<lower=1> C;                            // predictors (columns of X)
  int<lower=1> K;                            // groups
  vector[N] y;                               // observed effect sizes
  vector<lower=0>[N] sigma;                  // sampling standard deviations
  matrix[N, C] X;                            // one row per observation
  array[N] int<lower=1, upper=K> id;         // 1-based group index per observation
  real<lower=0> tau_prior_scale;             // scale of the half-normal prior on tau
}
parameters {
  vector[C] beta;
  vector[K] theta_raw;
  real<lower=0> tau;
}
transformed parameters {
  // Non-centered. Sampling theta_raw ~ N(0, 1) and scaling avoids the funnel
  // geometry theta ~ normal(0, tau) produces as tau approaches zero, which is
  // the dominant source of divergences when groups are few.
  vector[K] theta = tau * theta_raw;
}
model {
  theta_raw ~ std_normal();
  // Half-normal: the <lower=0> declaration on tau truncates the normal at zero.
  tau ~ normal(0, tau_prior_scale);
  // Equivalent to y ~ normal(X * beta + theta[id], sigma), but documented as the
  // faster form. The vector-alpha/vector-sigma overload takes a per-observation
  // intercept (the group effect) and scale (the sampling SD).
  y ~ normal_id_glm(X, theta[id], beta, sigma);
}
generated quantities {
  real<lower=0> tau2 = square(tau);
}
