// Bayesian hierarchical meta-regression.
//
//   y_i     ~ normal(x_i' beta + theta_{g(i)}, sigma_i)   i = 1..N
//   theta_g ~ normal(0, tau)                              g = 1..K
//
// This is the Stan User's Guide random-effects meta-analysis model (Measurement
// Error and Meta-Analysis, section "Meta-Analysis") with the guide's stated
// extension to trial-specific predictors: the per-observation effects are given
// a regression on X. sigma_i is the *known* sampling standard deviation of
// observation i, i.e. sqrt(v_i) -- Stan's normal() is parameterized by a scale,
// not a variance. tau is the between-group standard deviation, and tau2 = tau^2
// the between-group variance that every other PyMARE estimator reports.
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
  // Non-centered: sampling theta_raw ~ N(0, 1) and scaling by tau avoids the
  // funnel geometry that theta ~ normal(0, tau) produces when tau is near zero.
  // That geometry is the dominant source of divergences in small-K hierarchical
  // models, which is exactly this estimator's use case.
  vector[K] theta = tau * theta_raw;
}
model {
  theta_raw ~ std_normal();
  // Half-normal: the <lower=0> declaration on tau truncates the normal at zero.
  tau ~ normal(0, tau_prior_scale);
  // Equivalent to y ~ normal(X * beta + theta[id], sigma), but the GLM form has
  // hand-derived gradients and is documented as the faster of the two. The
  // vector-alpha/vector-sigma overload takes a per-observation intercept
  // (the group effect) and a per-observation scale (the sampling SD).
  y ~ normal_id_glm(X, theta[id], beta, sigma);
}
generated quantities {
  real<lower=0> tau2 = square(tau);
}
