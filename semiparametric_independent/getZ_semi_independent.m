function Z = getZ_semi_independent(M_samples, a, x, bin_centers, betas)
%getZ_semi_independent Estimates the partition function of a semiparametric
%independent model.
%The estimator uses annealed importance sampling (see e.g. Learning and
%Evaluating Boltzmann Machines, Salakhutdinov R, 2008) to estimate the
%partition function. The sampling chain starts at a model in which all 
%neurons are independent.
%
% Syntax: Z = getZ_semi_independent(M_samples, a, x, bin_centers, betas)
%
% Inputs:
%   M_samples: Number of AIS samples.
%   a: The 'biases' of the semiparametric independent model.
%   x: Parameter of the semiparametric in the model.
%   bin_centers: Hyperparameter of the nonlinearity in the model.
%   betas: Values of a parameter which interpolates between the initial
%          independent model and the actual model. "betas" must be an
%          increasing sequence from 0 to 1 (i.e. betas(1) = 0, 
%          betas(end) = 1). The more values between 0 and 1 the smaller 
%          the variance of the estimator.
%
% Outputs:
%   Z: Estimate of the partition function.
%
% Required m-files: sample_semi_independent.m

% Define function which returns V(E_(k-1)) - V(E_k) = log(p_k/p_(k-1))
n = size(a, 1);
x_linear = zeros(size(x));
x_linear(1) = (bin_centers(1) - bin_centers(2))/2;
x_linear(2) = 1;
V = @(E, x)monotone(x', bin_centers, E);
Venergy = @(samples, x)V(samples*a, x);
Venergy_diff = @(samples, b_k, b_k_1)(Venergy(samples, (1-b_k_1)*x_linear + b_k_1*x) - Venergy(samples, (1-b_k)*x_linear + b_k*x));
% Draw samples from p_0 and get the first probability ratio.
samples = rand(M_samples, n);
spike_probs = exp(-a') ./ (1 + exp(-a'));
samples = samples < repmat(spike_probs, [M_samples, 1]);
log_prob_ratios = Venergy_diff(samples, betas(2), betas(1));
% For every sample, perform AIS.
for k = 1:(length(betas) - 2)
    x_k = (1-betas(k+1))*x_linear + betas(k+1)*x;
    samples = sample_semi_independent(samples, a, @(E)V(E, x_k), n);
    log_prob_ratios = log_prob_ratios + Venergy_diff(samples, betas(k+2), betas(k+1));
end
% Calculate Z estimate.
Z_0 = exp(bin_centers(1))*exp(sum(log(1 + exp(-a))));
Z = Z_0 * mean(exp(log_prob_ratios));
end

