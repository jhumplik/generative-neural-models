function Z = getZ_RBM(M_samples, a, b, W, betas)
%getZ_RBM Estimates the partition function of a RBM.
%The estimator uses annealed importance sampling (see e.g. Learning and
%Evaluating Boltzmann Machines, Salakhutdinov R, 2008) to estimate the
%partition function. The sampling chain starts at a model in which all 
%neurons are independent.
%
% Syntax: Z = getZ_RBM(M_samples, a, b, W, betas)
%
% Inputs:
%   M_samples: Number of AIS samples.
%   a, b, W: RBM parameters.
%   betas: Values of a parameter which interpolates between the initial
%          independent model and the actual model. "betas" must be an
%          increasing sequence from 0 to 1 (i.e. betas(1) = 0, 
%          betas(end) = 1). The more values between 0 and 1 the smaller 
%          the variance of the estimator.
%
% Outputs:
%   Z: Estimate of the partition function.
%
% Required m-files: sample_RBM.m

% Define function which returns E_(k-1) - E_k = log(p_k/p_(k-1))
n = size(W, 1);
exp_in = @(samples)exp(bsxfun(@plus, samples*W, b'));
energy_diff = @(samples, beta_k, beta_k_1) ...
   sum(log((1+exp_in(samples).^beta_k)./(1+exp_in(samples).^beta_k_1)), 2);
% Draw samples from independent model and get the first probability ratio.
samples = rand(M_samples, n);
spike_probs = exp(a') ./ (1 + exp(a'));
samples = samples < repmat(spike_probs, [M_samples, 1]);
log_prob_ratios = energy_diff(samples, betas(2), betas(1));
% For every sample, perform AIS.
for k = 1:(length(betas) - 2)
    b_k = betas(k+1)*b;
    W_k = betas(k+1)*W;
    samples = sample_RBM(samples, a, b_k, W_k, 1);
    log_prob_ratios = log_prob_ratios + energy_diff(samples, betas(k+2), betas(k+1));
end
% Calculate Z estimate.
Z_0 = exp(size(W, 2)*log(2) + sum(log(1 + exp(a))));
Z = Z_0 * mean(exp(log_prob_ratios));
end

