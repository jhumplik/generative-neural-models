function neg_log_L = likelihood_RBM(data, M_samples, a, b, W, betas)
%likelihood_RBM Estimates the likelhood of data under RBM.
%Partition function is estimated using annealed importance sampling. 
%See getZ_RBM.m. 
%
% Syntax: neg_log_L = likelihood_RBM(data, M_samples, a, b, W, betas)
%
% Inputs:
%   data: Binary array of size number_of_samples x number_of_neurons.
%   M_samples: Number of AIS samples.
%   a, b, W: RBM parameters.
%   betas: Values of a parameter which interpolates between the initial
%          independent model and the actual model. "betas" must be an
%          increasing sequence from 0 to 1 (i.e. betas(1) = 0, 
%          betas(end) = 1). The more values between 0 and 1 the smaller 
%          the variance of the estimator.
%
% Outputs:
%   neg_log_L: Estimate of the negative log-likelihood of data under the
%              model.
%
% Required m-files: getZ_RBM.m

Z = getZ_RBM(M_samples, a, b, W, betas);
E = -data*a - sum(log(1+exp(bsxfun(@plus, data*W, b'))),2);
neg_log_L = (log(Z) + mean(E))/size(data, 2);

end

