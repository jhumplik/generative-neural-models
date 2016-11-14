function neg_log_L = likelihood_semi_pairwise(data, M_samples, J, ...
                                                    x, bin_centers, betas)
%likelihood_semi_pairwise Estimates the likelhood of data under a
%semiparametric pairwise model.
%Partition function is estimated using annealed importance sampling. 
%See getZ_semi_pairwise.m. 
%
% Syntax: neg_log_L = likelihood_semi_pairwise(data, M_samples, J, ...
%                                         x, bin_centers, betas)
%
% Inputs:
%   data: Binary array of size number_of_samples x number_of_neurons.
%   M_samples: Number of AIS samples.
%   J: Coupling matrix of the semiparametric pairwise model.
%   x: Parameter of the nonlinearity in the model.
%   bin_centers: Hyperparameter of the nonlinearity in the model.
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
% Required m-files: getZ_semi_pairwise.m

Z = getZ_semi_pairwise(M_samples, J, x, bin_centers, betas);
V = @(E)monotone(x', bin_centers, E);
neg_log_L = (log(Z) + mean(V(sum(data.*(data*J), 2)))) / size(J, 1);

end

