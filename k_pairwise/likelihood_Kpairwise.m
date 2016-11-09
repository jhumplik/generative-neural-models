function neg_log_L = likelihood_Kpairwise(data, M_samples, J, VK, betas)
%likelihood_Kpairwise Estimates the likelhood of data under a K-pairwise 
%model.
%Partition function is estimated using annealed importance sampling. 
%See getZ_Kpairwise.m.
%
% Syntax: neg_log_L = likelihood_Kpairwise(data, M_samples, J, VK, betas)
%
% Inputs:
%   data: Binary array of size number_of_samples x number_of_neurons.
%   M_samples: Number of AIS samples. See getZ_pairwise.m.
%   J: Coupling matrix of the K-pairwise model.
%   VK: The VKs of the K-pairwise model.
%   betas: Values of a parameter which interpolates between the initial
%          independent model and the actual model. "betas" must be an
%          increasing sequence from 0 to 1 (i.e. betas(1) = 0, 
%          betas(end) = 1). The more values between 0 and 1 the smaller 
%          the variance of the estimator. See getZ_pairwise.m.
%
% Outputs:
%   neg_log_L: Estimate of the negative log-likelihood of data under the
%              model.
%
% Required m-files: getZ_Kpairwise.m

Z = getZ_Kpairwise(M_samples, J, VK, betas);
neg_log_L = (log(Z) + mean(sum(data.*(data*J), 2) ...
                           + VK(sum(data, 2) + 1))) / size(J, 1);

end

