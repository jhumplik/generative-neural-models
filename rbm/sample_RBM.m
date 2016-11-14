function samples = sample_RBM(samples, a, b, W, n_steps)
%sample_RBM Samples RBM.
%Applies "n_steps" of Gibbs sampling steps to every row in samples.
%
% Syntax: samples = sample_RBM(samples, a, b, W, n_steps)
%
% Inputs:
%   samples: Initial samples for Gibbs sampling, 
%            size is number of samples x number of neurons.
%   a, b, W: RBM parameters.
%   n_steps: Number of Gibbs sampling steps to apply to every sample.
%
% Outputs:
%   samples: (Approximate) samples from the RBM.

[M, n] = size(samples);
n_hidden = length(b);
for j = 1:n_steps
    p_h_given_s = 1./(1+exp(-bsxfun(@plus, samples*W, b')));
    h_samples = rand(M, n_hidden) < p_h_given_s;
    p_s_given_h = 1./(1+exp(-bsxfun(@plus, h_samples*(W'), a')));
    samples = rand(M, n) < p_s_given_h;
end

end

