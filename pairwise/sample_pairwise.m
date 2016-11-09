function samples = sample_pairwise(samples, J, n_steps)
%sample_pairwise Samples pairwise model.
%Applies "n_steps" of Gibbs sampling steps to every row in samples.
%
% Syntax: samples = sample_pairwise(samples, J, n_steps)
%
% Inputs:
%   samples: Initial samples for Gibbs sampling, 
%            size is number of samples x number of neurons.
%   J: The coupling matrix of the pairwise model.
%   n_steps: Number of Gibbs sampling steps to apply to every sample.
%
% Outputs:
%   samples: (Approximate) samples from the pairwise model.

% Initialize.
[M, n] = size(samples);
J_offdiag = J;
J_offdiag(eye(n)==1)=0;
neuron_id = 1;
% Perform n_steps of Gibbs sampling.
for j = 1:n_steps
    delta_E = J(neuron_id, neuron_id) + 2*samples*J_offdiag(:,neuron_id);
    p_spike = 1./(1+exp(delta_E));
    samples(:,neuron_id) = rand(M,1) < p_spike;
    neuron_id = neuron_id + 1;
    if neuron_id == n+1
        neuron_id = 1;
    end
end

end

