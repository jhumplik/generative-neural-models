function samples = sample_semi_independent(samples, a, V, n_steps)
%sample_semi_independent Samples semiparametric independent model.
%Applies "n_steps" of Gibbs sampling steps to every row in samples.
%
% Syntax: samples = sample_semi_independent(samples, a, V, n_steps)
%
% Inputs:
%   samples: Initial samples for Gibbs sampling, 
%            size is number of samples x number of neurons.
%   a: The 'biases' of the semiparametric independent model.
%   V: A function V: R -> R of the semiparametric independent model.
%   n_steps: Number of Gibbs sampling steps to apply to every sample.
%
% Outputs:
%   samples: (Approximate) samples from the semiparametric 
%            independent model.

% Initialize.
[M,n] = size(samples);
neuron_id = 1;
E_samples = samples*a;
% Perform n_steps of Gibbs sampling.
for j = 1:n_steps
    % Calculate energy when current neuron = 0/1.
    deltaE = a(neuron_id);
    E0 = E_samples - samples(:, neuron_id).*deltaE;
    E1 = E_samples + (1 - samples(:, neuron_id)).*deltaE;
    % Calculate current neurons' spiking probability, flip neurons, and
    %update the energy of samples.
    p_spike = 1./(1+exp(V(E1) - V(E0)));
    flipped = rand(M,1) < p_spike;
    E_samples = E_samples + (flipped - samples(:,neuron_id)).*deltaE;
    samples(:,neuron_id) = flipped;
    % Move on to the next neuron.
    neuron_id = neuron_id + 1;
    if neuron_id == n+1
        neuron_id = 1;
    end
end

end

