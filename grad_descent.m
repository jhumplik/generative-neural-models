function pars = grad_descent(grad, pars0, learning_rate, ...
                             samples_batch, iter)
%grad_descent Gradient descent. Passes "samples_batch" from 
%iteration T to the gradient function which generates (and utilizes) 
%"samples_batch" at iteration T+1.
%
% Syntax: pars = grad_descent(grad, pars0, learning_rate, ...
%                             samples_batch, iter)
%
% Inputs:
%   grad: A callable [g, samples_batch] = grad(pars, samples_batch),
%         where g is the gradient of the optimized function at "pars".
%   pars0: Initial guess.
%   learning_rate: Learning rate.
%   samples_batch: An array of samples which are being updated by "grad".
%   iter: Number of iterations.
%
% Outputs:
%   pars: Whathever gradient descent converged to in "iter" iterations.

pars = pars0;
[g, samples_batch] = grad(pars0, samples_batch);
iteration = 0;
while iteration <= iter
    pars = pars - learning_rate.*g;
    [g, samples_batch] = grad(pars, samples_batch);
    iteration = iteration + 1;
    disp(['Iteration: ', num2str(iteration), ...
          ', ||Gradient||: ', num2str(max(abs(g)))]);
end
end