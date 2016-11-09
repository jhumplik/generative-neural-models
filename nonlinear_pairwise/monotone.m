function [fc, Dfc] = monotone(parameters, bin_centers, t)
%monotone Returns a twice differentiable monotone function. 
%The construction follows Ramsay, 1998. 
%Returns fc(t) = g_1 + g_2\int_{t_min}^t exp(\int_{t_min}^t' w(t'') dt'')
%dt', where g_1 = parameters(1), g_2 = parameters(2), and w(t) = \sum_i^K
%b_i I_i(t), where I_i(t) are indicator functions over bins with the
%specified bin centers, and b_i = parameters(i+2).
%Also returns the gradient of fc(t) with respect to all parameters at every
%point t.
%
% Syntax: [fc, Dfc] = monotone(parameters, bin_centers, t)
%
% Inputs:
%   parameters: Vector of parameters describing the monotone function.
%   bin_centers: Vector of size "size(parameters) - 2". Contains the
%                centers of bins described above.
%   t: Vector of points at which to evaluate the monotone function.
%
% Outputs:
%   fc: Values of the function at points "t".
%   Dfc: Gradient of the function at every point in "t" with respect to
%        "parameters".
%
% Example: f = @(t)monotone(parameters, bin_centers, t) returns a callable
%          which corresponds to the monotone function parameterized by 
%          "parameters" and "bin_centers".

%% Initialize.
if ((length(parameters)-2) ~= length(bin_centers))
    error('Number of parameters and number of bins do not match!');
end
b0 = parameters(1);
b1 = parameters(2);
w = parameters(3:end);
n_bins = length(w);
delta = bin_centers(2)-bin_centers(1);
% Check if t is column or row, code below is for row.
transpose_check = 0;
if ~isrow(t)
    t = t';
    transpose_check = 1;
end
% Function is evaluated differently for t <= t_min.
fc = zeros(1, length(t));
Dfc = zeros(length(parameters), length(t));
t_min = bin_centers(1) - delta/2;
minus_idx = (t <= t_min);
t_minus = t(minus_idx);
t = t(~minus_idx);

%% Calculate fc.
% Convert times to bin numbers.
t = t-t_min;
t_int = ceil(t/delta);
% Set w=0 when t outside of last bin.
w = [w, 0];
t_int(t_int > n_bins) = n_bins+1; 
% Evaluate fc(t).
q = delta*cumsum([0,w(1:(end-1))]);
% Integrate up to left bin boundary.
integral_tmp = (exp(q)./w).*(exp(delta*w)-1);
integral_tmp(w==0) = exp(q(w==0))*delta;
integral_up_to_boundary = [0,cumsum(integral_tmp)];
integral_up_to_boundary = integral_up_to_boundary(t_int);
% Integrate in the last bin.
integral_in_last_bin = (exp(q(t_int))./w(t_int)) ... 
                       .* (exp(w(t_int).*(t-(t_int-1)*delta))-1);
integral_in_last_bin(w(t_int) == 0) = exp(q(t_int(w(t_int) == 0))) ...
                     .* (t(w(t_int) == 0)-(t_int(w(t_int) == 0)-1)*delta);
fc_inside = integral_up_to_boundary + integral_in_last_bin;
fc(~minus_idx) = b0 + b1*fc_inside;
% Evaluate fc for t <= t_min.
fc(minus_idx) = b0 + b1*(t_minus - t_min);
% If t was column, then transpose.
if transpose_check == 1
    fc = fc';
end

%% Calculate gradient of fc with respect to b0, b1, and w
if nargout > 1
    Dfc_w = zeros(n_bins, length(t));
    for k = 1:n_bins
        Dfc_w(k, t_int < k) = 0;
        if w(k) ~= 0
            Dfc_w(k, t_int == k) = exp(q(k)) ... 
                *(1 + exp(w(k)*(t(t_int == k)-(k-1)*delta)) ... 
                      .* (w(k)*(t(t_int == k)-(k-1)*delta) - 1))/w(k)^2;
            if k < n_bins
                sum_tmp = [0, cumsum(integral_tmp((k+1):end))];
                Dfc_w(k, t_int > k) = exp(q(k)) ... 
                    *(1 + exp(w(k)*delta)*(w(k)*delta - 1))/w(k)^2 ... 
                    + delta*(sum_tmp(t_int(t_int > k) - k) ... 
                             + integral_in_last_bin(t_int > k));
            end
        else
            Dfc_w(k, t_int == k) = (1/2) * exp(q(k)) ...
                                         * (t(t_int == k)-(k-1)*delta).^2;
            if k < n_bins
                sum_tmp = [0, cumsum(integral_tmp((k+1):end))];
                Dfc_w(k, t_int > k) = (1/2)*exp(q(k))*delta^2 ... 
                    + delta*(sum_tmp(t_int(t_int > k) - k) ... 
                             + integral_in_last_bin(t_int > k));
            end
        end
    end
    % Set gradient when t>t_min.
    Dfc_w = b1*Dfc_w;
    Dfc_b1 = fc_inside;
    Dfc_b0 = ones(1,length(t));
    Dfc_inside = [Dfc_b0; Dfc_b1; Dfc_w];
    Dfc(:,~minus_idx) = Dfc_inside;
    % Set gradient when t<=t_min.
    Dfc_w = zeros(n_bins, length(t_minus));
    Dfc_b1 = t_minus - t_min;
    Dfc_b0 = ones(1, length(t_minus));
    Dfc_minus = [Dfc_b0; Dfc_b1; Dfc_w];
    Dfc(:,minus_idx) = Dfc_minus;
    % If t was column, then transpose.
    if transpose_check == 1
        Dfc = Dfc';
    end
end


end

