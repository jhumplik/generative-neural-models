function DfcDt = DmonotoneDt(parameters, bin_centers, t)
%DmonotoneDt Derivative of a twice differentiable monotone function.
%Returns a derivative at every t of the function returned by 
%"monotone(parameters, bin_centers, t)". See monotone.m.
%
% Syntax: DfcDt = DmonotoneDt(parameters, bin_centers, t)

% Initialize.
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
DfcDt = zeros(1, length(t));
t_min = bin_centers(1) - delta/2;
minus_idx = (t <= t_min);
t = t(~minus_idx);
% Convert times to bin numbers.
t = t-t_min;
t_int = ceil(t/delta);
t_int(t_int > n_bins) = n_bins; 
% Dvaluate DfcDt
q = delta*cumsum([0,w(1:(end-1))]);
DfcDt(~minus_idx) = b1*exp(q(t_int) + w(t_int).*(t-(t_int-1)*delta));
DfcDt(minus_idx) = b1;
if transpose_check == 1
    DfcDt = DfcDt';
end
end

