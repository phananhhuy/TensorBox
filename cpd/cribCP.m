function criball = cribCP(U,Weights,noisevar)
% Cramer Rao Induced bound of the squared angular error for estimation of
% loading components in a CANDECOMP/PARAFAC decomposition
% U: factor matrices U{1}, U{2}, ..., U{N}
% Weights:  indicator tensor of 0 (missing entries) and 1.
% noisevar : variance of Gaussian noise (1)
%
% Ref:
% P. Tichavsk?, A.-H. Phan, and Z. Koldovsk?, ?Cram?r-Rao-induced bounds
% for CANDECOMP/PARAFAC tensor decomposition,? IEEE Trans. Signal Process.,
% vol. 61, no. 8, pp. 1986?1997, 2013. 
% 
% TENSORBOX, 2014
%
if nargin < 2
    Weights = [];
end

if nargin < 3
    noisevar = 1;
end

N = numel(U);R = size(U{1},2);
criball = zeros(N,R);
for n = 1:N
    criball(n,:) = cribCP1(U([n 1:n-1 n+1:end]),Weights);
end
criball = criball*noisevar;
end