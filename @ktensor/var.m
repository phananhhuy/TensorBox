function v = var(Y)
% Variance of k-tensor Y
%
% Phan Anh-Huy
numelY = prod(size(Y));
v = (norm(Y)^2 - (2*numelY - 1)*mean(Y)^2)/numelY;