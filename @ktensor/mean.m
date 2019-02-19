function mu = mean(Y)
% Mean of all entries of a k-tensor Y
%
% Phan Anh-Huy
%

mu = double(full(ktensor(Y.lambda,cellfun(@(x) sum(x,1),Y.u,'uni',0))))/prod(size(Y));
