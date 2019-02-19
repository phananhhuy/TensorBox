function Z = convn(X,Y)
% Convolution between a Ktensor X and a tensor Y
%
% Phan Anh Huy, 2016
%

Nx = ndims(X);Ny = ndims(Y);
SzX = size(X);
SzY = size(Y);

if isa(Y,'ktensor')
    Z = convn(Y,X);
else
    Z = convn(double(X),double(Y));
end