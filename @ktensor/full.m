function t = full(t)
%FULL Convert a ktensor to a (dense) tensor.
%
%   T = FULL(C) converts a ktensor to a (dense) tensor.
%
%   Examples
%   X = ktensor([3; 2], rand(4,2), rand(5,2), rand(3,2));
%   Y = full(A) %<-- equivalent dense tensor
%
%   See also KTENSOR, TENSOR.
%
%MATLAB Tensor Toolbox.
%Copyright 2009, Sandia Corporation. 

% This is the MATLAB Tensor Toolbox by Brett Bader and Tamara Kolda. 
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2009) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in tensor_toolbox/LICENSE.txt
% $Id: full.m,v 1.12 2009/07/07 23:48:56 tgkolda Exp $

% After loop, M = number of rows in the result
A = t.u;
M = 1;
N = size(A{1},2); 
matorder = length(A):-1:1;
for i = matorder
    if ndims(A) ~= 2
        error('Each argument must be a matrix');
    end
    if (N ~= size(A{i},2))
        error('All matrices must have the same number of columns.')
    end
    M = M * size(A{i},1);
end

% Preallocate
% Loop through all the columns
    data = zeros(M,1);
    for n = 1:N
        % Loop through all the matrices
        ab = A{matorder(1)}(:,n);
        for i = matorder(2:end)
            % Compute outer product of nth columns
            
            ab = A{i}(:,n) * ab(:).';
        end
        % Fill nth column of P with reshaped result
        data = data+ t.lambda(n)*ab(:);
    end
    t = tensor(data,size(t));
