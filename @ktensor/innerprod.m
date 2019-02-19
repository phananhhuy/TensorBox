function res = innerprod(X,Y)
%INNERPROD Efficient inner product with a ktensor.
%
%   R = INNERPROD(X,Y) efficiently computes the inner product between
%   two tensors X and Y.  If Y is a ktensor, the inner product is
%   computed using inner products of the factor matrices, X{i}'*Y{i}.
%   Otherwise, the inner product is computed using ttv with all of
%   the columns of X's factor matrices, X{i}.
%
%   See also KTENSOR, KTENSOR/TTV
%
%MATLAB Tensor Toolbox.
%Copyright 2010, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by Brett Bader and Tamara Kolda.
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2010) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in tensor_toolbox/LICENSE.txt
% $Id: innerprod.m,v 1.12 2010/03/19 23:46:30 tgkolda Exp $

if ~isequal(size(X),size(Y))
    error('X and Y must be the same size.');
end

% X is a ktensor
switch class(Y)
    
    case {'ktensor'}
        if (~any(strcmp(fieldnames(X), 'weights'))) ||...
                (any(strcmp(fieldnames(X), 'weights')) ...
                && (isempty(X.weights) || isscalar(X.weights)) ...
                && (isempty(Y.weights) || isscalar(Y.weights)))
            M = X.lambda * Y.lambda';
            for n = 1:ndims(X)
                M = M .* (X.u{n}' * Y.u{n});
            end
            res = sum(M(:));
        else
            if any(strcmp(fieldnames(X), 'weights')) && ~isempty(X.weights) 
                Weights = X.weights; 
            else
                Weights = 1; 
            end
            if any(strcmp(fieldnames(Y), 'weights')) && ~isempty(Y.weights),
                if isscalar(Weights)
                    Weights = Y.weights;
                else
                    Weights = Weights.*Y.weights;
                end
            end
            
            Rx = size(X.u{1},2);Ry = size(Y.u{1},2);
            res = 0;
            for r = 1:Rx
                for s = 1:Ry
                    ur = cellfun(@(x,y) x(:,r).*y(:,s),X.u(:),Y.u(:),'uni',0);
                    res = res + X.lambda(r).*Y.lambda(s) * ttv(Weights,ur);
                end
            end
        end
        
    case {'sptensor','ttensor'}
        if any(strcmp(fieldnames(X), 'weights')) && ~(isempty(X.weights) || isscalar(X.weights))
            %Y = X.weights.* Y;
            Y(~reshape(double(X.weights),[],1)) = 0;
        end
        
        R = length(X.lambda);
        vecs = cell(1,ndims(X));
        res = 0;
        for r = 1:R
            for n = 1:ndims(X)
                vecs{n} = X.u{n}(:,r);
            end
            res = res + X.lambda(r) * ttv(Y,vecs);
        end
        
    case {'tensor'}
        if any(strcmp(fieldnames(X), 'weights')) && any(strcmp(fieldnames(X), 'weights'))  && ~(isempty(X.weights) || isscalar(X.weights))
            %Y = X.weights.* Y;
            Y(~reshape(double(X.weights),[],1)) = 0;
        end
        R = length(X.lambda);
        sz = size(Y);N = ndims(Y);
        for n = N:-1:1
            if n == N
                res = reshape(Y.data,[],sz(n));
                res = conj(res) * X.u{n};  % Fixed for complex-valued tensor
            else
                res = reshape(res,[],sz(n),R);
                res = bsxfun(@times,res,reshape(X.u{n},1,[],R));
                res = sum(res,2);
            end
        end
        res = X.lambda.'*res(:);
         
        
    otherwise
        disp(['Inner product not available for class ' class(Y)]);
end

return;
