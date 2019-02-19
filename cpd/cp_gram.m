function P = cp_gram(X,R,complexloading)
% Extended GRAM and direct trilinear decomposition to higher order CPD.
% This decomposition is often used to initialize CP algorithms.
%
% REF:
% [1] E. Sanchez and B.R. Kowalski, ?Tensorial resolution: a direct trilinear
%     decomposition,? J. Chemometrics, vol. 4, pp. 29?45, 1990.
%
% [2] C.A. Andersson and R. Bro, ?The N-way toolbox for MATLAB,?
%     Chemometrics and Intelligent Laboratory Systems, vol. 52, no. 1, pp.
%     1-4, 2000.
% [3] Anh-Huy  Phan, Petr Tichavsky, Andrzej Cichocki, "CANDECOMP/PARAFAC
%     Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
%     http://arxiv.org/abs/1211.3796, 2012
%
% See also: cp_init, cp_fcp, dtld and gram of N-way toolbox
%
% TENSOR BOX, v1. 2012, 2013
% Copyright 2011, Phan Anh Huy.

if nargin<3
    complexloading = 0;
end

if ndims(X) > 3
    opts = cp_fcp;
    opts.compress_param.compress = false;
    % opts.compress_param.maxiters = 4;
    % opts.compress_param.tol = 1e-5;
    % opts.Rfold = R;
    
    opts.var_thresh = 0.9999;
    opts.refine = 1;%N-1;
    opts.fullrefine = false;
    opts.TraceFit = true;
    opts.foldingrule = 'direct';%[1 1 4]; % 'one' 'half' 'direct'
    opts.mindim = 3;
    
    opts.cp_param.complex = complexloading;
    opts.cp_func = @cp_mdltd;  % @cp_fLMa_v2; %@cp_fastals; % @cp_fLMa_v2 cp_fastals
    opts.cp_reffunc = @cp_fastals;
    
    % ts = tic;
    [P,output,BagofOut] = cp_fcp(X,R,opts);
    % t3 = toc(ts);
else
    [P,fit] = dtld(X,R,complexloading);
    output.Fit = [1 fit];
end
end

%% ######################################################################
function [P,output] = cp_mdltd(X,R,opts)
% X: order-3 tensor
[P,fit] = dtld(X,R,opts.complex);
output.Fit = [1 fit];
end

%% ######################################################################
function [P,fit]=dtld(X,F,Complexloading)

%DTLD direct trilinear decomposition
%
% See also:
% 'gram', 'parafac'
%
%
% DIRECT TRILINEAR DECOMPOSITION
%
% calculate the parameters of the three-
% way PARAFAC model directly. The model
% is not the least-squares but will be close
% to for precise data with little model-error
%
% This implementation works with an optimal
% compression using least-squares Tucker3 fitting
% to generate two pseudo-observation matrices that
% maximally span the variation of all samples. per
% default the mode of smallest dimension is compressed
% to two samples, while the remaining modes are
% compressed to dimension F.
%
% For large arrays it is fastest to have the smallest
% dimension in the first mode
%
% INPUT
% [A,B,C]=dtld(X,F);
% X is the I x J x K array
% F is the number of factors to fit
% An optional parameter may be given to enforce which
% mode is to be compressed to dimension two
%

% Copyright (C) 1995-2006  Rasmus Bro & Claus Andersson
% Copenhagen University, DK-1958 Frederiksberg, Denmark, rb@life.ku.dk
%
% This program is free software; you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation; either version 2 of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
% You should have received a copy of the GNU General Public License along with
% this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
% Street, Fifth Floor, Boston, MA  02110-1301, USA.


% $ Version 2.00 $ May 2001 $ Changed to array notation $ RB $ Not compiled $
% $ Version 1.02 $ Date 28. July 1998 $ Not compiled $
% $ Version 1.03 $ Date 25. April 1999 $ Not compiled $
% Modified by Anh-Huy Phan, 2013

if nargin < 3
    Complexloading = 0;
end
DimX = size(X);
order = [];
if ~issorted(DimX)
    [DimX,order] = sort(DimX);
    X = permute(X,order);
end

Fac   = [2 F F];

f=F;
if F==1;
    Fac = [2 2 2];
    f=2;
end

if DimX(1) < 2
    error(' The smallest dimension must be > 1')
end

if any(DimX(2:3)-Fac(2:3)<0)
    error(' This algorithm requires that two modes are of dimension not less the number of components')
end


% Compress data into a 2 x F x F array. Only 10 iterations are used since
% exact SL fit is insignificant; only obtaining good truncated bases is
% important 
dimorder = find(DimX>Fac);
if ~isempty(dimorder)
    opts = struct('printitn',0,'tol',1e-6,'maxiters',5,'dimorder',dimorder);
    T = mtucker_als(X,Fac,opts);
    Gt = T.core.data;
    U = T.U; % note that two factor matrices U{n} are identity matrices, n # dimorder
else
    Gt = X.data;
    U{1} = eye(DimX(1));
    U{2} = eye(DimX(2));
    U{3} = eye(DimX(3));
end


% Fit GRAM to compressed data
[Bg,Cg,Ag]=gram(squeeze(Gt(1,:,:)),squeeze(Gt(2,:,:)),F,Complexloading);

U{2} = U{2}*Bg;
U{3} = U{3}*Cg;
CkB = bsxfun(@times,reshape(U{2},[],1,size(U{2},2)),reshape(U{3},1,[],size(U{3},2)));
CkB = reshape(CkB,[],size(U{3},2));
X = reshape(X.data,[DimX(1),prod(DimX(2:end))]);
U{1} = X*CkB * pinv((U{2}'*U{2}).*(U{3}'*U{3}));

fit = sum(sum(abs(X - U{1}*CkB.').^2));

P = ktensor(U);
if ~isempty(order)
    P = ipermute(P,order);
end
end


function [A,B,C]=gram(X1,X2,F,Complexloading)

%GRAM generalized rank annihilation method
%
% [A,B,C]=gram(X1,X2,F);
%
% cGRAM - Complex Generalized Rank Annihilation Method
% Fits the PARAFAC model directly for the case of a
% three-way array with only two frontal slabs.
% For noise-free trilinear data the algorithm is exact.
% If input is not complex, similarity transformations
% are used for assuring a real solutions (Henk Kiers
% is thanked for providing the similarity transformations)
%
% INPUTS:
% X1    : I x J matrix of data from observation one
% X2    : I x J matrix of data from observation two
% Fac   : Number of factors
%
% OUTPUTS:
% A     : Components in the row mode (I x F)
% B     : Components in the column mode (J x F)
% C     : Weights for each slab; C(1,:) are the component
%         weights for first slab such that the approximation
%         of X1 is equivalent to X1 = A*diag(C(1,:))*B.'
%

% Copyright (C) 1995-2006  Rasmus Bro & Claus Andersson
% Copenhagen University, DK-1958 Frederiksberg, Denmark, rb@life.ku.dk
%
% This program is free software; you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation; either version 2 of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
% You should have received a copy of the GNU General Public License along with
% this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
% Street, Fifth Floor, Boston, MA  02110-1301, USA.

% $ Version 1.02 $ Date 28. July 1998 $ Not compiled $
% $ Version 1.03 $ Date 22. February 1999 $ Not compiled $
% Modified by Anh-Huy Phan, 2013

if nargin<4
    Complexloading = 0;
end

if Complexloading == 0
    IsReal=0; % If complex data, complex solutions allowed.
    if all(isreal(X1))&&all(isreal(X2))
        IsReal=1;
    end
else
    IsReal = 0;
end

% Find optimal bases in F x F subspace
[U,s,V]=svd(X1+X2,0);
U=U(:,1:F);
V=V(:,1:F);

% Reduce to an F x F dimensional subspace
S1=U'*X1*V;
S2=U'*X2*V;

% Solve eigenvalue-problem and sort according to size
[k,l]=eig(S1\S2);
l=diag(l);
ii=abs(l)>eps;
k=k(:,ii);
l=l(ii);
p=length(l);
[l,ii]=sort(l);
j=p:-1:1;
l=l(j);
l=diag(l);
k=k(:,ii(j));
%k=k/norm(k);

if IsReal % Do not allow complex solutions if only reals are considered
    [k,l] = cdf2rdf(k,l); % simplified here 
end

C(1,:)=ones(1,F);
C(2,:)=diag(l)';
A = U*S1*k;
B=V/k';
end