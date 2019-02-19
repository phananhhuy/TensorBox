function [A B S] = bcd_ajd(X,R)
% TENSORBOX, 2018

F = sum(R(:,1));
% compress
DimX = size(X);
Fac = [F F 2];
% Compress data into a 2 x F x F array. Only 10 iterations are used since
% exact SL fit is insignificant; only obtaining good truncated bases is
% important
dimorder = find(DimX>Fac);
opts = struct('printitn',0,'tol',1e-6,'maxiters',5,'dimorder',dimorder);
T = mtucker_als(X,Fac,opts);
Gt = T.core.data;

U = T.U;
% Fit GRAM to compressed data
[Ag,Bg,Cg]=gram(Gt(:,:,1),Gt(:,:,2),F);

% Z1 = (Ag\Gt(:,:,1))/Bg;
% Z2 = (Ag\Gt(:,:,2))/Bg;
% Z = abs(Z1)+abs(Z2);
% [u,s,v] = svds(Z - mean(Z(:)),1);
% [u,ord1] = sort(u,'descend');
% [v,ord2] = sort(v,'descend');
% Ag=Ag(:,ord1); Bg=Bg(:,ord2);


AL = U{1}*Ag;
AR = U{2}*Bg;

A=inv(AL);
B=inv(AR);
S = ttm(X,{A B},[1 2]);
SS=sum(abs(double(S)),3);
[u,s,v] = svds(SS - mean(SS(:)),1);
[u,ord1] = sort(u,'descend');
[v,ord2] = sort(v,'descend');
AL=AL(:,ord1); AR=AR(:,ord2);

% ord=srovnej(SS+SS',LL);
% imagesc(SS(ord,ord))
% A=A(ord,:); B=B(ord,:);
il0=0;
for ib=1:length(R)
   ind=il0+1:il0+R(ib);
   [U Lam V]=svd(AL(:,ind)); % finding orthogonal basis of subspaces
   AL(:,ind)=U(:,1:R(ib));
   [U Lam V]=svd(AR(:,ind));
   AR(:,ind)=U(:,1:R(ib));
   il0=il0+R(ib);
end    

A = inv(AL);
B = inv(AR);
S = ttm(X,{A B},[1 2]);
end


function [A,B,C]=gram(X1,X2,F)

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

IsReal=0; % If complex data, complex solutions allowed.
if all(isreal(X1))&all(isreal(X2))
    IsReal=1;
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
k=k/norm(k);

if IsReal % Do not allow complex solutions if only reals are considered
    [k,l] = cdf2rdf(k,l); % simplified here
end

C(2,:)=ones(1,F);
C(1,:)=diag(l)';
A = U*S1*k;
B=V/k';
end