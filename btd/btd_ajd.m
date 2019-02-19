function [A B S]=btd(T,LL)
%
% Block tensor decomposition;
% T is assumed to be a cube of dimension m x m x m
% LL are sizes of the blocks, sum(LL)=m;
% A,B,C should be the estimated factor matrices
% S should be a diagonalized tensor (with blocks along its
%   main spatial diagonal) ... not yet implemented
%     
% TENSORBOX, 2018

[m m1 m2]=size(T);
eps=1e-6;
if ~(m==sum(LL)) %   'sizes of the blocks are not appropriate'
   LL=ones(1,m); %% default=CPD
end
[A1 B1]=I_DIEM_NS(T);
%P = cp_gram(tensor(T),m);A1 = P.U{1};B1 = P.U{2};
ir=0;
for i=1:m
   im=max(imag(A1(:,i)));
   if im>eps
      if ir==0
         A1(:,i)=real(A1(:,i));
         ir=1;
      else
         A1(:,i)=imag(A1(:,i));
         ir=0;
      end
   end
end    
ir=0;
for i=1:m
   im=max(imag(B1(:,i)));
   if im>eps
      if ir==0
         B1(:,i)=real(B1(:,i));
         ir=1;
      else
         B1(:,i)=imag(B1(:,i));
         ir=0;
      end
   end
end    
% A1 = bsxfun(@rdivide,A1,sqrt(sum(A1.^2,1)));
% B1 = bsxfun(@rdivide,B1,sqrt(sum(B1.^2,1)));
A=pinv(A1);
B=pinv(B1);
S=T;
for n=1:m
   S(:,:,n)=A*T(:,:,n)*B';
end 
SS=sum(abs(S),3);
[u,s,v] = svds(SS - mean(SS(:)),1);
% [u,s,v] = svds(SS+SS' - mean(SS(:)),1);
% [u,s,v] = svds(SS-mean(SS,2)*mean(SS,1),1);
[u,ord1] = sort(u,'descend');
[v,ord2] = sort(v,'descend');
A=A(ord1,:); B=B(ord2,:);

% ord=srovnej(SS+SS',LL);
% imagesc(SS(ord,ord))
% A=A(ord,:); B=B(ord,:);
il0=0;
for ib=1:length(LL)
   ind=il0+1:il0+LL(ib);
   [U Lam V]=svd(A(ind,:)); % finding orthogonal basis of subspaces
   A(ind,:)=V(:,1:LL(ib))';
   [U Lam V]=svd(B(ind,:));
   B(ind,:)=V(:,1:LL(ib))';
   il0=il0+LL(ib);
end    
S=T;
for n=1:m
   S(:,:,n)=A*T(:,:,n)*B';
end 

% S=permute(S,[3,2,1]);
% C=update(S,eye(d));

% imagesc(sum(abs(S),3))
end
%
function ord=srovnej(D,LL)
%
d=size(D,1);
nb=length(LL);
ord=1:d;
D0=D;
ind=0;
for k=1:nb-1
   sb=LL(k);
   [h m]=sort(-D);
   [h2 imd2]=sort(-h(sb+1,:));
   iny=imd2(ind+sb);
   im=m(1:sb,iny);
   ord(ind+1:ind+sb)=im;
   D(im,:)=-ones(sb,d);
   D(:,im)=-ones(d,sb);
   ind=ind+sb;
end 
aux=1:d;
aux(ord(1:ind))=[];
ord(ind+1:d)=aux;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ____________________                 
% ____  _/_____  /__(_)___________ ___ 
%  __  / _  __  /__  /_  _ \_  __ `__ \
% __/ /  / /_/ / _  / /  __/  / / / / /   DIagonalization using Equivalent
% /___/  \__,_/  /_/  \___//_/ /_/ /_/  Matrices (improved non-symmetrical
%                                                                 version)
%
%
% Usage: '[A_L,A_R] = I_DIEM_NS(T)'.
%
% Input:
%
% T- a N x N x K array collecting K (K>=N) (complex) 'target matrices' T_k.
%
% Output:
%
% A_L- the N x N left diagonalizing matrix.
% A_R- the N x N right diagonalizing matrix.
%
%
% This algorithm computes a non-iterative minimizing solution for 
% the direct-fit least-squares criterion:
%
%                  $$\sum_k \|T_k - A_L Delta_k A_R^H\|_F^2.$$
%
% By Gilles CHABRIEL and Jean BARRERE.
% 
% (c) 2009, IM2NP, UMR 6242 CNRS-USTV, FRANCE.


function [A_L,A_R] = I_DIEM_NS(T)

[N,N,K] = size(T);

if K<N 
   error(['At least ',num2str(N),' target matrices are necessary. Only ',num2str(K),' are available!']);
end

%% computes the N 'representative' matrices R_n

F = zeros(N*N);

for k = 1:K
   F = F + vec(T(:,:,k))*vec(T(:,:,k))';
end

[vecR,SvecR] = svd(F);

R = zeros(N,N,N);

for n=1:N
   R(:,:,n) = unvec(vecR(:,n));
end


%% computes the matrices V_n

F = zeros(N*N);

for n = 1:N
   F = F + vec(R(:,n,:))*vec(R(:,n,:))';
end

[vecV,SvecV,vecU] = svd(F);

%% computes the matrix E = C^{-1}

[E,S] = eig(unvec(vecV(:,2)),unvec(vecV(:,1)));

[E,S] = cdf2rdf(E,S); % 
%% computes, columnwise, the diagonalizing matrix A

colA = zeros(N,N,N);

for n = 1:N
    for ind = 1:N
        colA(:,ind,n) = unvec(vecV(:,ind))*E(:,n);
    end
end

A_L = zeros(N,N);

for n = 1:N
   [U,S,V] = svd(colA(:,:,n));
   A_L(n,:) = U(:,1).';
end

%% computes the matrices V_n

F = zeros(N*N);

for n = 1:N
   F = F + vec(R(n,:,:))*vec(R(n,:,:))';
end

[vecV,SvecV,vecU] = svd(F);

%% computes the matrix E = C^{-1}

%[E,S] = eig(unvec(vecV(:,2)),unvec(vecV(:,1)));

%% computes, columnwise, the diagonalizing matrix A

colA = zeros(N,N,N);

for n = 1:N
    for ind = 1:N
        colA(:,ind,n) = unvec(vecV(:,ind))*E(:,n);
    end
end

A_R = zeros(N,N);

for n = 1:N
   [U,S,V] = svd(colA(:,:,n));
   A_R(n,:) = U(:,1)';
end

A_R = A_R.';
A_L = A_L.';

end %% END I_DIEM_NS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% unvec(.) operator

function Y = unvec(y)

N = sqrt(length(y));

Y = reshape(y,N,N);
end

%% vec(.) operator

function y = vec(Y)

y = Y(:);
end



function A=update(T,B)
[m m1 p]=size(T);
S=zeros(m,m,m);
for n=1:m
    U=zeros(m,m);
    for i=1:p
        u=T(:,:,i)*B(n,:)';
        U=U+u*u';
    end
    S(:,:,n)=U;
end
V=sum(S,3);
for n=1:m
    [W Lam]=eig(S(:,:,n),V-S(:,:,n));
    [Lmin,imin]=max(diag(Lam));
    %    A(j,:)=A(j,:)/norm(A(j,:))
    A(n,:)=W(:,imin)';
    A(n,:)=A(n,:)/norm(A(n,:));
end
end