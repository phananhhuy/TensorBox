function [A B C S iter]=btd3(T,LL,met)
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
if ~(m==sum(LL))
    'sizes of the blocks are not appropriate'
    LL=ones(1,m); %% default=CPD
end
if nargin<3, met = 1; end
switch met
    case 1
%         [A B C S iter]=tedia3g(T,100);
        [A B C S iter]=tedia3c(T,100);
    case 2
        [A B C S iter]=tedia3b_pf(T,100);
    case 3
%         [A B C S iter]=tedia3b_pf1(T,100);
%     case 4
        [A B C S iter]=tedia3b_pf2(T,100);
    case 4
        [A B C S iter]=tediaX(T,100);
    case 5
        [A B C S iter]=tedia3f(T,100);

end

% %%
% % Sort components of A and B
% SS=sum(abs(S),3);
% [u,s,v] = svds(SS - mean(SS(:)),1);
% [u,ord1] = sort(u,'descend');
% [v,ord2] = sort(v,'descend');
% S = S(ord1,ord2,:);
% A = A(ord1,:);
% B = B(ord2,:);
% 
% SS = squeeze(sum(abs(S),2));
% [u,s,v] = svds(SS - mean(SS(:)),1);
% %[u,ord1] = sort(u,'descend');
% [v,ord2] = sort(v,'descend');
% S = S(:,:,ord2);
% C = C(ord2,:);

%%
SS=sum(abs(S),3)+squeeze(sum(abs(S),2))+squeeze(sum(abs(S),1));
ord=srovnej(SS+SS',LL);
% imagesc(SS(ord,ord))
A=A(ord,:); B=B(ord,:); C=C(ord,:);


il0=0;
for ib=1:length(LL)
    ind=il0+1:il0+LL(ib);
    [U Lam V]=svd(A(ind,:)); % finding orthogonal basis of subspaces
    A(ind,:)=V(:,1:LL(ib))';
    [U Lam V]=svd(B(ind,:));
    B(ind,:)=V(:,1:LL(ib))';
    [U Lam V]=svd(C(ind,:));
    C(ind,:)=V(:,1:LL(ib))';
    il0=il0+LL(ib);
end    
S=T;
for n=1:m
    S(:,:,n)=A*T(:,:,n)*B';
end 
for i=1:n
    S(i,:,:)=squeeze(S(i,:,:))*C';
end 
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

