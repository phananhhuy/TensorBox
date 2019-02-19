function [A0 B0 C0 T G X iter]=tedia4R(T0,numit,A0,B0,C0)
%
% Limit tensor diagonalization algorithm - version 1.0
%                     
% Input: cubic tensor of dimension n x n x n
% Former output: matrices A,B,C such that
%    T = T0 x1 A x2 B x3 C is nearly diagonal (diagonal as much as possible)
%
% Initial matrice A0, B0, C0 are optional (if available, they can be used).
%
% Normalization: A,B,C are normalized so that det(A)=det(B)=det(C)=1.
% There is no other normalization, therefore WN, without normalization.
%
% New output: A0=inv(A), B0=inv(B0), C0=inv(C) 
%             T .... diagonalized (core) tensor
%             G .... matrix of gradients (should be zero at the end)
%             X .... matrix computed from T, revealing (block-)diagonality
%             iter ... criterion versus iteration
%
%                               Programmed: Petr Tichavsky, November 2013
%
[n,n1,n2,n3]=size(T0); 
it=0;
if nargin<5
   A0 = eye(n); B0 = A0; C0=A0; T=T0;
else
   T=multi(T0,inv(A0),inv(B0),inv(C0));
end
eps=1e-6;
tol=10; 
iter=kolik3(T);   %%% criterion, counting sum of off-diagonal elements
if nargin<2
    numit=20;
end   
G=zeros(n,n);
while it<numit && tol>eps
    it=it+1;
    tol=0;
    A1=eye(n); B1=A1; C1=A1; T1=T;
    for i2=1:n
        for i1=[1:i2-1 i2+1:n]
           [A B C Ty val0 val1 nx S q]=onestepG(T,i1,i2);
           G(i1,i2)=norm(q);
           A1([i1 i2],:)=A*A1([i1 i2],:); B1([i1 i2],:)=B*B1([i1 i2],:); C1([i1 i2],:)=C*C1([i1 i2],:); 
           T=Ty;
           tol=tol+nx;
        end
    end
    A0=A0/A1; B0=B0/B1; C0=C0/C1;
    valy=kolik3(T); 
    iter=[iter valy];
end
% iter
Tx=sum(abs(T),4);
X=sum(Tx,3)+squeeze(sum(Tx,2))+squeeze(sum(Tx,1));
J=serad(X+X');
X=X(J,J);
T=T(J,J,J,:);
A0=A0(:,J);
B0=B0(:,J);
C0=C0(:,J);
end

function [A B C T val0 val nx HH gg]=onestepG(T,i1,i2)
[n,n1,n2,n3]=size(T);
eps=1e-6;
% val0=kolik3(T);
val0=kolik5c(T,i1,i2);
T1=T([i1 i2],[i1 i2],[i1,i2],:); T1=sum(T1.*conj(T1),4);
t=[T1(1,1,1)+T1(2,2,2),T1(1,2,2)+T1(2,1,1) T1(2,2,1)+T1(1,1,2),T1(2,1,2)+T1(1,2,1)];
[m it]=max(t);
switch it
   case 2
        T([i1,i2],:,:,:)=T([i2 i1],:,:,:);
   case 3
        T(:,:,[i1,i2],:)=T(:,:,[i2 i1],:);
   case 4
        T(:,[i1,i2],:,:)=T(:,[i2 i1],:,:);
end    
d=n;        
Ds=sum(reshape(T([i1,i2],[i1,i2],:,:),2,2,n2*n3).^2,3);
gg=sum(reshape(T(i2,i2,:,:).*T(i1,i1,:,:),1,n2*n3),2); Dx=sum(reshape(T(i1,i2,:,:).*T(i2,i1,:,:),1,n2*n3),2);
E=permute(T([i1 i2],:,[i1 i2],:),[1 3 2 4]);
Es=sum(reshape(E.^2,2,2,n2*n3),3); 
hh=sum(reshape(E(2,2,:,:).*E(1,1,:,:),1,n2*n3),2); Ex=sum(reshape(E(1,2,:,:).*E(2,1,:,:),1,n2*n3),2);
F=permute(T(:,[i1 i2],[i1 i2],:),[2 3 1 4]); 
Fs=sum(reshape(F.^2,2,2,n2*n3),3); %Fs=sum(F([1,2],[1,2],:).^2,3);
jj=sum(reshape(F(2,2,:,:).*F(1,1,:,:),1,n2*n3),2); Fx=sum(reshape(F(1,2,:,:).*F(2,1,:,:),1,n2*n3),2);
X=zeros(6,6);
h1=Dx+gg; h2=Ex+hh; h3=Fx+jj; 
aux=T(i1,:,:,:).*T(i2,:,:,:); ggg1=sum(aux(:));
aux=T(:,i1,:,:).*T(:,i2,:,:); ggg2=sum(aux(:));
aux=T(:,:,i1,:).*T(:,:,i2,:); ggg3=sum(aux(:));
aux=T(i1,:,:,:).^2; hhh(1)=sum(aux(:)); aux=T(i2,:,:,:).^2; hhh(2)=sum(aux(:));
aux=T(:,i1,:,:).^2; hhh(3)=sum(aux(:)); aux=T(:,i2,:,:).^2; hhh(4)=sum(aux(:));
aux=T(:,:,i1,:).^2; hhh(5)=sum(aux(:)); aux=T(:,:,i2,:).^2; hhh(6)=sum(aux(:));
hhx=hhh-sum([T(i1,i2,i2,:).^2 T(i2,i1,i1,:).^2 T(i2,i1,i2,:).^2 T(i1,i2,i1,:).^2 T(i2,i2,i1,:).^2 T(i1,i1,i2,:).^2],4);
X(1,4)=h1; X(2,3)=h1; X(1,6)=h2; X(2,5)=h2; X(3,6)=h3; X(4,5)=h3;
X(1,2)=0.5*(hhh(1)+hhh(2)-sum(T(i2,i2,i2,:).^2+T(i1,i1,i1,:).^2)); 
X(3,4)=0.5*(hhh(3)+hhh(4)-sum(T(i2,i2,i2,:).^2+T(i1,i1,i1,:).^2)); 
X(5,6)=0.5*(hhh(5)+hhh(6)-sum(T(i2,i2,i2,:).^2+T(i1,i1,i1,:).^2)); 
X(2,4)=h1 - sum(T(i1,i2,i1,:).*T(i2,i1,i1,:) + T(i1,i1,i1,:).*T(i2,i2,i1,:)); 
X(2,6)=h2 - sum(T(i1,i1,i2,:).*T(i2,i1,i1,:) + T(i1,i1,i1,:).*T(i2,i1,i2,:));
X(3,5)=h3 - sum(T(i2,i1,i1,:).*T(i2,i2,i2,:) + T(i2,i1,i2,:).*T(i2,i2,i1,:));
X(4,6)=h3 - sum(T(i1,i1,i2,:).*T(i1,i2,i1,:) + T(i1,i1,i1,:).*T(i1,i2,i2,:));
X(1,3)=h1 - sum(T(i1,i1,i2,:).*T(i2,i2,i2,:) + T(i1,i2,i2,:).*T(i2,i1,i2,:));
X(1,5)=h2 - sum(T(i1,i2,i1,:).*T(i2,i2,i2,:) + T(i1,i2,i2,:).*T(i2,i2,i1,:));
X=X+X';
gg=[ggg1-sum(T(i1,i2,i2,:).*T(i2,i2,i2,:)) ggg1-sum(T(i2,i1,i1,:).*T(i1,i1,i1,:)) ggg2-sum(T(i2,i1,i2,:).*T(i2,i2,i2,:)) ...
    ggg2-sum(T(i1,i2,i1,:).*T(i1,i1,i1,:)) ggg3-sum(T(i2,i2,i1,:).*T(i2,i2,i2,:)) ggg3-sum(T(i1,i1,i2,:).*T(i1,i1,i1,:))];
hh=gg;
HH=X+diag(hhx);
iH0=diag(HH)<eps;
ddH=zeros(6,1); ddH(iH0)=1; HH=HH+diag(ddH);
if norm(gg)<eps
   x=zeros(1,6); A=eye(2); B=A; C=A;
   val=val0;
else   
   x=-(HH\gg')';
dx1=sqrt(1+x(1)*x(2)); dx2=sqrt(1+x(3)*x(4)); dx3=sqrt(1+x(5)*x(6)); 
A=[dx1 x(2); x(1) dx1]; B=[dx2 x(4); x(3) dx2]; C=[dx3 x(6); x(5) dx3];
T0=T;
T([i1 i2],:,:,:) = reshape(A*reshape(T([i1 i2],:,:,:),2,[]),2,n1,n2,n3);
T(:,[i1 i2],:,:) = permute(reshape(B*reshape(permute(T(:,[i1 i2],:),[2 1 3 4]),2,[]),2,n,n2,n3),[2 1 3 4]);
T(:,:,[i1 i2],:) = permute(reshape(C*reshape(permute(T(:,:,[i1 i2],:),[3 1 2 4]),2,[]),2,n,n1,n3),[2 3 1 4]);
if min([x(1)*x(2) x(3)*x(4) x(5)*x(6)])<-1
    val=1e10;
else    
    val=kolik5c(T,i1,i2);
end    
end
nx=norm(x);
if val>val0
%   criterion did not decrease => switch to damped GN algorithm
%    nx=1;
    mu=min(diag(HH));
end    
while val>val0
      x=-((HH+mu*eye(6))\gg')';
      dx1=sqrt(1+x(1)*x(2)); dx2=sqrt(1+x(3)*x(4)); dx3=sqrt(1+x(5)*x(6)); 
      A=[dx1 x(2); x(1) dx1]; B=[dx2 x(4); x(3) dx2]; C=[dx3 x(6); x(5) dx3];
      T=T0;
      T([i1 i2],:,:,:) = reshape(A*reshape(T([i1 i2],:,:,:),2,[]),2,n1,n2,n3);
      T(:,[i1 i2],:,:) = permute(reshape(B*reshape(permute(T(:,[i1 i2],:),[2 1 3 4]),2,[]),2,n,n2,n3),[2 1 3 4]);
      T(:,:,[i1 i2],:) = permute(reshape(C*reshape(permute(T(:,:,[i1 i2],:),[3 1 2 4]),2,[]),2,n,n1,n3),[2 3 1 4]);
      if min([x(1)*x(2) x(3)*x(4) x(5)*x(6)])<-1
         val=1e10;
      else    
      %   val=kolik3(T);
          val=kolik5c(T,i1,i2);
      end      
      mu=2*mu;
end
switch it
    case 2
         A=A(:,[2 1]);
    case 3
         C=C(:,[2 1]);
    case 4
         B=B(:,[2 1]);
end  
end
    
function val=kolik3(T)
[n n2 n3 n4]=size(T);
T=reshape(T,n*n2*n3,n4);
T(1:n^2+n+1:n^3,:)=0;
val=sum(T(:).*conj(T(:)));
end

function val=kolik5c(T,i1,i2)
%
[n1 n2 n3 n4]=size(T);
T6=T([i1 i2],[i1 i2],:,:); T7=T(:,[i1 i2],[i1 i2],:); T8=T([i1 i2],:,[i1 i2],:);
T4=reshape(T([i1 i2],[i1 i2],[i1 i2],:),8,n4); 
T1=T([i1 i2],:,:,:); T2=T(:,[i1 i2],:,:); T3=T(:,:,[i1 i2],:);
val=sum(T1(:).*conj(T1(:)))+sum(T2(:).*conj(T2(:)))+sum(T3(:).*conj(T3(:)))...
-sum(T6(:).*conj(T6(:)))-sum(T7(:).*conj(T7(:)))-sum(T8(:).*conj(T8(:)))+sum(sum(T4(2:7,:).*conj(T4(2:7,:))));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ord=serad(X)
%
% Hierarchical clustering for tensor diagonalization
% Input: matrix of similarities (symmetric, nonnegative elements)
% Output: order of components such that X(ord,ord) is approximately block diagonal
%
%                         Coded: Petr Tichavsky, November 2013
[n n2]=size(X);
X(1:n+1:n^2)=0;
if sum(X(:).^2)<1e-6
   ord=1:n;
else   
srt=repmat((1:n)',1,n);
len=ones(n,1);
for in=n:-1:2
    [m1,ii]=max(X(1:in,1:in));
    [m2 i1]=max(m1);
    [m3 i2]=max(X(:,i1));
    if i1>i2
       aux=i1; i1=i2; i2=aux;
    end   
    lennew=len(i1)+len(i2);
    srtnew=[srt(i1,1:len(i1)) srt(i2,1:len(i2)) zeros(1,n-lennew)];
    indl=[1:i1-1 i1+1:i2-1 i2+1:in];
    Xnew=(len(i1)*X(i1,indl)+len(i2)*X(i2,indl))/lennew;
    X=[0 Xnew; Xnew' X(indl,indl)];
    srt=[srtnew; srt(indl,:)];
    len=[lennew; len(indl)];
end
ord=srt(1,:);
end
end

function T2=multi(T,A,B,C)
[n1 n2 n3]=size(T);
r1=size(A,1); r2=size(B,1); r3=size(C,1);
T1=A*reshape(T,n1,n2*n3);
T2=reshape(T1,r1*n2,n3)*C.';
T2=permute(reshape(B*reshape(permute(reshape(T2,r1,n2,r3),[2 1 3]),n2,r1*r3),r2,r1,r3),[2 1 3]);
end