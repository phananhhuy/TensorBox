function [A0 B0 T X rr blen iter]=tedia2P(T0,numit)
%
% Parallel implementation of two-side diagonalization of an order-3 tensor
%
% Input: cubic tensor of dimension d x d x n
% Former output: matrices A,B,C such that
%    T = T0 x1 A x2 B is nearly diagonal (diagonal as much as possible)
%                      in the sense that each slice T(:,:,k) should be
%                      nearly diagonal matrix.
%
% Normalization: A,B are normalized so that det(A)=det(B)=det(C)=1.
% There is no other normalization, therefore WN, without normalization.
%
% New output: A0=inv(A), B0=inv(B0), C0=inv(C)
%             T .... diagonalized (core) tensor
%             G .... matrix of gradients (should be zero at the end)
%             X .... matrix computed from T, revealing (block-)diagonality
%             rr ... multilinear rank
%             blen . estimated block sizes
%             iter ... criterion versus iteration
%
%                               Programmed: Petr Tichavsky, November 2013
%
eps=1e-8;
[n1,n2,n3]=size(T0);
if nargin<2
    numit=50;
end
Tc = T0;A1 = eye(n1); B1 = eye(n2);
r = n1;r1 = r; r2 = r;rr = [r1 r2];
T1=reshape(T0,n1,n2*n3);
%[A1 S]=eig(T1*T1'); %%% can replace the next line, if the tensor is large.
[A1,S,V]=svd(T1,0);
% r1=sum(diag(S)>eps);
% T1=reshape(S(1:r1,1:r1)*V(:,1:r1)',r1,n2,n3);
T1=reshape(permute(T1,[2 1 3]),n2,r1*n3);
[B1,S,V]=svd(T1);
% r2=sum(diag(S)>eps);
% T1=reshape(S(1:r2,1:r2)*V(:,1:r2)',r2,r1,n3);
% rr=[r1 r2];
% r=max(rr);
% Tc=zeros(r,r,n3);
% Tc(1:r1,1:r2,:)=T1;

[A2,B2,T,X,iter]=tedia2(Tc,numit);
A0=zeros(n1,r); A0(:,1:r)=A1(:,1:r)*A2; %A0(:,1:r-r1)=0;
B0=zeros(n2,r); B0(:,1:r)=B1(:,1:r)*B2; %B0(:,1:r-r2)=0;
% Tx=zeros(n1,n2,n3); Tx(1:r,1:r,1:r)=T;
%[T,ord1,ord2,blen]=seradY(Tx);
blen=0;
%A0=A0(:,ord1); B0=B0(:,ord2); C0=C0(:,ord3);
X=sum(abs(T),3);
if r1<r
    [ix,in]=sort(sum(X,2));
    A0(:,in(1:r-r1))=0;
end
if r2<r
    [ix,in]=sort(sum(X2,1));
    B0(:,in(1:r-r2))=0;
end
end
%
function [A0 B0 T X iter]=tedia2(T0,numit)
%
% Limit tensor diagonalization algorithm - version 1.0
%
% Input: cubic tensor of dimension n x n x n
% Former output: matrices A,B,C such that
%    T = T0 x1 A x2 B is nearly diagonal (diagonal as much as possible)
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
%                               Programmed: Petr Tichavsky, June 2014
%
[n n1 n2]=size(T0);
A0=eye(n); B0=eye(n); T=T0;
it=0;
eps=1e-8;
nx=10;
iter=kolik3(T);   %%% criterion, counting sum of squares of off-diagonal elements
if nargin<2
    numit=50;
end
% G=zeros(n,n);
%   A1=eye(n); B1=A1; C1=A1; T1=T;
%     for i2=1:n-1
%         for i1=i2+1:n %[1:i2-1 i2+1:n]
%             %Tbk = T;
% %             [A B Ty val0 val1 nx S q]=onestepG2(T,i1,i2);
%             [A B Ty val0 val1 nx S q]=onestepG2b(T,i1,i2);
%             G(i1,i2)=norm(q);
%             A1([i1 i2],:)=A*A1([i1 i2],:); B1([i1 i2],:)=B*B1([i1 i2],:);
%             T=Ty;
%         end
%     end
%     A0=A0/A1; B0=B0/B1;
%     valy=kolik3(T);
%     iter=[iter valy];
while it<numit && nx>eps
    it=it+1;
    A1=eye(n); B1=A1; T1=T;
    [Ar Br Ty nx val0 val]=onestepR(T);
    %[Ap Bp T1 nx val0 val]=onestepL(T1);
%     A1=eye(n); B1=A1; C1=A1; T1=T;
%     for i2=1:n-1
%         for i1=i2+1:n %[1:i2-1 i2+1:n]
%             %Tbk = T;
% %             [A B Ty val0 val1 nx S q]=onestepG2(T,i1,i2);
%             [A B Ty val0 val1 nx S q]=onestepG2b(T,i1,i2);
%             G(i1,i2)=norm(q);
%             A1([i1 i2],:)=A*A1([i1 i2],:); B1([i1 i2],:)=B*B1([i1 i2],:);
%             T=Ty;
%         end
%     end
%    A0=A0/A1; B0=B0/B1;
    A0=A0/Ar; B0=B0/Br;
    T=Ty;
    iter=[iter val];
end
% iter;
X=sum(abs(T),3);
J=serad(X+X');
X=X(J,J);
T=T(J,J,:);
A0=A0(:,J);
B0=B0(:,J);
end

function [A B T]=uprav2(T);
[n n1 n2]=size(T);
A=eye(n); B=eye(n);
S=sum(abs(T),3);
for in=n:-1:2
    [A1 in2]=max(reshape(abs(S(1:in,1:in)),in^2,1));
    in1=ceil(in2/in);
    S(:,1:in)=S(:,[1:in1-1 in1+1:in in1]);
    B(1:in,:)=B([1:in1-1 in1+1:in in1],:);
    in1=in2-(in1-1)*in;
    S(1:in,:)=S([1:in1-1 in1+1:in in1],:);
    A(1:in,:)=A([1:in1-1 in1+1:in in1],:);
end
end

function [A B T val0 val nx HH gg]=onestepG2(T,i1,i2)
[n n1 n2]=size(T);
eps=1e-10;
val0=kolik3(T);
% val0=kolik5c(T,i1,i2);
T1=T([i1 i2],[i1 i2],:); T1=T1.*conj(T1); T2=sum(T1,3);
t=[T2(1,1)+T2(2,2),T2(1,2)+T1(2,1)];
[m it]=max(t);
if it==2
    T([i1,i2],:,:)=T([i2 i1],:,:);
end
d=n;
T1=reshape(T([i1,i2],:,:),2,n1*n2);
alpha=conj(T1)*T1.';
T1=reshape(permute(T(:,[i1,i2],:),[2 1 3]),2,n*n2);
beta=conj(T1)*T1.';
v1=sum(conj(T(i2,i1,:)).*T(i1,i1,:),3);
v2=sum(conj(T(i1,i2,:)).*T(i2,i2,:),3);
v3=sum(conj(T(i1,i2,:)).*T(i1,i1,:),3);
v4=sum(conj(T(i2,i1,:)).*T(i2,i2,:),3);
v5=sum(conj(T(i2,i1,:)).*T(i2,i1,:),3);
v6=sum(conj(T(i1,i2,:)).*T(i1,i2,:),3);
v7=sum(conj(T(i1,i1,:)).*T(i2,i2,:),3);
gg=[alpha(2,1)-v1,alpha(1,2)-v2,beta(2,1)-v3,beta(1,2)-v4];
H=[alpha(2,2)-v5 0 0 conj(v7); 0 alpha(1,1)-v6 v7 0; 0 conj(v7) beta(2,2)-v6 0; v7 0 0 beta(1,1)-v5];
HH=H;
iH0=abs(diag(HH))<eps;
ddH=zeros(4,1); ddH(iH0)=1; HH=HH+diag(ddH);
if norm(gg)<eps
    x=zeros(1,4); A=eye(2); B=A; C=A;
    val=val0;
else
    x=-(HH\gg.').';
    dx1=sqrt(1+x(1)*x(2)); dx2=sqrt(1+x(3)*x(4));
    A=[dx1 x(1); x(2) dx1]; B=[dx2 x(3); x(4) dx2];
    T0=T;
    T([i1 i2],:,:) = reshape(A*reshape(T([i1 i2],:,:),2,[]),2,n1,n2);
    T(:,[i1 i2],:) = permute(reshape(reshape(permute(T(:,[i1 i2],:),[1 3 2]),[],2)*B.',n,n2,2),[1 3 2]);
    if min(abs([x(1)*x(2) x(3)*x(4)]))<-1
        val=1e10;
    else
        %  val=kolik5c(T,i1,i2);
        val=kolik3(T);
    end
end
nx=norm(x);
if val>val0
    %   criterion did not decrease => switch to damped GN algorithm
    %    nx=1;
    mu=min(diag(HH));
end
while val>val0 && mu<1e8
    x=-((HH+mu*eye(4))\gg.').';
    dx1=sqrt(1+x(1)*x(2)); dx2=sqrt(1+x(3)*x(4));
    A=[dx1 x(1); x(2) dx1]; B=[dx2 x(3); x(4) dx2];
    T=T0;
    T([i1 i2],:,:) = reshape(A*reshape(T([i1 i2],:,:),2,[]),2,n1,n2);
    T(:,[i1 i2],:) = permute(reshape(reshape(permute(T(:,[i1 i2],:),[1 3 2]),[],2)*B.',n,n2,2),[1 3 2]);
    if min(abs([x(1)*x(2) x(3)*x(4)]))<-1
        val=1e10;
    else
        val=kolik3(T);
        %    val=kolik5c(T,i1,i2);
        
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

function [A B T val0 val nx HH gg]=onestepG2b(T,i1,i2)
[n n1 n2]=size(T);
eps=1e-10;
% val0=kolik3(T);
T1=T([i1 i2],[i1 i2],:); T1=T1.*conj(T1); T2=sum(T1,3);
t=[T2(1,1)+T2(2,2),T2(1,2)+T1(2,1)];
[m it]=max(t);
if it==2
    T([i1,i2],:,:)=T([i2 i1],:,:);
end
d=n;
T1=reshape(T([i1,i2],:,:),2,n1*n2);
alpha=conj(T1)*T1.';
T1=reshape(permute(T(:,[i1,i2],:),[2 1 3]),2,n*n2);
beta=conj(T1)*T1.';
v1=sum(conj(T(i2,i1,:)).*T(i1,i1,:),3);
v2=sum(conj(T(i1,i2,:)).*T(i2,i2,:),3);
v3=sum(conj(T(i1,i2,:)).*T(i1,i1,:),3);
v4=sum(conj(T(i2,i1,:)).*T(i2,i2,:),3);
v5=sum(conj(T(i2,i1,:)).*T(i2,i1,:),3);
v6=sum(conj(T(i1,i2,:)).*T(i1,i2,:),3);
v7=sum(conj(T(i1,i1,:)).*T(i2,i2,:),3);
gg=[alpha(2,1)-v1,alpha(1,2)-v2,beta(2,1)-v3,beta(1,2)-v4];
H=[alpha(2,2)-v5 0 0 conj(v7); 0 alpha(1,1)-v6 v7 0; 0 conj(v7) beta(2,2)-v6 0; v7 0 0 beta(1,1)-v5];
HH=H;
iH0=abs(diag(HH))<eps;
ddH=zeros(4,1); ddH(iH0)=1; HH=HH+diag(ddH);
if norm(gg)<eps
    x=zeros(1,4); A=eye(2); B=A;  C=A;
    val0 = [];% no improvement
    val=val0;
else
    x=-(HH\gg.').';
    dx1=sqrt(1+x(1)*x(2)); dx2=sqrt(1+x(3)*x(4));
    A=[dx1 x(1); x(2) dx1]; B=[dx2 x(3); x(4) dx2];
    T0=T;
    val0=kolik5c(T,i1,i2);val = inf;   
    if min(abs([x(1)*x(2) x(3)*x(4)]))>=-1 
        T([i1 i2],:,:) = reshape(A*reshape(T([i1 i2],:,:),2,[]),2,n1,n2);
        T(:,[i1 i2],:) = permute(reshape(reshape(permute(T(:,[i1 i2],:),[1 3 2]),[],2)*B.',n,n2,2),[1 3 2]);
        val=kolik5c(T,i1,i2);
    end
    
    % criterion did not decrease => switch to damped GN algorithm or
    % conditions on x are not satisfied 
    if val>val0
        mu=min(real(diag(HH)));
        
        while val>val0 && mu<1e8
            x=-((HH+mu*eye(4))\gg.').';
            
            if min(abs([x(1)*x(2) x(3)*x(4)]))>=-1
                dx1=sqrt(1+x(1)*x(2)); dx2=sqrt(1+x(3)*x(4));
                A=[dx1 x(1); x(2) dx1]; B=[dx2 x(3); x(4) dx2];
                T=T0;
                T([i1 i2],:,:) = reshape(A*reshape(T([i1 i2],:,:),2,[]),2,n1,n2);
                T(:,[i1 i2],:) = permute(reshape(reshape(permute(T(:,[i1 i2],:),[1 3 2]),[],2)*B.',n,n2,2),[1 3 2]);
                
                val=kolik5c(T,i1,i2);
            end
            mu=2*mu;
        end
    end
end
switch it
    case 2
        A=A(:,[2 1]);
    case 3
        C=C(:,[2 1]);
    case 4
        B=B(:,[2 1]);
end
nx=norm(x);
end

function [A B T1 nx val0 val]=onestepR(T)
[n n1 n2]=size(T);
eps=1e-10;
in=reshape(1:n^2,n,n); in=(in<in'); in2=in+in';
% T2=sum(T.*conj(T),3);
% t=[T2(1,1)+T2(2,2),T2(1,2)+T1(2,1)];
% [m it]=max(t);
% if it==2
%     T([i1,i2],:,:)=T([i2 i1],:,:);
% end
d=n;
T1=reshape(T,n,n1*n2);
alpha=conj(T1)*T1.';
T1=reshape(permute(T,[2 1 3]),n1,n*n2);
beta=conj(T1)*T1.';
T1=reshape(T,n*n1,n2);
T1d=T1(1:n+1:n^2,:); 
T1e=reshape(repmat(T1d(:).',n,1),n,n1,n2);
T1f=reshape(repmat(T1d,n,1),n,n1,n2);
%v1=sum(conj(T(i2,i1,:)).*T(i1,i1,:),3)
%v1=sum(conj(permute(T,[2 1 3])).*T1f,3);
v1=sum(conj(T).*T1e,3).';
%v2=sum(conj(T(i1,i2,:)).*T(i2,i2,:),3);
%v2=sum(conj(T).*T1e,3);
v3=sum(conj(T).*T1f,3);
%v3=sum(conj(T(i1,i2,:)).*T(i1,i1,:),3);
%v4=sum(conj(T(i2,i1,:)).*T(i2,i2,:),3);
%v4=sum(conj(T).*T1f,3).';
%v5=sum(conj(T(i2,i1,:)).*T(i2,i1,:),3);
%v5=sum(conj(T).*T,3).';
v6=sum(conj(T).*T,3);
%v7=sum(conj(T(i1,i1,:)).*T(i2,i2,:),3);
v7=sum(conj(T1f).*T1e,3);
%gg=[alpha(2,1)-v1,alpha(1,2)-v2,beta(2,1)-v3,beta(1,2)-v4];
gg2=alpha-v1.';
gg3=beta.'-v3;
H22=repmat(diag(alpha),1,n)-v6;
iH0=abs(H22)<eps;
H22(iH0)=1;
dHH=H22.*H22'-v7.*conj(v7);
x2=-(H22'.*gg2-v7.*gg3)./dHH;
x3=-(H22.*gg3-conj(v7).*gg2)./dHH;
%A=(eye(n)+in.*x2)*(eye(n)+in'.*x2);
%B=((eye(n)+in'.*x3)*(eye(n)+in.*x3)).';
A=(eye(n)+in2.*x2).'; A=sprav(A);
B=(eye(n)+in2.*x3); B=sprav(B);
T1 = reshape(A*reshape(T,n,[]),n,n1,n2);
T1 = permute(reshape(reshape(permute(T1,[1 3 2]),[],n1)*B.',n,n2,n1),[1 3 2]);
nx=norm(x2(:))+norm(x3(:));
val0=kolik3(T);
val=kolik3(T1);
mu=max(H22(:)); nu=2;
A0=A; B0=B; 
while val>val0 && mu<1e8
   H22=H22+mu*ones(n,n);
   dHH=H22.*H22'-v7.*conj(v7);
   x2=-(H22'.*gg2-v7.*gg3)./dHH;
   x3=-(H22.*gg3-conj(v7).*gg2)./dHH;
   %A=(eye(n)+in.*x2)*(eye(n)+in'.*x2);
   %B=((eye(n)+in'.*x3)*(eye(n)+in.*x3)).';
   A=(eye(n)+in2.*x2).'; A=sprav(A); %dA=det(A); A=A/dA^(1/n);
   B=(eye(n)+in2.*x3); B=sprav(B); %dB=det(B); B=B/dB^(1/n);
   T1 = reshape(A*reshape(T,n,[]),n,n1,n2);
   T1 = permute(reshape(reshape(permute(T1,[1 3 2]),[],n1)*B.',n,n2,n1),[1 3 2]);
   val=kolik3(T1);
   mu=mu*nu; nu=2*nu;
end
if val>val0
   A=A0; B=B0; T1=T;
end
end

function A2=sprav(A)
n=size(A,1);
V=poly(A); V(n+1)=V(n+1)-1; r=roots(V); [val imin]=min(abs(r)); mu=r(imin); A2=A-mu*eye(n);
end
%%
function [A B T1 nx val0 val]=onestepL(T)
[n n1 n2]=size(T);
eps=1e-10;
in=reshape(1:n^2,n,n); in=(in<in'); in2=in+in';
% T2=sum(T.*conj(T),3);
% t=[T2(1,1)+T2(2,2),T2(1,2)+T1(2,1)];
% [m it]=max(t);
% if it==2
%     T([i1,i2],:,:)=T([i2 i1],:,:);
% end
d=n;
T1=reshape(T,n,n1*n2);
alpha=conj(T1)*T1.';
T1=reshape(permute(T,[2 1 3]),n1,n*n2);
beta=conj(T1)*T1.';
T1=reshape(T,n*n1,n2);
T1d=T1(1:n+1:n^2,:); 
T1e=reshape(repmat(T1d(:).',n,1),n,n1,n2);
T1f=reshape(repmat(T1d,n,1),n,n1,n2);
%v1=sum(conj(T(i2,i1,:)).*T(i1,i1,:),3)
%v1=sum(conj(permute(T,[2 1 3])).*T1f,3);
v1=sum(conj(T).*T1e,3).';
%v2=sum(conj(T(i1,i2,:)).*T(i2,i2,:),3);
%v2=sum(conj(T).*T1e,3);
v3=sum(conj(T).*T1f,3);
%v3=sum(conj(T(i1,i2,:)).*T(i1,i1,:),3);
%v4=sum(conj(T(i2,i1,:)).*T(i2,i2,:),3);
%v4=sum(conj(T).*T1f,3).';
%v5=sum(conj(T(i2,i1,:)).*T(i2,i1,:),3);
%v5=sum(conj(T).*T,3).';
v6=sum(conj(T).*T,3);
%v7=sum(conj(T(i1,i1,:)).*T(i2,i2,:),3);
v7=sum(conj(T1f).*T1e,3);
%gg=[alpha(2,1)-v1,alpha(1,2)-v2,beta(2,1)-v3,beta(1,2)-v4];
gg2=alpha-v1.';
gg3=beta.'-v3;
H22=repmat(diag(alpha),1,n)-v6;
iH0=abs(H22)<eps;
H22(iH0)=1;
dHH=H22.*H22'-v7.*conj(v7);
x2=-(H22'.*gg2-v7.*gg3)./dHH;
x3=-(H22.*gg3-conj(v7).*gg2)./dHH;
% A=(eye(n)+in.*x2)*(eye(n)+in'.*x2);
% B=((eye(n)+in'.*x3)*(eye(n)+in.*x3)).';
%A=eye(n)+in2.*x2; dA=det(A); A=A/dA^(1/n);
%B=(eye(n)+in2.*x3).'; dB=det(B); B=B/dB^(1/n);
A=eye(n)+in'.*x2; %dA=det(A); A=A/dA^(1/n);
B=(eye(n)+in'.*x3).'; %dB=det(B); B=B/dB^(1/n);
T1 = reshape(A*reshape(T,n,[]),n,n1,n2);
T1 = permute(reshape(reshape(permute(T1,[1 3 2]),[],n1)*B.',n,n2,n1),[1 3 2]);
nx=norm(x2(:))+norm(x3(:));
val0=kolik3(T);
val=kolik3(T1);
mu=max(H22(:)); nu=2;
A0=A; B0=B; 
while val>val0 && mu<1000
   H22=H22+mu*ones(n,n);
   dHH=H22.*H22'-v7.*conj(v7);
   x2=-(H22'.*gg2-v7.*gg3)./dHH;
   x3=-(H22.*gg3-conj(v7).*gg2)./dHH;
   %A=(eye(n)+in.*x2)*(eye(n)+in'.*x2);
   %B=((eye(n)+in'.*x3)*(eye(n)+in.*x3)).';
   %A=eye(n)+in2.*x2; dA=det(A); A=A/dA^(1/n);
   %B=(eye(n)+in2.*x3).'; dB=det(B); B=B/dB^(1/n);
   A=eye(n)+in'.*x2; %dA=det(A); A=A/dA^(1/n);
   B=(eye(n)+in'.*x3).'; %dB=det(B); B=B/dB^(1/n);
   T1 = reshape(A*reshape(T,n,[]),n,n1,n2);
   T1 = permute(reshape(reshape(permute(T1,[1 3 2]),[],n1)*B.',n,n2,n1),[1 3 2]);
   val=kolik3(T1);
   mu=mu*nu; nu=2*nu;
end
if val>val0
   A=A0; B=B0;
end
end
%%

function val=kolik3(T)
persistent diagidx d n;
if isempty(diagidx) ||  (d ~= size(T,1)) || (n ~= size(T,3))
    [d d1 n] = size(T);
    diagidx = 1:d+1:d^2;
    diagidx = bsxfun(@plus,diagidx',0:d^2:d^2*(n-1));
end
% [d d1 n]=size(T);
% ind0=zeros(1,d^2); ind0(1:d+1:d^2)=1;
% diagidx=find(repmat(ind0,1,n));
T(diagidx)=0;   %%% performing operation "off_2"
val=sum(T(:).*conj(T(:)));
end

function val=kolik5c(T,i1,i2)
%
n = size(T,1);
% Tht = reshape(T([i1 i2],:,:),2,[]);
% Tvt = reshape(permute(T(:,[i1 i2],:),[1 3 2]),[],2);
% Tht(:,i1:n:end) = 0;Tht(:,i2:n:end) = 0;
% Tvt(i1:n:end,1,:) = 0;Tvt(i2:n:end,2,:) = 0;
% val = norm(Tht(:))^2 +  norm(Tvt(:))^2;

Th = T([i1 i2],:,:);
Tv = T(:,[i1 i2],:);
Th(:,[i1 i2],:) = 0;
Tv(i1,1,:) = 0;
Tv(i2,2,:) = 0;
val = norm(Th(:))^2 + norm(Tv(:))^2;

% T6=T([i1 i2],[i1 i2],:);
% T1=T([i1 i2],:,:); T2=T(:,[i1 i2],:);
% val=sum(T1(:).*conj(T1(:)))+sum(T2(:).*conj(T2(:)))+sum(T3(:).*conj(T3(:)))...
%     -sum(T6(:).*conj(T6(:)))-sum(T7(:).*conj(T7(:)))-sum(T8(:).*conj(T8(:)))+sum(T4(2:7).*conj(T4(2:7)));
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