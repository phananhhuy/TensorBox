function [A,B,C,smax] = bestrank1_222(Y)
% Find best rank-1 tensor approximation to the tensor Y of size 2x2x2
% by solving a polynomial of degree-6
%
% Phan Anh-Huy, 2017
% August 16, 2017
% 

t1 = @(y) (sum(sum(y(:,:,1).^2)) - sum(sum(y(:,:,2).^2)))/2;
t2 = @(y) (sum(sum(y(:,:,1).^2)) + sum(sum(y(:,:,2).^2)))/2;
% t3 = @(y) sum(sum(y(:,:,1).*y(:,:,2)));
t4 = @(y) - y(1)*y(4) + y(2)*y(3) - y(5)*y(8) + y(6)*y(7);
t5 = @(y) - y(1)*y(4) + y(2)*y(3) + y(5)*y(8) - y(6)*y(7);
t6 = @(y) -y(1)*y(8) + y(2)*y(7) + y(3)*y(6) - y(4)*y(5);

% Find orthonormal rotation matrix along mode-3 of Y
[~,s,Z] = svd(reshape(Y,[],2));
Yz = reshape(reshape(Y,[],2)*Z,[2 2 2]); % Y x3 Z^T

t_1 = t1(Yz);
t_2 = t2(Yz);
% t_3 = t3(Yz);
t_4 = t4(Yz);
t_5 = t5(Yz);
t_6 = t6(Yz);

% Coefficients of the degree-6 polynomial 
c(1) = t_4*t_6^2 - t_5*t_6^2;
c(2) = 4*t_1^2*t_6 - 4*t_2*t_1*t_6 - 4*t_5^2*t_6 + 4*t_4*t_5*t_6 + 2*t_6^3;
c(3) = 4*t_1^2*t_5 + 4*t_4*t_1^2 - 8*t_2*t_1*t_5 - 4*t_5^3 + 4*t_4*t_5^2 + 11*t_5*t_6^2 - t_4*t_6^2;
c(4) = 16*t_5^2*t_6 - 4*t_6^3;
c(5) = - 4*t_1^2*t_5 + 4*t_4*t_1^2 - 8*t_2*t_1*t_5 + 4*t_5^3 + 4*t_4*t_5^2 - 11*t_5*t_6^2 - t_4*t_6^2;
c(6) = 4*t_1^2*t_6 + 4*t_2*t_1*t_6 - 4*t_5^2*t_6 - 4*t_4*t_5*t_6 + 2*t_6^3;
c(7) = t_4*t_6^2 + t_5*t_6^2;

% x = atan(z), whereas z is a root of p(z)

z = roots(c);
ireal=find(abs(imag(z))<1e-8);
x=[atan(real(z(ireal)));-pi/2;pi/2];

% smax=-1;
% for i1=1:length(x)
%     c=cos(x(i1)); s=sin(x(i1));
%     M=Yz(:,:,1)*c+Yz(:,:,2)*s;
%     [U,S,V]=svd(M);
%     s0=max(diag(S));
%     if s0>smax
%         A=U(:,1);
%         B=V(:,1);
%         C=[c;s];
%         smax=s0;
%     end
% end

% choose the best root which yields the largest singular value of Y.
% svd of the projected matrix is 
%  a(x) + sqrt(a^2 - b^2) 
%  or 
%     b(x)*b'(x)/a'(x)
%
a = @(x) t_1 *cos(2*x) + t_2; % t3 is omitted since t3= 0
b = @(x) t_5 *cos(2*x) + t_6 * sin(2*x) + t_4;

ax = a(x);
bx = b(x);
s = ax + sqrt(ax.^2 - bx.^2); % the best singular value is one of s.
[smax,ix] = max(sqrt(s/2));

c=cos(x(ix)); s=sin(x(ix));
M=Yz(:,:,1)*c+Yz(:,:,2)*s;
[U,S,V]=svd(M);
S = diag(S); [~,is] = max(S);
A=U(:,is);
B=V(:,is);
C = Z*[c;s]; % rotate c back by Z