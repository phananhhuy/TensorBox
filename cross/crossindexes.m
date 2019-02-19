%% Function that finds a good submatrix
function [indrows,indcols]=crossindexes(A,K)
%
% TENSORBOX, 2018

[m,n]=size(A);

indcols=zeros(K,1) ; %Initialize a null column indexes
indrows=zeros(K,1) ; %Initialize a null row indexes

C=zeros(m,K);
R=zeros(K,n);

%indrows(1)=ceil(m*rand()); %Selects the first row at random
indcols(1)=ceil(n*rand); %Selects the first column at random
% indcols(1)=1; %Selects the first column at random

%R(1,:)=A(indrows(1),:);
C(:,1)=A(:,indcols(1));
[aux,newrowind]=max((C(:,1))); % Selects the first row pivoting in maximum entry
indrows(1)=newrowind;
R(1,:)=A(indrows(1),:);

U=1/(A(indrows(1),indcols(1)));

%Aest=C(:,1)*U*R(1,:);
Eant=A(indrows(1),:);

%start a loop for columns and rows from number 2
current=2;

CU = C(:,1)*U;
UR = U*R(1,:);
while current <= K
    % Search for the best row to add
    indcols(current) = addrowCross_fast(indcols(1:current-1), Eant');
    c = A(:,indcols(current));
    x = c(indrows(1:current-1));
    CUx = CU * x;
    E=A(:,indcols(current)) - CUx;
    
    indrows(current) = addrowCross_fast(indrows(1:current-1), E);
    r = A(indrows(current),:);
    y = r(indcols(1:current-1));
    yUR = y*UR;
    Eant = A(indrows(current),:) - yUR;
    
    z = r(indcols(current));
    
    yU=y*U;                     %the product for repeated use
    Ux=U*x;                   %the (1,2)th block
    q = 1/(z-yU*x);             %the (2,2)th block
    
    t1 = q*(CUx - c);
    CU = [CU+t1*yU, -t1];
    
    t1 = q*(yUR-r);
    UR = [UR+Ux * t1 ; -t1];
    
    U = [U+q*Ux*yU -q*Ux; -yU*q q];   %updated inverse
    current=current+1;
end

function ind=addrowCross_fast(indrows,C)
ind = 1:numel(C);
ind(indrows) = [];
[maxC,loc] = max(abs(C(ind)));
ind = ind(loc);