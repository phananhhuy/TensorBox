function [H,S,s,err]=toeplitz_factor(X)
%TOEPLITZ_FACTOR Factorize X as X=H*S with H unstructured and S Toeplitz.
% Works for real and complex data.
% PROBLEM:
% Factorize an LxJ observed matrix X (J>L) as
%           X = H * S,
% where H is LxL and  S is an LxJ Toeplitz matrix
%
% INPUT: 
% - X :  full row-rank matrix LxJ (J>L)
%
% OUTPUTS:
% - H (LxL) : estimate of H
% - S (LxJ) : estimate of S
% - s = {s1,s2} : cell holding the generator vectors of S,
%                 s1 (1xL) is the first column of S,
%                 s2 (1xJ) is the first row of S, with s1(1)=s2(1),
%                 such that S=toeplitz(s1,s2);
% - err  : frobenius norm of residual, i.e., norm(X-H*S,'fro')
%
% REFERENCE:
% E. Moulines, P. Duhamel, J.-F. Cardoso and S. Mayrargue, "Subspace Methods for
% the Blind Identification of Multichannel FIR Filters", IEEE Trans. on Signal 
% Processing, vol. 43, no.2, Feb. 1995 
%
% EXAMPLE:
% J=20;
% L=6;
% SNR=inf;  % Signal To Noise Ratio in dB, choose SNR=inf for an exact model
% s1=randn(1,L)+j*randn(1,L);
% s2=randn(1,J)+j*randn(1,J);
% s2(1)=s1(1);
% s={s1,s2};
% S=toeplitz(s1,s2);
% H=randn(L,L)+j*randn(L,L);
% X=H*S;
% [H_est,S_est,s_est,err]=toeplitz_factor(X);

% @Copyright 2008
% Dimitri Nion (feedback: dimitri.nion@gmail.com)
%
% For non-commercial use only
%
% Fast version is implemented in Aug, 2014 by Phan Anh Huy


[L J]=size(X);
if L>J
    error('The input observed matrix has to be fat')
end
    
%----------------------------------------------
% STEP 1 : estimate noise subspace from SVD 
%----------------------------------------------
[U temp temp]=svd(X.');  

%--------------------------------
% STEP 2 : Compute quadartic form
%--------------------------------
% Fast computation : Anh Huy Phan Aug, 2014
if L> J/2    
    Q=zeros(J+L-1,J+L-1);
    Un=U(:,L+1:J);           
    for p=1:J-L
        u_gen=Un(:,p);
        Up=toeplitz([u_gen(end:-1:1).', zeros(1,L-1)],[Un(end,p),zeros(1,L-1)]);
        Q=Q+Up*Up';
    end
else
    Q=zeros(J+L-1,J+L-1);
    for p=1:L
        u_gen=U(:,p);
        Up=toeplitz([u_gen(end:-1:1).', zeros(1,L-1)],[U(end,p),zeros(1,L-1)]);
        Q=Q+Up*Up';
    end
    Q = diag([1:L-1 L*ones(1,J-L+1) L-1:-1:1]) - Q;
end

%-------------------------------------------------------------
% STEP 3 : Minimization of quadratic form
% take the eigenvector associated to the smallest eigenvalue
%-------------------------------------------------------------
% [Eq Dq]=eig(Q);
% Dq=diag(Dq);
% m=find(Dq==min(Dq));
% sg=Eq(:,m);        % noise eigenvector associated to smallest eigenvalue
[sg,dqs] = eigs(Q,1,'SM'); %  Anh Huy Phan Aug, 2014
sg=sg(end:-1:1).';   % generator vector of Toeplitz matrix
s1=sg(L:-1:1);       % first column of Toeplitz matrix
s2=sg(L:end);        % first row
s={s1,s2};           % output s
S=toeplitz(s1,s2);   % fat toeplitz matrix
H=X/S;               % Estimate matrix H
err=norm(X-H*S,'fro');