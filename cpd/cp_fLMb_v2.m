function [P,output] = cp_fLMb(X,R,opts)
% fLMb - Fast Damped Gauss-Newton (Levenberg-Marquard) algorithm factorizes
%       the N-way tensor X into factors of R components.
%       The code inverses the approximate Hessian with a cost of O(N^3R^6)
%       compared to O(R^3(I1+...+IN)) in other dGN/LM algorithms.
%
% INPUT:
%   X:  N-D data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'|cell array]
%     .printitn: Print fit every n iterations {1}
%     .fitmax
% Output:
%  P:  ktensor of estimated factors
%
% REF:
% [1] P. Tichavsky and Z. Koldovsky, Simultaneous search for all modes in
% multilinear models, ICASSP, 2010, pp. 4114 ? 4117.
%
% [2] A.-H. Phan, P. Tichavsky, A. Cichocki, "Low Complexity Damped
% Gauss-Newton Algorithms for CANDECOMP/PARAFAC", submitted to SIMAX, 2011
%
% [3] A. H. Phan, P. Tichavsk?y, and A. Cichocki, Fast damped Gauss-Newton
% algorithm for sparse and nonnegative tensor factorization, in ICASSP,
% 2011, pp. 1988-1991.
%
% The function uses the Matlab Tensor toolbox.
% See also: cp_als
%
% This code is to assist editors and reviewers in the evaluation the
% manuscript [2].
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors of [2].
%
%
% Anh Huy Phan, 07/2010
% v2 = v2.1 with fast CP gradients 11/2011
% v2.2 with fast Kronecker and Khatri-Rao product 03/2012
% v2.3 change Permutation matrix to vector of indices 03/2012


if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end
N = ndims(X); normX = norm(X);
normX2 = normX.^2; In = size(X);
Uinit = cp_init(X,R,param);  U = Uinit;

p_perm = [];
if ~issorted(In)
    [In,p_perm] = sort(In);
    X = permute(X,p_perm);
    U = U(p_perm);
end

%% Initialize U, damping parameters mu, nu
fprintf('\nCP_fLMb:\n');

ell2 = zeros(N,R); for n = 1:N,  ell2(n,:) = sum(abs(U{n}).^2); end
mm = zeros(N,R); for n = 1:N, mm(n,:) = prod(ell2([1:n-1 n+1:N],:),1); end
nu=2; mu=param.tau*max(mm(:));    % initialize damping mu, nu

warning off;
P = ktensor(U);
err=normX.^2 + norm(P).^2 - 2 * innerprod(X,P);
fit = 1-sqrt(err)/normX; fitold = fit; fitarr = fit;
%% Main Loop: Iterate until convergence
iter = 1;iter2 = 1;boostcnt = 0;flagtol = 0;
while (iter <= param.maxiters) && (iter2 <= 5*param.maxiters)
    %pause(.0001)
    iter2 = iter2+1; U1 = U;
    
    [U, d, g]=fastupdate(X,U,mu);      % fast update U
    
    P = ktensor(U);
    err2 = normX^2 + norm(P)^2 - 2 * innerprod(P,X);
    if err2>err                           % The step is not accepted
        U = U1;
        if mu < 1e30
            mu=mu*nu; nu=2*nu;
        else                              % recursive loop
            if (param.recursivelevel == param.MaxRecursivelevel) || (boostcnt == param.maxboost)
                break
            else
                boostcnt = boostcnt +1;
                [U,mu,nu,fit,err] = recurloop(U,mu,nu,err);
                iter= iter+1;
            end
        end
    else
        % update damping parameter mu, nu
        rho=real((err-err2)/(d'*(g+mu*d)));
        nu=2; mu=mu*max([1/3 1-(nu-1)*(2*rho-1)^3]);
        
        % Normalization factors U
        lambda = ones(1,R);
        for n=1:N
            am=sqrt(sum(U{n}.^2)); lambda = lambda.* am;
            U{n}=bsxfun(@rdivide,U{n},am);
        end
        for n=1:N
            U{n}=bsxfun(@times,U{n},lambda.^(1/N));
        end
        
        fit = 1-sqrt(err2)/normX;       %fraction explained by model
        fitchange = abs(fitold - fit);
        if mod(iter,param.printitn)==0
            fprintf('Iter %d: fit = %d delfit = %2.2e\n',iter,fit,fitchange);
        end
        
        if (iter > 1) && (fitchange < param.tol) % Check for convergence
            flagtol = flagtol + 1;
        else
            flagtol = 0;
        end
        if flagtol >= 10, break; end
        err = err2; fitold = fit; fitarr = [fitarr fit];
        iter = iter + 1;
    end
    % Check for convergence
    if (fit>= param.fitmax),  break; end
end

% Arrange the final tensor so that the columns are normalized.
P = ktensor(U); P = arrange(P);
if param.printitn>0
    fprintf(' Final fit = %e \n', fit);
end

if nargout >=2
    output = struct('Uinit',{Uinit},'Fit',fitarr,'mu',mu,'nu',nu);
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    function [U,mu,nu,fit,err] = recurloop(U,mu,nu,err)
        param.recursivelevel = param.recursivelevel+1;
        %fprintf('Recursive loop %d ... \n',param.recursivelevel)
        Ui = U; for n = 1:N, [Ui{n},foe] = qr(Ui{n},0); end
        
        % LM through recursive loop
        fit = 1-real(sqrt(err))/normX;
        opts.init = Ui;opts.recursivelevel = param.recursivelevel;
        opts.fitmax = fit;opts.alsinit = 0;
        opts.printitn = 0;opts.tol = 1e-8; %opts.maxiters = 1000;
        [P,output] = cp_fLMb(X,R,opts);
        
        err3=normX.^2 + norm(P).^2 - 2 * innerprod(X,P);
        if abs(err3 - err) < param.tol
            P = arrange(P);
            U = P.U;U{N} = bsxfun(@times,U{N},P.lambda(:).');
            nu = output.nu; mu = output.mu;
            err = err3; fit = 1-real(sqrt(err))/normX;
        end
        %fprintf('Exit recursive loop %d ... \n',param.recursivelevel)
        param.recursivelevel = param.recursivelevel-1;
    end
end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||ismember(x(1:4),{'rand' 'nvec' 'orth'})));
param.addOptional('alsinit',1);
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('tau',1e-3);
param.addOptional('maxboost',5);
param.addOptional('fullupdate',false);
param.addParamValue('MaxRecursivelevel',2);
param.addParamValue('recursivelevel',1);
param.addParamValue('TraceFit',false,@islogical);
param.addParamValue('TraceMSAE',true,@islogical);
param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end

%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function U = cp_init(X,R,param)
% Set up and error checking on initial guess for U.
N = ndims(X);
if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init;
        Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
            error('Wrong Initialization');
        end
    else
        %normX = sqrt(nansum(X(:).^2));%norm(X);
        bestfit = 0;Pbest = [];
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || ...
                (ischar(initk)  && ismember(initk(1:4),{'rand' 'nvec' 'orth'}))  % multi-initialization
                cp_fun = str2func(mfilename); %@cp_fastals_missingdata; 
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                [P,outputinit] = cp_fun(X,R,initparam);
                %fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
                fitinit = real(outputinit.Fit(end));
                if fitinit > bestfit
                    Pbest = P;
                    bestfit = fitinit;
                end
            end
        end
        Uinit = Pbest.U;
        Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
    end
elseif isa(param.init,'ktensor')
    Uinit = param.init.U;
    Uinit{end} = Uinit{end} * diag(param.init.lambda);
    Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
    if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
        error('Wrong Initialization');
    end    
elseif strcmp(param.init(1:4),'rand')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = randn(size(X,n),R);
    end
elseif strcmp(param.init(1:4),'orth')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = orth(randn(size(X,n),R));
    end    
elseif strcmp(param.init(1:4),'nvec') % changed for missing values
    Uinit = cell(N,1);
    for n = 1:N
        if R<=size(X,n)
            Uinit{n} = real(nvecs(X,n,R));
        else
            Uinit{n} = randn(size(X,n),R);
        end
    end
else
    error('Invalid initialization');
end

if param.alsinit
%     if isa(X,'tensor') && any(isnan(X.data(:)))
%         Weights = ~isnan(X.data(:));
%         Weights = reshape(Weights,size(X));
%         Options = [param.tol 10 0 0 0 10];
%         U = parafac(double(X),R,Options,0*ones(1,N),[],[],Weights);
%     else
        alsopts = struct('maxiters',5,'printitn',0,'init',{Uinit},'alsinit',0);
        try
            P = cp_fastals_missingdata(X,R,alsopts);
%             P = cp_fastals(X,R,alsopts);
        catch me            
            Weights = isnan(X.data(:));
            X(Weights) = 0;
            P = cp_als_ls(X,R,alsopts);
        end
        U = P.U;U{end} = bsxfun(@times, U{end},P.lambda.');
%     end
else
    U = Uinit;
end
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [U dv g]=fastupdate(X,U,mu)
% Fast inverse of the approximate Hessian, and update step size d
% g  gradient, dv step size,
% Inverse of the large Hessian is replaced by inverse of a smaller matrix
% (NR^2 x NR^2)

persistent Phi Ptr iGn iKnn Idx Km C w Uals orthflag ns;

N = ndims(X);In = size(X); R = size(U{1},2);R2 = R^2;
cIn = cumsum(In); cIn = [0 cIn];

if isempty(Phi) || (size(Phi,1) ~= N*R2)
    Phi = zeros(N*R2,N*R2);
    iGn = zeros(R,R,N);             % inverse of Gamma
    iKnn = zeros(R2,R2);            % diagonal blocks of inv(K)
    Ptr = per_vectrans(R,R);        % permutation P_RxR
    ir = 1:R^2;ic = ir;
    Idx = sub2ind([N*R^2,N*R^2],ir,ic);
    
    Km = zeros(R2,N,N);             % K as a partitioned matrix
    C = zeros(R,R,N);               % C in (4.5), Theorem 4.1
    w = zeros(N*R2,1);              % (4.30)
    Uals = U;
    orthflag = 0;
    
    % Find the first n* such that I1...In* > I(n*+1) ... IN
    Jn = cumprod(In); Kn = Jn(end)./Jn;
    ns = find(Jn>Kn,1);
    if ((ns >= (N-1)) && (ns > 2))
        ns = ns -1;
    end
end
dv = zeros(sum(In)*R,1);            % step size
g = zeros(sum(In)*R,1);             % gradient


for n = 1:N
    C(:,:,n) = U{n}'*(U{n});
end
orthflagnew = any(abs(C(:)) < 1e-10); % verify mutually orthogonal components
if orthflag ~= orthflagnew          % if K is invertible -> Phi as given in (4.38)
    Phi(:) = 0;                     % else Phi is given in (4.36)
    orthflag = orthflagnew;
end

Pmat = [];
Gxn = cell(N,1);
for n = [ns:-1:1 ns+1:N]
    if isa(X,'ktensor')
        Gxn{n} = mttkrp(X,U,n);   % compute CP gradients
    elseif isa(X,'tensor')
        [Gxn{n},Pmat] = cp_gradient(U,n,Pmat);   % compute CP gradients
    end
end

for n = 1:N
    % Kernel matrix
    for m = n+1:N
        gamma = prod(C(:,:,[1:n-1 n+1:m-1 m+1:N]),3);% Gamma(n,m) Eq. (4.5)
        if orthflag
            Km(:,n,m) = gamma(:); Km(:,m,n) = gamma(:);   % Eq.(4.4)
        else
            v = 1./(gamma(:)+eps)/(N-1);
            Phi(Idx+N*R2*(m-1)*R2+(n-1)*R2) = v; % inv(K) in (C.2), Phi in (4.38)
            Phi(Idx+N*R2*(n-1)*R2+(m-1)*R2) = v;
        end
    end
    if n~=N
        gamma = gamma .* C(:,:,N);      % Gamma(n,n)
    else
        gamma = prod(C(:,:,1:N-1),3);   % Gamma(n,n)
    end
    iGn(:,:,n) = inv(gamma+mu*eye(R));                % Eq. (4.20)
    
    if orthflag
        ZiGZ = kron(iGn(:,:,n),C(:,:,n)); ZiGZ = ZiGZ(:,Ptr);% Eq.(4.22)
        for m = [1:n-1 n+1:N]
            Phi((n-1)*R2+1:n*R2,(m-1)*R2+1:m*R2) = bsxfun(@times,ZiGZ,Km(:,n,m).'); % Eq.(4.36)
        end
        Phi((n-1)*R2+1:n*R2,(n-1)*R2+1:n*R2) = 0; % Eq.(4.36)
    else
        v= C(:,:,n)./(gamma+eps);
        iKnn = kron(iGn(:,:,n),C(:,:,n));iKnn = iKnn(:,Ptr);
        iKnn(1:R^2+1:end) = diag(iKnn) -(N-2)/(N-1)*v(:);    % iK(n,n), Eq. (C.2)
        Phi((n-1)*R2+1:n*R2,(n-1)*R2+1:n*R2) = iKnn;      % Phi in (4.38)
    end
    
    Ud = Gxn{n} - U{n} * gamma.';  % gradient Eq.(4.42)

    g(cIn(n)*R+1:cIn(n+1)*R) =  Ud(:);
    
    Uals{n} = Ud * iGn(:,:,n).' ;                     % Eq. (4.29)
    ww = U{n}'*Uals{n};                               % Eq. (4.30)
    w((n-1)*R2+1:n*R2) = ww(:);
end
if orthflag
    Phi(1:N*R2+1:end) = 1;                  % Eq.(4.36)
    F = Phi\w;                              % Eq.(4.35)
    
    F1 = reshape(F,R2,N);F = F1;
    for n = 1:N
        idx = [1:n-1 n+1:N];
        F(:,n) = sum(Km(:,idx,n) .* F1(:,idx),2)';
    end
else
    F = (Phi)\w;  % Phi\w              % Eq.(4.35)
end
F = reshape(F,R,R,N);                      % F in Eq.(4.31), Eq.(4.34)

for n = 1: N
    delU = Uals{n} - U{n} * (iGn(:,:,n) * F(:,:,n)).'; % deltaU = Unew - U;
    U{n} = U{n} + delU;                    % Eq.(4.33)
    dv(cIn(n)*R+1:cIn(n+1)*R) = delU(:);
end

%% CP Gradient with respect to mode n
    function [G,Pmat] = cp_gradient(A,n,Pmat)
        right = N:-1:n+1; left = n-1:-1:1;

        if n <= ns
            if n == ns
                if numel(right) == 1
                    KRP_right = A{right};
                elseif numel(right) > 1
                    KRP_right = khatrirao(A(right)); %KRP_right = khatrirao(A(right));
                end

                Pmat = reshape(X.data,[],prod(In(right))); % Right-side projection
                Pmat = Pmat * KRP_right ;
            else
                Pmat = reshape(Pmat,[],In(right(end)),R);
                if R>1
                    Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],In(right(end)),R));
                    Pmat = sum(Pmat,2);    % fast Right-side projection
                else
                    Pmat = Pmat * A{right(end)};
                end
            end

            if ~isempty(left)       % Left-side projection
                KRP_left = khatrirao(A(left));
                T = reshape(Pmat,prod(In(left)),In(n),[]);
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
                    T = sum(T,1);
                    G = squeeze(T);
                else
                    G = (KRP_left'*T)';
                end
            else
                G = squeeze(Pmat);
            end

        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left};
                elseif numel(left) > 1
                    KRP_left = khatrirao(A(left));
                end
                T = reshape(X.data,prod(In(left)),[]);
                Pmat = KRP_left' * T;   % Left-side projection
            else
                if R>1
                    Pmat = reshape(Pmat,R,In(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,In(left(1)),[]);
                    Pmat = A{left(1)}'* Pmat;
                end
            end

            if ~isempty(right)
                T = reshape(Pmat,[],In(n),prod(In(right)));
                KRP_right = khatrirao(A(right));
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
                    T = sum(T,3);
                    G = squeeze(T)';        % Right-side projection
                else
                    G = T * KRP_right;
                end
            else
                G = squeeze(Pmat)';
            end

        end
    end

end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Perm,P] = per_vectrans(m,n)
% vec(X_mn^T) = P vec(X)
Perm = reshape(1:m*n,[],n)'; Perm = Perm(:);
P = speye(m*n); P = P(Perm,:);
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function K = khatrirao(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1};
for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
end
K = reshape(K,[],R);
end

%%
function K = kron(A,B)
%  Fast implementation of Kronecker product of A and B
%
%   Copyright 2012 Phan Anh Huy
%   $Date: 2012/3/18$

if ndims(A) > 2 || ndims(B) > 2
    error(message('See ndkron.m'));
end
I = size(A); J = size(B);

if ~issparse(A) && ~issparse(B)
    K = bsxfun(@times,reshape(B,J(1),1,J(2),1),reshape(A,1,I(1),1,I(2)));
    K = reshape(K,I(1)*J(1),[]);
else
    K = kron(A,B);
end
end