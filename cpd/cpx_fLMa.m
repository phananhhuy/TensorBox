function [P,output] = cp_fLMa(X,R,opts)
% fLMa - Fast Damped Gauss-Newton (Levenberg-Marquard) algorithm factorizes
%       an N-way tensor X into factors of R components.
%       The code inverts the approximate Hessian with a cost of O(NR^6) or
%       O(N^3R^6) compared to O(R^3(I1+...+IN)) in other dGN/LM algorithms.
%
% INPUT:
%   X:  N-way data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'| ktensor| cell array]
%          init can be a cell array whose each entry specifies an intial
%          value. The algorithm will chose the best one after small runs.
%          For example,
%          opts.init = {'random' 'random' 'nvec'};
%     .printitn: Print fit every n iterations {1}
%     .fitmax
%     .TraceFit: check fit values as stoping condition.
%     .TraceMSAE: check mean square angular error as stoping condition
%
% OUTPUT:
%  P:  ktensor of estimated factors
%  output:
%      .Fit
%      .NoIters
%
% EXAMPLE
%   X = tensor(randn([10 20 30]));
%   opts = cp_fLMa;
%   opts.init = {'nvec' 'random' 'random'};
%   [P,output] = cp_fLMa(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
%
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "Low Complexity Damped
% Gauss-Newton Algorithms for CANDECOMP/PARAFAC", SIAM, SIAM, Jour-
% nal on Matrix Analysis and Applications, vol. 34, pp. 126?147, 2013.
%
% [2] A. H. Phan, P. Tichavsky, and A. Cichocki, Fast damped Gauss-Newton
% algorithm for sparse and nonnegative tensor factorization, in ICASSP,
% 2011, pp. 1988-1991.
% 
% [3] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition, IEEE TSP, in print,
% 2013, available online at http://arxiv.org/abs/1209.3215.
%
% [4] A.-H. Phan, P. Tichavsky, A. Cichocki, "Fast Alternating LS
% Algorithms for High Order CANDECOMP/PARAFAC Tensor Factorization",
% http://arxiv.org/abs/1204.1586. 
%
% [5] P. Tichavsky and Z. Koldovsky, Simultaneous search for all modes in
% multilinear models, ICASSP, 2010, pp. 4114 - 4117.
%
% The function uses the Matlab Tensor toolbox.
% See also: cp_fastals
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors.
%
% This algorithm is a part of the TENSORBOX, 2012.
%
% Anh-Huy Phan, 07/2010
% v2 = v2.1 with fast CP gradients 11/2011
% v2.2 with fast Kronecker and Khatri-Rao product 03/2012
% v2.3 change Permutation matrix to vector of indices 03/2012
% v3 approximate fLM for missing data 24/06/2012
% v3.2 full update for missing data 30/06/2012 (see full_update)

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0, P = param; return ; end

%% Initializing U
Uinit = cp_init(X,R,param);  U = Uinit;

fprintf('\nCP_fLMa:\n');

In = size(X);N = ndims(X);

p_perm = [];
if ~issorted(In)
    [In,p_perm] = sort(In);
    X = permute(X,p_perm);
    U = U(p_perm);
end
I = In;

normX = norm(X); normX2 = normX.^2;

if isa(X,'tensor')
    IsReal = isreal(X.data);
elseif isa(X,'ktensor') || isa(X,'ttensor')
    IsReal = all(cellfun(@isreal,X.u));
end

%% Damping parameters mu, nu
ell2 = zeros(N,R); for n = 1:N,  ell2(n,:) = sum(abs(U{n}).^2); end
mm = zeros(N,R); for n = 1:N, mm(n,:) = prod(ell2([1:n-1 n+1:N],:),1); end
nu=2; mu=param.tau*max(mm(:));    % initialize damping mu, nu

warning off;
% P = ktensor(U);

% Find the first n* such that I1...In* > I(n*+1) ... IN
Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
ns = find(Jn<=Kn,1,'last');

%% Main Loop: Iterate until convergence
iter = 1;iter2 = 1;boostcnt = 0;flagtol = 0;full_misscnt = 0;
fullupdate_flag = false;mu_inc_count = 0;


%% Correlation matrices Cn
C = zeros(R,R,N);Gamman = zeros(R,R,N);
for n = 1:N, C(:,:,n) = U{n}'*(U{n});end
for n = 1:N, Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3);end

% CP-gradient X(n) * Khatri-Rao(U) except Un
Gxn = cell(N,1);
Pmat = [];
Uc = cellfun(@conj,U,'uni',0);
for n = [ns:-1:1 ns+1:N]
    % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
    if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
        Gxn{n} = mttkrp(X,Uc,n);
    elseif isa(X,'tensor') || isa(X,'sptensor')
        if (n == ns) || (n == ns+1)
            [Gxn{n},Pmat] = cp_gradient(Uc,n,X);
        else
            [Gxn{n},Pmat] = cp_gradient(Uc,n,Pmat);
        end
    end
end

err=normX2 + sum(sum(Gamman(:,:,1).*C(:,:,1))) - 2 * real(sum(conj(Gxn{ns}(:)).*U{ns}(:)));
fit = 1-sqrt(err)/normX; fitold = fit; fitarr = [1 fit]; % iteration-fit

flag = false;
while (iter <= param.maxiters) && (iter2 <= 5*param.maxiters)
    
    %pause(.0001)
    iter2 = iter2+1; U1 = U;
    
    if param.updaterule~= 1
        flag = cellfun(@(x) any(abs(reshape(x'*x,[],1))<= 1e-8), U,'uni',1);
        flag = any(flag==1);%flag = 1;
    end
    
    if param.updaterule==1 || flag
        currupdaterule = 1;
        [U, d, g] = updaten3r6(X,U,mu,Gxn,C,Gamman);       % update U n3r6
        %     elseif opts.updaterule==3
        %         currupdaterule = 3;
        %         [U, d, g] = superfast_update3(X,U,mu,Gxn); % fast update U
    else
        %elseif (opts.updaterule==2 ) %|| flag
        currupdaterule = 2;
        [U, d, g] = updatenr6(X,U,mu,Gxn,C,Gamman); % update U nr6
        
        %     elseif opts.updaterule==3
        %         currupdaterule = 2;
        %         [U, d, g] = update_nr6(X,U,mu,Gxn,C,Gamman); % fast update U
    end
    
    % compute CP gradient with respect to U{ns}
    Uc = cellfun(@conj,U,'uni',0);
    if isa(X,'ktensor')
        Gxnnew = mttkrp(X,Uc,ns);   % compute CP gradient w.r.t U{ns}
    elseif isa(X,'tensor') ||  isa(X,'sptensor')
        [Gxnnew,Pmat] = cp_gradient(Uc,ns,X);% compute CP gradient w.r.t U{ns}
    end
    
    for n = 1:N, C(:,:,n) = U{n}'*(U{n}); end
    err2=normX2 + sum(sum(prod(C,3))) - 2 * real(sum(conj(Gxnnew(:)).*U{ns}(:)));
    
    if (err2>err)|| isnan(err2)                           % The step is not accepted
        U = U1;
        if (norm(d)> param.tol) ... %&& (sqrt(err2-err)>(1e-5*normX)) ...
                && (mu < 1e10) && (mu_inc_count < 20)
            mu=mu*nu; nu=2*nu; mu_inc_count = mu_inc_count+1;
        else                              % recursive loop
            if (param.recursivelevel == param.MaxRecursivelevel) || (boostcnt == param.maxboost)
                break
            else
                boostcnt = boostcnt +1;
                [U,mu,nu,fit,err] = recurloop(U,mu,nu,err);
                % Update C and Gamma
                for n = 1:N, C(:,:,n) = U{n}'*(U{n});end
                for n = 1:N, Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3);end
                iter= iter+1;
            end
            mu_inc_count = 0;
        end
    else
        mu_inc_count = 0;
        % update damping parameter mu, nu
        rho=real((err-err2)/(d'*(g+mu*d)));
        nu=2; mu=mu*max([1/3 1-(nu-1)*(2*rho-1)^3]);
        
        % Normalize factors U
        am = zeros(N,R); %lambda = ones(1,R);
        for n=1:N
            am(n,:) =sqrt(sum(abs(U{n}).^2));
            U{n}=bsxfun(@rdivide,U{n},am(n,:));
        end
        lambdans = prod(am([1:ns-1 ns+1:end],:),1);
        lambda = lambdans .* am(ns,:);
        for n=1:N
            U{n}=bsxfun(@times,U{n},lambda.^(1/N));
        end
        
        % Precompute Cn and Gamma_n
        for n = 1:N, C(:,:,n) = U{n}'*(U{n});end
        for n = 1:N, Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3); end
        
        % Fix CP gradient Gns due to normalization
        Gxn{ns} = bsxfun(@times,Gxnnew,lambda.^((N-1)/N)./lambdans);
        Pmat = bsxfun(@times,Pmat,lambda.^((N-ns)/N)./prod(am([ns+1:end],:),1));
        % Pre-computing CP gradients
        Uc = cellfun(@conj,U,'uni',0);
        for n = [ns-1:-1:1 ns+1:N]
            if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
                Gxn{n} = mttkrp(X,Uc,n);
            elseif isa(X,'tensor') || isa(X,'sptensor')
                if (n == ns) || (n == ns+1)
                    [Gxn{n},Pmat] = cp_gradient(Uc,n,X);
                else
                    [Gxn{n},Pmat] = cp_gradient(Uc,n,Pmat);
                end
            end
        end
        
        fit = 1-sqrt(err2)/normX;       %fraction explained by model
        fitchange = abs(fitold - fit);
        if mod(iter,param.printitn)==0
            str = {'Fast N3R6' 'Fast NR6' 'AppFast NR6' 'Full'};
            fprintf('Iter %d: fit = %d delfit = %2.2e. %s update rule.\n',iter,fit,...
                fitchange,str{currupdaterule});
        end
        
        if (iter > 1) && (fitchange < param.tol) % Check for convergence
            flagtol = flagtol + 1;
        else
            flagtol = 0;
        end
        
        if flagtol >= 3,
            break
        end
        err = err2; fitold = fit; fitarr = [fitarr; iter fit];
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
% Rearrange dimension of the estimation tensor
if ~isempty(p_perm)
    P = ipermute(P,p_perm);
end

if nargout >=2
    output = struct('Uinit',{Uinit},'Fit',fitarr,'mu',mu,'nu',nu);
end

%% CP Gradient with respect to mode n
    function [G,Pmat] = cp_gradient(A,n,Pmat)
        persistent KRP_right0;
        right = N:-1:n+1; left = n-1:-1:1;
        % KRP_right =[]; KRP_left = [];
        if n <= ns
            if n == ns
                if numel(right) == 1
                    KRP_right = A{right};
                elseif numel(right) > 2
                    [KRP_right,KRP_right0] = khatrirao(A(right));
                elseif numel(right) > 1
                    KRP_right = khatrirao(A(right));
                else
                    KRP_right = 1;
                end
                
                if isa(Pmat,'tensor')
                    Pmat = reshape(Pmat.data,[],prod(I(right))); % Right-side projection
                elseif isa(Pmat,'sptensor')
                    Pmat = reshape(Pmat,[prod(size(Pmat))/prod(I(right)),prod(I(right))]); % Right-side projection
                    Pmat = spmatrix(Pmat);
                else
                    Pmat = reshape(Pmat,[],prod(I(right))); % Right-side projection
                end
                Pmat = Pmat * KRP_right ;
            else
                Pmat = reshape(Pmat,[],I(right(end)),R);
                if R>1
                    Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],I(right(end)),R));
                    Pmat = sum(Pmat,2);    % fast Right-side projection
                else
                    Pmat = Pmat * A{right(end)};
                end
            end
            
            if ~isempty(left)       % Left-side projection
                KRP_left = khatrirao(A(left));
%                 if (isempty(KRP_2) && (numel(left) > 2))
%                     [KRP_left,KRP_2] = khatrirao(A(left));
%                 elseif isempty(KRP_2)
%                     KRP_left = khatrirao(A(left));
%                     %KRP_2 = [];
%                 else
%                     KRP_left = KRP_2; KRP_2 = [];
%                 end
                T = reshape(Pmat,prod(I(left)),I(n),[]);
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
                    T = sum(T,1);
                    %G = squeeze(T);
                    G = reshape(T,[],R);
                else
                    G = (KRP_left.'*T).';
                end
            else
                %G = squeeze(Pmat);
                G = reshape(Pmat,[],R);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left}.';
                elseif numel(left) > 1
                    KRP_left = khatrirao_t(A(left));
                    %KRP_left = khatrirao(A(left));KRP_left = KRP_left';
                else 
                    KRP_left = 1;
                end
                if isa(Pmat,'tensor')
                    T = reshape(Pmat.data,prod(I(left)),[]);
                elseif isa(Pmat,'sptensor')
                    T = reshape(Pmat,[prod(I(left)) prod(size(Pmat))/prod(I(left))]); % Right-side projection
                    T = spmatrix(T);
                else
                    T = reshape(Pmat,prod(I(left)),[]);
                end
                %
                Pmat = KRP_left * T;   % Left-side projection
            else
                if R>1
                    Pmat = reshape(Pmat,R,I(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}.');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,I(left(1)),[]);
                    Pmat = A{left(1)}.'* Pmat;
                end
            end
            
            if ~isempty(right)
                T = reshape(Pmat,[],I(n),prod(I(right)));
                
                if (n == (ns+1)) && (numel(right)>=2)
                    %KRP_right = KRP_right0;
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right0.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T, R,[]).';
                    else
                        %G = squeeze(T) * KRP_right0;
                        G = reshape(T,[],prod(I(right))) * KRP_right0;
                    end
                else
                    KRP_right = khatrirao(A(right));
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T,R,[]).';        % Right-side projection
                    else
                        %G = squeeze(T) * KRP_right;
                        G = reshape(T,I(n),[]) * KRP_right;
                    end
                end
            else
                %G = squeeze(Pmat)';
                G = reshape(Pmat,R,[]).';
            end
            
        end
        
    end




%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    function [U,mu,nu,fit,err] = recurloop(U,mu,nu,err)
        param.recursivelevel = param.recursivelevel+1;
        %fprintf('Recursive loop %d ... \n',param.recursivelevel)
        Ui = U;
        for n = 1:N,
            if In(n)>=R
                [Ui{n},foe] = qr(Ui{n},0);
            else
                Ui{n} = randn(In(n),R);
            end
        end
        
        % LM through recursive loop
        fit = 1-real(sqrt(err))/normX;
        opts.init = Ui;opts.recursivelevel = param.recursivelevel;
        opts.fitmax = fit;opts.alsinit = 0;
        opts.printitn = 0;opts.tol = 1e-8; %opts.maxiters = 1000;
        [P,output] = feval(mfilename,X,R,opts);
        
        err3=normX2 + norm(P).^2 - 2 * real(innerprod(X,P));
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

param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'orth' 'fibe' 'dtld'})));
param.addOptional('alsinit',1);
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('tau',1e-3);
param.addOptional('maxboost',5);
param.addOptional('fullupdate',false);
param.addOptional('complex',false);
param.addParamValue('MaxRecursivelevel',2);
param.addParamValue('recursivelevel',1);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addParamValue('updaterule',2);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [U dv g]= updaten3r6(X,U,mu,Gxn,C,Gamman)
% Fast inverse of the approximate Hessian, and update step size d
% g  gradient, dv step size,
persistent Phi w iGn Ptr Uals ns Km ; %Weights missingdata;

N = ndims(X);In = size(X); R = size(U{1},2);R2 = R^2;
cIn = cumsum(In); cIn = [0 cIn];


if isempty(Phi) || (size(Phi,1) ~= N*R2)
    Phi = zeros(N*R2,N*R2);
    w = zeros(N*R2,1);          %
    iGn = zeros(R,R,N);         % inverse of Gamma
    %C = zeros(R,R,N);   % C in (4.5), Theorem 4.1
    Ptr = per_vectrans(R,R); % permutation matrix Appendix A
    Uals = U;
    
    % Find the first n* such that I1...In* > I(n*+1) ... IN
    Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
    ns = find(Jn<=Kn,1,'last');
    
    Km = zeros(R2,N,N);      % matrix K as a partitioned matrix
end
dv = zeros(sum(In)*R,1);    % step size
g = zeros(sum(In)*R,1);     % gradient

for n = 1:N
    % Kernel matrix
    for m = n+1:N
        gamma = prod(C(:,:,[1:n-1 n+1:m-1 m+1:N]),3); % Gamma(n,m) Eq.(4.5)
        Km(:,n,m) = gamma(:); Km(:,m,n) = gamma(:);   % Eq.(4.4)
    end
    
    if n~=N
        gamma = gamma .* C(:,:,N) +eps;   %Gamma(n,n), Eq.(4.5)
    else
        gamma = prod(C(:,:,1:N-1),3)+eps; %Gamma(n,n), Eq.(4.5)
    end
    iGn(:,:,n) = inv(gamma+mu*eye(R));    % Eq.(4.20)
    
    % Phi = ZiGZ * K in Eq.(4.36)
    ZiGZ = kron(iGn(:,:,n),C(:,:,n)); ZiGZ = ZiGZ(:,Ptr);% Eq.(4.22)
    for m = [1:n-1 n+1:N]
        Phi((n-1)*R2+1:n*R2,(m-1)*R2+1:m*R2) = bsxfun(@times,ZiGZ,Km(:,n,m).'); % Eq.(4.36)
    end
    %     Phi((n-1)*R2+1:n*R2,[1:(n-1)*R2 n*R2+1:end]) = reshape(...
    %             bsxfun(@times,ZiGZ,reshape(Km(:,n,[1:n-1 n+1:N]),1,R2,[])),...
    %             R2,[]);
    
    Ud = Gxn{n} - (U{n} * gamma.');  % gradient Eq.(4.42)
    
    g(cIn(n)*R+1:cIn(n+1)*R) =  Ud(:);
    
    Uals{n} = Ud * iGn(:,:,n).';          % Uals - Uhat_als in Eq.(4.29)
    w((n-1)*R2+1:n*R2) = reshape(U{n}'*Uals{n},[],1); % Eq.(4.30)
end

Phi(1:N*R2+1:end) = 1;                    % Eq.(4.36)
w = Phi\w;                                % Eq.(4.34)

F1 = reshape(w,R2,N);F = F1;
for n = 1:N
    idx = [1:n-1 n+1:N];
    F(:,n) = sum(Km(:,idx,n) .* F1(:,idx),2).';
end
F = reshape(F,R,R,N);                     % F in Eq.(4.31), Eq.(4.34)

for n = 1: N
    delU = Uals{n} - U{n} * (iGn(:,:,n) * F(:,:,n)).'; % deltaU = Unew - U;
    U{n} = U{n} + delU;                    % Eq.(4.33)
    dv(cIn(n)*R+1:cIn(n+1)*R) = delU(:);
end

end

% %% Super fast (N+1) inverses of R^2 x R^2 matrices
% %% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function [U dv g] = superfast_update(X,U,mu)
% % Fast inverse of the approximate Hessian, and update step size d
% % g  gradient, dv step size,
% persistent C ns Ptr; %Weights missingdata;
% 
% N = ndims(X);In = size(X); R = size(U{1},2);R2 = R^2;
% cIn = cumsum(In); cIn = [0 cIn];
% cIR = cIn*R;
% 
% if isempty(C) || (size(C,1) ~= R) || (size(C,3) ~= N)
%     C = zeros(R,R,N);   % C in (4.5), Theorem 4.1
%     Ptr = per_vectrans(R,R); % permutation matrix Appendix A
%     
%     % Find the first n* such that I1...In* > I(n*+1) ... IN
%     Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
%     ns = find(Jn<=Kn,1,'last');
%     
% end
% g = zeros(sum(In)*R,1);     % gradient
% 
% for n = 1:N
%     C(:,:,n) = U{n}'*(U{n});
% end
% 
% Pmat = [];
% Gxn = cell(N,1);
% for n = [ns:-1:1 ns+1:N]
%     if isa(X,'ktensor')
%         Gxn{n} = mttkrp(X,U,n);   % compute CP gradients
%     elseif isa(X,'tensor')
%         if (n == ns) || (n == ns+1)
%             [Gxn{n},Pmat] = cp_gradient(X,U,n);   % compute CP gradients
%         else
%             [Gxn{n},Pmat] = cp_gradient(Pmat,U,n);   % compute CP gradients
%         end
%     end
% end
% 
% %%
% Ir2 = eye(R^2);
% 
% 
% %% Method 3
% ZdH0Zd = 0 ;
% %iH0g = zeros(sum(In)*R,1);
% H0Zdtg = 0;
% iGamman = zeros(R,R,N);
% Gamman = zeros(R,R,N);
% Gn2 = zeros(R,R,N);
% iKkkn = zeros(R^2*N,R^2);
% for n = 1:N
%     Gamma = prod(C(:,:,[1:n-1 n+1:N]),3);
%     iGamma = inv(Gamma + mu * eye(R)); % N R xR (9) (10)
%     
%     Gamman(:,:,n) = Gamma;
%     iGamman(:,:,n) = iGamma;
%     iGnkrC = kron(iGamma,C(:,:,n));
%     omega = bsxfun(@times,iGnkrC,Gamma(:));
%     %omega = bsxfun(@times,iGnkrC,Gamma(:)./reshape(C(:,:,n),[],1));
%     omega = omega(Ptr,:);
%     
%     % inver iKKk
%     omega(1:R^2+1:end) = omega(1:R^2+1:end) - reshape(C(:,:,n),1,[]);
%     iKkk = inv(-omega);   % R^2 x R^2  inv(dvec(Cn) - dvec(Gam_n)* Prr * (inv(Gam_n) o C_n))
%     
%     gn2 = (U{n}'*Gxn{n} - C(:,:,n) * Gamma.')*iGamma;  % iGnkrU' * vec(gn)
%     
%     gn2 = iKkk' * gn2(:);  % iKkk' * iGnkrU' * vec(gn)
%     
%     H0Zdtg = H0Zdtg + gn2;
%     
%     gn2 = Gamma.*reshape(gn2,[],R)';
%     Gn2(:,:,n) = gn2;
%     
%     %foe = omega*iKkk
%     foe = C(:,:,n) * reshape(iKkk,R,[]);
%     foe = reshape(foe,R,R,[]);
%     foe = permute(foe,[2 1 3]);
%     foe = iGamma * reshape(foe,R,[]);
%     foe = reshape(foe,R,R,[]);
%     foe = ipermute(foe,[2 1 3]);
%     foe = reshape(foe,R^2,[]);
%     foe = bsxfun(@times,foe,Gamma(:));
%     %foe = foe(Ptr,:);
%     
%     ZdH0Zd = ZdH0Zd + foe;
%     iKkkn((n-1)*R^2+1:n*R^2,:) = iKkk;
% end
% 
% Gamma = prod(C,3);
% foe = (Ir2 + ZdH0Zd(Ptr,:)')\H0Zdtg;
% foe = foe.*Gamma(:);
% foe = foe(Ptr);
% foe = iKkkn * foe;
% foe = reshape(foe,R,R,N);
% 
% dv = zeros(sum(In)*R,1);
% for n = 1: N
%     gn = Gxn{n} - U{n} * Gamman(:,:,n).';
%     
%     
%     g(cIR(n)+1:cIR(n+1)) = gn(:);  % gradient Eq.(4.42)
%     
%     temp = U{n} * (Gn2(:,:,n) - foe(:,:,n)) + gn;
%     temp = temp * iGamman(:,:,n);
%     dv(cIR(n)+1:cIR(n+1)) =  temp(:);
%     U{n} = U{n} + temp;                    % Eq.(4.33)
% end
% 
% end


%% Super fast (N+1) inverses of R^2 x R^2 matrices
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [U dv g] = updatenr6(X,U,mu,Gxn,C,Gamman)
% Fast inverse of the approximate Hessian, and update step size d
% g  gradient, dv step size,
persistent Ptr Pr_idx iJn iGamman; %Weights missingdata;

N = ndims(X);In = size(X); R = size(U{1},2);%R2 = R^2;
cIn = cumsum(In); cIn = [0 cIn];
cIR = cIn*R;

if (size(iJn,1) ~= R^2)
    %C = zeros(R,R,N);   % C in (4.5), Theorem 4.1
    
    [Ptr,Pr] = per_vectrans(R,R); % permutation matrix Appendix A
    Pr_idx = Pr==1;
    
    iJn = zeros(R^2,R^2,N);
    
    % Find the first n* such that I1...In* > I(n*+1) ... IN
    %Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
    %ns = find(Jn<=Kn,1,'last');
    
    %     C0 = zeros(R,R,N);
    %     TT0 = zeros(R,R,N);
    %     TT1 = zeros(R^2,R^2,N);
    %     TT2 = zeros(R^2,R^2,N);
    iGamman = zeros(R,R,N);
end
g = zeros(sum(In)*R,1);     % gradient

%% Method 3

Su = 0;

fullinverse = 1;
% if mu < 1e-2
%     fullinverse = 0;
% end
% Approximate 1(s+mu) = 1/mu - 1/mu^2 *s, Jn = iGamman otime Cn
% if fullinverse == 0
%     if norm(C(:) - C0(:)) > 1e-6
%         for n = 1:N
%             [uu,ss] = eig(Gamman(:,:,n));
%             %TT0(:,:,n) = uu*uu';
%             %TT1(:,:,n) = kron(eye(R),C(:,:,n));
%             %TT2(:,:,n) = kron(Gamman(:,:,n),C(:,:,n));
%         end
%     end
%     C0 = C;
% end

for n = 1:N
%     if fullinverse == 1
        iGamman(:,:,n) = inv(Gamman(:,:,n) + mu * eye(R)); % N R xR (9) (10)
        Jn = kron(iGamman(:,:,n),C(:,:,n));
        
        dvecC_Gamma = C(:,:,n)./Gamman(:,:,n);dvecC_Gamma = dvecC_Gamma.';
        Jn(Pr_idx) = Jn(Pr_idx) - dvecC_Gamma(:);
        
        % invert iKKk
        iJn(:,:,n) = inv(Jn);   % R^2 x R^2  inv(dvec(Cn) - dvec(Gam_n)* Prr * (inv(Gam_n) o C_n))
        
%     else % Fast approximate of inverse of Jn
%         iGamman(:,:,n) = inv(Gamman(:,:,n) + mu * eye(R)); % N R xR (9) (10)
%         %iGamman(:,:,n) = 1/mu*eye(R)- 1/mu^2*Gamman(:,:,n) ;%+ 1/mu^3*Gamman(:,:,n)^2 - 1/mu^4*Gamman(:,:,n)^3; % N R xR (9) (10)
%         %Jn = 1/mu*TT1(:,:,n) - 1/mu^2 * TT2(:,:,n);
%         %Jn = 1/mu*TT1(:,:,n);
%         
%         Jn = 1/mu*kron(eye(R),C(:,:,n));
%         dvecC_Gamma = C(:,:,n)./Gamman(:,:,n);
%         Jn(Pr_idx) = Jn(Pr_idx) - dvecC_Gamma(:);
%         %         for r1 = 1:R
%         %             Jn(Pr_idx) = Jn(Pr_idx) - dvecC_Gamma(:,r);
%         %         end
%         
%         % inver iKKk
%         iJn(:,:,n) = inv(Jn);   % R^2 x R^2  inv(dvec(Cn) - dvec(Gam_n)* Prr * (inv(Gam_n) o C_n))
%     end
    
    vGn = reshape(Gamman(:,:,n).',[],1);
    temp = bsxfun(@rdivide,iJn(:,:,n),vGn);
    vGn = reshape(Gamman(:,:,n),1,[]);
    temp = bsxfun(@rdivide,temp,vGn);
    Su = Su + temp;
end
Gamma = Gamman(:,:,1).*C(:,:,1);
% Su = -(N-1)*diag(1./Gamma(:)) * Pr - Pr * Su * Pr;
% % Su(Ptr,:) = -Su;
% % Su(1:R^2+1:end) = Su(1:R^2+1:end) -(N-1)*(1./Gamma(:)');
% % Su = Su(:,Ptr);
% % Su - (-(N-1)*diag(1./Gamma(:)) * Pr-Pr*sum(bsxfun(@times,bsxfun(@times,iJn,reshape(1./Gamman,[],1,N)),reshape(1./Gamman,1,[],N)),3)*Pr)

% change sign of Su and LutZtGugn
Su = Su(:,Ptr);
Su(1:R^2+1:end) = Su(1:R^2+1:end) + (N-1)*(1./reshape(Gamma.',1,[]));
Su = Su(Ptr,:);

%%
LutZtGugn = 0;
dv = zeros(sum(In)*R,1);

for n = 1: N
    gn = Gxn{n} - U{n} * Gamman(:,:,n).'; % gradient
    g(cIR(n)+1:cIR(n+1)) = gn(:);  % gradient Eq.(4.42)
    
    Gugn = gn*iGamman(:,:,n).';       % term1
    ZtGugn = U{n}' * Gugn ;
    iJuZtGugn = iJn(:,:,n) * ZtGugn(:);
    GuZiJuZtGugn = U{n}*reshape(iJuZtGugn,R,R) * iGamman(:,:,n).';    % term2
    
    LutZtGugn = LutZtGugn + bsxfun(@rdivide,iJuZtGugn,reshape(Gamman(:,:,n).',[],1));
    % Eq.(4.33)
    
    dv(cIR(n)+1:cIR(n+1)) = Gugn(:) - GuZiJuZtGugn(:);
end
%LutZtGugn = -Pr * LutZtGugn;
% LutZtGugn(Ptr,:) = -LutZtGugn; %sign and permut. have been changed in Sur
w = Su\LutZtGugn(Ptr,:);
Gw = -bsxfun(@rdivide,reshape(w(Ptr),R,R),Gamman);
% Gw = -bsxfun(@rdivide,reshape(w,R,R),Gamman);
% Gw = permute(Gw,[2 1 3]);
Gw = reshape(Gw,[],N);

for n = 1:N
    temp = iJn(:,:,n)*Gw(:,n);
    %temp = iJn(:,:,n)*(-Pr) * diag(1./reshape(Gamman(:,:,n),[],1)) * w;
    temp = U{n} * reshape(temp,R,R) * iGamman(:,:,n).';
    
    dv(cIR(n)+1:cIR(n+1)) = dv(cIR(n)+1:cIR(n+1)) - temp(:);
    U{n}(:) = U{n}(:) + dv(cIR(n)+1:cIR(n+1));
end
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Perm,P] = per_vectrans(m,n)
% vec(X_mn^T) = P vec(X)
Perm = reshape(1:m*n,[],n)'; Perm = Perm(:);
if nargout ==2
    P = speye(m*n); P = P(Perm,:);
end
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

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = khatrirao_t(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
%
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1}.';

for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i}.',R,[]),reshape(K,R,1,[]));
end
K = reshape(K,R,[]);

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