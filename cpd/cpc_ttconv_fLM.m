function [P,output] = cp_fLMa(X,R,opts)
% fLMa - Fast Damped Gauss-Newton (Levenberg-Marquard) algorithm factorizes
%       an N-way tensor X into factors of R components.
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

%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

maxiters = param.maxiters;
N = ndims(X); SzX = size(X);

%%
if param.printitn ~=0
    fprintf('\nTT-Conversion for CPD:\n');
end

if nargout >=2
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%% U{n}: core tensors of the TT-tensor X
% B{n}   factor matrices of the estimated K-tensor 
%% Stage 1: compression of the data tensor by a TT-tensor of rank-(R,...,R)
% If X is not a TT-tensor, first fit it by a TT-tensor of rank <= R.
tt_opts = struct('tol',1e-9,'maxiters',100);
if ~isa(X,'tt_tensor')
    tinit = tic;
    Xtt1 = tt_tensor(double(X),tt_opts.tol,SzX,[1 R*ones(1,N-1) 1]');
    Xtt = round(Xtt1,tt_opts.tol,[1 R*ones(1,N-1) 1]');
    Xtt = tt_als(Xtt1,Xtt,tt_opts.maxiters);
    tinit = toc(tinit);
else
    % if input is a TT-tensor, round its rank to R, or not exceed R
    % round the input TT-tensor to rank R
    tinit = tic;
    Xtt = round(X,tt_opts.tol,[1 R*ones(1,N-1) 1]');
    % Further improvement by ALS with 10 sweeps
    Xtt = tt_als(X,Xtt,tt_opts.maxiters);
    tinit = toc(tinit);
end

%% Stage 2: Initialize factor matrices and conversion from a TT to K-tensor
if (ischar(param.init) && strcmp(param.init,'exact_tt_to_cp')) || isa(X,'tt_tensor')
    Binit = exact_tt_to_cp(Xtt,R);
else
    param.cp_func = @cp_fastals;
    Binit = cp_init(X,R,param);
end
B = Binit;

%% Stage 3: % Fit a TT tensor of B to TTtensor of G using the LM algorithm
C = cellfun(@(x) x'*x,B,'uni',0);
C = reshape(cell2mat(C(:)'),[R,R,N]);

normX = norm(Xtt);normX2 = normX^2;
% Get core tensors of Xtt
Ux = core2cell(Xtt);
Ux{1} = reshape(Ux{1},size(Ux{1},2),[]);
rankX = Xtt.r;

%% Damping parameters mu, nu
ell2 = zeros(N,R); for n = 1:N,  ell2(n,:) = sum(abs(B{n}).^2); end
mm = zeros(N,R); for n = 1:N, mm(n,:) = prod(ell2([1:n-1 n+1:N],:),1); end
nu=2; mu=param.tau*max(mm(:));    % initialize damping mu, nu

warning off;


%% Main Loop: Iterate until convergence
iter = 1;iter2 = 1;boostcnt = 0;flagtol = 0;
mu_inc_count = 0;


%% Correlation matrices Cn
Gamman = zeros(R,R,N);
for n = 1:N, Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3);end

% Precompute the products X_(n) * KhatriRao(B,-n)
[Gxn,Phi_left,Phi_right] = cp_gradients(Ux,B); % Gxn{n} = mttkrp(Xd,B,n);

% Approximation error 
err=normX2 + sum(sum(Gamman(:,:,1).*C(:,:,1))) - 2 * real(sum(conj(Gxn{1}(:)).*B{1}(:)));
fit = 1-sqrt(err)/normX; fitold = fit; fitarr = [1 fit]; % iteration-fit

flag = false;
while (iter <= param.maxiters) && (iter2 <= 5*param.maxiters)
    
    %pause(.0001)
    iter2 = iter2+1; B1 = B;
    
    % Check condition of the update rule NR6
    if param.updaterule~= 1
        flag = cellfun(@(x) any(abs(reshape(x'*x,[],1))<= 1e-8), B,'uni',1);
        flag = any(flag==1);%flag = 1;
    end
    
    if param.updaterule==1 || flag
        currupdaterule = 1;
        [B, d, g] = updaten3r6(Xtt,B,mu,Gxn,C,Gamman);       % update U n3r6
    else
        currupdaterule = 2;
        [B, d, g] = updatenr6(Xtt,B,mu,Gxn,C,Gamman); % update U nr6        
    end
    
    %% Compute the new estimation error             
    P = ktensor(B);
    Phi_right = update_Phiright(Ux,B,Phi_right);
    Gxnnew{1} = Ux{1} * Phi_right{1};
    err2=normX2 + norm(P).^2 - 2 * real(sum(conj(Gxnnew{1}(:)).*B{1}(:)));
    
    %% Check if the cost function descreases
    if (err2>err)|| isnan(err2)                           % The step is not accepted
        B = B1;
        if (norm(d)> param.tol) ... %&& (sqrt(err2-err)>(1e-5*normX)) ...
                && (mu < 1e10) && (mu_inc_count < 20)
            mu=mu*nu; nu=2*nu; mu_inc_count = mu_inc_count+1;
        else                              % recursive loop
            if (param.recursivelevel == param.MaxRecursivelevel) || (boostcnt == param.maxboost)
                break
            else
                boostcnt = boostcnt +1;
                [B,mu,nu,fit,err] = recurloop(B,mu,nu,err);
                % Update C and Gamma
                for n = 1:N, C(:,:,n) = B{n}'*(B{n});end
                for n = 1:N, Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3);end
                [Gxn,Phi_left,Phi_right] = cp_gradients(Ux,B,Phi_left,Phi_right);

                iter= iter+1;
            end
            mu_inc_count = 0;
        end
    else
        mu_inc_count = 0;
        % update damping parameter mu, nu
        rho=real((err-err2)/(d'*(g+mu*d)));
        nu=2; mu=mu*max([1/3 1-(nu-1)*(2*rho-1)^3]);
        
        % Normalize factor matrices B
        am = zeros(N,R); %lambda = ones(1,R);
        for n=1:N
            am(n,:) =sqrt(sum(abs(B{n}).^2));
            B{n}=bsxfun(@rdivide,B{n},am(n,:));
        end
        lambda = prod(am,1);
        for n=1:N
            B{n}=bsxfun(@times,B{n},lambda.^(1/N));
        end
        
        % Precompute Cn and Gamma_n
        for n = 1:N, C(:,:,n) = B{n}'*(B{n});end
        for n = 1:N, Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3); end
         
        % Pre-computing CP gradients
        [Gxn,Phi_left,Phi_right] = cp_gradients(Ux,B,Phi_left,Phi_right);

        
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
P = ktensor(B); P = arrange(P);
if param.printitn>0
    fprintf(' Final fit = %e \n', fit);
end

if nargout >=2
    output = struct('Uinit',{Binit},'Fit',fitarr,'mu',mu,'nu',nu);
end

%%
% Pre-computing CP gradients


    function Phi_right = update_Phiright(Ux,B,Phi_right)
        Phi_right{N-1} = Ux{N}*(B{N}); %fixed for complex data
        for n = N-2:-1:1% RIght-to-Left
            %dir = 'RL';upto = n+1; % upto = 2, 3, 4
            %Phi_right{n} = innerprod(Xtte,Btte, dir, upto ); % contraction between cores k of X and B where k = n+1:N
            Phi_right{n} = fmttkrp(Ux{n+1}, (B{n+1}),Phi_right{n+1},1); %
        end
    end

    function Phi_left = update_Phileft(Ux,B,Phi_left)
        for n = 2:N% RIght-to-Left
             if n == 2
                Phi_left{n} = Ux{n-1}'*B{n-1};
            elseif n>2
                Phi_left{n} = fmttkrp(Ux{n-1},Phi_left{n-1},B{n-1},3);
            end
        end
    end

    function [Grad_n,Phi_left,Phi_right] = cp_gradients(Ux,B,Phi_left,Phi_right)
        %% Compute the product  mttkrp(X_cp,B,n);
        % % Pre-Compute Phi_left % matrices of size R x R, products of all cores form
        % % 1:n-1 between two tensor Xtt and Btt
        if nargin <3
            Phi_left = cell(N,1); Phi_right = cell(N,1);
            Phi_left{1} = 1;Phi_right{N} = 1;
        end
        
        % Pre-Compute Phi_right % matrices of size R x R, products of all cores form
        Phi_right = update_Phiright(Ux,B,Phi_right);
        Phi_left = update_Phileft(Ux,B,Phi_left);
        
        for n = 1:N
            %%
            %if n == 2
            %    Phi_left{n} = Ux{n-1}'*B{n-1};
            %elseif n>2
            %    Phi_left{n} = fmttkrp(Ux{n-1},Phi_left{n-1},B{n-1},3);
            %end
            
            XQleft = Phi_left{n};
            XQright = Phi_right{n};
            
            %% Update B{n} and assess Cost value
            if n == 1 % right
                % The term "squeeze(Xtte.U{n}) * XQright" is mttkrp(X_cp,B,n);
                Grad_n{n} = Ux{n} * (XQright);
            elseif n == N
                % THe term squeeze(Xtte.U{n})'*XQleft is mttkrp(X_cp,B,n);
                Grad_n{n} = Ux{N}' * (XQleft);
            else
                % This term is  mttkrp(X_cp,B,n)
                %Grad_n = mttkrp(tensor(Ux{n}),{XQleft,B{n},XQright},2);
                Grad_n{n} = fmttkrp(Ux{n},XQleft,XQright,2);
            end
        end
    end

         
  
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    function [B,mu,nu,fit,err] = recurloop(B,mu,nu,err)
        param.recursivelevel = param.recursivelevel+1;
        %fprintf('Recursive loop %d ... \n',param.recursivelevel)
        Bi = B;
        for n = 1:N,
            if SzX(n)>=R
                [Bi{n},foe] = qr(Bi{n},0);
            else
                Bi{n} = randn(SzX(n),R);
            end
        end
        
        % LM through recursive loop
        fit = 1-real(sqrt(err))/normX;
        opts.init = Bi;opts.recursivelevel = param.recursivelevel;
        opts.fitmax = fit;opts.alsinit = 0;
        opts.printitn = 0;opts.tol = 1e-8; %opts.maxiters = 1000;
        [P,output] = feval(mfilename,Xtt,R,opts);
        Bnew = P.u;Bnew{N} = bsxfun(@mtimes,Bnew{N},P.lambda');
        
        Phi_right = update_Phiright(Ux,Bnew,Phi_right);
        Gxnnew{1} = Ux{1} * Phi_right{1};
        err3=normX2 + norm(P).^2 - 2 * sum(Gxnnew{1}(:).*Bnew{1}(:));
        
        if abs(err3 - err) < param.tol
            B = Bnew;
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
param.addOptional('init','exact_tt_to_cp',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'exac'})));
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
    [Ptr,Pr] = per_vectrans(R,R); % permutation matrix Appendix A
    Pr_idx = Pr==1;
    
    iJn = zeros(R^2,R^2,N);    
    iGamman = zeros(R,R,N);
end
g = zeros(sum(In)*R,1);     % gradient

%% Method 3
Su = 0;
for n = 1:N
    iGamman(:,:,n) = inv(Gamman(:,:,n) + mu * eye(R)); % N R xR (9) (10)
    Jn = kron(iGamman(:,:,n),C(:,:,n));
    
    dvecC_Gamma = C(:,:,n)./Gamman(:,:,n);dvecC_Gamma = dvecC_Gamma.';
    Jn(Pr_idx) = Jn(Pr_idx) - dvecC_Gamma(:);
    
    % invert iKKk
    iJn(:,:,n) = inv(Jn);   % R^2 x R^2  inv(dvec(Cn) - dvec(Gam_n)* Prr * (inv(Gam_n) o C_n))
    
    vGn = reshape(Gamman(:,:,n).',[],1);
    temp = bsxfun(@rdivide,iJn(:,:,n),vGn);
    vGn = reshape(Gamman(:,:,n),1,[]);
    temp = bsxfun(@rdivide,temp,vGn);
    Su = Su + temp;
end
Gamma = Gamman(:,:,1).*C(:,:,1);

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


function G = fmttkrp(X,A,B,mode_n)
%  X_(n)^T * khatrirao(B,A)
%
szX = size(X);
if numel(szX)<3, szX(3) = 1;end
R = size(A,2);
switch mode_n
    case 1
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[1 szX(2) R]));
        G = squeeze(sum(G,2));
    case 2
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[szX(1) 1 R]));
        G = reshape(sum(G,1),size(G,2),[]);
    case 3
        tmp = A.'*reshape(X,[szX(1)  szX(2)*szX(3)]);
        G = bsxfun(@times, reshape(tmp,[R szX(2) szX(3)]), B.');
        G = squeeze(sum(G,2)).';
end
    
end
