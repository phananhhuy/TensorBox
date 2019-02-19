function [U,output,A] = bcd_als(X,LR,rankRdim,opts)
% Multiplicative algorithm for Nonnegative BTD 
% X = X1 o F1 + X2 o F2 + ... + XR o FR
% where Xr are P-D CP tensor of rank-Lr, r = 1, 2,..., R
%   Xr = I x1 Ur1 x2 Ur2 ... xP UrP , 
% Urp : Ip x Lr, p = 1, 2,..., P 
% 
% and Fr are (N-P)-D rank-1 CP tensor, r = 1, 2,..., R 
%   Fr = cr1 o cr2 o ... o crK   ,  K = N-P.
%  
% INPUT:
%   X:  N-D data
%   L:  vector specifies size of CP-rank of the pattern
%   P:  dimensions of tha pattern tensors (P < N)
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'|cell array]
%     .printitn: Print fit every n iterations {1}
%     .fitmax
% Output:
%  U:  cell array of factors  (Nx 1)
%  U{1} = [U11 ... Ur1 ... UR1 c11 ... cr1 ... cR1]
%  U{n} = [U1n ... Urn ... URn c1n ... crn ... cRn]
% 
%
%% REF: (need to be corrected later)
%
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "Low Complexity Damped
% Gauss-Newton Algorithms for CANDECOMP/PARAFAC", submitted to SIMAX, 2011
% 
%
% The function is a parZt of the TENSORBOX.
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors of [2].
%
% Anh Huy Phan, 04/2012

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    U = param; return
end

N = ndims(X); normX = norm(X); normX2 = normX.^2; I = size(X);

updatemodes = param.updatemodes;
if isempty(updatemodes)
    updatemodes = 1:N;
end

nonnegativity = param.nonnegativity;
if numel(nonnegativity) ==1 
    nonnegativity = nonnegativity(ones(1,N));
end

orthogonal = param.orthogonal;
if numel(orthogonal) ==1 
    orthogonal = orthogonal(ones(1,N));
end

R = sum(prod(LR,1)); % total rank structured CPD

NoPat = size(LR,2);

p_perm = [];
if ~issorted(I)
    [I,p_perm] = sort(I);
    X = permute(X,p_perm);
    [foe,rankRdim] = ismember(rankRdim,p_perm);
    nonnegativity = nonnegativity(p_perm);
    orthogonal = orthogonal(p_perm);

    
    if (numel(param.init) == N) && (iscell(param.init) && all(cellfun(@isnumeric,param.init)))
        param.init = param.init(p_perm);
    end
end
rankLdim = setdiff(1:N,rankRdim);

Eleft = [];
Eright = [];
for p = 1:NoPat
    Eleft = blkdiag(Eleft,kron(eye(LR(1,p)),ones(1,LR(2,p))));
    Eright = blkdiag(Eright,kron(ones(1,LR(1,p)),eye(LR(2,p))));
end


Uinit = bcd_init(X,LR,rankRdim,rankLdim,Eleft,Eright,param);
U = Uinit;

%% Initializing U, damping parameters mu, nu
fprintf('\nBCD_rank(LoR)_ALS:\n');

A = cell(N,1);
for n = 1:N
    if ismember(n,rankLdim)
        A{n} = U{n} * Eleft;
    else
        A{n} = U{n} * Eright;
    end
end

AtA = zeros(R,R,N);
for n = 1: N
    AtA(:,:,n) = A{n}'*A{n};
end

warning off;
fitold = 0;fitarr = [];
%% Main Loop: Iterate until convergence
iter = 1;iter2 = 1;%boostcnt = 0;
flagtol = 0;


Jn = cumprod(I); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
ns = find(Jn<=Kn,1,'last');
updateorder = [ns:-1:1 ns+1:N];
boostcnt = 0;
while (iter <= param.maxiters) && (iter2 <= 5*param.maxiters)
    %pause(.0001)
    iter2 = iter2+1; %U1 = U;
    
    for n = updateorder
        
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
            gn = mttkrp(X,A,n); 
        elseif isa(X,'tensor') || isa(X,'sptensor')
            if (n == ns) || (n == ns+1)
                [gn,Pmat] = cp_gradient(A,n,X);
            else
                [gn,Pmat] = cp_gradient(A,n,Pmat);
            end
        end
        
        %gn = mttkrp(X,A,n);
        if ismember(n,updatemodes)
            if nonnegativity(n) == 1
                if ismember(n,rankRdim)
                    gnrnk1 = gn * Eright'; %gn = gn * Emsk';
                    U{n} = U{n}.*gnrnk1./(U{n} * Eright * prod(AtA(:,:,setdiff(1:N,[n])),3) * Eright' + 1e-10);
                    U{n} = max(1e-10,U{n});
                    A{n} = U{n} * Eright;
                    
                else % ismember(n,rankLdim)
                    gnrnk1 = gn * Eleft'; %gn = gn * Emsk';
                    U{n} = U{n}.*gnrnk1./(U{n} * Eleft * prod(AtA(:,:,setdiff(1:N,[n])),3) * Eleft' + 1e-10);
                    U{n} = max(1e-10,U{n});
                    A{n} = U{n} * Eleft;
                end
            elseif orthogonal(n) == 1
                if ismember(n,rankRdim)
                    gnrnk1 = gn * Eright'; %gn = gn * Emsk';
                    [uu,ss,vv] = svd(gnrnk1,0);
                    U{n} = uu*vv';
%                     Gamma = (Eright * prod(AtA(:,:,setdiff(1:N,[n])),3) * Eright');
%                     if rcond(Gamma) < 1e-8
%                         U{n} = gnrnk1*pinv(Gamma);
%                     else
%                         U{n} = gnrnk1/Gamma;
%                     end
                    A{n} = U{n} * Eright;
                else % ismember(n,rankRdim)
                    gnrnk1 = gn * Eleft'; %gn = gn * Emsk';
                    [uu,ss,vv] = svd(gnrnk1,0);
                    U{n} = uu*vv';
                    A{n} = U{n} * Eleft;
                end
            else
                if ismember(n,rankRdim)
                    gnrnk1 = gn * Eright'; %gn = gn * Emsk';
                    Gamma = (Eright * prod(AtA(:,:,setdiff(1:N,[n])),3) * Eright');
                    if rcond(Gamma) < 1e-8
                        U{n} = gnrnk1*pinv(Gamma);
                    else
                        U{n} = gnrnk1/Gamma;
                    end
                    A{n} = U{n} * Eright;
                else % ismember(n,rankRdim)
                    gnrnk1 = gn * Eleft'; %gn = gn * Emsk';
                    U{n} = gnrnk1/(Eleft * prod(AtA(:,:,setdiff(1:N,[n])),3) * Eleft');
                    A{n} = U{n} * Eleft;
                end
            end
            AtA(:,:,n) = A{n}'*A{n};
        end
    end
    
    %Xhat = ktensor(A);
    %err2 = normX2 + norm(Xhat)^2 - 2 * innerprod(Xhat,X);
    err = normX2 + sum(sum(prod(AtA,3))) - 2 * sum(sum(gn.*A{n}));  
    
     % Normalization factors U
    lambda = ones(1,sum(LR(1,:)));
    for n=rankLdim
        am=sqrt(sum(U{n}.^2)); lambda = lambda.* am;
        U{n}=bsxfun(@rdivide,U{n},am);
    end
    for n=rankLdim
        U{n}=bsxfun(@times,U{n},lambda.^(1/numel(rankLdim)));
    end
    
    lambda = ones(1,sum(LR(2,:)));
    for n=rankRdim
        am=sqrt(sum(U{n}.^2)); lambda = lambda.* am;
        U{n}=bsxfun(@rdivide,U{n},am);
    end
    for n=rankRdim
        U{n}=bsxfun(@times,U{n},lambda.^(1/numel(rankRdim)));
    end
    
    fit = 1-sqrt(err)/normX;       %fraction explained by model
    fitchange = abs(fitold - fit);
    if mod(iter,param.printitn)==0
        fprintf('Iter %d: fit = %d delfit = %2.2e\n',iter,fit,fitchange);
    end
    
    if (iter > 1) && (fitchange < param.tol) % Check for convergence
        flagtol = flagtol + 1;
    else
        flagtol = 0;
    end
    if flagtol >= 10, 
        if (param.recursivelevel == param.MaxRecursivelevel) || (boostcnt == param.maxboost)
            break
        else
            boostcnt = boostcnt +1;
            [U,fit,err] = recurloop(U,err);
            A = cell(N,1);
            for n = 1:N
                if ismember(n,rankLdim)
                    A{n} = U{n} * Eleft;
                else
                    A{n} = U{n} * Eright;
                end
            end
            
            AtA = zeros(R,R,N);
            for n = 1: N
                AtA(:,:,n) = A{n}'*A{n};
            end
            iter= iter+1;
        end
        mu_inc_count = 0;
    end
    fitold = fit; fitarr = [fitarr fit];
    iter = iter + 1;
    
    % Check for convergence
    if (fit>= param.fitmax),  break; end
end

% Arrange the final tensor so that the columns are normalized.
% Xhat = ktensor(A); Xhat = arrange(P);

if param.printitn>0
    fprintf(' Final fit = %e \n', fit);
end

if nargout >=2
    output = struct('Uinit',{Uinit},'Fit',fitarr);
end

A = cell(N,1);
for n = 1:N
    if ismember(n,rankLdim)
        A{n} = U{n} * Eleft;
    else
        A{n} = U{n} * Eright;
    end
end

% Rearrange dimension of the estimation tensor 
if ~isempty(p_perm)
    %P = ipermute(P,p_perm);
    [foe,ip_perm] = sort(p_perm);
    U = U(ip_perm); A = A(ip_perm);
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    function [U,fit,err] = recurloop(U,err)
        param.recursivelevel = param.recursivelevel+1;
        %fprintf('Recursive loop %d ... \n',param.recursivelevel)
        Ui = U; 
        for n = 1:N, 
            [Ui{n},foe] = qr(Ui{n},0); 
            Ui{n} = Ui{n} * randn(size(U{n},2));
            %Ui{n} = randn(size(U{n})) * foe;
        end
        
        % LM through recursive loop
        fit = 1-real(sqrt(err))/normX;
        opts.init = Ui;opts.recursivelevel = param.recursivelevel;
        opts.fitmax = fit;opts.alsinit = 0;
        opts.printitn = 0;opts.tol = 1e-8;opts.maxiters = 1000;
        [Unew,output] = feval(mfilename,X,LR,rankRdim,opts);
        Anew = cell(N,1);
        for n = 1:N
            if ismember(n,rankLdim)
                Anew{n} = Unew{n} * Eleft;
            else
                Anew{n} = Unew{n} * Eright;
            end
        end
        P = ktensor(Anew);
        
        err3=normX2 + norm(P).^2 - 2 * innerprod(X,P);
        if err3 < err
            U = Unew;
            err = err3; fit = 1-real(sqrt(err))/normX;
        end
        %fprintf('Exit recursive loop %d ... \n',param.recursivelevel)
        param.recursivelevel = param.recursivelevel-1;
    end



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
                    G = (KRP_left'*T)';
                end
            else
                %G = squeeze(Pmat);
                G = reshape(Pmat,[],R);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left}';
                elseif numel(left) > 1
                    KRP_left = khatrirao_t(A(left));
                    %KRP_left = khatrirao(A(left));KRP_left = KRP_left';
                else
                    KRP_left = 1;
                end
                if isa(Pmat,'tensor')
                    T = reshape(Pmat.data,prod(I(left)),[]);
                else
                    T = reshape(Pmat,prod(I(left)),[]);
                end
                %
                Pmat = KRP_left * T;   % Left-side projection
            else
                if R>1
                    Pmat = reshape(Pmat,R,I(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,I(left(1)),[]);
                    Pmat = A{left(1)}'* Pmat;
                end
            end
            
            if ~isempty(right)
                T = reshape(Pmat,[],I(n),prod(I(right)));
                
                if (n == (ns+1)) && (numel(right)>=2)
                    %KRP_right = KRP_right0;
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right0',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T, R,[])';
                    else
                        %G = squeeze(T) * KRP_right0;
                        G = reshape(T,[],prod(I(right))) * KRP_right0;
                    end
                else
                    KRP_right = khatrirao(A(right));
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T,R,[])';        % Right-side projection
                    else
                        %G = squeeze(T) * KRP_right;
                        G = reshape(T,I(n),[]) * KRP_right;
                    end
                end
            else
                %G = squeeze(Pmat)';
                G = reshape(Pmat,R,[])';
            end
            
        end
        
        %         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
        %             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
        %             size(KRP_left,1),size(KRP_left,2),...
        %             size(KRP_right,1),size(KRP_right,2))
    end
end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','random',@(x) (iscell(x)||ismember(x(1:4),{'rand' 'nvec' 'tsvd'})));
param.addOptional('alsinit',1);
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('tau',1e-3);
param.addOptional('maxboost',5);
param.addOptional('nonnegativity',0);
param.addOptional('orthogonal',0);
param.addOptional('updatemodes',[]);

param.addParamValue('MaxRecursivelevel',2);
param.addParamValue('recursivelevel',1);
param.addParamValue('TraceFit',false,@islogical);
param.addParamValue('TraceMSAE',true,@islogical);
param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end

%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function U = bcd_init(X,LR,rankRdim,rankLdim,Eleft,Eright,param)
% Set up and error checking on initial guess for U.
N = ndims(X);
nonnegativity = param.nonnegativity;
if numel(nonnegativity) ==1 
    nonnegativity = nonnegativity(ones(1,N));
end
if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init;
        Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || ...
                (~all(Sz(rankLdim,2)==sum(LR(1,:))))  || ...
                (~all(Sz(rankRdim,2)==sum(LR(2,:))))
            error('Wrong Initialization');
        end
    else
%         normX = norm(X);
%         bestfit = 0;Pbest = [];
%         for ki = 1:numel(param.init)
%             initk = param.init{ki};
%             if iscell(initk) || ...
%                 (ischar(initk)  && ismember(initk(1:4),{'rand' 'nvec'}))  % multi-initialization
%                 cp_fun = str2func(mfilename);
%                 initparam = param;initparam.maxiters = 10;
%                 initparam.init = initk;
%                 P = cp_fun(X,R,initparam);
%                 fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
%                 if fitinit > bestfit
%                     Pbest = P;
%                     bestfit = fitinit;
%                 end
%             end
%         end
%         Uinit = Pbest.U;
%         Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
         Uinit = param.init{1};
    end
elseif strcmp(param.init(1:4),'rand')
    Uinit = cell(N,1);
    for n = 1:N
        if ismember(n,rankLdim)
            Uinit{n} = randn(size(X,n),sum(LR(1,:)));
        else
            Uinit{n} = randn(size(X,n),sum(LR(2,:)));
        end
    end
elseif strcmp(param.init(1:4),'nvec')
    Uinit = cell(N,1);
    for n = 1:N
        if nonnegativity(n)
            if ismember(n,rankLdim)
                if sum(LR(1,:)) <= size(X,n)
                    Uinit{n} = real(nvecs(X,n,sum(LR(1,:))));
                else
                    Uinit{n} = randn(size(X,n),sum(LR(1,:)));
                end
            else
                if sum(LR(2,:)) <= size(X,n)
                    Uinit{n} = real(nvecs(X,n,sum(LR(2,:))));
                else
                    Uinit{n} = randn(size(X,n),sum(LR(2,:)));
                end
            end
            Uinit{n} = abs(Uinit{n});
        else
            
            if ismember(n,rankLdim)
                if sum(LR(1,:)) <= size(X,n)
                    if nonnegativity(n) == 1
                        Uinit{n} = rand(size(X,n),sum(LR(1,:)));
                    else
                        Uinit{n} = real(nvecs(X,n,sum(LR(1,:))));
                    end
                else
                    if nonnegativity(n) == 1
                        Uinit{n} = rand(size(X,n),sum(LR(1,:)));
                    else
                        Uinit{n} = randn(size(X,n),sum(LR(1,:)));
                    end
                end
            else
                if sum(LR(2,:)) <= size(X,n)
                    Uinit{n} = real(nvecs(X,n,sum(LR(2,:))));
                else
                    Uinit{n} = randn(size(X,n),sum(LR(2,:)));
                end
            end
        end
    end
elseif strcmp(param.init(1:4),'tsvd')
    
    SzX = size(X);N = ndims(X);
    
    % Transform tensor X to  matrix of (Left-dim x Righgt-dim) 
    Xm = tenmat(X,rankLdim); % tensor unfolding 
    Xm = double(Xm);
    
    P = size(LR,2); % no of components 
    
    % SVD of the unfolded matrix 
    [FacL,s,FacR] = svd(Xm,0);
    FacR = FacR * s; % correct the scaling
    
    % initialize factor matrices 
    %opts = cp_fastals;
    opts = cp_fLMa;
    opts.init = {'dtld' 'nvec' 'rand'};
    opts.maxiters = 1000;
    opts.tol = 1e-7;
    V = cell(N,P);
    for k = 1:P
        
        % Construct factor matrices for the left tensor A_r
        % for Left-factor tensors
        if numel(rankLdim) > 1
            % Reshape the p-th leading singular vector 
            % to tensor of sizes defined by rankLeft_dim
            Tleft = reshape(FacL(:,k),SzX(rankLdim));
            
            % Approximate T_left by low-rank tensor of rank L(k)
            if all(nonnegativity(rankLdim) == 0)
                Pleft = cp_fLMa(tensor(Tleft),LR(1,k),opts);
            else
                Pleft = ncp_hals(tensor(abs(Tleft)),LR(1,k),opts);
            end
            U = Pleft.U;
            U{1} = U{1}*diag(Pleft.lambda);
            V(rankLdim,k) = U;
        else
            if all(nonnegativity(rankLdim) == 0)
                V{rankLdim,k} = FacL(:,k);
            else
                V{rankLdim,k} = abs(FacL(:,k));
            end
        end
        
        % Construct factor matrices for the left tensor X_r
        % for right-factor tensors
        if numel(rankRdim) > 1
             % Reshape the p-th leading singular vector 
            % to tensor of sizes defined by rankRight_dim
            Tright = reshape(FacR(:,k),SzX(rankRdim));
            
             % Approximate T_left by low-rank tensor of rank L(k)
            if all(nonnegativity(rankRdim) == 0)
                Pright = cp_fLMa(tensor(Tright),LR(2,k),opts);
            else
                Pright = ncp_hals(tensor(abs(Tright)),LR(2,k),opts);
            end
            U = Pright.U;
            U{1} = U{1}*diag(Pright.lambda);
            V(rankRdim,k) = U;
        else
            if all(nonnegativity(rankRdim) == 0)
                V{rankRdim,k} = FacR(:,k);
            else
                V{rankRdim,k} = abs(FacR(:,k));
            end
        end
        
    end
    
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = cell2mat(V(n,:));
        if nonnegativity(n) == 1
            Uinit{n} = abs(Uinit{n});
        end
    end
    
        
else
    error('Invalid initialization');
end

if param.alsinit
    alsopts = struct('maxiters',2,'printitn',0,'init',{Uinit},'alsinit',0);
    try
        U = bcd_als_ls(X,LR,rankRdim,alsopts);
    catch me
        U = bcd_als(X,LR,rankRdim,alsopts);
    end
    %U = P.U;%U{end} = bsxfun(@times, U{end},P.lambda.');
    
    for n = 1:N
        if nonnegativity(n)
            U{n} = abs(U{n});
        end
    end
else
    U = Uinit;
end
end
  
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = khatrirao(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1};
if nargout == 1
    for i = 2:numel(A)
        K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
    end
elseif numel(A) > 2
    for i = 2:numel(A)-1
        K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
    end
    K2 = reshape(K,[],R);
    K = bsxfun(@times,reshape(A{end},[],1,R),reshape(K,1,[],R));
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
K = A{1}';

for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i}',R,[]),reshape(K,R,1,[]));
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