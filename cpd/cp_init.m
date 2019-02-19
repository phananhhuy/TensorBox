%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function Uinit = cp_init(X,R,opts)
% The following guidelines are provided for CPD with/without constraints
%  
% TENSORBOX implements several simple and efficient methods to initialize
% factor matrices.
%   - (random) Random initialization which is always the simplest and fastest, but
%    not the most efficient method.
%
%   - (nvec) SVD-based initialization which initializes factor matrices by
%    leading singular vectors of mode-n unfoldings of the tensor. The
%    method is similar to higher order SVD or multilinear SVD.
%    This method is often better than using random values in term of
%    convergence speed.
%    However, this method is less efficient when factor matrices comprise
%    highly collinear components. 
%   
%   - (dtld) Direct trilinear decomposition (DTLD) and its extended version for
%   higher CPD using the FCP algorithm are recommended for most CPD. This
%   initialization may consume time for compression, but it initial values
%   are always much better than those using other methods.
%     
%   - (fiber) Using fibers for initialization is also suggested.
%   
%   - Multi-initialization with some small number of iterations are often
%   performed. The component matrices with the lowest approximation
%   error are selected.
%   
%   - (orth) Orthogonal factor matrices
% 
% Factor matrices can be initialized outside algorithms using the routine
% "cp_init" 
%   
%   opts = cp_init;  
%   opts.init = 'dtld';
%   Uinit = cp_init(X,R,opts);
%
% then passed into the algorithm through the optional parameter "init" 
%   opts = cp_fastals;
%   opts.init = {Uinit}; % or can directly set  opts.init = 'dtld';
%   P = cp_fastals(X,R,opts);
%
% For the fLM algorithm, ALS with small runs can be employed before the main
% algorithm by setting
%   opts.alsinit = 1;
%
% For nonnegative CPD, set
%    opts.nonnegative = true
%
% See comparison of initialization methods in "demo_CPD_3.m".
%
% See also:  cp_als, nvecs, Nway and PLS toolbox
%
% TENSOR BOX, v1. 2012
% Copyright 2008, 2011, 2012, 2013 Phan Anh Huy.

%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));   
param.addOptional('alsinit',0);
param.addOptional('complex',0);
param.addOptional('cp_func',@cp_fastals,@(x) isa(x,'function_handle'));
param.addOptional('nonnegative',false);

if ~exist('opts','var'), opts = struct; end
param.parse(opts);param = param.Results;
if nargin == 0
    Uinit = param; return
end

if isa(X,'tensor')
    IsReal = isreal(X.data);
elseif isa(X,'ktensor') || isa(X,'ttensor')
    IsReal = all(cellfun(@isreal,X.u));
end

if param.complex 
    IsReal = false;
end

% Set up and error checking on initial guess for U.
N = ndims(X); In = size(X);
if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init(:);
        Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),In')) || (~all(Sz(:,2)==R))
            error('Wrong Initialization');
        end
    else % small iteratons to find the best initialization
        normX = norm(X);
        bestfit = -inf;Pbest = [];
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || isa(initk,'ktensor') || ...
                    (ischar(initk)  && ismember(initk(1:4), ...
                    {'rand' 'nvec' 'fibe' 'orth' 'dtld'}))  % multi-initialization
                if ischar(initk)
                    cprintf('blue','Init. %d - %s\n',ki,initk)
                else
                    cprintf('blue','Init. %d - %s\n',ki,class(initk))
                end
                
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                try 
                    P = param.cp_func(X,R,initparam);
                    fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
                catch 
                    continue;
                end
                if real(fitinit) > bestfit
                    Pbest = P;
                    bestfit = fitinit;kibest = ki;
                end
            end
        end
        cprintf('blue','Choose the best initial value: %d.\n',kibest);
        Uinit = Pbest.U;
        Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
    end
elseif isa(param.init,'ktensor')
    Uinit = param.init.U;
    Uinit{end} = Uinit{end} * diag(param.init.lambda);
    Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
    if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),In')) || (~all(Sz(:,2)==R))
        error('Wrong Initialization');
    end
elseif strcmp(param.init(1:4),'rand')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = randn(In(n),R);
    end
elseif strcmp(param.init(1:4),'orth')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = orth(randn(In(n),R));
        if R > In(n)
            Uinit{n}(:,end+1:R) = randn(In(n),R-In(n));
        end
    end
    
    UtU = zeros(R,R,N);
    for n = 1:N
        UtU(:,:,n) = Uinit{n}'*Uinit{n};
    end
    
    % Loop over small iterations
    p_perm = [];
    if ~issorted(In)
        [In,p_perm] = sort(In);
        X = permute(X,p_perm);
        Uinit = Uinit(p_perm);
    end

    Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
    ns = find(Jn<=Kn,1,'last');
    updateorder = [ns:-1:1 ns+1:N];
    
    for n = updateorder %[1 3 2] %
        
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
            G = mttkrp(X,Uinit,n); 
        elseif isa(X,'tensor') || isa(X,'sptensor')
            if (n == ns) || (n == ns+1)
                [G,Pmat] = cp_gradient2(Uinit,n,X);
            else
                [G,Pmat] = cp_gradient2(Uinit,n,Pmat);
            end
        end
        if R <= size(Uinit{n},1)
            sigma = sum(Uinit{n} .* G);
            [uu, foe, vv] = svd(G*diag(sigma), 0);
            Uinit{n} = uu*vv';
        else
            Uinit{n} = double(G)/prod(UtU(:,:,[1:n-1 n+1:N]),3);
        end
    end
    % Rearrange dimension of the estimation tensor
    if ~isempty(p_perm)
        [foe,ip_perm] = sort(p_perm);
        Uinit = Uinit(ip_perm);
    end

        
elseif strcmp(param.init(1:4),'nvec') % changed for missing values
    Uinit = cell(N,1);
    for n = 1:N
        if R<=In(n)
            Uinit{n} = (nvecs(X,n,R));
            if IsReal
                Uinit{n} = real(Uinit{n});
            end
        else
            Uinit{n} = randn(In(n),R);
        end
    end
elseif strcmp(param.init(1:4),'fibe') % select fibers from the data tensor
    Uinit = cell(N,1); 
    %Xsquare = X.data.^2;
    for n = 1:N
        Xn = double(tenmat(X,n));
        %proportional to row/column length
        part1 = sum(Xn.^2,1);
        probs = part1./sum(part1);
        probs = cumsum(probs);
        % pick random numbers between 0 and 1
        rand_rows = rand(R,1);
        ind = [];
        for i=1:R,
            msk = probs > rand_rows(i);
            msk(ind) = false;
            ind(i) = find(msk,1);
        end
        Uinit{n} = Xn(:,ind);
        Uinit{n} = bsxfun(@rdivide,Uinit{n},sqrt(sum(Uinit{n}.^2)));
    end
    
elseif strcmp(param.init(1:4),'dtld') % direct trilinear decomposition
        
    if isa(X,'tensor') && (N>=3)
        if (N==3) && (numel(find(R <= In))<2)
            warning('DTLD does not support this decomposition. Try another initialization.')
            fixparam = param;fixparam.init = 'nvec';
            Uinit = cp_init(X,R,fixparam);
        else
            Pi = cp_gram(X,R,param.complex);Pi = arrange(Pi);
            Uinit = Pi.U;Uinit{1} = bsxfun(@times,Uinit{1},Pi.lambda(:).');
        end
    else
        param2 = param; param2.init = 'nvecs';
        Uinit = cp_init(X,R,param2);
    end 
else
    error('Invalid initialization');
end


if param.nonnegative
    Uinit = cellfun(@abs,Uinit,'uni',0); 
    % or re-estimate K-tensor using an algorithm for nCPD 
    % P = ncp_hals(ktensor(Unit),R);
    % Unit = P.U;Unit{1} = bsxfun(@times,Unit{1},P.lambda,');
end

if param.alsinit
    cprintf('blue','Initialize by ALS.\n')
    %     if isa(X,'tensor') && any(isnan(X.data(:)))
    %         Weights = ~isnan(X.data(:));
    %         Weights = reshape(Weights,size(X));
    %         Options = [param.tol 10 0 0 0 10];
    %         U = parafac(double(X),R,Options,0*ones(1,N),[],[],Weights);
    %     else
    alsopts = struct('maxiters',5,'printitn',0,'init',{Uinit},'alsinit',0);
    try
        %             P = cp_fastals_missingdata(X,R,alsopts);
        P = cp_fastals(X,R,alsopts);
    catch me
        Weights = isnan(X.data(:));
        X(Weights) = 0;
        P = cp_als_ls(X,R,alsopts);
    end
    Uinit = P.U;Uinit{end} = bsxfun(@times, Uinit{end},P.lambda.');
    %     end
end

end


% %% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function U = cp_init(X,R,opts)
% param = inputParser;
% param.KeepUnmatched = true;
% param.addOptional('init','random',@(x) (iscell(x)||ismember(x(1:4),{'rand' 'nvec'})));
% param.addOptional('alsinit',1);
% param.addOptional('cp_func',@cp_fastals,@(x) isa(x,'function_handle'));
% param.parse(opts);
% 
% 
% param = param.Results;
% % Set up and error checking on initial guess for U.
% N = ndims(X);
% if iscell(param.init)
%     if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
%         Uinit = param.init;
%         Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
%         if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
%             error('Wrong Initialization');
%         end
%     else
%         normX = norm(X);
%         bestfit = 0;Pbest = [];
%         for ki = 1:numel(param.init)
%             initk = param.init{ki};
%             if iscell(initk) || ...
%                 (ischar(initk)  && ismember(initk(1:4),{'rand' 'nvec'}))  % multi-initialization
%                 initparam = param;initparam.maxiters = 10;
%                 initparam.init = initk;
%                 P = param.cp_func(X,R,initparam);
%                 fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
%                 if fitinit > bestfit
%                     Pbest = P;
%                     bestfit = fitinit;
%                 end
%             end
%         end
%         Uinit = Pbest.U;
%         Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
%     end
% elseif isa(param.init,'ktensor')
%     Uinit = param.init.U;
%     Uinit{end} = Uinit{end} * diag(param.init.lambda);
%     Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
%     if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
%         error('Wrong Initialization');
%     end    
% elseif strcmp(param.init(1:4),'rand')
%     Uinit = cell(N,1);
%     for n = 1:N
%         Uinit{n} = randn(size(X,n),R);
%     end
% elseif strcmp(param.init(1:4),'nvec')
%     Uinit = cell(N,1);
%     for n = 1:N
%         if R<=size(X,n)
%             Uinit{n} = real(nvecs(X,n,R));
%         else
%             Uinit{n} = randn(size(X,n),R);
%         end
%     end
% else
%     error('Invalid initialization');
% end
% 
% if param.alsinit
%     alsopts = struct('maxiters',2,'printitn',0,'init',{Uinit});
%     try
%         P = cp_als_ls(X,R,alsopts);
%     catch me
%         P = cp_als(X,R,alsopts);
%     end
%     U = P.U;U{end} = bsxfun(@times, U{end},P.lambda.');
% else
%     U = Uinit;
% end
% end