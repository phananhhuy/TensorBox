function [P,output] = cpo_als(X,R,opts)
% ALS for orthgonally constrained CPD
%
% INPUT:
%   X:  N-way data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .orthomodes : modes corresponding to orthogonal factor matrices.
%     .init: Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'|'dtld'| ktensor| cell array]
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
%   opts = cpo_als1;
%   opts.init = {'nvec' 'random' 'random'};
%   opts.orthomodes = 1;
%   [P,output] = cpo_als1(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
% [1] M. Sorensen, L. Lathauwer, P. Comon, S. Icart, and L. Deneire,
% "Canonical polyadic decomposition with a columnwise orthonormal
% factor matrix,"SIAM Journal on Matrix Analysis and Applications,
% vol. 33, no. 4, pp. 1190-1213, 2012.
%
% See also: cp_fastals, cp_als, cpo_als1, cp_fcp
%
% TENSOR BOX, v1. 2013
% Phan Anh Huy.



%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

if param.linesearch
    param.TraceFit = true;
end

N = ndims(X); I = size(X);
if isempty(param.normX)
    normX = norm(X);
else
    normX = param.normX;
end
%% Initialize factors U
Uinit = cp_init(X,R,param); U = Uinit;

%% Output
fprintf('\n CPO_ALS1:\n');

if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%% Permute tensor dimension (tranpose) so that I1<=I2<= ... <= IN
% p_perm = [];
% if ~issorted(I)
%     [I,p_perm] = sort(I);
%     X = permute(X,p_perm);
%     U = U(p_perm);
% end

% Find the first n* such that I1...In* > I(n*+1) ... IN
% Jn = cumprod(I); Kn = Jn(end)./Jn;
% ns = find(Jn>=Kn,1);
% if ((ns >= (N-1)) && (ns > 2))
%     ns = ns-1;
% end

% Jn = cumprod(I); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
% ns = find(Jn<=Kn,1,'last');
% updateorder = [ns:-1:1 ns+1:N];

normX2 = norm(X)^2;
%%
if param.verify_convergence == 1
    %lambda = ones(R,1);
    %P = ktensor(U);
    %err=normX.^2 + norm(P).^2 - 2 * innerprod(X,P);
    %fit = 1-sqrt(err)/normX;
    fit = 0;
    %if param.TraceFit
    %    output.Fit = fit;
    %end
    if param.TraceMSAE
        msae = (pi/2)^2;
    end
end

%% Main Loop: Iterate until convergence
% Pmat = [];Pls = false;
nonorthomodes = setdiff(1:N,param.orthomodes);
% Xm = tenmat(X,nonorthomodes);Xm = Xm.data;
if isempty(param.orthomodes)
    UtU = zeros(R,R,N);
    for n = 1:N
        UtU(:,:,n) = U{n}'*U{n};
    end
end
Uc = cellfun(@conj,U,'uni',0);
for iter = 1:param.maxiters

    if param.verify_convergence==1
        if param.TraceFit, fitold = fit;end
        if param.TraceMSAE, msaeold = msae;end
    end
    %
    if (param.verify_convergence==1) %|| (param.linesearch == true)
        Uold = U;
    end
    
    % Iterate over all N modes of the tensor
    for n = param.orthomodes %[1 3 2] %
        G = mttkrp(X,Uc,n);
        [un,sn,vn] = svd(G,0);
        U{n} = un*vn';
        Uc{n} = conj(U{n});
    end
     
    %CPOALS1
    for n = nonorthomodes
        G = mttkrp(X,Uc,n);
        if ~isempty(param.orthomodes)
            U{n} = G;
        else
            U{n} = G/prod(UtU(:,:,[1:n-1 n+1:end]),3).';
        end
        if n~=nonorthomodes(end)
            U{n} = bsxfun(@rdivide,U{n},sqrt(sum(abs(U{n}).^2)));
        end
        if isempty(param.orthomodes)
            UtU(:,:,n) = U{n}'*U{n};
        end
        Uc{n} = conj(U{n});
    end
    
    if param.verify_convergence==1
        if param.TraceFit
            P = ktensor(U);
            normresidual = sqrt( normX^2 + norm(P)^2 - 2 * real(innerprod(X,P))); %
            %normresidual = sqrt( normX^2 + sum(sum(prod(UtU,3))) - 2*innXXhat);
            
            fit = 1 - (normresidual/ normX); %fraction explained by model
            fitchange = abs(fitold - fit);
            stop(1) = fitchange < param.tol;
            stop(3) = fit >= param.fitmax;
            if nargout >=2
                output.Fit = [output.Fit; iter fit];
            end
        end
        
        if param.TraceMSAE
            msae = SAE(U,Uold);
            msaechange = abs(msaeold - msae); % SAE changes
            stop(2) = msaechange < param.tol*abs(msaeold);
            if nargout >=2
                output.MSAE = [output.MSAE; msae];
            end
        end
        
        if mod(iter,param.printitn)==0
            fprintf(' Iter %2d: ',iter);
            if param.TraceFit
                fprintf('fit = %e fitdelta = %7.1e ', fit, fitchange);
            end
            if param.TraceMSAE
                fprintf('msae = %e delta = %7.1e', msae, msaechange);
            end
            fprintf('\n');
        end
        
        % Check for convergence
        if (iter > 1) && any(stop)
            break;
        end
    end
end

%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = ktensor(U);

% Normalize factors and fix the signs
P = arrange(P);P = fixsigns(P);

if param.printitn>0
    normresidual = sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod(X,(P))));
    fit = 1 - (normresidual / normX); %fraction explained by model
    fprintf(' Final fit = %e \n', fit);
end

% % Rearrange dimension of the estimation tensor
% if ~isempty(p_perm)
%     P = ipermute(P,p_perm);
%     %[foe,ip_perm] = sort(p_perm);
%     %Uinit = Uinit(ip_perm);
% end
if nargout >=2
    output.NoIters = iter;
end

% %%
%     function [F G] = fevalCP(Un,X,U,lambda,n)
%         UtU = zeros(R,R,N);
%         for n = 1:N
%             UtU(:,:,n) = U{n}'*U{n};
%         end
%         P = ktensor(lambda,U);
%         F = (normX^2 + norm(P)^2 - 2 * innerprod(X,P))/2;
%         G = -mttkrp(X,U,n)*diag(lambda) + Un *diag(lambda)* prod(UtU(:,:,[1:n-1 n+1:end]),3)*diag(lambda);
%     end
% 
%     function [F G] = fastfevalCP(Un,Gn,U,lambda)
%         F = (normX2 + sum(lambda.^2) - 2 * sum(sum(Un.*(Gn))))/2;
%         G = Gn - Un *diag(lambda.^2);
%         G = -G;
%     end
% 
% %% CP Gradient with respect to mode n
%     function [G,Pmat] = cp_gradient(A,n,Pmat)
%         persistent KRP_right0;
%         right = N:-1:n+1; left = n-1:-1:1;
%         % KRP_right =[]; KRP_left = [];
%         if n <= ns
%             if n == ns
%                 if numel(right) == 1
%                     KRP_right = A{right};
%                 elseif numel(right) > 2
%                     [KRP_right,KRP_right0] = khatrirao(A(right));
%                 elseif numel(right) > 1
%                     KRP_right = khatrirao(A(right));
%                 else
%                     KRP_right = 1;
%                 end
%                 
%                 if isa(Pmat,'tensor')
%                     Pmat = reshape(Pmat.data,[],prod(I(right))); % Right-side projection
%                 else
%                     Pmat = reshape(Pmat,[],prod(I(right))); % Right-side projection
%                 end
%                 Pmat = Pmat * KRP_right ;
%             else
%                 Pmat = reshape(Pmat,[],I(right(end)),R);
%                 if R>1
%                     Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],I(right(end)),R));
%                     Pmat = sum(Pmat,2);    % fast Right-side projection
%                 else
%                     Pmat = Pmat * A{right(end)};
%                 end
%             end
%             
%             if ~isempty(left)       % Left-side projection
%                 KRP_left = khatrirao(A(left));
%                 %                 if (isempty(KRP_2) && (numel(left) > 2))
%                 %                     [KRP_left,KRP_2] = khatrirao(A(left));
%                 %                 elseif isempty(KRP_2)
%                 %                     KRP_left = khatrirao(A(left));
%                 %                     %KRP_2 = [];
%                 %                 else
%                 %                     KRP_left = KRP_2; KRP_2 = [];
%                 %                 end
%                 T = reshape(Pmat,prod(I(left)),I(n),[]);
%                 if R>1
%                     T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
%                     T = sum(T,1);
%                     %G = squeeze(T);
%                     G = reshape(T,[],R);
%                 else
%                     G = (KRP_left'*T)';
%                 end
%             else
%                 %G = squeeze(Pmat);
%                 G = reshape(Pmat,[],R);
%             end
%             
%         elseif n >=ns+1
%             if n ==ns+1
%                 if numel(left) == 1
%                     KRP_left = A{left}';
%                 elseif numel(left) > 1
%                     KRP_left = khatrirao_t(A(left));
%                     %KRP_left = khatrirao(A(left));KRP_left = KRP_left';
%                 else
%                     KRP_left = 1;
%                 end
%                 if isa(Pmat,'tensor')
%                     T = reshape(Pmat.data,prod(I(left)),[]);
%                 else
%                     T = reshape(Pmat,prod(I(left)),[]);
%                 end
%                 %
%                 Pmat = KRP_left * T;   % Left-side projection
%             else
%                 if R>1
%                     Pmat = reshape(Pmat,R,I(left(1)),[]);
%                     Pmat = bsxfun(@times,Pmat,A{left(1)}');
%                     Pmat = sum(Pmat,2);      % Fast Left-side projection
%                 else
%                     Pmat = reshape(Pmat,I(left(1)),[]);
%                     Pmat = A{left(1)}'* Pmat;
%                 end
%             end
%             
%             if ~isempty(right)
%                 T = reshape(Pmat,[],I(n),prod(I(right)));
%                 
%                 if (n == (ns+1)) && (numel(right)>=2)
%                     %KRP_right = KRP_right0;
%                     if R>1
%                         T = bsxfun(@times,T,reshape(KRP_right0',R,1,[]));
%                         T = sum(T,3);
%                         %G = squeeze(T)';        % Right-side projection
%                         G = reshape(T, R,[])';
%                     else
%                         %G = squeeze(T) * KRP_right0;
%                         G = reshape(T,[],prod(I(right))) * KRP_right0;
%                     end
%                 else
%                     KRP_right = khatrirao(A(right));
%                     if R>1
%                         T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
%                         T = sum(T,3);
%                         %G = squeeze(T)';        % Right-side projection
%                         G = reshape(T,R,[])';        % Right-side projection
%                     else
%                         %G = squeeze(T) * KRP_right;
%                         G = reshape(T,I(n),[]) * KRP_right;
%                     end
%                 end
%             else
%                 %G = squeeze(Pmat)';
%                 G = reshape(Pmat,R,[])';
%             end
%             
%         end
%         
%         %         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
%         %             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
%         %             size(KRP_left,1),size(KRP_left,2),...
%         %             size(KRP_right,1),size(KRP_right,2))
%     end
end


% %% Khatri-Rao xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function krp = khatrirao(A,B)
% if nargin==2
%     R = size(A,2);
%     krp = zeros(size(A,1)*size(B,1),R);
%     for r = 1:R
%         d = B(:,r) * A(:,r)';
%         krp(:,r) = d(:);
%     end
% else
%
%     krp = A{1};
%     I = cellfun(@(x) size(x,1),A);
%     R = size(A{1},2);
%     for k = 2:numel(A)
%         temp = zeros(size(krp,1)*I(k),R);
%         for r = 1:R
%             d = A{k}(:,r) * krp(:,r)';
%             temp(:,r) = d(:);
%         end
%         krp = temp;
%     end
% end
% end
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

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('linesearch',true);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addOptional('normX',[]);
param.addOptional('orthomodes',[]);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end

% %% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function Uinit = cp_init(X,R,param)
% % Set up and error checking on initial guess for U.
% N = ndims(X);
% if iscell(param.init)
%     if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
%         Uinit = param.init;
%         Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
%         if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
%             error('Wrong Initialization');
%         end
%     else % small iteratons to find the best initialization
%         normX = norm(X);
%         bestfit = 0;Pbest = [];
%         for ki = 1:numel(param.init)
%             initk = param.init{ki};
%             if iscell(initk) || isa(initk,'ktensor') || ...
%                     (ischar(initk)  && ismember(initk(1:4), ...
%                     {'rand' 'nvec' 'fibe' 'orth'}))  % multi-initialization
%                 if ischar(initk)
%                     cprintf('blue','Init. %d - %s',ki,initk)
%                 else
%                     cprintf('blue','Init. %d - %s',ki,class(initk))
%                 end
%                 
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
%         cprintf('blue','Chose the best initial value.\n')
%         Uinit = Pbest.U;
%         Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
%     end
% elseif isa(param.init,'ktensor')
%     Uinit = param.init.U;
%     Uinit{end} = Uinit{end} * diag(param.init.lambda);
%     Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
%     if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
%         error('Wrong Initialization');
%     end
% elseif strcmp(param.init(1:4),'rand')
%     Uinit = cell(N,1);
%     for n = 1:N
%         Uinit{n} = randn(size(X,n),R);
%     end
% elseif strcmp(param.init(1:4),'orth')
%     Uinit = cell(N,1);
%     for n = 1:N
%         Uinit{n} = orth(randn(size(X,n),R));
%         if R > size(X,n)
%             Uinit{n}(:,end+1:R) = randn(size(X,n),R-size(X,n));
%         end
%     end
% elseif strcmp(param.init(1:4),'nvec') % changed for missing values
%     Uinit = cell(N,1);
%     for n = 1:N
%         if R<=size(X,n)
%             Uinit{n} = real(nvecs(X,n,R));
%         else
%             Uinit{n} = randn(size(X,n),R);
%         end
%     end
% elseif strcmp(param.init(1:4),'fibe') % select fibers from the data tensor
%     Uinit = cell(N,1);
%     %Xsquare = X.data.^2;
%     for n = 1:N
%         Xn = double(tenmat(X,n));
%         %proportional to row/column length
%         part1 = sum(Xn.^2,1);
%         probs = part1./sum(part1);
%         probs = cumsum(probs);
%         % pick random numbers between 0 and 1
%         rand_rows = rand(R,1);
%         ind = [];
%         for i=1:R,
%             msk = probs > rand_rows(i);
%             msk(ind) = false;
%             ind(i) = find(msk,1);
%         end
%         Uinit{n} = Xn(:,ind);
%         Uinit{n} = bsxfun(@rdivide,Uinit{n},sqrt(sum(Uinit{n}.^2)));
%     end
%     
% else
%     error('Invalid initialization');
% end
% end

%%

% function [Pnew,param_ls] = cp_linesearch(X,P,U0,param_ls)
% % Simple line search adapted Rasmus Bro's approach.
% 
% alpha = param_ls.alpha^(1/param_ls.acc_pow);
% U = P.U;
% U{1} = U{1} * diag(P.lambda);
% 
% Unew = cellfun(@(u,uold) uold + (u-uold) * alpha,U,U0,'uni',0);
% Pnew = ktensor(Unew);
% mod_newerr = norm(Pnew)^2 - 2 * innerprod(X,Pnew);
% 
% if mod_newerr>param_ls.mod_err
%     param_ls.acc_fail=param_ls.acc_fail+1;
%     Pnew = false;
%     if param_ls.acc_fail==param_ls.max_fail,
%         param_ls.acc_pow=param_ls.acc_pow+1+1;
%         param_ls.acc_fail=0;
%     end
% else
%     param_ls.mod_err = mod_newerr;
% end
% 
% end

%%

function [msae,msae2,sae,sae2] = SAE(U,Uh)
% Square Angular Error
% sae: square angular error between U and Uh
% msae: mean over all components
%
% [1] P. Tichavsky and Z. Koldovsky, Stability of CANDECOMP-PARAFAC
% tensor decomposition, in ICASSP, 2011, pp. 4164?4167.
%
% [2] P. Tichavsky and Z. Koldovsky, Weight adjusted tensor method for
% blind separation of underdetermined mixtures of nonstationary sources,
% IEEE Transactions on Signal Processing, 59 (2011), pp. 1037?1047.
%
% [3] Z. Koldovsky, P. Tichavsky, and A.-H. Phan, Stability analysis and fast
% damped Gauss-Newton algorithm for INDSCAL tensor decomposition, in
% Statistical Signal Processing Workshop (SSP), IEEE, 2011, pp. 581?584.
%
% Phan Anh Huy, 2011

N = numel(U);
R = size(U{1},2);
sae = zeros(N,size(Uh{1},2));
sae2 = zeros(N,R);
for n = 1: N
    C = U{n}'*Uh{n};
    C = C./(sqrt(sum(abs(U{n}).^2))'*sqrt(sum(abs(Uh{n}).^2)));
    C = acos(min(1,abs(C)));
    sae(n,:) = min(C,[],1).^2;
    sae2(n,:) = min(C,[],2).^2;
end
msae = mean(sae(:));
msae2 = mean(sae2(:));
end