function [X,fcost] = linreg_lrmx_bals(Y,U,R,opts)
% Alternating algorithm to solve the linear regression problem 
%
%       min sum_l 1/2 * \| yl - U' * vec(A * Bl^T) \|^2 + mu * \|X\|_F^2
%
% where 
%   Y = [y1, ..., yL] is a matrix of size K x L
%   U is a matrix (I*J) x K which can be in CP format of two factor matrices
%
%   X is a tensor I x J x L,  X = A \o B
%
% Ouput:  X{1} = A  of size I x R
%         X{2} = [B1' B2' ... BL']' is array of size J*L x R
%
% Parameters
%
%    init: 'nvec'  initialization method
%    maxiters: 200     maximal number of iterations
%    mu: damping parameter
%
%    printitn: 0
%    tol: 1.0000e-06
%
% Phan Anh Huy, 2017
%


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    X = param; return
end


%%
if param.printitn ~=0
    fprintf('\nAlternating Single Core Update Algorithm for Single-term Multivariate Polynomial Regression\n');
    fprintf(' min  sum_k | y_k -  < X , (uk1 o uk2 o ... o ukN)> |^2\n')
    fprintf(' subject to : X = A * B''')
end

N = numel(U);
[K,L] = size(Y);
szX = cellfun(@(x) size(x,1),U)';
szX(2)= szX(2)*L;
% X1 = A is of size I x R
% X2 = B is of size J*L x R,   Bl: J x R 

 
damping = param.damping;

% Initialization
if isa(param.init,'ktensor')
    X = param.init;
    X = X.U;
elseif iscell(param.init)
    X = param.init;
    % elseif isa(param.init,'TTeMPS')
    %     X = param.init;
elseif isstruct(param.init)
    X = {param.init.U param.init.V};
else
    switch param.init
        case 'rand'
            X = arrayfun(@(n) randn(n,R),szX,'uni',0)';
            X = cellfun(@(x) x/norm(x,'fro'),X,'uni',0);
    end
end

X{1} = orth(X{1});

%%  
KRaostruct = iscell(U);

if ~KRaostruct
    % U is a numerical array 
     
    % min  x'*(UU+mu*I) *x - 2 g'*x + Y'*Y
    % where x = vec(A*B'), g = U*Y
    %
    % or  x'*UU*x  - 2*trace(A'*G*B)
    %
    UU = U*U';
    if damping ~= 0
        UU(1:prod(szX)+1:end) = UU(1:prod(szX)+1:end)+damping;
    end
    UU = tensor(reshape(UU,[szX szX]));
    g = U*Y;
    G = reshape(g,szX);
end
normy2 = norm(Y,'fro')^2;

%% Iterate the main algorithm
fcost = zeros(param.maxiters,1);
cnt = 0;
eps_ = 1e-7;
 
for kiter = 1:param.maxiters
    
    for n = [2 1]
        cnt = cnt +1;

        % alternating update X1 and X2
        if n == 2
            m = 1;
        else
            m = 2;
        end
         
        %% least squares        
         
        if KRaostruct 
            % If U is a Khatri-Rao structured matrix, i.e.
            %    min \| Y - P^T * vec(X) \|^2 + mu/2 * \|X\|_F^2  (1)
            % P = khatrirao(U2,U1)
            %
             
            switch n
                case 2 % m = 1
                    % In order to update X2, X1 is orthogonalized, then
                    % solve the sub-problem for X2
                    %    min \| Y -  P2^T * [B1(:) ... BL(:)] \|^2 + mu * \|X2\|_F^2
                    %
                    % P2 = khatrirao(U2,X1'*U1)
                    
                    Zm = X{m}'*U{m};
                    Pn = khatrirao(U{n},Zm);
                    
                    % norm(Y - Pn'*reshape(X{2}',[],L),'fro')^2 + damping*norm(X{1}*X{2}','fro')
                    % norm(Y - khatrirao(U,'r')'* reshape(X{1}*X{2}',[],L),'fro')^2 + damping*norm(X{1}*X{2}','fro')
                    gn = Pn*Y;
                    PnPn = Pn*Pn';
                    if damping ~= 0
                        PnPn(1:size(PnPn,1)+1:end) = PnPn(1:size(PnPn,1)+1:end) + damping;
                    end
                    xn = PnPn\gn;
                    
                    Xn = reshape(xn,R,[])'; % Xn : J*L  x R
                    
                    fcost(cnt) = normy2 - gn(:)'*xn(:);
                    
                case 1
                    
                    % In order to update X1, X2 is orthogonalized, then
                    % solve the sub-problem for X1
                    %    min \| Y -  P2^T * vec(X1) \|^2 + mu * \|X1\|_F^2
                    %
                    % P2 = [khatrirao(B1'*U2,U1)
                    %       ...
                    %       khatrirao(BL'*U2,U1)]
                    Zm = reshape(X{m},[],L*R)'*U{m}; % L*R x K
                    Pn = khatrirao(Zm,U{n}); % I*L*R x K
                    Pn = reshape(Pn,[],L,R*K);  %
                    Pn = permute(Pn,[1 3 2]);   %IRKxL
                    Pn = reshape(Pn,[],K*L);    %I*R x K*L

                    gn = Pn*Y(:);
                    PnPn = Pn*Pn';
                    if damping ~= 0
                        PnPn(1:size(PnPn,1)+1:end) = PnPn(1:size(PnPn,1)+1:end) + damping;
                    end
                    xn = PnPn\gn;
                    
                    Xn = reshape(xn,[],R); % Xn :I x R
                    
                    fcost(cnt) = normy2 - gn'*xn;
                    
            end
                  
            % min | Y - Pn*x|^2 + mu/2 |x|^2
            
            
            
        else % a numerical array 
             % Update X1
             %   min vec(X1)^T * Q1 * vec(X1) - 2 trace(X1'*G*X2)
             %
             % where Q1 = (X2 ox I)^T * UU *(X2 ox I)
             % Update X2
             %   min vec(X2)^T * Q2 * vec(X2) - 2 trace(X1'*G*X2)
             %
             % where Q2 = (I ox X1)^T * UU *(I ox X1)
            Qn = ttm(UU,{X{m} X{m}},[m m+2],'t');
            Qn = reshape(double(Qn),szX(n)*R,[]);
            
            if n == 2 % update X2
                gn = X{m}'*G;
            else % update X1
                gn = G*X{m};
            end
            xn = Qn\gn(:);
            
            if n == 1 % update X2
                Xn = reshape(xn,[],R);
            else
                Xn = reshape(xn,R,[])';
            end
             
            fcost(cnt) = normy2 - gn(:)'*xn(:);
        end
        
        % orthogonalise Xn 
        [QQ,RR] = qr(Xn,0);
        % check and truncate the rank
        ss = sum(abs(RR).^2,2);
        ixs = ss > eps_*sum(ss);
        R = sum(ixs); %  % rank may change
        X{n} = QQ(:,ixs);X{m} = X{m}*RR(ixs,:)';
    end
    
    %
    if mod(kiter,param.printitn)==0
        fprintf('(%d,%d)- %d\n',kiter,cnt,fcost(cnt))
    end
    
    
    % check stopping criteria
    if (cnt > 2) && (abs(fcost(cnt) - fcost(cnt-2))<param.tol)
        break
    end
end
fcost(cnt+1:param.maxiters) = [];

    
end
  

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addParameter('init','rand',@(x) (iscell(x) || isa(x,'ktensor')||...
    isa(x,'TTeMPS') || ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'exac'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
% param.addOptional('compression',true);
% param.addOptional('compression_accuracy',1e-6);
% param.addOptional('noise_level',1e-6);% variance of noise
param.addOptional('printitn',0);
param.addOptional('normX',[]);
param.addOptional('damping',0);
param.addParameter('compression',true);   % 0 1 


param.parse(opts);
param = param.Results;


end
% 
 %%
function xn = ppa_solver(Un,Zm,xn,K,In,R,damping)
% Parallel proximal to solve 
%    min \| Y - Pn^T * x \|^2 + mu/2 \|x\|^2
% 
%  where Pn is long matrix 
%
idx = 1:K;
% split the K entries into blocks of IJ
blksize = min(K/2,In*R*20);
no_blks = floor(K/(blksize));
ix_blks = [1:blksize:K-blksize K+1];

% Parallel Proximal algorithm
F = cell(no_blks,1);
mu = damping/no_blks;
for kblk = 1:no_blks
    i_kblk = idx(ix_blks(kblk):ix_blks(kblk+1)-1);
    Pn_k = khatrirao(Zm(:,i_kblk),Un(:,i_kblk));
    param_k.Y = Y(i_kblk);
    param_k.A = @(x) Pn_k'*x;
    param_k.At = @(x) Pn_k*x;
    Pnky = Pn_k*Y(i_kblk);
    PnPnk = Pn_k*Pn_k';
    F{kblk}.eval = @(x) 1/2*norm(param_k.Y - param_k.A(x))^2 + mu/2*norm(x)^2;
    %F{kblk}.prox = @(x,gamma)  prox_l2(x, gamma, param_k);
    F{kblk}.prox = @(x,gamma)  (gamma*(PnPnk)+(gamma*mu+1)*eye(size(Pn_k,1)))\(x + gamma * Pnky);
end
ppa_param.gamma = 1;
ppa_param.do_ts = @(x) log_decreasing_ts(x, 10, 0.1, 80);
[xn, info] = ppxa(xn(:), F,ppa_param);
end