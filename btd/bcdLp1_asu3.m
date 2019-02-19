function [a,U,lambda,cost,output] = bcdLp1_asu3(Y,R,opts)
% Tensor deflation using the Alternating Subspace Update algorithm
%
% The algorithm decomposes a tensor into a rank-1 tensor and
% multilinear-(R-1,R-1,...,R-1) tensor.
%
%    Y = lambda * a1 o a2 o ... o aN + [[G; U1, U2, ..., UN]].
%
% The algorithm can be used to extract a rank-1 tensor from a rank-R
% tensor.
%
% INPUT:
%   Y:  N-way data.
%   R:  rank of the tensor, should not exceed multilinear rank of Y.
%   OPTS: optional parameters
%     .tol:      tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {2000}
%     .printitn: Print fit every n iterations {1}
%
%     .alsinit : run the ALS algorithm with a small number of iterations
%                (3) to initialize the decomposition
%
%     .init:     Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'|
%                'dtld' | 'tdiag' | 'ceig' | 'pegi' | 'osvd' | cell array]
%                init can be a cell array whose each entry specifies an
%                intial value or type. The algorithm will choose the best
%                one after some small runs.
%                For example,
%                opts.init = {'ceig' 'random' 'random' 'nvec'};
%
%     .normX  :   precompute ell-2 norm of X
%     .correct_badinitial : (true) correct the initial loading components
%                of the rank-1 tensor if they violate the uniqueness
%                condition, i.e., there are at least (N-1) angles between
%                loading components a_n and subspaces U_n sufficiently small.
%
%     .pre_select: (false) pre-selection of loading components for the
%                  rank-1 tensor among R initial points after some small runs.
%
%     .verify_rank:  verify if initial rank exceeds multilinear rank of Y
%     .reference:  rank-1 tensor for reference. If reference is given,
%                  angular errors between two subspaces are used to select
%                  the similar initial point.
%
%     .tracking_param:  struct for tracking problem.
%
%
% OUTPUT:
%  a:   loading of the rank-1 tensor
%  U:   loading of the multilinear rank-(R-1) tensor
%  lambda :   scaling coefficient of the rank-1 tensor
%  cost :  array of cost values vs iterations
%  output: other output such as initial value.
%
%
% EXAMPLE: see ex_rank1_extraction.m
%
% REF:
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "DEFLATION METHOD FOR
% CANDECOMP/PARAFAC TENSOR DECOMPOSITION", ICASSP 2014.
% [2] A.-H. Phan, P. Tichavsky, A. Cichocki, "Tensor Deflation for
% CANDECOMP/PARAFAC. Part 1: Algorithms", IEEE Transaction on Signal
% Processing, 63(12), pp. 5924-5938, 2015.
% [3] A.-H. Phan, P. Tichavsky, A. Cichocki, "Tensor deflation for
% CANDECOMP/PARAFAC. Part 2: Initialization and Error Analysis?, IEEE
% Transaction on Signal Processing, 63(12), pp. 5939-5950, 2015.
%
% See also: cp_als,
%
% TENSOR BOX, v1. 2014
% Copyright 2014, Phan Anh Huy.

% dbstop if naninf;
% dbstop if error;
% dbstop if warning ;

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    a = param; return
end

I = size(Y);
N = ndims(Y);
normY = norm(Y);
normY2 = normY^2;

%% Check if R exceeds multilinear rank of the tensor.

if param.verify_rank
    mrank = zeros(1,N);
    for n = 1:N
        mrank(n) = rank(double(tenmat(Y,n)));
    end
    R = min(mrank,R);
elseif numel(R) == 1
    R = R(ones(1,N));
end


%% Compression
compress_mode = find(I > R);

if ~isempty(compress_mode)
    comp_opts = struct('init','nvecs','maxiters',10,'dimorder',compress_mode,...
        'tol',1e-6,'printitn',0);
    Ycomp = mtucker_als(Y,R(compress_mode),comp_opts);
    Y = Ycomp.core;
    I = size(Y);
    
    % Correct predefined initial value
    if (isequal(size(param.init),[N 2])) && all(all(cellfun(@isnumeric,param.init))) % pre-defined initialization
        a = param.init(:,1);
        U = param.init(:,2);
        
        Ifac = cellfun(@(x) size(x,1),a(:));
        if any(Ifac(compress_mode)' > I(compress_mode))
            for n = compress_mode
                if ~isempty(a{n})
                    a{n} = Ycomp.u{n}'*a{n};
                end
                if ~isempty(U{n})
                    U{n} = Ycomp.u{n}'*U{n};
                end
            end
        end
        param.init = [a U];
    end
end

%% Initialization

w = [];
[a,U] = init_factor(Y,R,param);
[ai,Ui] = deal(a,U);

% % Correct predefined initialization
% if ~isempty(compress_mode)
%     Ifac = cellfun(@(x) size(x,1),a(:));
%     if any(Ifac(compress_mode)' > I(compress_mode))
%         for n = compress_mode
%             a{n} = Ycomp.u{n}'*a{n};
%             U{n} = Ycomp.u{n}'*U{n};
%         end
%     end
% end
% U = findU_givena(Y,a);

%% Preselection of components of the rank-1 tensor 
if param.pre_select
    [a,U] = init_preselect(Y,R,a,U);
end

if any(isnan(cell2mat(a)))
    fprintf('Nan\n')
end

%% Rotate factors to be orthognal 
[a,U] = rotate_factor(a,U);

%% Refine the initial points
if param.refine_a
    %     [a,U] = refine_factor(Y,R,a,U);
    [a,U] = refine_factor_0814(Y,R,a,U);
    %     [a,U] = refine_factor_0914(Y,R,a,U);
    
    
    %     u = cellfun(@(x) x(:,1),U,'uni',0);
    %     [a,u] = refine_factor2(Y,R,a,u);
    %     rho = cellfun(@(x,y) x'*y(:,1),a,u);
    %     w = cell(N,1);
    %     for n = 1:N
    %         if (abs(1-rho(n)))>1e-5
    %             w{n} = (a{n}-rho(n)*u{n})/sqrt(1-rho(n)^2);
    %         else
    %             [wn,e] = qr(U{n});
    %             w{n} = wn(:,end);
    %         end
    %         [Un,rr] = qr([w{n} u{n}]); U{n} = [u{n} Un(:,3:end)];
    %     end
    
    %     Costref = zeros(10,1);
    %     for ki = 1:100
    %         [a,U] = refine_factor(Y,R,a,U);
    %         U = findU_givena(Y,a);
    %         costref= bcdLp1_cost(Y,a,U);
    %         Costref(ki) = costref;
    %
    %         if ki> 1
    %             if abs(costref - costold)<1e-5 * costold
    %                 break
    %             end
    %         end
    %         costold = costref;
    %     end
end

%% ALS initiation. 
if param.alsinit == true
    [a,U] = als_init(Y,R,a,U,param);
end

% %% Rotate factors to be orthognal
% [a,U] = rotate_factor(a,U);

%% Compute mutual angles between a_n and u_n
rho = cellfun(@(x,y) x'*y(:,1),a,U);
if param.correct_badinitial
    % if there are at leat (N-1) rho_n == 1, try another initial point
    if sum(abs(1-abs(rho))<1e-4) >= (N-1)
        [a,rho,w,U] = finda_givenU2(Y,U);
    end
end

%% Precomputing paramaters 

Y = tensor(Y);cost = zeros(param.maxiters,1);
u = cellfun(@(x) x(:,1),U,'uni',0);
% type_a = 1;

% Precompute Phin
Cn = cell(1,N);
for n = 1:N
    Zn = double(tenmat(Y,n));
    Cn{n} = (Zn*Zn');
end

if isempty(w)
    w = cell(N,1);
    for n = 1:N
        if (abs(1-rho(n)))>1e-5
            w{n} = (a{n}-rho(n)*u{n})/sqrt(1-rho(n)^2);
        else
            [wn,e] = qr(U{n});
            w{n} = wn(:,end);
        end
    end
end

% mu = tol_angle;
cntdone = 0; maxndone = 5;
tol_angle = 1e-6;

% Predefine mode-n matricization of tensor Y
Y1 = reshape(Y.data,I(1),[]);
Y3 = reshape(Y.data,[I(1)*I(2),I(3)]);
Y2 = double(tenmat(Y,2)).';

Yw2 = reshape(Y2*(w{2}),I(1),[]);

%% Main loop to estimate components of the rank-1 tensors 

for iter = 1: param.maxiters
    
    % if there are at leat (N-1) rho_n == 1, try another initial point
    % This correction makes the random initialization to be more robust
    if sum(abs(1-abs(rho))<1e-5) >= (N-1)
        %         [a,u] = refine_factor_0814(Y,R,a,u);
        %         u = cellfun(@(x) x(:,1),U,'uni',0);
        %         rho = cellfun(@(x,y) x'*y(:,1),a,u);
        %         for n = 1:N
        %             if (abs(1-rho(n)))>1e-5
        %                 w{n} = (a{n}-rho(n)*u{n})/sqrt(1-rho(n)^2);
        %             else
        %                 [wn,e] = qr(U{n});
        %                 w{n} = wn(:,end);
        %             end
        %         end
        
        
        % %         %for m = 1:N
        % %         %    [Un,rr] = qr([w{m} u{m}]); U{m} = [u{m} Un(:,3:end)];
        % %         %end
        % % %         cost0 = bcdLp1_cost(Y,a,u);
        %         [a,rho,w,u] = finda_givenU2(Y,u,w);
        %         Yw2 = reshape(Y2*(w{2}),I(1),[]);
    end
    
    for n = 1:3
        switch n
            case 1
                % Precompute some parameters
                %Yw3 = double(ttv(Y,w{3},3)); Yw2 = double(ttv(Y,w{2},2));
                %Yu3 = double(ttv(Y,u{3},3)); %Yu2 = ttv(Y,w{2});
                %Y3 = reshape(Y.data,[I(1)*I(2),I(3)]);
                Yw3 = reshape(Y3*w{3},[I(1),I(2)]);Yu3 = reshape(Y3*u{3},[I(1),I(2)]);
                %Yw2 = squeeze(sum(bsxfun(@times,Y.data,w{2}.'),2));
                
                Ya3 = sqrt(1-rho(3)^2)* Yw3 + rho(3)*Yu3;
                
                Yw23 = Yw3 * w{2};
                zn = Yu3 * u{2};
                sn = Ya3*a{2};
                Qn = Cn{n} - Yw2*Yw2' - Yw3*Yw3' + Yw23 * Yw23';
                
            case 2
                %Yw1 = double(ttv(Y,w{1},1));
                Yw1 = reshape(w{1}'*Y1,I(2),I(3));
                Yw13 = (w{1}'*Yw3)';
                zn = (u{1}'*Yu3)';
                sn = (a{1}'*Ya3)';
                Qn = Cn{n} - Yw1*Yw1' - Yw3'*Yw3 + Yw13 * Yw13';
                
            case 3
                
                %Yw2 = double(ttv(Y,w{2},2)); Yu2 = double(ttv(Y,u{2},2));
                %Yw2 = squeeze(sum(bsxfun(@times,Y.data,w{2}.'),2));
                %Yu2 = squeeze(sum(bsxfun(@times,Y.data,u{2}.'),2));
                Yw2 = reshape(Y2*(w{2}),I(1),[]);
                Yu2 = reshape(Y2*(u{2}),I(1),[]);
                
                Yw12 = (w{1}'*Yw2)';
                zn = (u{1}'*Yu2)';
                Ya2 = sqrt(1-rho(2)^2)* Yw2 + rho(2)*Yu2;
                sn = (a{1}'*Ya2)';
                
                Qn = Cn{n} - Yw1'*Yw1 - Yw2'*Yw2 + Yw12 * Yw12';
        end
        
        rho_ne = min(1,prod(rho([1:n-1 n+1:end])));
        
        %         flag_n = (abs(1-abs(rho(n)))>tol_angle); % check rho_n  ~= 1
        %         flag_ne = (abs(1-abs(rho_ne)) > tol_angle);% check rho_(-n) ~= 1
        
        %         if flag_ne
        dn = (sn - rho_ne *zn)/sqrt(1-rho_ne^2);
        %             if norm(dn) < 1e-8
        %                 flag_ne = false;
        %             end
        %         end
        
        
        %         if flag_n && flag_ne % rho_n  ~= 1 and rho_{-n}  ~= 1
        
        [an,wn,un,rho_n,cost_new] = update_wu_case1(Qn,sn,dn);
        %             if (1-abs(rho_n))<1e-4 % re-solve w_n un and rho_n
        %                 [an,wn,un,rho_n,cost_new] = update_wu_case2(Qn,dn);
        %             end
        %         elseif ~flag_n && flag_ne   % rho_n  == 1  and  rho_{-n}  ~= 1
        %             [an,wn,un,rho_n,cost_new] = update_wu_case2(Qn,dn);
        %
        %         elseif flag_n && ~flag_ne
        %             fprintf('Iteration %d. There are %d loading a_n (n ~= %d) lying in subspaces of Un.\nTry other initialization or a lower rank.\n',iter,N-1,n)
        %             %break;
        %
        %             [an,wn,un,rho_n,cost_new] = update_wu_case3(Qn,sn);
        %
        %         else %(abs(1-rho(n))<1e-6) && (abs(1-rho_ne)<1e-6)
        %             fprintf('Iteration %d. All loading a_n lie in subspaces of Un.\n Try other initialization or a lower rank.\n',iter)
        %
        %             try
        %                 [an,wn,un,rho_n,cost_new] = update_wu_case4(Qn);
        %             catch me;
        %                 % If there is any error, skip this update, and correct
        %                 % rho_n and a_n with given Un and wn
        %                 continue;
        %             end
        %         end
        
        if iter>1
            if ((n >1) && cost_new > cost(iter,n-1)) || ((n==1) && cost_new > cost(iter-1,end))
                % skip this update, rho(n) perhaps is very close to 1
                if (n >1)
                    cost(iter,n) = cost(iter,n-1);
                else
                    cost(iter,n) = cost(iter-1,end);
                end
                continue
            end
        end
        a{n} = an; w{n} = wn; u{n} = un; rho(n) = rho_n;
        cost(iter,n) = cost_new;
        
    end
    
    %% Check convergence  and print the approximation error 
    
    if mod(iter,param.printitn) == 0
        fprintf('Iter %d Cost %d \n',iter,cost(iter,end))
    end
    
    if iter>1
        done = (abs(cost(iter,end)-cost(iter-1,end))<= param.tol*abs(cost(iter-1,end)));
        if done
            cntdone = cntdone +1;
            if cntdone < maxndone
                done = false;
            end
        else
            done = false;
            cntdone = 0;
        end
        if done
            break
        end
    end
    
end


% Compute factor matrices of the block of multilinear rank-(R-1,...,R-1)
for n = 1:N
    [Un,rr] = qr([w{n} u{n}]); U{n} = [u{n} Un(:,3:end)];
end

rhoall = prod(rho);
try
    lambda = (a{end}'*sn - rhoall*u{end}'*zn)/(1-rhoall^2);
catch
    lambda = [];
end
% U = cellfun(@(x) x(:,2:end),Vc,'uni',0);
cost(iter+1:end,:) = [];


% Convert the solution back to the orignal space (i.e. non-compression
% data)
if ~isempty(compress_mode)
    a = cellfun(@(x,y) x*y,Ycomp.U,a,'uni',0);
    U = cellfun(@(x,y) x*y,Ycomp.U,U,'uni',0);
    % Need to correct the cost value
end

if nargout> 4
    output = struct('ai',{ai},'Ui',{Ui});
end

%%

    function [an,wn,un,rho_n,cost_new] = update_wu_case1(Qn,sn,dn)
        % rho_n # 1 and prod(rho_k # n) < 1
        Qx = Qn-sn*sn' + dn*dn';
        
        
        % Update wn
        if R < 10000
            [wn,e0] = eig(Qx);
            e0 = diag(e0);
            id1 = find(e0~=0);
            [e0,id] = min(e0(id1));
            wn = wn(:,id1(id));
        else
            [wn,e0] = eigs(Qx,1,'SM');
        end
        
        % Update un
        alpha = wn'*dn;dn2 = dn'*dn;
        un = (dn - wn*alpha)/sqrt(dn2 - alpha^2);
        
        cost_new = sqrt(max(0,normY2 - trace(Qn) + e0 - dn2))/normY;
        
        Vn = [wn un];
        % Update rho_n
        Ytu = un'*zn;
        x = Vn'*double(sn);x(2) = x(2) - rho_ne*Ytu;
        t = sqrt(x(1).^2*(1-rho_ne^2)^2+x(2).^2);
        rho_n = x(2)/t;
        
        an = (wn*x(1)*(1-rho_ne^2) + un*x(2))/t;
        wn = wn * sign(x(1));
    end


    function [an,wn,un,rho_n,cost_new] = update_wu_case2(Qn,dn)
        % rho_n ~~ 1 and prod(rho_k) < 1 for k # n
        
        Qx = Qn + dn*dn';
        
        % Update wn
        if R < 10000
            [wn,e0] = eig(Qx);
            e0 = diag(e0);
            id1 = find(e0~=0);
            [e0,id] = min(e0(id1));
            wn = wn(:,id1(id));
        else
            [wn,e0] = eigs(Qx,1,'SM');
        end
        
        % Update un
        alpha = wn'*dn;dn2 = dn'*dn;
        un = (dn - wn*alpha)/sqrt(dn2 - alpha^2);
        
        cost_new = sqrt(normY2 - trace(Qn) + e0 - dn2)/normY;
        
        Vn = [wn un];
        % Update rho_n
        Ytu = un'*zn;
        x = Vn'*double(sn);x(2) = x(2) - rho_ne*Ytu;
        t = sqrt(x(1).^2*(1-rho_ne^2)^2+x(2).^2);
        if x(2) < 0
            un = -un;
            x(2) = -x(2);
        end
        if x(1) < 0
            wn = -wn;
            x(1) = -x(1);
        end
        rho_n = x(2)/t;
        an = (wn*x(1)*(1-rho_ne^2) + un*x(2))/t;
        %         wn = wn * sign(x(1));
    end


    function [an,wn,un,rho_n,cost_new] = update_wu_case3(Qn,sn)
        % rho_n < 1 and rho_k ~~ 1 for k # n
        
        fprintf('Iteration %d. There are %d loading a_n (n ~= %d) lying in subspaces of Un.\nTry other initialization or a lower rank.\n',iter,N-1,n)
        %break;
        Qx = Qn-sn*sn';
        % Update wn
        if R < 10000
            [wn,e0] = eig(Qx);
            e0 = diag(e0);
            id1 = find(e0~=0);
            [e0,id] = min(e0(id1));
            wn = wn(:,id1(id));
        else
            [wn,e0] = eigs(Qx,1,'SM');
        end
        
        % Update Un
        [Un,rr] = qr(wn); %U{n} = Un(:,2:end);
        un = Un(:,2);
        cost_new = sqrt(normY2 - trace(Qn) + e0)/normY;
        
        % Update rho_n
        rho_n = 0;
        
        % Update a_n
        an = wn;
    end

    function [an,wn,un,rho_n,cost_new] = update_wu_case4(Qn)
        % rho_k ~~ 1  for all k
        
        % Update wn
        if R < 10000
            [wn,e0] = eig(Qn);
            e0 = diag(e0);
            id1 = find(e0~=0);
            [e0,id] = min(e0(id1));
            wn = wn(:,id1(id));
        else
            [wn,e0] = eigs(Qn,1,'SM');
        end
        
        % Update Un
        [Un,rr] = qr(wn); %U{n} = Un(:,2:end);
        un = Un(:,2);
        cost_new = sqrt(normY2 - trace(Qn) + e0)/normY;
        
        % Update rho_n
        rho_n = 0;
        
        % Update a_n
        an = wn;
    end
end


%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','tdiag',@(x) (iscell(x) ||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'tdia' 'ceig' 'peig' 'osvd' 'lamb'})));
param.addOptional('maxiters',2000);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('linesearch',false);
param.addOptional('refine_a',false);
param.addOptional('alsinit',true);
param.addOptional('pre_select',false);
param.addOptional('verify_rank',false);
param.addOptional('reference',[]);
param.addOptional('normX',[]);

param.addOptional('correct_badinitial',true);

% Tracking parameter
track_param = inputParser;
track_param.addOptional('tracking_function',[],@(x) isempty(x) || isa(x,'function_handle'));% Function to evaluate tracking error
track_param.addOptional('tracking_tol',4);
track_opts = [];
if isfield(opts,'tracking_param')
    track_opts = opts.tracking_param;
end
track_param.parse(track_opts);
track_param = track_param.Results;
param.addOptional('tracking_param',track_param);


param.parse(opts);
param = param.Results;
end

%% Rotate factors to be orthognal
function [a,U] = rotate_factor(a,U)
for n =1:numel(U);
    a{n} = a{n}/norm(a{n});
    
    [temp,rr] = qr(U{n},0);
    U{n} = temp;
    
    [u,s,v] = svd(a{n}'*U{n},0);
    
    a{n} = a{n} * u;
    U{n} = U{n} * v;
end
end

%% Refine factor components
%%
function [a,U] = refine_factor_0814(Y,R,a,U)
N = ndims(Y);

% Uniball parameters
uniopts.record = 0;
uniopts.mxitr  = 100;
uniopts.gtol = 1e-6;
uniopts.xtol = 1e-6;
uniopts.ftol = 1e-6;
uniopts.tau = 1e-3;

d = size(Y,1);
Cni = zeros(d,d,N);

for k = 1:N
    Yn = tenmat(Y,k);Yn = double(Yn);
    Cn = Yn*Yn';
    Cni(:,:,k) = inv(Cn);
end

% for n = 1:N
%     a{n} = nvecs(Y,n,1);
% end
% P = tucker_als(Y,R);
% for r = 1:R
%     a = cellfun(@(x) x(:,r) , P.u,'uni',0);
%     fr(r) = est_a_spherical_cnt(cell2mat(a(:)'),double(Y),Cni);
% end
% [foe,r] = min(fr);
% a = cellfun(@(x) x(:,r) , P.u,'uni',0);

am = cell2mat(a(:)');
for ktr = 1:1
    %     if ktr >1
    %         am = randn(d,N);
    %         am = bsxfun(@rdivide,am,sqrt(sum(am.^2)));
    %     end
    
    [am,foe,out]= OptManiMulitBallGBB(am, @est_a_spherical_cnt, uniopts, Y,Cni);
    if ktr == 1
        fval = abs(out.fval);
        ambest = am;
    elseif abs(out.fval)<fval
        fval = abs(out.fval);
        ambest = am;
    end
end
am = ambest;
a = mat2cell(am,size(am,1),ones(1,N));
a = a(:);

% Find w
% w = inv(Phi)*A*sum(inv(A^T*inv(Phi)*A),2)
Phi = [];s = [];A = [];
alpha = zeros(N,1);
beta = zeros(N,1);
gamma = zeros(N,1);
for k = 1:N
    Yn = tenmat(Y,k);Yn = double(Yn);
    Cn = Yn*Yn';
    Phi = blkdiag(Phi,Cn);
    
    sn = double(ttv(Y,a,-k));
    s = [s; sn];
    A = blkdiag(A,a{k});
    
    
    alpha(k) = 1./(a{k}'*Cni(:,:,k) * a{k});
    beta(k) = a{k}'*Cni(:,:,k) * sn;
    gamma(k) = sn'*Cni(:,:,k) * sn;
end
Phi = Phi - 1/N*(s*s');
iPhiA = Phi\A;
w = iPhiA*sum(inv(A'*iPhiA),2);

% cost = w'*Phi*w;
% lambda = (w'*s)/N;
w = reshape(w,size(Y,1),[]);
%
% cost2 = 0;
% for k = 1:N
%     err = norm(ttv(Y,w(:,k),k) - full(ktensor(lambda,a([1:k-1 k+1:end]))))^2;
%     cost2 = cost2 +err;
% end
%
% cost3 = sum(alpha) - (alpha.'*beta)^2/(N - sum(gamma) + alpha.'*beta.^2);
%
% 1;

rho = sqrt(1 - 1./sum(w.^2));
u = cell(N,1);
for n = 1:N
    u{n} = (a{n} - w(:,n) * (1 - rho(n).^2))/rho(n);
    [U{n},rr] = qr([w(:,n)/norm(w(:,n)) u{n}]);
    U{n} = [u{n} U{n}(:,3:end)];
end

%% Rotate factors to be orthognal
if nargin>=4
    [a,U] = rotate_factor(a,U);
else
    U = [];
end
end


%%
function [a,U] = refine_factor_0914(Y,R,a,U)
N = ndims(Y);

% Uniball parameters
uniopts.record = 1;
uniopts.mxitr  = 1000;
uniopts.gtol = 1e-12;
uniopts.xtol = 1e-12;
uniopts.ftol = 1e-12;
uniopts.tau = 1e-3;

d = size(Y,1);
Cni = zeros(d,d,N);

for k = 1:N
    Yn = tenmat(Y,k);Yn = double(Yn);
    Cn = Yn*Yn';
    Cni(:,:,k) = inv(Cn);
end

am = cell2mat(a(:)');
% for ktr = 1:1
%     if ktr >1
%         am = randn(d,N);
%         am = bsxfun(@rdivide,am,sqrt(sum(am.^2)));
%         a = mat2cell(am,size(am,1),ones(1,size(am,2)))';
%     end
for kit = 1:100
    for k = 1:N
        [am_k,foe,out]= OptManiMulitBallGBB(am(:,k), @est_a_spherical_cnt, uniopts, Y,Cni,k,a);
        am(:,k) = am_k;a{k} = am_k;
    end
end
%         if ktr == 1
%             fval = abs(out.fval);
%             ambest = am;
%         elseif abs(out.fval)<fval
%             fval = abs(out.fval);
%             ambest = am;
%         end
% end
% am = ambest;
% a = mat2cell(am,size(am,1),ones(1,N));
% a = a(:);

% Find w
% w = inv(Phi)*A*sum(inv(A^T*inv(Phi)*A),2)
Phi = [];s = [];A = [];
alpha = zeros(N,1);
beta = zeros(N,1);
gamma = zeros(N,1);
for k = 1:N
    Yn = tenmat(Y,k);Yn = double(Yn);
    Cn = Yn*Yn';
    Phi = blkdiag(Phi,Cn);
    
    sn = double(ttv(Y,a,-k));
    s = [s; sn];
    A = blkdiag(A,a{k});
    
    
    alpha(k) = 1./(a{k}'*Cni(:,:,k) * a{k});
    beta(k) = a{k}'*Cni(:,:,k) * sn;
    gamma(k) = sn'*Cni(:,:,k) * sn;
end
Phi = Phi - 1/N*(s*s');
iPhiA = Phi\A;
w = iPhiA*sum(inv(A'*iPhiA),2);

% cost = w'*Phi*w;
% lambda = (w'*s)/N;
w = reshape(w,size(Y,1),[]);
%
% cost2 = 0;
% for k = 1:N
%     err = norm(ttv(Y,w(:,k),k) - full(ktensor(lambda,a([1:k-1 k+1:end]))))^2;
%     cost2 = cost2 +err;
% end
%
% cost3 = sum(alpha) - (alpha.'*beta)^2/(N - sum(gamma) + alpha.'*beta.^2);
%
% 1;

rho = sqrt(1 - 1./sum(w.^2));
u = cell(N,1);
for n = 1:N
    u{n} = (a{n} - w(:,n) * (1 - rho(n).^2))/rho(n);
    [U{n},rr] = qr([w(:,n)/norm(w(:,n)) u{n}]);
    U{n} = [u{n} U{n}(:,3:end)];
end

%% Rotate factors to be orthognal
if nargin>=4
    [a,U] = rotate_factor(a,U);
else
    U = [];
end
end


% function [f,g] = refine_a_given_w(a,Y,Cni)
% % f: cost function and
% % g gradient of f w.r.t a1,a2,a3
% % Zn = Y x_{n} wn
% Sz = size(Y);N = 3;
% d = Sz(1);
% u2 = mat2cell(a,d,ones(1,N));
%
% alpha2 = zeros([N 1]);
% beta2 = zeros([N 1]);
% gamma2 = zeros([N 1]);
% g = zeros(d,N);
%
% v = cell(N,1);
% Csn = cell(N,1);
% for n = 1:N
%     sn = double(ttv(tensor(Y),u2,-n));
%
%     v{n} = Cni(:,:,n) * u2{n};
%     Csn{n} = Cni(:,:,n) * sn;
%
%     alpha2(n) = 1/(u2{n}.'*v{n});
%     beta2(n) = sn.' * v{n};
%     gamma2(n) = sn.'*Csn{n};
%
% end
%
% fnum = (alpha2.'*beta2);
% fden = N + alpha2.'*beta2.^2 - sum(gamma2);
% f = sum(alpha2) - fnum^2/fden;
%
% zn = zeros(d,N);
% Fnk = zeros(d,d,N);
%
% % Gradient
% for n = 1:N
%
%     zn(:,n) = 0;
%     Fnk(:,:,n) = 0;
%     for k = [1:n-1 n+1:N]
%         Ytu = ttv(tensor(Y),u2,-[n k]);
%         Ytu = double(Ytu);
%         if k < n
%             Ytu = Ytu';
%         end
%         YtuC = Ytu* Cni(:,:,k);
%         zn(:,k) = YtuC * u2{k};
%         Fnk(:,:,k) = YtuC* Ytu';
%     end
%
%
%     temp2 = v{n} * beta2(n);
%     temp3 = alpha2(n) * Csn{n};
%
%     gnum = 2*fnum*(2*temp2 * (-alpha2(n)^2)+ temp3  + zn*alpha2);
%     gden = 2*(temp2* beta2(n) * (-alpha2(n)^2)+ temp3 * beta2(n) +  zn*diag(beta2)*alpha2 - sum(Fnk,3)*u2{n});
%
%     g(:,n) = v{n} *2 * (-alpha2(n)^2) - (gnum * fden - fnum^2 * gden)/fden^2;
% end
%
% % g = g;
%
%
% % Ytu = zeros(d,d,N);
% % v = cell(N,1);
% % for n = 1:N
% %     Ytu(:,:,n) = double(ttv(Y,u2,n));
% %     v{n} = Cni(:,:,n) * u2{n};
% % end
% %
% % for n = 1:N
% %     zn(:,n) = 0;
% %     Fnk(:,:,n) = 0;
% %     for k = [1:n-1 n+1:N]
% %         m = setdiff(1:N,[n k]);
% %         if k < n
% %             Ytun =  Ytu(:,:,m)';
% %         else
% %             Ytun =  Ytu(:,:,m);
% %         end
% %         zn(:,k) = Ytun * v{k};
% %
% %         YtuC = Ytun* Cni(:,:,k);
% %         Fnk(:,:,k) = YtuC* Ytun';
% %     end
% %     sn = Ytun*u2{k};
% %
% %     Csn = Cni(:,:,n) * sn;
% %
% %     alpha2(n) = 1/(u2{n}.'*v{n});
% %     beta2(n) = sn.' * v{n};
% %     gamma2(n) = sn.'*Csn;
% %
% %     fnum = (alpha2.'*beta2);
% %     temp3 = alpha2(n) * Csn;
% %
% %     gnum = 2*fnum*(2*v{n} * beta2(n) * (-alpha2(n)^2) + temp3  + zn*alpha2);
% %     fden = N + alpha2.'*beta2.^2 - sum(gamma2);
% %     gden = 2*(v{n} * beta2(n)^2 * (-alpha2(n)^2)+ temp3 * beta2(n) +  zn*diag(beta2)*alpha2 - sum(Fnk,3)*u2{n});
% %
% %     g(:,n) = v{n} *2 * (-alpha2(n)^2) - (gnum * fden - fnum^2 * gden)/fden^2;
% % end
% % % g = g;
% % f = sum(alpha2) - fnum^2/fden;
%
% end


function [a,U] = refine_factor3(Y,R,a,U)

% Basis of row spaces of Yn
N = ndims(Y);SzY = size(Y);
Za = cell(N,1);
for n = 1:N
    Yn = tenmat(Y,n);Yn = double(Yn);
    
    %[Fn,~] = qr(Yn',0);
    [Fn,E1,foe] = svd(Yn',0);
    %     Cn = Yn*Yn';Cn = max(Cn,Cn');
    %     [Fn,foe] = svd(Yn'*inv(Cn)*Yn,0);
    %     Fn = Fn(:,1:SzY(n));
    
    Zn = tensor(Fn',SzY([n 1:n-1 n+1:end]));
    Zn = ipermute(Zn,[n 1:n-1 n+1:N]);
    Za{n} = Zn;
end

ah = a;
cost_a = zeros(20,1);
for iter = 1:100
    
    for n = 1:N
        % Find a{1} as largest eigen vector of Xa
        Qn = 0;
        for m = [1:n-1 n+1:N]
            tnm = double(ttv(Za{m},ah,-[n m]));
            if n < m
                Qn = Qn + tnm*tnm';
            else
                Qn = Qn + tnm'*tnm;
            end
        end
        
        [ah{n},ea] = eigs(Qn,1);
    end
    
    % cost here will be not accurate for higher N.
    cost = 0;
    for n = 1:N
        bn = double(ttv(Za{n},ah,-n));
        cost = cost - bn'*bn;
    end
    %     b1 = t12*ah{3};%double(ttv(Ta1,ah,[2 3]));
    %     b2 = t21*ah{3};%double(ttv(Ta2,ah,[1 3]));
    %     b3 = t31'*ah{2};%double(ttv(Ta3,ah,[1 2]));
    
    cost_a(iter) = cost;
    if (iter>1) && ((abs(cost_a(iter)-cost_a(iter-1))<1e-5*abs(cost_a(iter-1))) || ...
            (abs(cost_a(iter)-cost_a(iter-1))<1e-5))
        break
    end
end

a = ah;

%% Rotate factors to be orthognal
[a,U] = rotate_factor(a,U);
end


%%
function [a,u] = refine_factor2(Y,R,a,u)
N = ndims(Y);
rho = cellfun(@(x,y) x'*y,a,u);

w = cell(N,1);C = cell(N,1);
for n = 1:N
    w{n} = (a{n} - u{n}*rho(n))/(1-rho(n)^2);
    Yn = tenmat(Y,n);
    C{n} = inv(double(Yn*Yn'));%C1 = max(C1,C1');
end

cost_a = zeros(20,1);
for iter = 1:200
    
    for n = 1:N
        
        lambda = 0;
        for m = 1:N
            temp = ttv(Y,a,-m);
            temp = double(temp)'*w{m};
            lambda = lambda + temp;
        end
        lambda = lambda/3;
        
        w{n} = lambda * C{n}*double(ttv(Y,a,-n));
        
        an = 0;
        for m = [1:n-1 n+1:N]
            temp = ttv(Y,a,-[n m]);
            temp = double(temp);
            if n < m
                temp = temp * w{m};
            else
                temp = w{m}'*temp;
            end
            an = an + temp(:);
        end
        an = an/lambda/(N-1);
        a{n} = an/norm(an);
        w{n} = w{n}/(w{n}'*a{n});
    end
    
    % Cost
    cost_new = 0;
    for n = 1:N
        temp = norm(ttv(Y,w,n) - full(ktensor(lambda,a([1:n-1 n+1:end]))))^2;
        cost_new = cost_new + temp;
    end
    cost_a(iter) = cost_new;
    if (iter>1) && ((abs(cost_a(iter)-cost_a(iter-1))<1e-8*abs(cost_a(iter-1))) || ...
            (abs(cost_a(iter)-cost_a(iter-1))<1e-5))
        break
    end
end

for n = 1:N
    rho(n) = sqrt(1 - 1/norm(w{n})^2);
    u{n} = (a{n} - w{n}* (1-rho(n)^2))/rho(n);
end

end


%%

function [a,U] = refine_factor(Y,R,a,U)
Y1 = double(tenmat(Y,1));
Y2 = double(tenmat(Y,2));
Y3 = double(tenmat(Y,3));

% [A1,E1] = qr(Y1',0);
% [A2,E2] = qr(Y2',0);
% [A3,E3] = qr(Y3',0);


% % Basis of row space of Yn;  This often gives better results
C1 = Y1*Y1';C1 = max(C1,C1');
C2 = Y2*Y2';C2 = max(C2,C2');
C3 = Y3*Y3';C3 = max(C3,C3');
[A1a,E1] = svd(Y1'*inv(C1)*Y1,0);
A1 = A1a(:,1:R); %A1c =  A1a(:,R+1:end);
[A2a,E2] = svd(Y2'*inv(C2)*Y2,0);
A2 = A2a(:,1:R);%A2c =  A2a(:,R+1:end);
[A3a,E3] = svd(Y3'*inv(C3)*Y3,0);
A3 = A3a(:,1:R);%A3c =  A3a(:,R+1:end);

% if size(Y1,2)<10000
% [A1a,E1] = svd(Y1'*inv(C1)*Y1,0);
% A1 = A1a(:,1:R); %A1c =  A1a(:,R+1:end);
% [A2a,E2] = svd(Y2'*inv(C2)*Y2,0);
% A2 = A2a(:,1:R);%A2c =  A2a(:,R+1:end);
% [A3a,E3] = svd(Y3'*inv(C3)*Y3,0);
% A3 = A3a(:,1:R);%A3c =  A3a(:,R+1:end);
% % else
% %     [cA1a,E1] = eigs(C1,R,'LA');
% %     A1 = (diag(1./sqrt(diag(E1)))*cA1a'*Y1)';
% %     [cA2a,E2] = eigs(C2,R,'LA');
% %     A2 = (diag(1./sqrt(diag(E2)))*cA2a'*Y2)';
% %     [cA3a,E3] = eigs(C3,R,'LA');
% %     A3 = (diag(1./sqrt(diag(E3)))*cA3a'*Y3)';
% % end

% % % Basis of row spaces of Yn
% [A1,E1,foe] = svd(Y1',0);
% [A2,E2,foe] = svd(Y2',0);
% [A3,E3,foe] = svd(Y3',0);

Ta1 = reshape(A1',R);
Ta2 = itenmat(A2',2,R);
Ta3 = itenmat(A3',3,R);

Ta1 = tensor(Ta1);
Ta2 = tensor(Ta2);
Ta3 = tensor(Ta3);

ah = a;

cost_a = zeros(20,1);
for iter = 1:20
    % Find a{1} as largest eigen vector of Xa
    t23 = double(ttv(tensor(Ta2),ah{3},3));
    Xa = t23*t23';
    t32 = double(ttv(tensor(Ta3),ah{2},2));
    Xa = Xa + t32*t32';
    
    [ah{1},ea] = eigs(Xa,1);
    
    % Find b{1} as largest eigen vector of Xb
    t13 = double(ttv(tensor(Ta1),ah{3},3));
    Xb = t13'*t13;
    t31 = double(ttv(tensor(Ta3),ah{1},1));
    Xb = Xb + t31*t31';
    
    [ah{2},eb] = eigs(Xb,1);
    
    % Find c{1} as largest eigen vector of Xc
    t12 = double(ttv(tensor(Ta1),ah{2},2));
    Xc = t12'*t12;
    t21 = double(ttv(tensor(Ta2),ah{1},1));
    Xc = Xc + t21'*t21;
    [ah{3},ec] = eigs(Xc,1);
    
    b1 = t12*ah{3};%double(ttv(Ta1,ah,[2 3]));
    b2 = t21*ah{3};%double(ttv(Ta2,ah,[1 3]));
    b3 = t31'*ah{2};%double(ttv(Ta3,ah,[1 2]));
    %e_ah = cellfun(@(x) x'*x,ah,'uni',1);
    %         cost(iter) = norm(Ta1)^2 - b1'*b1 +...
    %             norm(Ta2)^2 - (b2'*b2) +...
    %             norm(Ta3)^2 - (b3'*b3) ;
    cost_a(iter) =  - (b1'*b1 + (b2'*b2) + (b3'*b3));
    if (iter>1) && ((abs(cost_a(iter)-cost_a(iter-1))<1e-8*abs(cost_a(iter-1))) || ...
            (abs(cost_a(iter)-cost_a(iter-1))<1e-5))
        break
    end
end

a = ah;

%% Rotate factors to be orthognal
if nargin>=4
    [a,U] = rotate_factor(a,U);
else
    U = [];
end
end

%% Factor initialization

function [a,U] = init_factor(Y,R,param)
ni = numel(param.init);
I = size(Y);
N = numel(I);

if ischar(param.init) && strcmp(param.init(1:min(ni,4)),'rand')
    a = cell(N,1);U = cell(N,1);
    for n = 1:N
        a{n} = randn(I(n),1); a{n} = a{n}/norm(a{n});
        U{n} = randn(I(n),R(n)-1);
    end
elseif ischar(param.init) && strcmp(param.init(1:min(ni,5)),'tdiag')
    init = 'cp_init';
    % ts = tic;
    [A0 B0 C0 S]=tediaWN(double(Y),100,init,'normal');
    % tinit = toc(ts);
    
    %     [A0, B0, C0]=tedia4RS(double(Y),100);
    
    aU = [mat2cell(A0,I(1),[R(1)-1 1]);
        mat2cell(B0,I(2),[R(2)-1 1]);
        mat2cell(C0,I(3),[R(3)-1 1])];
    
    a = aU(:,2);
    U = aU(:,1);
    
    
elseif ischar(param.init) && strcmp(param.init(1:min(ni,4)),'dtld')
    opts_init.init = {'nvecs' 'dtld'};
    V = cp_init(tensor(Y),max(R),opts_init);
    if any(isnan(cell2mat(V(:))))
        fprintf('DTLD returns nan.\n')
    end
    Pi = ktensor(V); Pi = arrange(Pi);V = Pi.u;
    a = cell(N,1);U = cell(N,1);
    for n = 1:N
        %a{n} = V{n}(:,1);U{n} = V{n}(:,2:end);
        a{n} = V{n}(:,end);U{n} = V{n}(:,1:end-1);
    end
    [a,U] = init_preselect(Y,R,a,U,param.reference);
    
    %     for n = 1:N
    %         %a{n} = V{n}(:,1);U{n} = V{n}(:,2:end);
    %         a{n} = V{n}(:,end);U{n} = V{n}(:,1:end-1);
    %     end
    %     aU = [mat2cell(V{1},I(1),[1 R-1]);
    %         mat2cell(V{2},I(2),[1 R-1]);
    %         mat2cell(V{3},I(3),[1 R-1])];
    %
    %     a = aU(:,1);
    %     U = aU(:,2);
    
    
elseif ischar(param.init) && (strcmp(param.init(1:min(ni,5)),'nvec2') || strcmp(param.init(1:min(ni,5)),'nveca'))
    opts_init = struct('maxiters',3,'init','nvec');
    [T,fitarr] = mtucker_als(Y,R,opts_init);
    
    pairselection= false;%pairselection= false;
    if pairselection
        Uh = T.u;
        for r1 = 1:R
            for r2 = [1:r1-1 r1+1:R]
                
                w = cellfun(@(x) x(:,r1),Uh,'uni',0);
                u = cellfun(@(x) x(:,r2),Uh,'uni',0);
                rho = rand(1,N);
                ah = cell(N,1);
                for n = 1:N
                    ah{n} = w{n} *sqrt(1-rho(n)^2) + u{n}*rho(n);
                end
                
                Ytu = double(ttv(Y,u));
                noiters = 10;
                cost = zeros(noiters,1);
                % Estimate rho
                for krho = 1:noiters
                    for n = 1:N
                        sn = double(ttv(Y,ah,-n));
                        
                        x = [w{n} u{n}]'*double(sn);
                        rho_ne = prod(rho([1:n-1 n+1:N]));
                        if abs(x(1))<1e-7 || abs(1-abs(rho_ne))< 1e-6
                            rho(n) = 1;
                        else
                            t = (x(2)-rho_ne*Ytu)/(x(1)*(1-rho_ne^2));
                            rho(n) = (t)./sqrt(1+t.^2);
                        end
                        
                        ah{n} = w{n}*sqrt(1-rho(n)^2) + u{n}*rho(n);
                    end
                    s = ah{n}'*sn;
                    rho_all = prod(rho);
                    cost(krho) = -(s-rho_all * Ytu)^2/(1-rho_all^2);
                end
                
                cost_rs(r1,r2) = cost(end);
            end
        end
        [foe,ix] = min(cost_rs(:));
        [r1,r2] = ind2sub(size(cost_rs),ix);
        
        ind = [r1 r2 setdiff(1:R,[r1 r2])];
    else
        ind = 1:R;
        %ind = R:-1:1;
    end
    %     aU = [mat2cell(T.U{1}(:,ind),I(1),[1 R-1])
    %         mat2cell(T.U{2}(:,ind),I(2),[1 R-1])
    %         mat2cell(T.U{3}(:,ind),I(3),[1 R-1])];
    %
    %     w = aU(:,1);a = w;
    %     U = aU(:,2);
    
    a = cell(N,1);U = cell(N,1);
    for n = 1:N
        a{n} = T.u{n}(:,ind(1));
        U{n} = T.u{n}(:,ind(2:end));
    end
    
    %     % The following steps estimates rho_n, and are equipvalent to using
    %     % the "finda_givenU2" routine.
    %     w = cell(N,1);U = cell(N,1);
    %     for n = 1:N
    %         w{n} = T.u{n}(:,ind(1));
    %         U{n} = T.u{n}(:,ind(2:end));
    %     end
    %
    %     u = cellfun(@(x) x(:,1),U,'uni',0);
    %     rho = rand(1,N);
    %     a = w;
    %     for n = 1:N
    %         a{n} = w{n}*sqrt(1-rho(n)^2) + u{n}*rho(n);
    %     end
    %
    %     Ytu = double(ttv(Y,u));
    %     % Estimate rho
    %     tol_angle = 1e-5;
    %     noiters = 10;
    %     for krho = 1:noiters
    %         for n = 1:N
    %             sn = double(ttv(Y,a,-n));
    %
    %             x = [w{n} u{n}]'*double(sn);
    %             rho_ne = prod(rho([1:n-1 n+1:N]));
    %             if abs(x(1))<tol_angle || (abs(1-rho_ne)<tol_angle)
    %                 rho(n) = 1;
    %             else
    %                 t = (x(2)-rho_ne*Ytu)/(x(1)*(1-rho_ne^2));
    %                 rho(n) = (t)./sqrt(1+t.^2);
    %             end
    %
    %             a{n} = w{n}*sqrt(1-rho(n)^2) + u{n}*rho(n);
    %         end
    %     end
    
elseif ischar(param.init) && strcmp(param.init(1:min(ni,4)),'nvec')
    opts_init = struct('maxiters',3,'init','nvec');
    [T,fitarr] = mtucker_als(Y,R,opts_init);
    % Fix U and find an
    U0 = T.U;
    %     U0 = U;
    costri = zeros(R(1),1);
    
    
    exec_func = str2func(mfilename);
    opts = exec_func();
    opts.maxiters = 10;
    opts.tol = 1e-8;
    opts.refine_a = false;
    opts.alsinit = false;
    opts.printitn = 0;
    opts.pre_select = false;
    opts.verify_rank = false;
    opts.correct_badinitial = false;
    %curr_cost = [];
    bestcost = [];
    ri = 1;
    for ri = 1:R(1)
        w = cellfun(@(x) x(:,ri), U0,'uni',0);
        U = cellfun(@(x) x(:,[1:ri-1 ri+1:end]), U0,'uni',0);
        [a,rho,w,U] = finda_givenU2(Y,U,w);
        
        
        opts.init = [a U]; %;%'nvec';%
        
        try
            [a,U,lambda,cost] = exec_func(Y,R,opts);
            costri(ri) = cost(end);
            
            if isempty(bestcost)
                [abest,Ubest,ribest] = deal(a,U,ri);
                bestcost = costri(ri);
            elseif costri(ri) < bestcost
                [abest,Ubest,ribest] = deal(a,U,ri);
                bestcost = costri(ri);
            end
            
        catch me;
        end
        
        
        %         % cost
        %         s = ttv(Y,a);rhoall = prod(rho);
        %         z = ttv(Y,cellfun(@(x) x(:,1), U,'uni',0));
        %         lambda = (s - rhoall*z)/(1-rhoall^2);
        %         G = ttm(Y,U,'t'); G(1) = G(1) - lambda*rhoall;
        %
        %         costri(ri) = norm(Y - full(ktensor(lambda,a)) - full(ttensor(tensor(G),U)));
        %
        %         if ri == 1
        %             [abest,Ubest,ribest] = deal(a,U,ri);
        %             bestcost = costri(ri);
        %         elseif costri(ri) < bestcost
        %             [abest,Ubest,ribest] = deal(a,U,ri);
        %             bestcost = costri(ri);
        %         end
    end
    [a,U,ri] = deal(abest,Ubest,ribest);
    
    
elseif ischar(param.init) && strcmp(param.init(1:min(ni,4)),'osvd')
    w = cell(N,1);Cn = zeros(size(Y,1),size(Y,1),N);
    V = cell(N,1);
    u = cell(N,1);
    for n = 1:N
        Yn = double(tenmat(Y,n));
        Cn(:,:,n) = Yn*Yn';
        [V{n},e] = eig(Cn(:,:,n));
        [e,id] = sort(diag(e),'descend');
        V{n} = V{n}(:,id);
        
        % dimensionality selection
        % convexfull
        %         fite = cumsum(e)/sum(e);
        %         outfit = [-ones(R,2) (1:R)',fite(:)];
        %         rs= DimSelectorCP(outfit);
        %         Hullrank = 2;
        % RAE
        rae = log(e(1:end-1)./e(2:end));
        [foe,rs] = max(rae);
        w{n} = V{n}(:,rs);
    end
    
    
    for n = 1:N
        dn = double(ttv(Y,w,-n));
        u{n} = dn - w{n} * (w{n}'*dn);
        u{n} = u{n}/norm(u{n});
    end
    
    a = w;
    U = cell(N,1);
    for n = 1:N
        [U{n},rr] = qr([w{n} u{n}]);
        U{n} = [u{n} U{n}(:,3:end)];
    end
    
    % Preselection process
    exec_func = str2func(mfilename);
    
    % Find a good initial point for ah
    sopts = exec_func();
    sopts.maxiters = 10;
    sopts.tol = 1e-8;
    sopts.refine_a = false;
    sopts.alsinit = false;
    sopts.printitn = 0;
    sopts.pre_select = false;
    sopts.verify_rank = false;
    
    for r = R:-1:1
        w = cellfun(@(x) x(:,r),V,'uni',0);
        for n = 1:N
            dn = double(ttv(Y,w,-n));
            u{n} = dn - w{n} * (w{n}'*dn);u{n} = u{n}/norm(u{n});
        end
        a = w;
        %[a,rho,w,u] = finda_givenU2(Y,u,w);
        U = findU_givena_exact(Y,a);
        
        U = cell(N,1);
        for n = 1:N
            [U{n},rr] = qr([w{n} u{n}]);
            U{n} = [u{n} U{n}(:,3:end)];
        end
        
        sopts.init = [a U]; %;%'nvec';%
        [a,U,lambda,costr] = exec_func(Y,R,sopts);
        cost(r) = costr(end);
        
        if r == R
            bestcost = cost(r);
            ab = a;Ub = U;
        elseif bestcost > cost(r);
            ab = a;Ub = U;
            bestcost = cost(r);
        end
        %[a,rho,w,U] = finda_givenU2(Y,U,w);
        u = cellfun(@(x) x(:,1),U,'uni',0);
    end
    [foe,ri] = min(cost);
    a = ab; U = Ub;
    
    
    %     bestcost = [];
    %     for r1 = R:-1:1
    %         w{1} = V{1}(:,r1);
    %         for r2 = R:-1:1
    %             w{2} = V{2}(:,r2);
    %             for r3 = R:-1:1
    %                 w{3} = V{3}(:,r1);
    %
    %                 %w = cellfun(@(x) x(:,r),V,'uni',0);
    %                 for n = 1:N
    %                     dn = double(ttv(Y,w,-n));
    %                     u{n} = dn - w{n} * (w{n}'*dn);u{n} = u{n}/norm(u{n});
    %                 end
    %                 a = w;
    %                 %[a,rho,w,u] = finda_givenU2(Y,u,w);
    %
    %                 U = cell(N,1);
    %                 for n = 1:N
    %                     [U{n},rr] = qr([w{n} u{n}]);
    %                     U{n} = [u{n} U{n}(:,3:end)];
    %                 end
    %
    %                 sopts.init = [a U]; %;%'nvec';%
    %                 [a,U,lambda,costr] = exec_func(Y,R,sopts);
    %                 newcost = costr(end);
    %
    %                 if isempty(bestcost)
    %                     bestcost = newcost;
    %                     ab = a;Ub = U;
    %                 elseif bestcost > newcost;
    %                     ab = a;Ub = U;
    %                     bestcost = newcost;
    %                 end
    %                 %[a,rho,w,U] = finda_givenU2(Y,U,w);
    %                 u = cellfun(@(x) x(:,1),U,'uni',0);
    %
    %                 cost(r1,r2,r3) = newcost;
    %             end
    %         end
    %     end
    %     [foe,ri] = min(cost(:));
    %     a = ab; U = Ub;
    
elseif ischar(param.init) && strcmp(param.init(1:min(ni,4)),'ceig') % direct rank-1 decomposition
    % Initialization based on (approximate) common eigenvalues
    [a,U] = rk1ext_ceig(Y,'common_eig');
    U = findU_givena_exact(Y,a);
elseif ischar(param.init) && strcmp(param.init(1:min(ni,4)),'peig') % direct rank-1 decomposition
    % Initialization based on matching pairs of eigenvectors
    [a,U] = rk1ext_ceig(Y,'pair_eig');
    
elseif ischar(param.init) && strcmp(param.init(1:min(ni,6)),'lambda') % lambda_constrained
    % a new initialization method which constrains the weight lambda not to
    % be high by imposing additional constraints 
    %     <Yn , Ya > = 1 for n = 1, ..., N
    % where Yn = Y x_n inv(Yn*Yn')
    
    [a,U] = lambda_constrained(Y);
    
elseif iscell(param.init)
    if (isequal(size(param.init),[N 2])) && all(all(cellfun(@isnumeric,param.init))) % pre-defined initialization
        a = param.init(:,1);
        U = param.init(:,2);
        
        if any(cellfun(@isempty,a))
            [a,rho,w,U] = finda_givenU(Y,U);
        end
        
        if any(cellfun(@isempty,U))
            U = findU_givena(Y,a);
        end
        
    else % small iteratons to find the best initialization
        exec_func = str2func(mfilename);
        Pbest = [];besterror = inf;
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || (ischar(initk)  && ismember(initk(1:4), ...
                    {'rand' 'nvec' 'fibe' 'orth' 'dtld' 'tdia' 'ceig' 'peig' 'osvd'}))  % multi-initialization
                if ischar(initk)
                    cprintf('blue','Init. %d - %s\n',ki,initk)
                else
                    cprintf('blue','Init. %d - %s\n',ki,class(initk))
                end
                
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                
                if ischar(initk)  && ismember(initk(1:4),{'nvec' 'dtld' 'tdia'})
                    initparam.pre_select = 1;
                else
                    initparam.pre_select = 0;
                end
                
                try
                    [a,U,lambda,cost] = exec_func(Y,R,initparam);
                    % If reference is given, angular errors between two subspaces are
                    % used to select the similar initial point, instead of cost
                    % function
                    if ~isempty(param.reference)
                        Uref = param.reference(:,2);
                        aref = param.reference(:,1);
                        sae_aU = nan(N,2);
                        for m = 1:N
                            if ~isempty(Uref{m})
                                sae_aU(m,1) = subspace(Uref{m},U{m});
                                sae_aU(m,2) = 1-real(aref{m}'*a{m}/norm(aref{m})/norm(a{m}))^2;
                            end
                        end
                        angle_error = nanmean(nanmean(sae_aU,2));
                        if angle_error < besterror
                            Pbest = {a U};
                            besterror = angle_error;kibest = ki;
                        end
                    else
                        %fitinit = 1-cost(end);
                        if real(cost(end)) < besterror
                            Pbest = {a U};
                            besterror = cost(end);kibest = ki;
                        end
                    end
                catch
                    continue
                end
            end
        end
        cprintf('blue','Choose the best initial value: %d.\n',kibest);
        a = Pbest{1};
        U = Pbest{2};
    end
end

end

%%
function [a,U] = init_preselect(Y,R,a,U,ref)

if (nargin < 5) || isempty(ref)
    ref = [];
    aref = [];
    Uref = [];
else
    aref = ref(:,1);
    if size(ref,2)>1
        Uref = ref(:,2);
    else
        Uref  = [];
    end
end

exec_func = str2func(mfilename);

% Find a good initial point for ah
Ai = cellfun(@(x,y) ([x y]),a,U,'uni',0);
N = ndims(Y);
opts = exec_func();
opts.maxiters = 10;
opts.tol = 1e-8;
opts.refine_a = false;
opts.alsinit = false;
opts.printitn = 0;
opts.pre_select = false;
% cost_i = zeros(R,1);
A_init = cell(N,2);
R = min(R);
curr_cost = [];
for ri = 1:R
    permind = [ri 1:ri-1 ri+1:R];
    
    for n = 1:N
        A_init(n,:) = mat2cell(Ai{n}(:,permind),size(Ai{n},1),[1 size(Ai{n},2)-1]);
    end
    opts.init = A_init; %;%'nvec';%
    
    try
        [ah,Uh,lambda,cost] = exec_func(Y,R,opts);
        
        % If reference is given, angular errors between two subspaces are
        % used to select the similar initial point, instead of cost
        % function
        if ~isempty(Uref) && ~isempty(aref)
            sae_aU = nan(N,2);
            for m = 1:N
                if ~isempty(Uref{m})
                    sae_aU(m,1) = subspace(Uref{m},Uh{m});
                    sae_aU(m,2) = 1-abs(aref{m}'*ah{m}/norm(aref{m})/norm(ah{m}))^2;
                end
            end
            cost = nanmean(nanmean(sae_aU,2)); %angular_error
            
        elseif ~isempty(aref)
            sae_a = nan(N,1);
            for m = 1:N
                if ~isempty(aref{m})
                    sae_a(m) = 1-abs(aref{m}'*ah{m}/norm(aref{m})/norm(ah{m}))^2;
                end
            end
            cost = nanmean(sae_a); %angular_error
        end
        
        if isempty(curr_cost)
            curr_cost = cost(end);
            a = ah; U = Uh;
        elseif curr_cost > cost(end)
            curr_cost = cost(end);
            a = ah; U = Uh;
        end
    catch me;
        continue;
    end
end
% [mcost,ri] = min(cost_i);
% permind = [ri 1:ri-1 ri+1:R];
%
% for n = 1:N
%     A_init(n,:) = mat2cell(Ai{n}(:,permind),size(Ai{n},1),[1 size(Ai{n},2)-1]);
% end
% a = A_init(:,1);
% U = A_init(:,2);
end

%%
function [a,U] = als_init(Y,R,a,U,param)
%% ALS init

alsopts = param;
alsopts.updateterm = false;
alsopts.updatefactor = true;
alsopts.correctfactor = false;
alsopts.linesearch = 'none';

u = cellfun(@(x) x(:,1),U,'uni',0);
rho = cellfun(@(x,y) x'*y(:,1),a,U);
rhoall = prod(rho);
lambda = (ttv(Y,a) - rhoall*ttv(Y,u))/(1-rhoall^2);

G = ttm(Y,U,'t');G = G.data;
G(1) = G(1) - lambda *rhoall;
G = tensor(G);

Xp{1} = ktensor(lambda,a);
if size(U{1},2)> 1
    Xp{2} = ttensor(G,U);
else
    Xp{2} = ktensor(double(G),U);
end
alsopts.init = Xp;
alsopts.maxiters = 3;
N = numel(a);
if numel(R) == 1
    Rb = [ones(1,N);(R-1)*ones(1,N)];
else
    Rb = [ones(1,N);(R-1)];
end
Xp = b2d_rR(Y,Rb,alsopts);
a = Xp{1}.u;
U = Xp{2}.u;

%% Rotate factors to be orthognal
[a,U] = rotate_factor(a,U);
end


%% Estimate U when a is given

function U = findU_givena(Y,a)

%% FInd Un with given a
N = ndims(Y);

U = cell(N,1);

Cm = cell(N,1);Tm = cell(N,1);
p = zeros(1,5);
for km = 1:N
    
    Ym = double(tenmat(Y,km));
    C = Ym*Ym';
    a1 = a{km}; a2 = double(ttv(Y,a,-km));
    T = [a1 a2];
    Q = (T'/(C))*T;
    %Qi = inv(Q);
    
    pk = [Q(1,1) - det(Q), -2*Q(1,2), 1];
    %p = p + (det(C))^2*conv(pk,pk);
    p = p + conv(pk,pk);
    %p = p + det(C)*det(Q)*[Qi(2,2) - 1, 2*Qi(1,2), - Qi(1,2)^2 + Qi(1,1)*Qi(2,2)];
    Cm{km} = C;
    Tm{km} = T;
end
alpha1 = roots(polyder(p));
alpha1 = max(real(alpha1));

for mode1 = 1:N
    Yd = Cm{mode1} + Tm{mode1}*[alpha1^2 -alpha1; -alpha1 0]*Tm{mode1}';
    
    a1 = a{mode1};
    [U1,s] = eig(Yd);
    s = diag(s); [s,is] = sort(s,'descend');
    U1 = U1(:,is(1:end-1));
    
    d = a1'*U1;[u1,s1,v1] = svd(d,0);
    U1 = U1*v1';
    
    U{mode1} = U1;
end
end



%% This function is to find rho_n with given Un for n = 1,2,..., N
function [f,g] = rho_givenU(a,y)
% a: 2 x N
% a(1,n) = xi_n = a_n^T * w_n and a(2,n) = rho_n = a_n^T * u_n

u = a(:,1); v = a(:,2); w = a(:,3);

rho_all = u(2)*v(2)*w(2);
yt = kron(kron(w,v),u);
yt(end) = [];

if abs(1 - abs(rho_all)) < 1e-5
    f = -0.5;
    g = zeros(2,3);
else
    f = -((y.'*yt)^2+eps)/(eps+1-rho_all^2)/2;
    G = @(u1,u2,v1,v2,w1,w2,y1,y2,y3,y4,y5,y6,y7)reshape([-((v1.*w1.*y1+v2.*w1.*y3+v1.*w2.*y5+v2.*w2.*y7).*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7))./(u2.^2.*v2.^2.*w2.^2-1.0),-((v1.*w1.*y2+v2.*w1.*y4+v1.*w2.*y6).*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7))./(u2.^2.*v2.^2.*w2.^2-1.0)+u2.*v2.^2.*w2.^2.*1.0./(u2.^2.*v2.^2.*w2.^2-1.0).^2.*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7).^2,-((u1.*w1.*y1+u2.*w1.*y2+u1.*w2.*y5+u2.*w2.*y6).*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7))./(u2.^2.*v2.^2.*w2.^2-1.0),-((u1.*w1.*y3+u2.*w1.*y4+u1.*w2.*y7).*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7))./(u2.^2.*v2.^2.*w2.^2-1.0)+u2.^2.*v2.*w2.^2.*1.0./(u2.^2.*v2.^2.*w2.^2-1.0).^2.*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7).^2,-((u1.*v1.*y1+u2.*v1.*y2+u1.*v2.*y3+u2.*v2.*y4).*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7))./(u2.^2.*v2.^2.*w2.^2-1.0),-((u1.*v1.*y5+u2.*v1.*y6+u1.*v2.*y7).*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7))./(u2.^2.*v2.^2.*w2.^2-1.0)+u2.^2.*v2.^2.*w2.*1.0./(u2.^2.*v2.^2.*w2.^2-1.0).^2.*(u1.*v1.*w1.*y1+u2.*v1.*w1.*y2+u1.*v2.*w1.*y3+u1.*v1.*w2.*y5+u2.*v2.*w1.*y4+u2.*v1.*w2.*y6+u1.*v2.*w2.*y7).^2],[2,3]);
    
    g = -G(u(1),u(2),v(1),v(2),w(1),w(2),y(1),y(2),y(3),y(4),y(5),y(6),y(7));
end


end


function [a,rho,w,U] = finda_givenU(Y,U,w)
N = ndims(Y);
if nargin < 3
    w = cell(N,1);
    for n = 1:N
        [wn,e] = qr(U{n});
        w{n} = wn(:,end);
    end
end

% Find new a{n} through new rho_n
% addpath('/Users/phananhhuy/Documents/Matlab/Ortho_opt/FOptM-share')

% Uniball parametres
uniopts.record = 0;
uniopts.mxitr  = 1000;
uniopts.gtol = 1e-9;
uniopts.xtol = 1e-9;
uniopts.ftol = 1e-8;
uniopts.tau = 1e-3;


V = cellfun(@(x, y) [x y(:,1)],w,U,'uni',0); % Vn = [wn un]

%VtU = cellfun(@(x,y) (x'*y),V,U,'uni',0);
%T = double(ttm(Y,V,'t') - ttm(ttm(Y,U,'t'),VtU));
T = ttm(Y,V,'t') - ttm(Y,cellfun(@(x) [zeros(size(x,1),1) x(:,1)],U,'uni',0),'t');
T = double(T);

% rhoi = [sqrt(1-rho'.^2);rho'];
rhoi = rand(2,3);
rhoi = bsxfun(@rdivide,rhoi,sqrt(sum(rhoi.^2)));
[rho_e, foe,out]= OptManiMulitBallGBB(rhoi, @rho_givenU, uniopts, T(1:end-1)');
rho = real(rho_e(2,:));

a = cell(N,1);
for n = 1:N
    a{n} = V{n}*rho_e(:,n);
    if rho_e(2,n)<0
        U{n}(:,1) = -U{n}(:,1);
        rho(n) = -rho(n);
    end
end
end



function [a,rho,w,U,cost] = finda_givenU2(Y,U,w)
% U is an array of Un which can comprise only un if wn are given.
% New version of the finda_givenU without using the Crank-Nicholson algrithm
N = ndims(Y);
if nargin < 3
    w = cell(N,1);
    for n = 1:N
        [wn,e] = qr(U{n});
        w{n} = wn(:,end);
    end
end

% Find new a{n} through new rho_n

V = cellfun(@(x, y) [x y(:,1)],w,U,'uni',0); % Vn = [wn un]

%VtU = cellfun(@(x,y) (x'*y),V,U,'uni',0);
%T = double(ttm(Y,V,'t') - ttm(ttm(Y,U,'t'),VtU));
T = ttm(Y,V,'t');% - ttm(Y,cellfun(@(x) [zeros(size(x,1),1) x(:,1)],U,'uni',0),'t');
T = double(T);T(end) = 0;

rho_e = rand(2,N);
rho_e = bsxfun(@rdivide,rho_e,sqrt(sum(rho_e.^2)));
mxitr = 200;
cost = zeros(mxitr,1);

for iter = 1:mxitr
    for n = 1:N
        Tn = T;
        for m = N:-1:n+1
            Tn = reshape(Tn,[],2);
            Tn = Tn * rho_e(:,m);
        end
        
        for m = 1:n-1
            Tn = reshape(Tn,2,[]);
            Tn = rho_e(:,m)'*Tn;
        end
        rho_ne = prod(rho_e(2,[1:n-1 n+1:end]));
        rho_e(2,n) = abs(Tn(2))/sqrt(Tn(2)^2 + Tn(1)^2*(1- rho_ne^2)^2);
        rho_e(1,n) = sqrt(1-rho_e(2,n)^2);
        
        % a{n} = V{n}*rho_e(:,n);
    end
    
    cost(iter) = -1/2*(Tn*rho_e(:,N))^2/(1-prod(rho_e(2,:))^2);
    if (iter > 1) && (abs(cost(iter)-cost(iter-1)) < 1e-6)
        break;
    end
end

rho = rho_e(2,:);
a = cell(N,1);
for n = 1:N
    a{n} = V{n}*rho_e(:,n);
    if rho_e(2,n)<0
        U{n}(:,1) = -U{n}(:,1);
        rho(n) = -rho(n);
    end
end
end



function cost = bcdLp1_cost(Y,a,u)
N = ndims(Y);
rho = cellfun(@(x,y) x'*y(:,1),a,u);
if (size(u{1},2)==1) && (size(u{1},2) < size(u{1},1)-1)
    U = cell(N,1);
    w = cell(N,1);
    for n = 1:N
        w{n} = (a{n}-rho(n)*u{n})/sqrt(1-rho(n)^2);
        [Un,rr] = qr([w{n} u{n}]); U{n} = [u{n} Un(:,3:end)];
    end
else
    U = u;
    u = cellfun(@(x) x(:,1),U,'uni',0);
end


s = ttv(Y,a); z = ttv(Y,u);
rhoall = prod(rho);
lambda = (s - rhoall*z)/(1-rhoall^2);

G = ttm(Y,U,'t');G(1) = G(1) - lambda * rhoall;
Yhat = full(ktensor(lambda,a)) + full(ttensor(G,U));
cost = norm(Y - Yhat);
end



function [f,g] = est_a_spherical_cnt(a,Y,Cni,mode,a0)
% f: cost function and
% g gradient of f w.r.t a1,a2,a3
% Zn = Y x_{n} wn
Sz = size(Y);N = 3;
d = Sz(1);
if nargin< 4
    mode = 1:N;
    u2 = mat2cell(a,d,ones(1,N));
else
    u2 = a0;
    u2{mode} = a;
end


alpha2 = zeros([N 1]);
beta2 = zeros([N 1]);
gamma2 = zeros([N 1]);


v = cell(N,1);
Csn = cell(N,1);
for n = 1:N
    sn = double(ttv(tensor(Y),u2,-n));
    
    v{n} = Cni(:,:,n) * u2{n};
    Csn{n} = Cni(:,:,n) * sn;
    
    %     alpha2(n) = (u2{n}.'*v{n});
    alpha2(n) = 1/(u2{n}.'*v{n});
    
    beta2(n) = sn.' * v{n};
    gamma2(n) = sn.'*Csn{n};
end

fnum = (alpha2.'*beta2);
fden = N + alpha2.'*beta2.^2 - sum(gamma2);
f = sum(alpha2) - fnum^2/fden;

zn = zeros(d,N);
Fnk = zeros(d,d,N);

% Gradient

g = zeros(d,numel(mode));
for kn = 1:numel(mode)
    n = mode(kn);
    
    zn(:,n) = 0;
    Fnk(:,:,n) = 0;
    for k = [1:n-1 n+1:N]
        Ytu = ttv(tensor(Y),u2,-[n k]);
        Ytu = double(Ytu);
        if k < n
            Ytu = Ytu';
        end
        YtuC = Ytu* Cni(:,:,k);
        zn(:,k) = YtuC * u2{k};
        Fnk(:,:,k) = YtuC* Ytu';
    end
    
    
    temp2 = v{n} * beta2(n);
    temp3 = alpha2(n) * Csn{n};
    
    %     gnum = 2*fnum*(2*temp2 + temp3  + zn*alpha2);
    %     gden = 2*(temp2* beta2(n) + temp3 * beta2(n) +  zn*diag(beta2)*alpha2 - sum(Fnk,3)*u2{n});
    %     g(:,n) = v{n} *2  - (gnum * fden - fnum^2 * gden)/fden^2;
    
    gnum = 2*fnum*(2*temp2 * (-alpha2(n)^2)+ temp3  + zn*alpha2);
    gden = 2*(temp2* beta2(n) * (-alpha2(n)^2)+ temp3 * beta2(n) +  zn*diag(beta2)*alpha2 - sum(Fnk,3)*u2{n});
    g(:,kn) = v{n} *2 * (-alpha2(n)^2) - (gnum * fden - fnum^2 * gden)/fden^2;
end

% f = -f;
% g = -g;

% Ytu = zeros(d,d,N);
% v = cell(N,1);
% for n = 1:N
%     Ytu(:,:,n) = double(ttv(Y,u2,n));
%     v{n} = Cni(:,:,n) * u2{n};
% end
%
% for n = 1:N
%     zn(:,n) = 0;
%     Fnk(:,:,n) = 0;
%     for k = [1:n-1 n+1:N]
%         m = setdiff(1:N,[n k]);
%         if k < n
%             Ytun =  Ytu(:,:,m)';
%         else
%             Ytun =  Ytu(:,:,m);
%         end
%         zn(:,k) = Ytun * v{k};
%
%         YtuC = Ytun* Cni(:,:,k);
%         Fnk(:,:,k) = YtuC* Ytun';
%     end
%     sn = Ytun*u2{k};
%
%     Csn = Cni(:,:,n) * sn;
%
%     alpha2(n) = 1/(u2{n}.'*v{n});
%     beta2(n) = sn.' * v{n};
%     gamma2(n) = sn.'*Csn;
%
%     fnum = (alpha2.'*beta2);
%     temp3 = alpha2(n) * Csn;
%
%     gnum = 2*fnum*(2*v{n} * beta2(n) * (-alpha2(n)^2) + temp3  + zn*alpha2);
%     fden = N + alpha2.'*beta2.^2 - sum(gamma2);
%     gden = 2*(v{n} * beta2(n)^2 * (-alpha2(n)^2)+ temp3 * beta2(n) +  zn*diag(beta2)*alpha2 - sum(Fnk,3)*u2{n});
%
%     g(:,n) = v{n} *2 * (-alpha2(n)^2) - (gnum * fden - fnum^2 * gden)/fden^2;
% end
% % g = g;
% f = sum(alpha2) - fnum^2/fden;

end



function U = findU_givena_exact(Y,a)
N = ndims(Y);
%Ci = zeros(size(Y,1),size(Y,1),N);%v = cell(N,1);
U = cell(N,1);
for n = 1:N
    sn = double(ttv(Y,a,-n));
    
    Yn = double(tenmat(Y,n));
    vn = (Yn*Yn')\sn;
    
    beta = a{n}'*vn;
    
    ldar = 1/beta;
    rho_n = sqrt(1 - 1/(ldar^2 * (vn'*vn)));
    
    wn = vn/norm(vn);
    
    un = (a{n} - wn * beta/norm(vn))/rho_n;
    [qq,rr] = qr([wn un]);
    U{n} = [un qq(:,3:end)];
end
end



%%
function [x, g, out]= OptManiMulitBallGBB(x, fun, opts, varargin)
%-------------------------------------------------------------------------
% Line search algorithm for optimization on manifold:
%
%   min f(X), s.t., ||X_i||_2 = 1, where X \in R^{n,p}
%       g(X) = grad f(X)
%   X = [X_1, X_2, ..., X_p]
%
%
%   each column of X lies on a unit sphere
% Input:
%           X --- ||X_i||_2 = 1, each column of X lies on a unit sphere
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptManiMulitBallGBB(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%
% Output:
%           x --- solution
%           g --- gradient of x
%         Out --- output information
%
% -------------------------------------
% For example, consider the maxcut SDP:
% X is n by n matrix
% max Tr(C*X), s.t., X_ii = 1, X psd
%
% low rank model is:
% X = V'*V, V = [V_1, ..., V_n], V is a p by n matrix
% max Tr(C*V'*V), s.t., ||V_i|| = 1,
%
% function [f, g] = maxcut_quad(V, C)
% g = 2*(V*C);
% f = sum(dot(g,V))/2;
% end
%
% [x, g, out]= OptManiMulitBallGBB(x0, @maxcut_quad, opts, C);
%
% -------------------------------------
%
% Reference:
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
% Author: Zaiwen Wen, Wotao Yin
%   Version 1.0 .... 2010/10
%-------------------------------------------------------------------------

%% Size information
% termination rule
if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end


if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
    if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
    end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
    if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
    end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
    if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
    end
else
    opts.gamma = 0.85;
end



if isfield(opts, 'tau')
    if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
    end
else
    opts.tau = 1e-3;
end


% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'M')
    opts.M = 10;
end

if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end


if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;
    end
else
    opts.mxitr = 1000;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end

%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
ftol = opts.ftol;
gtol = opts.gtol;
rho  = opts.rho;
M     = opts.M;
STPEPS = opts.STPEPS;
eta   = opts.eta;
gamma = opts.gamma;

record = opts.record;

nt = opts.nt;
crit = ones(nt, 3);

% normalize x so that ||x||_2 = 1
[n,p] = size(x); nrmx = dot(x,x,1);
if norm(nrmx - 1,'fro')>1e-8; x = bsxfun(@rdivide, x, sqrt(nrmx)); end;

%% Initial function value and gradient
% prepare for iterations
% tmp = cputime;
[f,g] = feval(fun, x, varargin{:});    out.nfe = 1;
xtg = dot(x,g,1);   gg = dot(g,g,1);
xx = dot(x,x,1);    xxgg = xx.*gg;
dtX = bsxfun(@times, xtg, x) - g;    nrmG = norm(dtX, 'fro');

Q = 1; Cval = f; tau = opts.tau;
%% Print iteration header if debug == 1
if (record >= 1)
    fprintf('----------- Gradient Method with Line search ----------- \n');
    fprintf('%4s \t %10s \t %10s \t  %10s \t %5s \t %9s \t %7s \n', 'Iter', 'tau', 'f(X)', 'nrmG', 'Exit', 'funcCount', 'ls-Iter');
    fprintf('%4d \t %3.2e \t %3.2e \t %5d \t %5d	\t %6d	\n', 0, 0, f, 0, 0, 0);
end

if record == 10; out.fvec = f; end

%% main iteration
for itr = 1 : opts.mxitr
    xp = x;     fp = f;     gp = g;   dtXP =  dtX;
    
    nls = 1; deriv = rho*nrmG^2;
    while 1
        % calculate g, f,
        tau2 = tau/2;     beta = (1+(tau2)^2*(-xtg.^2+xxgg));
        a1 = ((1+tau2*xtg).^2 -(tau2)^2*xxgg)./beta;
        a2 = -tau*xx./beta;
        x = bsxfun(@times, a1, xp) + bsxfun(@times, a2, gp);
        
        %if norm(dot(x,x,1) - 1) > 1e-6
        %    error('norm(x)~=1');
        %end
        
        %f = feval(fun, x, varargin{:});   out.nfe = out.nfe + 1;
        [f,g] = feval(fun, x, varargin{:});   out.nfe = out.nfe + 1;
        
        if f <= Cval - tau*deriv || nls >= 5
            break
        end
        tau = eta*tau;
        nls = nls+1;
    end
    
    if record == 10; out.fvec = [out.fvec; f]; end
    % evaluate the gradient at the new x
    %[f,g] = feval(fun, x, varargin{:});  %out.nfe = out.nfe + 1;
    
    xtg = dot(x,g,1);   gg = dot(g,g,1);
    xx = dot(x,x,1);    xxgg = xx.*gg;
    dtX = bsxfun(@times, xtg, x) - g;    nrmG = norm(dtX, 'fro');
    s = x - xp; XDiff = norm(s,'fro')/sqrt(n);
    FDiff = abs(fp-f)/(abs(fp)+1);
    
    if (record >= 1)
        fprintf('%4d \t %3.2e \t %7.6e \t %3.2e \t %3.2e \t %3.2e \t %2d\n', ...
            itr, tau, f, nrmG, XDiff, FDiff, nls);
    end
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    
    %if ((XDiff < xtol) || (nrmG < gtol) ) %|| abs((fp-f)/max(abs([fp,f,1]))) < 1e-20;
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
        out.msg = 'converge';
        break;
    end
    
    %y = g - gp;
    y = dtX - dtXP;
    sy = sum(sum(s.*y));    tau = opts.tau;
    sy = abs(sy);
    if sy > 0;
        %tau = sum(sum(s.*s))/sy;
        %tau = sy/sum(sum(y.*y));
        %tau = sum(sum(s.*s))/sy + sy/sum(sum(y.*y));
        if mod(itr,2)==0; tau = sum(sum(s.*s))/sy;
        else tau = sy/sum(sum(y.*y)); end
        
        % safeguarding on tau
        tau = max(min(tau, 1e20), 1e-20);
    end
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
end

if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(dot(x,x,1)-1);
if out.feasi > 1e-14
    nrmx = dot(x,x,1);      x = bsxfun(@rdivide, x, sqrt(nrmx));
    [f,g] = feval(fun, x, varargin{:});   out.nfe = out.nfe + 1;
    out.feasi = norm(dot(x,x,1)-1);
end

out.nrmG = nrmG;
out.fval = f;
out.itr = itr;



end


%% Mar 26, 2015

function [a,U] = lambda_constrained(Y,a)
% a new initialization method which constrains the weight lambda not to
% be high by imposing additional constraints
%     <Yn , Ya > = 1 for n = 1, ..., N
% where Yn = Y x_n inv(Yn*Yn')
%  
% THe method leads to solving the constrained QP
%  min at'*at - 2 * at^T * z     
%  subject    Z'* at = 1
%  Z = [z1 ...zN],  where zn = Yn xn {inv(Cn)}  x_{-n} {a_{k #n}}
%  z = Y x_[-n] a
%
% April 2015, Phan Anh-Huy

%% Initialize a
N= ndims(Y);
R = size(Y,1);
if nargin < 2
    a = cell(N,1);
end
maxiters = 100;
tol = 1e-6;
normY2  = norm(Y)^2;
%%
Yn = cell(N,1);
for n = 1:N
    Ym = double(tenmat(Y,n));
    
    Cn = double(Ym*Ym');
    
    % Initialize a if not given
    if nargin < 2
        [a{n},foe] = svds(Cn,1,'L');
        %[a{n},foe] = eigs(Cn,1,'LM');
    end
    
    Yn{n} = ttm(Y,inv(Cn),n);
end

cp_opts = cp_fLMa;
cp_opts.init = a;
P = cp_fLMa(Y,1,cp_opts);
a = P.u;

%%
costinit = zeros(maxiters,1);
 
for iter = 1:maxiters
    % estimate a    
    for n = 1:N
        z0 = double(ttv(Y,a,-n));
        
        Z = cell2mat(cellfun(@(x) double(ttv(x,a,-n)),Yn,'uni',0)');
        K = rank(Z'*Z);
        Z = Z(:,1:K);
        
        a{n} = z0 - Z*((Z'*Z)\(Z'*z0 - 1));
        lda = norm(a{n});
        a{n} = a{n}/lda;
    end
    
    cost = normY2 + lda^2 - 2 * lda * a{n}'*z0;
     
    costinit(iter) = cost;
    
    if iter > 1
        if abs(costinit(iter) - costinit(iter-1)) < tol
%             fita = true;
            break
        end
    end
end
% costinit = costinit(1:iter);
% cellfun(@(x) ttv(x,a)*lda - 1,Yn)

%%
w = cell(N,1);
u = cell(N,1);U = cell(N,1);
rho_n = zeros(N,1);
for n = 1:N
    w{n} = double(ttv(Yn{n},a,-n));
    norm_wn = norm(w{n});
    w{n} = w{n}/norm_wn;
    rho_n(n) = sqrt(1 - 1/(lda^2 *norm_wn^2));
    u{n} = (a{n} - sqrt(1-rho_n(n)^2) * w{n})/rho_n(n);
    [Un,foe] = qr([w{n} u{n}]);
    U{n} = [u{n} Un(:,3:end)];
end
end