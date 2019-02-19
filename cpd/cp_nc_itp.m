function [P,output] = cp_nc_itp(X,R,delta,opts)
% IPT or SQP or trust-region-reflective algorithm for the CPD with a target error bound
%
%  min     sum_n lambda_n^2
%  subjec to  |X - P|_F < delta
%
% wheere Un are normalized to have unit norm
%
%  P is a ktensor of [lambda, U1, ..., UN]
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
%   opts = cp_fastals;
%   opts.init = {'nvec' 'random' 'random'};
%   [P,output] = cp_fastals(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.

%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

N = ndims(X); szX = size(X);

if isempty(param.normX)
    normX = norm(X);
else
    normX = param.normX;
end
delta2 = delta^2;

bound_ell2_constraint = param.bound_ell2_constraint;

ineqnonlin = param.ineqnonlin;
% true for the inequality contraint  |X - X_hat(\theta)|_F <= delta
% false for the equality contraint  |X - X_hat(\theta)|_F = delta

Correct_Hessian = param.Correct_Hessian;

%% Initialize factors U
param.cp_func = @cp_fastals;%str2func(mfilename);
Uinit = cp_init(X,R,param); U = Uinit;

%% Output
if param.printitn ~=0
    fprintf('\n ITP for EPC\n');
end
[U,gamma] = normalizeU(U);
% initial point
x = cell2mat(cellfun(@(x) x(:),U,'uni',0));

% upper bound of |theta|_2^2
if bound_ell2_constraint
    xi = N*(R*norm(gamma))^(2/N); % used for the constraint |\theta\|_2^2 <= xi
end

% map a vectorization to factor matrices
vec2mat = @(x) cellfun(@(x) reshape(x,[],R),mat2cell(x,szX*R),'uni',0);


% default parameters
% different fmincon routines may have different parameters
try
    options = optimoptions('fmincon','Display','iter',...
        ...%'CheckGradients',true,...
        'FiniteDifferenceType','central',...
        'SpecifyObjectiveGradient',true,...
        'SpecifyConstraintGradient',true',... %
        'HessianFcn',@compact_hessian_fc,...
        'MaxFunctionEvaluations',sum(szX)*R*1000,...
        ...%'Algorithm','sqp',...
        'MaxIterations',min(param.maxiters,sum(szX)*R*100),...
        'OutputFcn',@outfun,...
        'ConstraintTolerance',param.tol,...
        'OptimalityTolerance',param.tol,...
        'StepTolerance',param.tol);
catch
    options = optimoptions('fmincon','Display','iter',...
        ...%'CheckGradients',true,...
        'FinDiffType','central',...
        'GradObj','on',...
        'GradConstr','on',... %
        'HessFcn',@compact_hessian_fc,...
        'MaxFunEvals',sum(szX)*R*1000,...
        'MaxIter',min(param.maxiters,sum(szX)*R*100),...
        'OutputFcn',@outfun,...
        'ConstraintTolerance',param.tol,...
        'OptimalityTolerance',param.tol,...
        'StepTolerance',param.tol);
end

if param.history == false
    options.OutputFcn = [];
end

% options.HessFcn = [];

switch param.Hessian
    case 'approximation'
        try
            options.HessianFcn = [];
        catch
            options.HessFun = [];
        end
    otherwise
        try
            options.HessianFcn = @compact_hessian_fc;%@hessian_ff;
            %options.HessianFcn = @compact_hessian_fc;
        catch
            options.HessFun = @compact_hessian_fc;%@hessian_ff;
        end
        %options.HessFun = @compact_hessian_fc;
end

switch param.Algorithm
    case {'SQP' 'sqp'}
        options.Algorithm = 'sqp';
        Correct_Hessian = true;
end


% for relatively large scale problem
no_params = R*sum(szX);
if no_params>500
    %options.HessianMultiplyFcn = @hessianmultiply_ff;
    Correct_Hessian = false;
    options.Algorithm = 'interior-point';        
    options.HessianMultiplyFcn = @hessianmultiply_fc; % new routine to compute Hessian x v (vector)
    options.SubproblemAlgorithm = 'cg';
    try
        options.HessianFcn = [];
    catch
        options.HessFun = [];
    end
end
%'SubproblemAlgorithm','cg','HessianMultiplyFcn',@hessianmultiply_fc,'HessianFcn',[],...


% Set up shared variables with OUTFUN
history.fval = [];
history.constrviolation = [];
history.firstorderopt=[];
history.gf=[];
history.gc=[];
history.H=[];
history.lambda=[];
history.c =[];
history.x =[];
history.ss =[];
history.stepsize = [];

Prr = per_vectrans(R,R);

%% checked Hessian
chech_Hessian = false;
if chech_Hessian
    lambda = struct('ineqnonlin',1);
    Correct_Hessian = false;
    v = randn(size(x));
    H = compact_hessian_fc(v,lambda);
    % H2 = hessian_ff(x,lambda);
    U = vec2mat(v);
    U = normalizeU(U);
    [D,Z1,K,Ho] = cp_hessian_obj(U);
    [G,Z,K0,V,F] = cp_hessian(U);
    Hc = G + Z*K0*Z' + V*F*V';
    norm(H-Ho-Hc,'fro')
    
    %% checked Hessian x v
    lambda = struct('ineqnonlin',1);
    Correct_Hessian = false;
    % compute the Hessian explicit and multiply it with v
    H = compact_hessian_fc(x,lambda);
    t1 = H*v;
    %fast computation H x v
    %t2 = hessianmultiply_ff(x,lambda,v);
    t2 = hessianmultiply_fc(x,lambda,v);
    norm(H*v-t2)
end
%%
[x,fval,exitflag] = fmincon(@cost,x,[],[],[],[],[],[],@constraints,options);

U = vec2mat(x);
P = ktensor(U);
output = history;

%%

    function  [f,g] = cost(x)
        % objective function and its gradient
        % f = sum_{r=1}^{R} Frobenius_norm_of_rank1tensor_rth
        %   = sum_{r=1}^R  \lambda_r^2
        
        U = vec2mat(x);
        % balanced normalization
        %U = normalizeU(U); % Note that normalization makes the gradient
        %         inconsistent with the finite derivatives, but it prevents the
        %         loading components significantly large compared to the others.
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        f = 1/2*sum(prod(Unorm,1));
        
        if nargout>1
            % gradient
            G = [];
            for n = 1:N
                Gamma = prod(Unorm([1:n-1 n+1:N],:));
                Gn = U{n} * diag(Gamma);
                G = [G; Gn(:)];
            end
            g = G(:);
            history.gf = [history.gf g];
        end
    end

    function [cineq, ceq,gradineq,gradceq] = constraints(x)
        %
        % constraint
        %       |X - X_hat|_F^2 <= delta^2
        %       |theta| <= xi
        %
        % if ineqnonlin == true
        %   |X - X_hat|_F^2 <= delta^2
        % otherwise
        %   |X - X_hat|_F^2 = delta^2
        %
        % cineq(x) <= 0
        % ceq(x) == 0
        %
        % If bound_ell2_constraint == true
        %    |theta|_2^2 < N*(R*norm(gamma))^(2/N)
        
        U = vec2mat(x);
        %U = normalizeU(U);
        P = ktensor(U);
        
        % Approximation error |X-P|_F^2 - delta^2
        cineq = (normX^2 + norm(P)^2 - 2* innerprod(X,P) - delta2)/2;
        ceq = [];
        
        gradceq = [];gradineq=[];
        if nargout >2
            gradceq = [];
            
            % Gradient of the constraint function X-P|_F^2 < delta^2
            UtU = cellfun(@(x) x'*x,U,'uni',0);
            UtU = reshape(cell2mat(UtU(:)'),R,R,[]);
            
            gradineq = [];
            for n = 1:N
                Gamma_n = prod(UtU(:,:,[1:n-1 n+1:N]),3);
                gradn = -mttkrp(X,U,n) + U{n} * Gamma_n;
                gradineq = [gradineq ; gradn(:)];
            end
            gradineq = gradineq(:);
        end
        
        if ineqnonlin == false
            [gradceq,gradineq] = deal(gradineq,gradceq);
            [cineq, ceq] = deal(ceq,cineq);
        end
        
        % for the constraint sum_r |u{nr}|^2 <  xi
        if bound_ell2_constraint
            cineq_mu = (norm(x)^2 - xi)/2;
            grad_cmu = x;
            gradineq = [gradineq grad_cmu];
            cineq = [cineq ; cineq_mu];
        end
        
        if ineqnonlin == false
            history.gc = [history.gc gradceq];
            history.c = [history.c ceq];
        else
            history.gc = [history.gc gradineq];
            history.c = [history.c cineq];
        end
    end

%%
    function H = compact_hessian_fc(x,lambda)
        % Hessian of the objective function
        %if ineqnonlin == true
        %    lambda = lambda.ineqnonlin;
        %else
        %    lambda = lambda.eqnonlin;
        %end
        lambda = [lambda.eqnonlin;lambda.ineqnonlin];
        
        U = vec2mat(x);
        U = normalizeU(U);
        
        %[G,Z,Kc,V,F,Kcm,Gamma,Gammas] = cp_hessian(U);
        
        Gamma_n = zeros(R,R,N);
        for n = 1: N
            Gamma_n(:,:,n) = U{n}'*U{n};
        end
        Gamma = prod(Gamma_n,3);
        
        V = [];
        for n = 1:N
            temp = bsxfun(@rdivide,kron(eye(R),U{n}),reshape(Gamma_n(:,:,n),1,[]));
            V = [V;temp];
        end
        
        beta = diag(Gamma);
        Gamma_lda = 2*diag(beta) + lambda(1) * Gamma;
        Vc = mat2cell(V,szX(:)*R,R^2);
        Blk1 = [];
        %ev = zeros(N,1);
        for n = 1:N
            Gamma_nneg = prod(Gamma_n(:,:,[1:n-1 n+1:end]),3);
            beta_n = diag(Gamma_nneg);
            
            Gamma_nneg_lda = diag(beta_n) + lambda(1) * Gamma_nneg;
            
            Blk1_n = kron(Gamma_nneg_lda,eye(szX(n))) - Vc{n} * Prr*diag(Gamma_lda(:))*Vc{n}';
            %Blk1_n = (Blk1_n+Blk1_n)/2;
            
            Blk1 = blkdiag(Blk1,Blk1_n);
            %ev(n) = eigs(Blk1_n,1,'SR');
        end
        Lradjterm = V * Prr*diag(Gamma_lda(:)) * V';
        H = Blk1 + Lradjterm;
        % end of Hessian of the objective function 
        %       f(theta) = sum norm_of_rank_1_tensors
        % + the constraint  c(theta) =  |X-Xhat|_F^2 <= delta^2
        
        %% Hessian of the constraint |\theta|^2 < xi
        if bound_ell2_constraint
            H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + lambda(2);
        end
        
        H = max(H,H');
        if Correct_Hessian
            %H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + 1e-10;
            dE=eig(H,'nobalance');
            aeps = 1e-8;
            pert=abs(min(dE))+aeps*max(dE);%/(1-aeps);
            H=H+pert*eye(size(H));
        end
        
        %[G2,Z2,K2,V2,F2] = cp_hessian(U);
        %H2 = G2 + Z2*K2*Z2'+V2*F2*V2';
        
        %% this part verifies the solution
        history.H = [history.H {H}];
        history.lambda = [history.lambda lambda];
        history.x = [history.x x];
    end

%%
    function H = hessian_ff(x,lambda)
        % hessian of the objective function for order-3 tensors
        U = vec2mat(x);
        U = normalizeU(U);
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        % D is a diagonal matrix
        D = diag([kron(prod(Unorm([2 3],:),1),ones(1,szX(1))),...
            kron(prod(Unorm([1 3],:),1),ones(1,szX(2))),...
            kron(prod(Unorm([1 2],:),1),ones(1,szX(3)))]);
        
        za = [];zb = [];zc = [];
        for r = 1:R
            za = blkdiag(za,U{1}(:,r));
            zb = blkdiag(zb,U{2}(:,r));
            zc = blkdiag(zc,U{3}(:,r));
        end
        Z1 = blkdiag(za,zb,zc);
        
        K = [zeros(R) diag(Unorm(3,:)) diag(Unorm(2,:));
            diag(Unorm(3,:)) zeros(R) diag(Unorm(1,:))
            diag(Unorm(2,:)) diag(Unorm(1,:)) zeros(R)] ;
        H = D + 2 * Z1*K*Z1.';
        
        % [D2,Z2,K2,Ho] = cp_hessian_obj(U);
        
        % hessian of the constraint function
        if 0
            [G,Z,K0,V,F] = cp_hessian(U);
            % G is a blkdiag matrix
            % Z * K0 * Z' is a blkdiagonal matrix
            % V is a partition matrix of (I_R \ox U)*diag(Dn)
            % of size InR x R^2
            % F: Pr * diag(Gamma) : R^2 x R^2
            % Hc = G + Z*K0*Z'+ V*F*V';
            %
        else
            % Jacobian
            J = [];
            for n = 1: N
                Pn = permute_vec(szX,n);
                Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(szX(n)));
                J = [J Pn'*Jn];
            end
            % Hessian
            Hc = J'*J ;
        end
        
        if ineqnonlin == true
            H = H + lambda.ineqnonlin(1)* Hc;
        else
            H = H + lambda.eqnonlin(1)* Hc;
        end
        
        % for the constraint |\theta|^2 < xi
        if bound_ell2_constraint
            H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + lambda.ineqnonlin(2);
        end
        
        H = max(H,H');
        if Correct_Hessian
            %H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + 1e-10;
            dE=eig(H,'nobalance');
            aeps = 1e-8;
            pert=abs(min(dE))+aeps*max(dE);%/(1-aeps);
            H=H+pert*eye(size(H));
        end
    end
%%
    function [D,Z1,K,H] = cp_hessian_obj(U)
        % Ho = D + 2 * Z1 * K * Z1'
        %         U = normalizeU(U);
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        % D is a diagonal matrix
        %         D = diag([kron(prod(Unorm([2 3],:),1),ones(1,szX(1))),...
        %             kron(prod(Unorm([1 3],:),1),ones(1,szX(2))),...
        %             kron(prod(Unorm([1 2],:),1),ones(1,szX(3)))]);
        
        D = [];
        for n = 1:N
            D = [D,kron(prod(Unorm([1:n-1 n+1:end],:),1),ones(1,szX(n)))];
        end
        D = diag(D);
        
        Z1 = [];
        for n = 1:N
            zn = [];
            for r = 1:R
                zn = blkdiag(zn,U{n}(:,r));
            end
            Z1 = blkdiag(Z1,zn);
        end
        
        %         K = [zeros(R) diag(Unorm(3,:)) diag(Unorm(2,:));
        %             diag(Unorm(3,:)) zeros(R) diag(Unorm(1,:))
        %             diag(Unorm(2,:)) diag(Unorm(1,:)) zeros(R)] ;
        K =[];
        for n1 = 1:N
            Kn = [];
            for n2 = 1:N
                if n2 ~= n1
                    beta_n = prod(Unorm(setdiff(1:N,[n1 n2]),:),1);
                    Kn = [Kn diag(beta_n)];
                else
                    Kn = [Kn zeros(R)];
                end
            end
            K = [K; Kn];
        end
        %%
        H = D + 2 * Z1*K*Z1.';
    end

%%
    function [G,Z,K0,V,F] = cp_hessian(U)
        % Hessian of the constraint function
        % H = G + Z* K * Z' + V* F * V';
        %
        Prr = per_vectrans(R,R);
        Gamma_n = zeros(R,R,N);
        for n = 1: N
            Gamma_n(:,:,n) = U{n}'*U{n};
        end
        
        G =[];
        for n = 1:N
            gnn = prod(Gamma_n(:,:,[1:n-1 n+1:N]),3);
            G = blkdiag(G,kron(gnn, eye(szX(n))));
        end
        
        Z = [];
        for n = 1: N
            Z = blkdiag(Z,kron(eye(R),U{n}));
        end
        
        Gamma = prod(Gamma_n,3);
        F = Prr*diag(Gamma(:));
        
        K0 = -diag(reshape(bsxfun(@rdivide,Gamma,Gamma_n.^2),[],1));
        K0 = kron(eye(N),Prr) * K0;
        
        %  kron(eye(N),Prr) *D = D * Prr;
        V = [];
        for n = 1:N
            Dn = diag(1./reshape(Gamma_n(:,:,n),[],1));
            temp = kron(eye(R),U{n}) * Dn;
            V = [V;temp];
        end
        % H = G + Z * K0 * Z' + V * F * V';
    end

%%
    function Hv = hessianmultiply_ff(x,lambda,v)
        % This script is to check
        % H = Ho + lambda * Hc
        % Ho = D + 2 * Z1 * K * Z1'
        % Hc = G + Z*K1 * Z + V * F* V';
        %
        V = vec2mat(v);
        
        U = vec2mat(x);
        U = normalizeU(U);
        % compute Hessian time v for the objective function
        
        Z2v = zeros(R,R,N);
        Z1v = zeros(R,N);
        Gamma = zeros(R,R,N);
        Unorm = zeros(N,R);
        for n = 1:N
            Gamma(:,:,n) = U{n}'*U{n};
            Unorm(n,:) = diag(Gamma(:,:,n));
            Z2v(:,:,n) = U{n}'*V{n};
            Z1v(:,n) = sum(U{n}.*V{n},1);
        end
        
        % D * v
        DnVn = [];
        for n = 1:N
            DnVn = [DnVn
                reshape(bsxfun(@times,V{n},prod(Unorm([1:n-1 n+1:N],:),1)),[],1)];
        end
        
        % K*Z1'*v
        KZ1v = [];ZKZ1v =[];
        for n1 = 1:N
            KZ1v_n1 = 0;
            for n2 = [1:n1-1 n1+1:N]
                beta_n1n2 = prod(Unorm(setdiff(1:N,[n1 n2]),:),1);
                KZ1v_n1 = KZ1v_n1 + beta_n1n2(:).*Z1v(:,n2);
            end
            KZ1v = [KZ1v ; KZ1v_n1];
            ZKZ1v = [ZKZ1v;
                reshape(bsxfun(@times,U{n1},KZ1v_n1'),[],1)];
        end
        
        %         KZ1v = [sum(Unorm([3 2],:).' .* Z1v(:,[2 3]),2) ...
        %                 sum(Unorm([3 1],:).' .* Z1v(:,[1 3]),2) ...
        %                 sum(Unorm([2 1],:).' .* Z1v(:,[1 2]),2)];
        % Z1*K1*Z1'*v
        
        %         ZKZ1v = [reshape(bsxfun(@times,U{1},KZ1v(:,1)'),[],1)
        %                  reshape(bsxfun(@times,U{2},KZ1v(:,2)'),[],1)
        %                  reshape(bsxfun(@times,U{3},KZ1v(:,3)'),[],1)];
        
        
        % Hv = [Vn* diag(gamma{-n} + 2 * Un * diag(beta_n)]
        Hv = DnVn + 2 * ZKZ1v;
        
        %[D2,Z2,K2,Ho2] = cp_hessian_obj(U);
        %norm(Hv - Ho2*v) % checked
        
        % compute Hessian time v for the contraint function
        %         [G,Z,K0,Vh,F] = cp_hessian(U);
        %         Hc = G + Z*K0*Z' + Vh*F*Vh';
        
        %         %% hessian of the constraint function
        %             [G,Z2,K2,V,F] = cp_hessian(U);
        %             % G is a blkdiag matrix
        %             % Z * K0 * Z' is a blkdiagonal matrix
        %             % V is a partition matrix of (I_R \ox U)*diag(Dn)
        %             % of size InR x R^2
        %             % F: Pr * diag(Gamma) : R^2 x R^2
        %             % Hc = G + Z2*K2*Z2'+ V*F*V';
        %             %
        
        
        %Gv = [reshape(V{1} * prod(Gamma(:,:,[2 3]),3),[],1)
        %      reshape(V{2} * prod(Gamma(:,:,[1 3]),3),[],1)
        %      reshape(V{3} * prod(Gamma(:,:,[1 2]),3),[],1)];
        Gv = [];
        for n = 1:N
            Gv = [Gv;
                reshape(V{n} * prod(Gamma(:,:,[1:n-1 n+1:end]),3),[],1)];
        end
        
        %         K2Z2v = cat(3,sum(Gamma(:,:,[3 2]).* Z2v(:,:,[2 3]),3), ...
        %                       sum(Gamma(:,:,[3 1]).* Z2v(:,:,[1 3]),3),...
        %                       sum(Gamma(:,:,[2 1]).* Z2v(:,:,[1 2]),3));
        
        K2Z2v = [];
        Z2KZ2v = [];
        Gamma_p = prod(Gamma,3);
        for n1=1:N
            temp = -(Gamma_p./Gamma(:,:,n1).^2).*Z2v(:,:,n1);
            K2Z2v = cat(3,K2Z2v,temp');
            Z2KZ2v = [Z2KZ2v;
                reshape(U{n1} * temp',[],1)];
        end
        
        % Z1*K1*Z1'*v
        %         Z2KZ2v = [reshape(U{1} * K2Z2v(:,:,1),[],1)
        %             reshape(U{2} * K2Z2v(:,:,2),[],1)
        %             reshape(U{3} * K2Z2v(:,:,3),[],1)];
        
        %% V2*v
        V2v = sum(Z2v./Gamma,3);
        F2V2v = (Gamma_p .* V2v)';
        V2F2V2v = [];
        for n1=1:N
            V2F2V2v = [V2F2V2v ;
                reshape(U{n1} * (F2V2v./Gamma(:,:,n1)),[],1)];
        end
        %%
        Hcv = Gv + Z2KZ2v+V2F2V2v;
        
        %         % H*v of the contraint function
        %         [G2,Z2,K2,V2,F2] = cp_hessian(U);
        %         Hc = G2 + Z2*K2*Z2' + V2*F2*V2';
        %         norm(Gv - G2*v)
        %         norm(Z2KZ2v - (Z2*K2*Z2')*v)
        %         % V2*F*V2'*v
        %         norm(V2'*v - reshape(sum(Z2v./Gamma,3),[],1))
        %         norm(F2*V2'*v - reshape(F2V2v,[],1))
        %         norm(V2*F2*V2'*v - reshape(V2F2V2v,[],1))
        %         norm(Hcv - Hc*v)
        
        
        %%
        if ineqnonlin == true
            Hv = Hv + lambda.ineqnonlin* Hcv;
        else
            Hv = Hv + lambda.eqnonlin* Hcv;
        end
        
    end

%%
    function Hv = hessianmultiply_fc(x,lambda,v)
        % This script computes the product H(x) x v
        % H = Ho + lambda * Hc
        % Ho = D + 2 * Z1 * K * Z1'
        % Hc = G + Z*K1 * Z + V * F* V';
        %
        lambda = [lambda.eqnonlin;lambda.ineqnonlin];
%         if ineqnonlin == true
%             lambda = lambda.ineqnonlin;
%         else
%             lambda = lambda.eqnonlin;
%         end
        
        V = vec2mat(v);
        U = vec2mat(x);
        U = normalizeU(U);
        
        % compute Hessian time v for the objective function
        Z2v = zeros(R,R,N);
        Z1v = zeros(R,N);
        Gamma = zeros(R,R,N);
        Unorm = zeros(N,R);
        for n = 1:N
            Gamma(:,:,n) = U{n}'*U{n};
            Unorm(n,:) = diag(Gamma(:,:,n));
            Z2v(:,:,n) = U{n}'*V{n};
            Z1v(:,n) = diag(Z2v(:,:,n));%sum(U{n}.*V{n},1);
        end
        
        % D * v
        DnVn = [];
        for n = 1:N
            DnVn = [DnVn
                reshape(bsxfun(@times,V{n},prod(Unorm([1:n-1 n+1:N],:),1)),[],1)];
        end
        
        % K*Z1'*v
        ZKZ1v =[];
        for n1 = 1:N
            KZ1v_n1 = 0;
            for n2 = [1:n1-1 n1+1:N]
                beta_n1n2 = prod(Unorm(setdiff(1:N,[n1 n2]),:),1);
                KZ1v_n1 = KZ1v_n1 + beta_n1n2(:).*Z1v(:,n2);
            end
            ZKZ1v = [ZKZ1v;
                reshape(bsxfun(@times,U{n1},KZ1v_n1'),[],1)];
        end
        % H x v
        Hv = DnVn + 2 * ZKZ1v;
        
        %[D2,Z2,K2,Ho2] = cp_hessian_obj(U);%norm(Hv - Ho2*v) % checked
        
        % compute Hessian time v for the contraint function
        % [G,Z,K0,Vh,F] = cp_hessian(U);
        % Hc = G + Z*K0*Z' + Vh*F*Vh';
        
        %         Gv = [];
        %         for n = 1:N
        %             Gv = [Gv;
        %                 reshape(V{n} * prod(Gamma(:,:,[1:n-1 n+1:end]),3),[],1)];
        %         end
        
        % Z2*K2*Z2'*v
        Gamma_p = prod(Gamma,3);
        Gamma_pd = bsxfun(@rdivide,Gamma_p,Gamma);
        Z2v_d = bsxfun(@rdivide,Z2v,Gamma);
        V2v = sum(Z2v_d,3);
        
        ZplusV = []; Gv = [];
        for n = 1:N
            %%
            temp = Gamma_pd(:,:,n) .* ( V2v' - Z2v_d(:,:,n)');
            ZplusV = [ZplusV; reshape(U{n} * temp,[],1)];
            Gv = [Gv; reshape(V{n} * Gamma_pd(:,:,n),[],1)];
        end
        
        %% Hcv1 = Gv + Z2KZ2v+V2F2V2v;
        Hcv = Gv + ZplusV;
        
        %% Construct the whole Hessian from Hessian matrices of the objective and the constraint functions.
        Hv = Hv + lambda(1)* Hcv;
        if bound_ell2_constraint
            Hv = Hv + lambda(2)*v;
        end
        
    end

%%
    function Hv = hessianmultiply_fc_old(x,lambda,v)
        % This script computes the product H(x) x v
        % H = Ho + lambda * Hc
        % Ho = D + 2 * Z1 * K * Z1'
        % Hc = G + Z*K1 * Z + V * F* V';
        %
        V = vec2mat(v);
        U = vec2mat(x);
        U = normalizeU(U);
        
        % compute Hessian time v for the objective function
        Z2v = zeros(R,R,N);
        Z1v = zeros(R,N);
        Gamma = zeros(R,R,N);
        Unorm = zeros(N,R);
        for n = 1:N
            Gamma(:,:,n) = U{n}'*U{n};
            Unorm(n,:) = diag(Gamma(:,:,n));
            Z2v(:,:,n) = U{n}'*V{n};
            Z1v(:,n) = sum(U{n}.*V{n},1);
        end
        
        % D * v
        DnVn = [];
        for n = 1:N
            DnVn = [DnVn
                reshape(bsxfun(@times,V{n},prod(Unorm([1:n-1 n+1:N],:),1)),[],1)];
        end
        
        % K*Z1'*v
        KZ1v = [];ZKZ1v =[];
        for n1 = 1:N
            KZ1v_n1 = 0;
            for n2 = [1:n1-1 n1+1:N]
                beta_n1n2 = prod(Unorm(setdiff(1:N,[n1 n2]),:),1);
                KZ1v_n1 = KZ1v_n1 + beta_n1n2(:).*Z1v(:,n2);
            end
            %KZ1v = [KZ1v ; KZ1v_n1];
            ZKZ1v = [ZKZ1v;
                reshape(bsxfun(@times,U{n1},KZ1v_n1'),[],1)];
        end
        
        % Hv = [Vn* diag(gamma{-n} + 2 * Un * diag(beta_n)]
        Hv = DnVn + 2 * ZKZ1v;
        
        %[D2,Z2,K2,Ho2] = cp_hessian_obj(U);
        %norm(Hv - Ho2*v) % checked
        
        % compute Hessian time v for the contraint function
        %         [G,Z,K0,Vh,F] = cp_hessian(U);
        %         Hc = G + Z*K0*Z' + Vh*F*Vh';
        
        Gv = [];
        for n = 1:N
            Gv = [Gv;
                reshape(V{n} * prod(Gamma(:,:,[1:n-1 n+1:end]),3),[],1)];
        end
        
        % Z2*K2*Z2'*v
        Gamma_p = prod(Gamma,3);
        %         V2v = sum(Z2v./Gamma,3);
        %         F2V2v = Gamma_p .* V2v';
        %         Z2KZ2v = []; V2F2V2v = [];
        %         for n1=1:N
        %             K2Z2v_n = -(Gamma_p./Gamma(:,:,n1).^2).*Z2v(:,:,n1);
        %             Z2KZ2v = [Z2KZ2v;
        %                 reshape(U{n1} * K2Z2v_n',[],1)];
        %
        %             %% V2*v
        %             V2F2V2v = [V2F2V2v ;
        %                 reshape(U{n1} * (F2V2v./Gamma(:,:,n1)),[],1)];
        %         end
        
        Gamma_pd = bsxfun(@rdivide,Gamma_p,Gamma);
        Z2v_d = bsxfun(@rdivide,Z2v,Gamma);
        V2v = sum(Z2v_d,3);
        
        ZplusV = [];
        for n1 = 1:N
            %%
            temp = Gamma_pd(:,:,n1) .* ( V2v' - Z2v_d(:,:,n1)');
            ZplusV = [ZplusV; reshape(U{n1} * temp,[],1)];
        end
        
        %%
        %         Hcv1 = Gv + Z2KZ2v+V2F2V2v;
        Hcv = Gv + ZplusV;
        
        %         % H*v of the contraint function
        %         [G2,Z2,K2,V2,F2] = cp_hessian(U);
        %         Hc = G2 + Z2*K2*Z2' + V2*F2*V2';
        %         norm(Hcv - Hc*v)
        
        %         norm(Gv - G2*v)
        %         norm(Z2KZ2v - (Z2*K2*Z2')*v)
        %         % V2*F*V2'*v
        %         norm(V2'*v - reshape(sum(Z2v./Gamma,3),[],1))
        %         norm(F2*V2'*v - reshape(F2V2v,[],1))
        %         norm(V2*F2*V2'*v - reshape(V2F2V2v,[],1))
        
        
        %%
        if ineqnonlin == true
            Hv = Hv + lambda.ineqnonlin* Hcv;
        else
            Hv = Hv + lambda.eqnonlin* Hcv;
        end
        
    end

%%

    function Hv = hessianmultiply_ff_3rd(x,lambda,v)
        % H = Ho + lambda * Hc
        % Ho = D + 2 * Z1 * K * Z1'
        % Hc = G + Z*K1 * Z + V * F* V';
        %
        % hessian of the objective function
        V = vec2mat(v);
        
        U = vec2mat(x);
        U = normalizeU(U);
        
        Z2v = zeros(R,R,N);
        Z1v = zeros(R,N);
        Gamma = zeros(R,R,N);
        Unorm = zeros(N,R);
        for n = 1:N
            Gamma(:,:,n) = U{n}'*U{n};
            Unorm(n,:) = diag(Gamma(:,:,n));
            Z2v(:,:,n) = U{n}'*V{n};
            Z1v(:,n) = sum(U{n}.*V{n},1);
        end
        
        DnVn = [reshape(bsxfun(@times,V{1},prod(Unorm([2 3],:),1)),[],1)
            reshape(bsxfun(@times,V{2},prod(Unorm([1 3],:),1)),[],1)
            reshape(bsxfun(@times,V{3},prod(Unorm([1 2],:),1)),[],1)];
        %         DnVn = [];
        %         KZ1v = [];
        %         for n = 1:N
        %             DnVn = [DnVn
        %                 reshape(bsxfun(@times,V{n},prod(Unorm([1:n-1 n+1:N],:),1)),[],1)];
        %             KZ1v = [KZ1v
        %                 sum(Unorm([3 2],:).' .* Z1v(:,[2 3]),2)];
        %         end
        
        % KZ1v
        KZ1v = [sum(Unorm([3 2],:).' .* Z1v(:,[2 3]),2) ...
            sum(Unorm([3 1],:).' .* Z1v(:,[1 3]),2) ...
            sum(Unorm([2 1],:).' .* Z1v(:,[1 2]),2)];
        % Z1*K1*Z1'*v
        
        ZKZ1v = [reshape(bsxfun(@times,U{1},KZ1v(:,1)'),[],1)
            reshape(bsxfun(@times,U{2},KZ1v(:,2)'),[],1)
            reshape(bsxfun(@times,U{3},KZ1v(:,3)'),[],1)];
        % Hv = [Vn* diag(gamma{-n} + 2 * Un * diag(beta_n)]
        Hv = DnVn + 2 * ZKZ1v;
        
        
        %         %% hessian of the constraint function
        %             [G,Z2,K2,V,F] = cp_hessian(U);
        %             % G is a blkdiag matrix
        %             % Z * K0 * Z' is a blkdiagonal matrix
        %             % V is a partition matrix of (I_R \ox U)*diag(Dn)
        %             % of size InR x R^2
        %             % F: Pr * diag(Gamma) : R^2 x R^2
        %             % Hc = G + Z2*K2*Z2'+ V*F*V';
        %             %
        
        
        Gv = [reshape(V{1} * prod(Gamma(:,:,[2 3]),3),[],1)
            reshape(V{2} * prod(Gamma(:,:,[1 3]),3),[],1)
            reshape(V{3} * prod(Gamma(:,:,[1 2]),3),[],1)];
        
        K2Z2v = cat(3,sum(Gamma(:,:,[3 2]).* Z2v(:,:,[2 3]),3), ...
            sum(Gamma(:,:,[3 1]).* Z2v(:,:,[1 3]),3),...
            sum(Gamma(:,:,[2 1]).* Z2v(:,:,[1 2]),3));
        % Z1*K1*Z1'*v
        
        Z2KZ2v = [reshape(U{1} * K2Z2v(:,:,1),[],1)
            reshape(U{2} * K2Z2v(:,:,2),[],1)
            reshape(U{3} * K2Z2v(:,:,3),[],1)];
        
        Hcv = Gv + Z2KZ2v;
        
        
        %%
        if ineqnonlin == true
            Hv = Hv + lambda.ineqnonlin* Hcv;
        else
            Hv = Hv + lambda.eqnonlin* Hcv;
        end
        
    end


%     % approximation the constraint function
%     function  [cineq, ceq,gradineq,gradceq]  = constraints(x)
%         ceq = [];
%         if 0
%             dtheta  = x - theta_0;
%             cineq = f0 + dtheta'* g(:) + 1/2 * dtheta'*H*dtheta - delta^2;
%         end
%         Hx = H*x;
%         cineq = f_0 + x'* g_Hx0 + 1/2* x'*Hx ;
%         if nargout >2
%             gradceq = [];
%             gradineq = g_Hx0 + Hx;
%         end
%     end
%
%
    function stop = outfun(x,optimValues,state)
        stop = false;
        
        switch state
            case 'init'
                % hold on
            case 'iter'
                % Concatenate current point and objective function
                % value with history. x must be a row vector.
                history.fval = [history.fval; optimValues.fval];
                history.constrviolation = [history.constrviolation; optimValues.constrviolation];
                history.firstorderopt = [history.firstorderopt; optimValues.firstorderopt];
                history.stepsize = [history.stepsize optimValues.stepsize];
                
                if size(history.gc,2)>1
                    debug_ = 0;
                    if debug_
                        kf = 1;
                        gc = history.gc(:,end-kf);
                        gf = history.gf(:,end-kf);
                        c =  history.c(end-kf);
                        H = history.H{end-kf};
                        lambda = history.lambda(end-kf);
                        ss = -[H gc; gc' 0]\[gf-lambda(1)*gc;c];
                        err = norm(history.x(:,end-kf+1)-(history.x(:,end-kf) + ss(1:end-1)));
                        ss = -[H gc; gc' 1]\[gf-lambda(1)*gc;c];
                        err2 = norm(history.x(:,end-kf+1)-(history.x(:,end-kf) + history.stepsize(end-kf+1)* ss(1:end-1)/norm(ss(1:end-1))));
                        if err> 1e-5
                            1
                        end
                        [err err2]
                    end
                end
                
                %history.x = [history.x; x];
                % Concatenate current search direction with
                % searchdir.
                %                 searchdir = [searchdir;...
                %                     optimValues.searchdirection'];
                %                 plot(x(1),x(2),'o');
                %                 % Label points with iteration number.
                %                 % Add .15 to x(1) to separate label from plotted 'o'
                %                 text(x(1)+.15,x(2),num2str(optimValues.iteration));
            case 'done'
                %hold off
            otherwise
        end
    end
end




%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;

param.addParameter('Hessian','full',@(x) ismember(x,{'approximation' 'full'}));


param.addParameter('init','rand',@(x) (iscell(x) || isa(x,'ktensor')||...
    (ischar(x) && ismember(x(1:min(numel(x),4)),{'rand' 'nvec' 'fibe' 'orth' 'dtld'}))));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('normX',[]);
param.addOptional('ineqnonlin',true,@islogical);
% true for the inequality contraint  |X - X_hat(\theta)|_F <= delta
% false for the equality contraint   |X - X_hat(\theta)|_F = delta
%
param.addOptional('Algorithm','interior-point'); %'sqp'
param.addOptional('history',false,@islogical);

param.addOptional('bound_ell2_constraint',false,@islogical); % if constraint |\theta|_2^2 < \xi
param.addOptional('Correct_Hessian',true,@islogical);
% Hessian of the contraint function (and the augmented Lagragian function)
% is ill-condition, and should be corrected if the algorithm is Sequential
% Quadratic Programming see cp_nc_sqp
%
param.parse(opts);
param = param.Results;
end


function P = permute_vec(In,n)
% vec(X_n) = P * vec(X)
Tt = tensor(1:prod(In),In);
Tn = tenmat(Tt,n);
Perm = Tn(:);
P = eye(prod(In));
P(:,Perm) = P;
end


function [U,gamma] = normalizeU(U)
P = ktensor(U);
P = normalize(P);
gamma= P.lambda;
U = P.U;
N = numel(U);
for n = 1:N
    U{n} = U{n} * diag(gamma.^(1/N));
end
end


%% Compute the approximation function of f(U) at the initial point
% U = mat2cell(reshape(x,[],R),szX(:),R);
% % Jacobian
% J = [];
% for n = 1: N
%     Pn = permute_vec(szX,n);
%     Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(szX(n)));
%     J = [J Pn'*Jn];
% end
% % Hessian
% H = J'*J ;
% % gradient
% P = ktensor(U); Yx = full(P);
% g = -J'*(X(:) - Yx(:));
%
% f0 = max(0,normX^2 + norm(P)^2 - 2*innerprod(tensor(X),P));

%         %
%         UtU = cellfun(@(x) x'*x,U,'uni',0);
%         UtU = reshape(cell2mat(UtU(:)'),R,R,[]);
%         gradineq = [];
%         for n = 1:N
%             Gamma_n = prod(UtU(:,:,[1:n-1 n+1:N]),3);
%             gradn = -mttkrp(Yf,U,n) + U{n} * Gamma_n;
%             gradineq = [gradineq ; gradn(:)];
%         end
%
% theta_0 = cell2mat(cellfun(@(x) x(:),U,'uni',0));
%
% f_0 = f0 - theta_0'*g + 1/2*theta_0'*H*theta_0 - delta^2;
% g_Hx0 = g - H*theta_0;

