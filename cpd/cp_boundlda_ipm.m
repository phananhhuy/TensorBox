function [P,output] = cp_boundlda_ipm(X,R,delta,opts)
% Fast ALS for constrained CPD
%
%  min         \|X - P\|_F 
%  subject to   sum_r  \lambda_r^2  < delta^2.
% 
% lambda_r: intensity of rank-1 tensors
%
% where Un are normalized to have unit norm
%
%  P is a ktensor of U1, ..., UN
% 
% TENSOR BOX, v1. 2018
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
min_error = 1-param.fitmax;

bound_ell2_constraint = param.bound_ell2_constraint;

ineqnonlin = param.ineqnonlin;
% true for the inequality contraint  |X - X_hat(\theta)|_F <= delta
% false for the equality contraint  |X - X_hat(\theta)|_F = delta

Correct_Hessian = param.Correct_Hessian;

%% Initialize factors U
param.cp_func = str2func(mfilename);
Uinit = cp_init(X,R,param); U = Uinit;

%% Output
if param.printitn ~=0
    fprintf('\n Interial Point algorithm for CPD with bound norm constraint\n');
end
% map a vectorization to factor matrices 
vec2mat = @(x) cellfun(@(x) reshape(x,[],R),mat2cell(x,szX*R),'uni',0);

[U,gamma] = normalizeU(U);
% initial point 
x = cell2mat(cellfun(@(x) x(:),U,'uni',0));

% Set up shared variables with OUTFUN
history.fval = [];
history.constrviolation = [];
history.firstorderopt=[];

[f,g] = obj_func(x);
optimValues = struct('fval',f,'firstorderopt',g,'constrviolation',0);
outfun(x,optimValues,'iter');

% upper bound of |theta|_2^2 
if bound_ell2_constraint
    xi = N*(R*norm(gamma))^(2/N); % used for the constraint |\theta\|_2^2 <= xi
end

% default parameters
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
        'OutputFcn',@outfun);
catch
    options = optimoptions('fmincon','Display','iter',...
        ...%'CheckGradients',true,...
        'FinDiffType','central',...
        'GradObj','on',...
        'GradConstr','on',... %
        'HessFcn',@compact_hessian_fc,...
        'MaxFunEvals',sum(szX)*R*1000,...
        'MaxIter',min(param.maxiters,sum(szX)*R*100),...
        'OutputFcn',@outfun);
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

options.OptimalityTolerance = param.tol;
options.StepTolerance = min(options.StepTolerance,param.tol);
options.ConstraintTolerance = min(options.ConstraintTolerance,param.tol);
switch param.Algorithm 
    case {'SQP' 'sqp'}
        options.Algorithm = 'sqp';        
        Correct_Hessian = true;
end
 
%'SubproblemAlgorithm','cg','HessianMultiplyFcn',@hessianmultiply_ff,'HessianFcn',[],...
Prr = per_vectrans(R,R);

%% 
[x,fval,exitflag] = fmincon(@obj_func,x,[],[],[],[],[],[],@constraint_func,options);

%%
U = vec2mat(x);
P = ktensor(U);
output = history; 

%%

    function [f,g] = obj_func(x)
        %  
        % f =  1/2 * |X - X_hat|_F^2 
        
        U = vec2mat(x);
        U = normalizeU(U);
        P = ktensor(U);
        
        f = (normX^2 + norm(P)^2 - 2* innerprod(X,P))/2;
         
        if nargout>1
             
            UtU = cellfun(@(x) x'*x,U,'uni',0);
            UtU = reshape(cell2mat(UtU(:)'),R,R,[]);

            g = [];
            for n = 1:N
                Gamma_n = prod(UtU(:,:,[1:n-1 n+1:N]),3);
                gradn = -mttkrp(X,U,n) + U{n} * Gamma_n;
                g = [g ; gradn(:)];
            end
            g = g(:);
        end
          
%         % for the constraint sum_r |u{nr}|^2 <  xi
%         if bound_ell2_constraint
%             cineq_mu = (norm(x)^2 - xi)/2;
%             grad_cmu = x;
%             gradineq = [gradineq grad_cmu];
%             cineq = [cineq ; cineq_mu];
%         end
    end

    function [cineq, ceq,gradineq,gradceq] = constraint_func(x)
        % c(x) = sum_{r=1}^{R} Frobenius_norm_of_rank1tensor_rth
        %      = sum_{r=1}^R  \lambda_r^2
        % if ineqnonlin == true
        %   c(x) <= delta^2
        % otherwise
        %   c(x)  = delta^2
        %
        % cineq(x) <= 0
        % ceq(x) == 0
        U = vec2mat(x);
        U = normalizeU(U);
         
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        cineq = 1/2*(sum(prod(Unorm,1)) - delta2);
        ceq = [];
        
        gradceq = [];gradineq=[];
        if nargout >2
            gradceq = [];
             
            % gradient
            gradineq = [];
            for n = 1:N
                Gamma = prod(Unorm([1:n-1 n+1:N],:));
                Gn = U{n} * diag(Gamma);
                gradineq = [gradineq; Gn(:)];
            end
            gradineq = gradineq(:);
        end
        
        if ineqnonlin == false
            [gradceq,gradineq] = deal(gradineq,gradceq);
            [cineq, ceq] = deal(ceq,cineq);
        end
    end


    function H = hessian_ff(x,lambda)
        % hessian of the objective function
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
        Hc = D + 2 * Z1*K*Z1.';
        
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
        else % construct Hessian form the Jacobian
            % Jacobian
            J = [];
            for n = 1: N
                Pn = permute_vec(szX,n);
                Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(szX(n)));
                J = [J Pn'*Jn];
            end
            % Hessian
            H = J'*J ;
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
% 
%     function  [f,g] = cost(x)
%         % objective function and its gradient 
%         % f = sum_{r=1}^{R} Frobenius_norm_of_rank1tensor_rth
%         %   = sum_{r=1}^R  \lambda_r^2
%          
%         U = vec2mat(x);
%         % balanced normalization
%         U = normalizeU(U);
%         
%         Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
%         Unorm = cell2mat(Unorm(:));
%         
%         f = 1/2*sum(prod(Unorm,1));
%         
%         if nargout>1
%             % gradient
%             G = [];
%             for n = 1:N
%                 Gamma = prod(Unorm([1:n-1 n+1:N],:));
%                 Gn = U{n} * diag(Gamma);
%                 G = [G; Gn(:)];
%             end
%             g = G(:);
%         end
%     end
% 
%     function [cineq, ceq,gradineq,gradceq] = constraints(x)
%         % if ineqnonlin == true
%         %   |X - X_hat|_F^2 <= delta^2
%         % otherwise  
%         %   |X - X_hat|_F^2 = delta^2
%         %
%         % cineq(x) <= 0
%         % ceq(x) == 0
%         U = vec2mat(x);
%         U = normalizeU(U);
%         P = ktensor(U);
%         
%         cineq = (normX^2 + norm(P)^2 - 2* innerprod(X,P) - delta2)/2;
%         ceq = [];
%         
%         gradceq = [];gradineq=[];
%         if nargout >2
%             gradceq = [];
%             UtU = cellfun(@(x) x'*x,U,'uni',0);
%             UtU = reshape(cell2mat(UtU(:)'),R,R,[]);
%             
%             gradineq = [];
%             for n = 1:N
%                 Gamma_n = prod(UtU(:,:,[1:n-1 n+1:N]),3);
%                 gradn = -mttkrp(X,U,n) + U{n} * Gamma_n;
%                 gradineq = [gradineq ; gradn(:)];
%             end
%             gradineq = gradineq(:);
%         end
%         
%         if ineqnonlin == false
%             [gradceq,gradineq] = deal(gradineq,gradceq);
%             [cineq, ceq] = deal(ceq,cineq);
%         end
%         
%         % for the constraint sum_r |u{nr}|^2 <  xi
%         if bound_ell2_constraint
%             cineq_mu = (norm(x)^2 - xi)/2;
%             grad_cmu = x;
%             gradineq = [gradineq grad_cmu];
%             cineq = [cineq ; cineq_mu];
%         end
%     end


%     function H = hessian_ff(x,lambda)
%         % hessian of the objective function
%         U = vec2mat(x);
%         U = normalizeU(U);
%         
%         Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
%         Unorm = cell2mat(Unorm(:));
%         
%         % D is a diagonal matrix
%         D = diag([kron(prod(Unorm([2 3],:),1),ones(1,szX(1))),...
%             kron(prod(Unorm([1 3],:),1),ones(1,szX(2))),...
%             kron(prod(Unorm([1 2],:),1),ones(1,szX(3)))]);
%         
%         za = [];zb = [];zc = [];
%         for r = 1:R
%             za = blkdiag(za,U{1}(:,r));
%             zb = blkdiag(zb,U{2}(:,r));
%             zc = blkdiag(zc,U{3}(:,r));
%         end
%         Z1 = blkdiag(za,zb,zc);
%         
%         K = [zeros(R) diag(Unorm(3,:)) diag(Unorm(2,:));
%             diag(Unorm(3,:)) zeros(R) diag(Unorm(1,:))
%             diag(Unorm(2,:)) diag(Unorm(1,:)) zeros(R)] ;
%         H = D + 2 * Z1*K*Z1.';
%         
%         % hessian of the constraint function
%         if 0
%             [G,Z,K0,V,F] = cp_hessian(U);
%             % G is a blkdiag matrix
%             % Z * K0 * Z' is a blkdiagonal matrix
%             % V is a partition matrix of (I_R \ox U)*diag(Dn) 
%             % of size InR x R^2
%             % F: Pr * diag(Gamma) : R^2 x R^2
%             % Hc = G + Z*K0*Z'+ V*F*V';
%             %
%         else
%             % Jacobian
%             J = [];
%             for n = 1: N
%                 Pn = permute_vec(szX,n);
%                 Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(szX(n)));
%                 J = [J Pn'*Jn];
%             end
%             % Hessian
%             Hc = J'*J ;
%         end
%         
%         if ineqnonlin == true
%             H = H + lambda.ineqnonlin(1)* Hc;
%         else
%             H = H + lambda.eqnonlin(1)* Hc;
%         end
%         
%         % for the constraint |\theta|^2 < xi
%         if bound_ell2_constraint
%             H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + lambda.ineqnonlin(2);
%         end
%         
%         H = max(H,H');
%         if Correct_Hessian
%             %H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + 1e-10;
%             dE=eig(H,'nobalance');
%             aeps = 1e-8;
%             pert=abs(min(dE))+aeps*max(dE);%/(1-aeps);
%             H=H+pert*eye(size(H));
%         end
%     end



    function H = compact_hessian_fc(x,lambda)
        % hessian of the objective function
        if ineqnonlin == true
            lambda = lambda.ineqnonlin;
        else
            lambda = lambda.eqnonlin;
        end
        
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
        Gamma_lda = 2*diag(lambda(1) *beta) + Gamma;
        Vc = mat2cell(V,szX(:)*R,R^2);
        Blk1 = [];
        %ev = zeros(N,1);
        for n = 1:N
            Gamma_nneg = prod(Gamma_n(:,:,[1:n-1 n+1:end]),3);
            beta_n = diag(Gamma_nneg);
            
            Gamma_nneg_lda = diag(lambda(1) * beta_n) + Gamma_nneg;
            
            Blk1_n = kron(Gamma_nneg_lda,eye(szX(n))) - Vc{n} * Prr*diag(Gamma_lda(:))*Vc{n}';
            %Blk1_n = (Blk1_n+Blk1_n)/2;
            
            Blk1 = blkdiag(Blk1,Blk1_n);
            %ev(n) = eigs(Blk1_n,1,'SR'); 
        end
        Lradjterm = V * Prr*diag(Gamma_lda(:)) * V';
        H = Blk1 + Lradjterm;
 
        % for the constraint |\theta|^2 < xi
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
    end

%%

    function [G,Z,K0,V,F] = cp_hessian(U)
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

    function Hv = hessianmultiply_ff(x,lambda,v)
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
        stop = optimValues.fval <= min_error;
    end
end




%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;

param.addParameter('Hessian','full',@(x) ismember(x,{'approximation' 'full'}));

param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));
param.addOptional('maxiters',inf);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('normX',[]);
param.addOptional('fitmax',1-1e-10);
param.addOptional('ineqnonlin',true,@islogical);
% true for the inequality contraint  |X - X_hat(\theta)|_F <= delta
% false for the equality contraint   |X - X_hat(\theta)|_F = delta
%
param.addOptional('Algorithm','interior-point'); %'sqp'

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

