function [P,output] = cp_boundlda_sqp(X,R,delta,opts)
% Sequential Quadratic Programming algorithm for solving the bounded CP
%
%  min     sum_n lambda_n^2
%  subjec to  |X - P|_F < delta
%
%  P is a ktensor of U1, ..., UN
%
% INPUT:
%   X:  N-way data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%  
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'| ktensor| cell array]
%          init can be a cell array whose each entry specifies an intial
%          value. The algorithm will chose the best one after small runs.
%          For example,
%          opts.init = {'random' 'random' 'nvec'};
%     .printitn: Print fit every n iterations {1}
%
% OUTPUT:
%  P:  ktensor of estimated factors
%  output:
%      .Fit
%      .NoIters
%
% EXAMPLE
%
% REF:
 
%
% See also: cp_als
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.

%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

N = ndims(X); I = size(X);szX = I;
if isempty(param.normX)
    normX = norm(X);
else
    normX = param.normX;
end

bound_ell2_constraint = param.bound_ell2_constraint;
Correct_Hessian = param.Correct_Hessian;

%% Initialize factors U
param.cp_func = str2func(mfilename);
Uinit = cp_init(X,R,param); U = Uinit;

%% Output
if param.printitn ~=0
    fprintf('\n Sequential Quadratic Programming for CPD with bound norm constraint\n');
end

delta2 = delta^2;
min_error = 1-param.fitmax;

[U,gamma] = normalizeU(U);
% upper bound of |theta|_2^2
if (bound_ell2_constraint ~= 0)
    xi = bound_ell2_constraint;
    if islogical(xi)
        xi = N*(R*norm(gamma))^(2/N); % used for the constraint |\theta\|_2^2 <= xi
    end
    bound_ell2_constraint = true;
    mu = xi;
end


%
x = cell2mat(cellfun(@(x) x(:),U,'uni',0));

vec2mat = @(x) cellfun(@(x) reshape(x,[],R),mat2cell(x,szX*R),'uni',0);

% Set up shared variables with OUTFUN
history.fval = [];
history.constrviolation = [];
history.firstorderopt=[];
% ineqnonlin = true;

[f,g] = fobjconst(x);
optimValues = struct('fval',f,'firstorderopt',g);
outfun(x,optimValues,'iter');

Opts.foptions = foptions;
Opts.foptions(1) = 1;%Display parameter
Opts.foptions(2) = param.tol; %Termination tolerance for X
Opts.foptions(3) = param.tol; %-Termination tolerance on F.(Default: 1e-4).
Opts.foptions(14) =  min(param.maxiters,sum(I)*R*100);

if param.ineqnonlin == false
    Opts.foptions(13) =  1;
else
    Opts.foptions(13) =  0;
end

switch param.Hessian 
    case 'approximation'
        Opts.HessFun = [];
    otherwise
        Opts.HessFun = @compact_hessian_fc;        
        %Opts.HessFun = @hessian_ff;
end

% Opts.HessFun = @compact_hessian_fc;
Opts.OutputFcn = @outfun;
if ~isempty(param.LagrangeMultipliers)
    Opts.LagrangeMultipliers = param.LagrangeMultipliers;
end
Prr = per_vectrans(R,R);
        
% status = -1;
% while status ~= 0
[x,opts,v,H,status] = msqp(@fobjconst,x,Opts,[],[],@fgrad);
% end

% Closed form update for lambda and delta_x
% Hgf = H\gf(:,1);
% Hgc = H\gc(:,1);
% lambda_new = (c(1) - Hgc'*gf(:,1))/(Hgc'*gc(:,1));
% delta_x = -(Hgf + delta*Hgc);

% %% MOSQP
% clear Problem;
% Problem.objfun=@cost;    % objective function
% Problem.objgrd=@fobjgrad;    % objective gradient function
% Problem.objhes=@hessian_fobj;    % objective Hessian function
% Problem.bl = -norm(lambda)*ones(size(x0))';
% Problem.bu = norm(lambda)*ones(size(x0))';
%
% Problem.confun =@fcconst;    %-> nonlinear constraints function
% Problem.conjac =@fcgrad;   %-> nonlinear Jacobian function
%
% Problem.nobj=1;              % number of objective functions
% Problem.nc=1;                % number of nonlinear constraints
% Problem.ncl=0;          % lower bound on nonlinear constraints
% Problem.ncu=delta^2/2;   % upper bound on nonlinear constraints
% Problem.init=1;
% Problem.output=0;
% Problem.display=1;
% % Problem.X0  = x0;      %-> column-wise set of point to include in the
%
% [x,F]=MOSQP(Problem);


%%

U = vec2mat(x);
P = ktensor(U);
output = history;

    function [f,fc] = fobjconst(x)
        
        U = vec2mat(x);
        U = normalizeU(U);
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        % objective function
        f = 1/2*sum(prod(Unorm,1));
        
        
        if bound_ell2_constraint  % For min f(x) + mu/2 * 
            f = f + mu/2 * norm(x)^2;
        end
        
        
        % constraint values
        P = ktensor(U);
        fc = (normX^2 + norm(P)^2 - 2* innerprod(X,P))/2;

        [f,fc] = deal(fc,f);
        
        fc = fc - delta2/2;
        
%         if f<0
%             1
%         end
        % for the constraint sum_r |u{nr}|^2 <  xi
%         if bound_ell2_constraint
%             fc(2) = (norm(x)^2 - xi)/2;
%         end
    end


    function [gf,gfc] = fgrad(x)
        U = vec2mat(x);
        U = normalizeU(U);
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        % gradient of the objective function
        G = [];
        for n = 1:N
            Gamma = prod(Unorm([1:n-1 n+1:N],:));
            Gn = U{n} * diag(Gamma);
            G = [G; Gn(:)];
        end
        gf = G(:);
          
        
        % gradient of the constraint function
        UtU = cellfun(@(x) x'*x,U,'uni',0);
        UtU = reshape(cell2mat(UtU(:)'),R,R,[]);
        gfc = [];
        for n = 1:N
            Gamma_n = prod(UtU(:,:,[1:n-1 n+1:N]),3);
            gradn = -mttkrp(X,U,n) + U{n} * Gamma_n;
            gfc = [gfc ; gradn(:)];
        end
        gfc = gfc(:);
        
        % for the constraint sum_r |u{nr}|^2 <  xi
%         if bound_ell2_constraint
%             gfc(:,2) = x;
%         end

        [gf,gfc] = deal(gfc,gf);
        if bound_ell2_constraint   % min f(x) + mu/2 * |x|^2
            gf = gf + mu*x;
        end
    end


    function H = hessian_ff(x,lambda)
        % hessian of the objective function
        U = vec2mat(x);
        U = normalizeU(U);
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        D = diag([kron(prod(Unorm([2 3],:),1),ones(1,szX(1))),...
            kron(prod(Unorm([1 3],:),1),ones(1,szX(2))),...
            kron(prod(Unorm([1 2],:),1),ones(1,szX(3)))]);
        
        za = [];zb = [];zc = [];
        for r = 1:R
            za = blkdiag(za,U{1}(:,r));
            zb = blkdiag(zb,U{2}(:,r));
            zc = blkdiag(zc,U{3}(:,r));
        end
        Zo = blkdiag(za,zb,zc);
        
        Ko = [zeros(R) diag(Unorm(3,:)) diag(Unorm(2,:));
            diag(Unorm(3,:)) zeros(R) diag(Unorm(1,:))
            diag(Unorm(2,:)) diag(Unorm(1,:)) zeros(R)] ;
        H = D + 2 * Zo*Ko*Zo.';
        
        if 0
            [H,Zob,Ko] = hessian_objfunction(U);
        end
        
        %%
        % hessian of the constraint function
        if 0
            [G,Z,Kc,V,F,Kcm,Gamma] = cp_hessian(U);
            % norm(V(:,1:R+1:end) - Zob,'fro')
            % Hc = G + Z*K0*Z'+ V*F*V';
            
            Prr = per_vectrans(R,R);
            % K0 = bdiag(Prr*diag(K0m(n,:))
            % F = Prr * diag(Gamma)
            
            % B : blockdiagonal matrix
            % Bn = Gamma_{-n} \ox I_In  + (I \ox Un) * Prr * diag(K0_n) *  (I \ox Un')
            %
            %% Z * K0 * Z' = blkdiag(kron(Un,Un) * diag(Kn) * Prr)
            %t1 = Z(1:szX(1)*R,1:R^2)*Prr*diag(reshape(K0m(:,:,1),[],1))*Z(1:szX(1)*R,1:R^2)';
            ZKZ = [];
            for n = 1:N
                tn = bsxfun(@times,kron(U{n},U{n}),reshape(Kcm(:,:,n),[],1)');
                tn = reshape(tn*Prr,[szX(n)  szX(n) R R]);
                tn = double(tenmat(tn,[1 3]));
                ZKZ = blkdiag(ZKZ,tn);
            end
            % Hc = blkdiag(Gamma_n \ox I_(In) +  ZKZ_n) + V*F*V'
            
            
        else % construct the Hessian from the Jacobian 
            % Jacobian
            J = [];
            for n = 1:N
                Pn = permute_vec(szX,n);
                Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(szX(n)));
                J = [J Pn'*Jn];
            end
            % Hessian
            Hc = J'*J ;
        end
        
        [Hc,H] = deal(H,Hc);
         
        % for the constraint |\theta|^2 < xi
        if bound_ell2_constraint
            H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + mu;
        end
        
        %%
        H = H + lambda(1) * Hc;
        
        %% check the full form of H
        if 0
            [G,Z,Kc,V,F,Kcm,Gamma,Gammas] = cp_hessian(U);
            Prr = per_vectrans(R,R);
            beta = diag(Gamma);
            Gamma_lda = 2*lambda(1)*diag(beta) +  Gamma;
            Vc = mat2cell(V,szX(:)*R,R^2);
            Blk1 = [];
            for n = 1:N
                Gamma_n = prod(Gammas(:,:,[1:n-1 n+1:end]),3);
                beta_n = diag(Gamma_n);
                
                Gamma_n_lda =lambda(1)* diag(beta_n) +   Gamma_n;
                
                Blk1_n = kron(Gamma_n_lda,eye(szX(n))) - Vc{n} * Prr*diag(Gamma_lda(:))*Vc{n}';
                Blk1 = blkdiag(Blk1,Blk1_n);
            end
            
            
            Lradjterm = V * Prr*diag(Gamma_lda(:)) * V';
            Hlda = Blk1 + Lradjterm;
            norm(H- Hlda,'fro')
            
        end
        
        %%
        
%         % for the constraint |\theta|^2 < xi
%         if bound_ell2_constraint
%             H(1:size(H,1)+1:end) = H(1:size(H,1)+1:end) + lambda(2);
%         end
        
        %% Correct the Hessian 
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
    function H = compact_hessian_fc(x,lambda)
        % hessian of the objective function
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
        Gamma_lda = 2*diag(lambda(1) * beta) +  Gamma;
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
%         norm(H- Hlda,'fro')
         
        %%
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
    function [H,Zob,beta] = hessian_objfunction(U)
        % H = D + 2*Z*K*Z'
        % or
        %
        % H = blkdiag_n(kron(beta_{-n},1_In)) ...
        %     + blkdiag_n( blkdiag(u_nr * u_nr' * beta_r/beta_{n,r}^2)
        %     + W * diag(beta) * W'
        %
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        D = diag([kron(prod(Unorm([2 3],:),1),ones(1,szX(1))),...
            kron(prod(Unorm([1 3],:),1),ones(1,szX(2))),...
            kron(prod(Unorm([1 2],:),1),ones(1,szX(3)))]);
        
        za = [];zb = [];zc = [];
        for r = 1:R
            za = blkdiag(za,U{1}(:,r));
            zb = blkdiag(zb,U{2}(:,r));
            zc = blkdiag(zc,U{3}(:,r));
        end
        
        % Form 1
        if 0
            Zo = blkdiag(za,zb,zc);
            
            Ko = [zeros(R) diag(Unorm(3,:)) diag(Unorm(2,:));
                diag(Unorm(3,:)) zeros(R) diag(Unorm(1,:))
                diag(Unorm(2,:)) diag(Unorm(1,:)) zeros(R)] ;
            H = D + 2 * Zo*Ko*Zo.';
        end
        
        % FORM 2
        %Kom = -bsxfun(@rdivide,beta,Unorm.^2);
        %Kom =  diag(reshape(Kom',[],1));
        %Kz = [diag(1./Unorm(1,:));
        %      diag(1./Unorm(2,:));
        %      diag(1./Unorm(3,:))];
        % Ko = Kom + Kz*diag(Beta)*Kz'
        beta = prod(Unorm);
        za = za*diag(1./Unorm(1,:));
        zb = zb*diag(1./Unorm(2,:));
        zc = zc*diag(1./Unorm(3,:));
        Zob = [za;zb;zc];
        
        H = D - 2* blkdiag(za*diag(beta)*za',zb*diag(beta)*zb',zc*diag(beta)*zc') ...
            + 2* Zob* diag(beta)*Zob';
    end


    function [G,Z,K0,V,F,K0m,Gamma,Gamma_n] = cp_hessian(U)
        %
        % G is a blkdiag matrix
        % Z * K0 * Z' is a blkdiagonal matrix
        % V is a partition matrix of (I_R \ox U)*diag(Dn)
        % of size InR x R^2
        % F: Pr * diag(Gamma) : R^2 x R^2
        % Hc = G + Z*K0*Z'+ V*F*V';
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
        
        K0m = -bsxfun(@rdivide,Gamma,Gamma_n.^2);
        K0 = kron(eye(N),Prr) * diag(reshape(K0m,[],1));
        
        %  kron(eye(N),Prr) *D = D * Prr;
        V = [];
        for n = 1:N
            temp = bsxfun(@rdivide,kron(eye(R),U{n}),reshape(Gamma_n(:,:,n),1,[]));
            V = [V;temp];
        end
        % H = G + Z * K0 * Z' + V * F * V';
    end

%% MOSQP
    function  [f] = cost(x,~)
        %U = mat2cell(reshape(x,[],R),szX(:),R);
        U = vec2mat(x);
        U = normalizeU(U);
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        f = 1/2*sum(prod(Unorm,1));
    end

    function gf = fobjgrad(x,~)
        U = vec2mat(x);
        U = normalizeU(U);
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        % gradient of the objective function
        G = [];
        for n = 1:N
            Gamma = prod(Unorm([1:n-1 n+1:N],:));
            Gn = U{n} * diag(Gamma);
            G = [G; Gn(:)];
        end
        gf = G(:);
    end


    function [fc] = fcconst(x)
        U = vec2mat(x);
        U = normalizeU(U);
        
        % constraint values
        P = ktensor(U);
        fc = (normX^2 + norm(P)^2 - 2* innerprod(X,P))/2;
    end


    function gfc = fcgrad(x)
        U = vec2mat(x);
        U = normalizeU(U);
        
        % gradient of the constraint function
        UtU = cellfun(@(x) x'*x,U,'uni',0);
        UtU = reshape(cell2mat(UtU(:)'),R,R,[]);
        gfc = [];
        for n = 1:N
            Gamma_n = prod(UtU(:,:,[1:n-1 n+1:N]),3);
            gradn = -mttkrp(X,U,n) + U{n} * Gamma_n;
            gfc = [gfc ; gradn(:)];
        end
        gfc = gfc(:)';
    end


    function H = hessian_fobj(x,~)
        % hessian of the objective function
        U = vec2mat(x);
        U = normalizeU(U);
        
        Unorm = cellfun(@(x) sum(x.^2), U,'uni',0);
        Unorm = cell2mat(Unorm(:));
        
        D = blkdiag(kron(diag(prod(Unorm([2 3],:),1)),eye(szX(1))),...
            kron(diag(prod(Unorm([1 3],:),1)),eye(szX(2))),...
            kron(diag(prod(Unorm([1 2],:),1)),eye(szX(3))));
        
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
    end
 
    function stop = outfun(x,optimValues,state)
        stop = false;
        
        switch state
            case 'init'
                % hold on
            case 'iter'
                % Concatenate current point and objective function
                % value with history. x must be a row vector.
                history.fval = [history.fval; optimValues.fval];
                %history.constrviolation = [history.constrviolation; optimValues.constrviolation];
                history.firstorderopt = [history.firstorderopt; optimValues.firstorderopt];
                
                %history.x = [history.x; x];
            case 'done'
                %hold off
            otherwise
        end
        if optimValues.fval <= min_error
            stop = true;
        end
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
param.addOptional('fitmax',1-1e-10);
param.addOptional('linesearch',true);
%param.addParamValue('verify_convergence',true,@islogical);
param.addOptional('ineqnonlin',true,@islogical);
% true for the inequality contraint  |X - X_hat(\theta)|_F <= delta
% false for the equality contraint   |X - X_hat(\theta)|_F = delta
%

param.addOptional('normX',[]);

param.addOptional('bound_ell2_constraint',0); % if constraint |\theta|_2^2 < \xi
param.addOptional('Correct_Hessian',true,@islogical);
param.addOptional('LagrangeMultipliers',[]);

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