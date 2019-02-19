function [X,out,cost,cone_d] = linreg_lrmx(y,A,szX,delta, opts)
% Solving the least squares minimization
% 
%   min rank(X) + mu * \| X \|_F^2 
% 
% 
%  subject to   \| y - A*vec(X)\| < delta 
%
%  where y is a vector of length K, 
%  A can be a structure matrix A = khatrirao(V,U)'.
%          or an numerical matrix of size IJ x K  
%
% 
% Problem 1 : When mu = 0, th optimization problem can be solved using the
% PPA algorithm or SPDT through CVX 
%
%           minimize     nuclear_norm(X)   
%           subject to   | y  - A(X)| <= delta
% 
% Problem 2: when mu ~= 0, 
%
%           minimize     nuclear_norm(X) + mu * \| X \|_F^2 
%           subject to   | y  - A(X)| <= delta
%
%  can be solved using the SDPT through CVX or TFOCS.
%
%
% Problem 3:  When X is with fixed rank 
%   
%           minimize    | y - A(X)|^2  
%           subject to  rank(X) = R
% 
%  This problem is solved using ALS or the trust-region algorithm on the Riemanian
%  manifold. See linreg_lrmx_als or linreg_lrmx_als.
%
%
% Phan Anh Huy, 24/03/2017
%

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    X = param; return
end

K = numel(y);

%% COmpression of K exceeds number of elements of X
%  A = Q*F
%  \| y - A*x\| = \|Q'*y - F*x\|
compression = (K > 2*prod(szX)) && (param.compression == true);



if compression   
    % Find subspace of A
    % compression when K >> I*J
    if isstruct(A)
        A = khatrirao(A.V,A.U)';
    end
    Q = A'*A;
    [Vq,Sq] = eig(Q);
    Sq = max(diag(Sq),0);
    Sq = sqrt(Sq);
    %Uq = A*Vq'/Sq;
    y0 = y;
    delta0 = delta;
    if min(Sq) == 0
        iss = find(Sq>1e-8);
        y = (Vq(:,iss)'*(A'*y))./Sq(iss); % compressed y
        Ax = diag(Sq(iss))*Vq(:,iss)';
    else
        y = (Vq'*(A'*y))./(Sq);
        Ax = diag(Sq)*Vq';
    end
 
    % define linear operator 
    
    Amap = @(X) Ax*X(:);% A* X(:)
    ATmap = @(b) reshape(Ax'*b,szX); % A'*b
    % adjust the delta 
    %   \|y - A*x\|^2 = \|yx - Ax*x\|^2 + \|y\|^2 - \|yx|^2 
    delta = sqrt(max(0,delta0^2 - norm(y0)^2 + norm(y)^2));

else
    if isstruct(A)
        % Amap   = A*X(:)
        Amap = @(X) sum(A.U.*(X*A.V),1)';
        % Atmap = A'*y
        %ATmap = @(b) reshape(A.U*bsxfun(@times,A.V',b),[],1);% a vector
        ATmap = @(b) reshape(A.U*bsxfun(@times,A.V',b),szX);% a vector
    elseif isnumeric(A)
        Amap = @(X) A*X(:);% A* X(:)
        ATmap = @(b) reshape(A'*b,szX); % A'*b
    end
end

X0  = param.init;

if iscell(X0)
    X0 = X0{1}*X0{2}';
end

if ischar(param.init)
    switch param.init
        case 'empty'
            X0  = [];
            
        case 'lowrank1'  % low rank to reshape(A'*b.szX)
            G=ATmap(b);
            err = norm(G - Amap(G));
            if err < delta
                X0 = G;
            else
                [u,s,v] = svd(G);s = diag(s);
                Rr = min(szX);
                AF = zeros(K,Rr);
                for r = 1:Rr
                    AF(:,r) = Amap(u(:,r) * v(:,r)');
                end
                Qf = AF'*AF;
                b2 = AF'*b;
                err_r = zeros(Rr,1);
                for r = 1:Rr
                    err_r(r) = s(1:r)'*Qf(1:r,1:r)*s(1:r) - 2*b2(1:r)'*s(1:r);
                end
                R = find(err_r<delta^2,1,'first');
                X0 = u(:,1:R) * diag(s(1:R))*v(:,1:R)';
            end
    end
end

% if param.mu == 0
%     param.method = 'sdpt';
% end


switch param.method
    case 'ppa'
        %  min          sum(svd(X))
        %  subject to   norm(b1-A1(X)) <= delta        
        param.plotyes = 0;
       
        [X,iter,time,sd,hist] = dualPPA(Amap,ATmap,y,delta,szX(1),szX(2),numel(y),0,0,param);
        
        out = hist.obj;
        
    case 'admm'
        
        %%
        % Define the prox of f2 see the function proj_B2 for more help
        tau=1;
        %param.gamma=1e1; % stepsize

        clear param2 param3;
        param3.A=Amap;
        param3.At=ATmap;
        param3.epsilon=delta;
        param3.y=y;
        param3.tight = 0;
        param3.maxit = 2000;
        param3.tol = 1e-6;
        f2.prox=@(x,T) proj_b2(x,T,param3);
        f2.eval=@(x) norm(Amap(x)-y)^2;
          
        % setting the function f1 (nuclear norm)
        param2.verbose=1;         
        f1.prox=@(x, T) prox_nuclearnorm(x, tau*T, param2);
        f1.eval=@(x) tau*norm_nuclear(x);   
        
        % setting the frame L
        L = @(x) x;
        %param.maxit = param.maxiters;
        % solving the problem
        [X,infos,objectiv,y_n,z_n]=admm(X0,f1,f2,L,param);
        out = objectiv;
        %%
         
        
    case {'sdpt' 'cvx'} %       via CVX
%         global cvx___;

%         cvx___.extra_params = X0;
        
        if param.mu == 0

            cvx_begin
                cvx_precision best
                variable X(szX(1),szX(2))
                minimize norm_nuc(X)
                subject to
                norm(Amap(X) - y ) <= delta
            cvx_end

%             optval =  cvx_optval;
%             cond_K = norm(Amap(X) - y );

        else
            mu = param.mu;
            if ~isempty(X0)
                X = X0;
            end
            % x0 = zeros(szX);

            cvx_begin
            cvx_precision best
                variable X(szX(1),szX(2))
                minimize norm_nuc(X) + mu/2 * sum_square(vec(X))  
                subject to
                norm(Amap(X) - y ) <= delta
            cvx_end
 %             optval =  cvx_optval;
%             cond_K = norm(Amap(X) - y );
        end
%         cvx___.extra_params = [];
        out = cvx_optval;
    case 'tfocs'
        %  minimize norm_nuc(X) + 0.5*mu*norm(X-X0,'fro').^2
        %        s.t.     ||A * x - b || <= epsilon

            
        delta = max(1e-6,delta);

        if isempty(param.mu) | (param.mu==0)
            param.mu = 1;
        end
        
        % for TFOCCS
        %ATmap2 = @(y) reshape(ATmap(y),szX);% same size of X
        tf_opts = [];
        tf_opts.alg    = 'AT'; % gradien descend 'AT'  'LLM'; 'GRA' N83
        tf_opts.maxIts = param.maxiters;
        tf_opts.continuation = (param.continuation == 1) || (param.continuation == true);
         
        z0 =[];
        %z0 = Amap(X0)-y;
        Aop = linop_handles({ szX [numel(y), 1]}, Amap, ATmap, 'R2R');
       %[X, out, opts ] = solver_sNuclearDN(Aop, y, delta, param.mu, [], z0, tf_opts ); 
       cone_d_ = [];
       while 1 
           % loop until the cone condition is satisfied 
           %  |y - A*x|
           [X,out] = tfocs_SCD( prox_nuclear, { Aop, -y }, prox_l2(delta), param.mu, [], z0, tf_opts);
           cone_d = norm(y-Amap(X));
           cone_d_ = [cone_d_; cone_d];
           if abs(cone_d - delta)<1e-4
               break
           else
               z0 = out.dual;
           end
           if (numel(cone_d_)> 3) && std(cone_d_(max(1,numel(cone_d_) - 3):end)) < 1e-2
               break
           end
       end
       
        
%         optval =  norm_nuclear(X) + param.mu/2*norm(X,'fro')^2;
%         cond_K = norm(Amap(X) - y );
        
%         vec = @(x) x(:);
%         mat = @(x) reshape(x,szX);
%         
%         if compression
%             z0 = Amap(X0)-yx;
%             %Aop = linop_handles({ szX [prod(szX), 1]}, Amap, ATmap, 'R2R');
%             
% %             Aop   = linop_matrix( Ax, 'R2R' ); % n1 x n2 is size of x
% %             It  = linop_handles({ szX,[prod(szX),1] }, vec, mat, 'R2R' );
% %             Aop   = linop_compose( Aop, It );
%             
%             [X, out, opts ] = solver_sNuclearDN2(Aop, yx, deltax, param.mu, [], z0, tf_opts );
%         else
%             z0 = Amap(X0)-y;
%             %Aop = linop_handles({ szX [K, 1]}, Amap, ATmap, 'R2R');
%             
%             Aop   = linop_matrix( A, 'R2R' ); % n1 x n2 is size of x
%             It  = linop_handles({ szX,[K,1] }, vec, mat, 'R2R' );
%             Aop   = linop_compose( Aop, It );
%             
% 
%             [X, out, opts ] = solver_sNuclearDN2(Aop, y, delta, param.mu, [], z0, tf_opts );
%         end

end
  
cost = norm_nuclear(X);
if param.mu 
    cost = cost + param.mu/2*norm(X,'fro')^2;
end
cone_d = norm(y-Amap(X));
if compression
     cone_d = sqrt(cone_d^2 + norm(y0)^2 - norm(y)^2);
end
end
 

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.StructExpand = true;
param.addParameter('init','empty',@(x) isnumeric(x) || iscell(x) || ismember(x,{'rand' 'ppa' 'empty'}));
param.addParameter('maxiters',2000);
param.addParameter('tol',1e-6);
param.addParameter('verbose',0);  
param.addParameter('mu',0); %in mu * |X - X0|_F^2
param.addParameter('gamma',1); %parameter used in ADMM algrithm 
param.addParameter('method','sdpt',@(x) ismember(x(1:3),{'sdp' 'cvx' 'ppa' 'tfo' 'adm'}));    % sdpt ppa  tfocs
param.addParameter('continuation',0);   % 0 1 
param.addParameter('compression',true);   % 0 1 

param.parse(opts);
param = param.Results;
end
 

 
function n = norm_nuclear(x)
if issparse(x)
    n = sum(svds(x, min(size(x))));
else
    n = sum(svd(x, 'econ'));
end

end
