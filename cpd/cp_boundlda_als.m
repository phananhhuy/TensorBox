function [P,output] = cp_boundlda_als(X,R,delta,opts)
% Fast ALS for constrained CPD
%
%  min         \|X - P\|_F 
%  subject to   sum_r  \lambda_r^2  < delta^2.
% 
% lambda_r: intensity of rank-1 tensors
%
%  P is a ktensor of U1, ..., UN
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
% See also: cp_als
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.
% 2012, Fast CP gradient
% 2013, Fix fast CP gradients for sparse tensor
% 2013, Extend GRAM intialization for higher order CPD



%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

if param.linesearch
    param.TraceFit = true;
end
%% Parameters for linesearch
param_ls = struct('acc_pow',2, ... % Extrapolate to the iteration^(1/acc_pow) ahead
    'acc_fail',0, ... % Indicate how many times acceleration have failed
    'max_fail',4); % Increase acc_pow with one after max_fail failure

%%
N = ndims(X); I = size(X);
if isempty(param.normX)
    normX = norm(X);
else
    normX = param.normX;
end

if isa(X,'tensor')
    IsReal = isreal(X.data);
elseif isa(X,'ktensor') || isa(X,'ttensor')
    IsReal = all(cellfun(@isreal,X.u));
end

%% Initialize factors U
param.cp_func = str2func(mfilename);
Uinit = cp_init(X,R,param); U = Uinit;

%% Output
if param.printitn ~=0
    fprintf('\n ALS algorithm for CPD with bound norm constraint\n');
end

if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%% Reorder tensor dimensions and update order of factor matrices
% Permute the tensor to an apprppriate model with low computational cost 
if N< 14
    [Is,permI] = sort(I);
    [ns,bestord] = cost_als(Is);
    bestord = permI(bestord);
    p_perm =[];
    if any(bestord~=(1:N))
        p_perm = bestord;
        X = permute(X,p_perm);
        U = U(p_perm);
        I = I(p_perm);
    end
else
    p_perm = [];
    if ~issorted(I)
        [I,p_perm] = sort(I);
        X = permute(X,p_perm);
        U = U(p_perm);
    end
    Jn = cumprod(I); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
    ns = find(Jn<=Kn,1,'last');
end
updateorder = [ns:-1:1 ns+1:N];

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


P = arrange(ktensor(U));
lambda= P.lambda;
U = P.U;


UtU = zeros(R,R,N);
for n = 1:N
    UtU(:,:,n) = U{n}'*U{n};
end
% lambda = ones(R,1);

%% Main Loop: Iterate until convergence
Pmat = [];Pls = false;flagtol = 0;
lambda_ = lambda.';
cost_ = norm(lambda)^2;

for iter = 1:param.maxiters
    
%     % Do linesearch
%     if (param.linesearch == true) && (iter>5)
%         %Uold2 = Uold;Uold = U;
%         param_ls.alpha = iter;
%         P = ktensor(lambda,U);
%         if ~param.TraceFit
%             param_ls.mod_err = norm(P)^2 - 2 * real(innerprod(X,P));
%         else
%             param_ls.mod_err = normresidual^2 - normX^2;
%         end
%         [Pls,param_ls] = cp_linesearch(X,P,Uold,param_ls);
%     end
%     
    if param.verify_convergence==1
        if param.TraceFit, fitold = fit;end
        if param.TraceMSAE, msaeold = msae;end
    end
%     
%     if (param.verify_convergence==1) || (param.linesearch == true)
%         Uold = U;
%     end
%     
%     if (param.linesearch == true) && isa(Pls,'ktensor')
%         U = Pls.U;U{1} = U{1} * diag(Pls.lambda);
%         for n = 1:N,  UtU(:,:,n) = U{n}'*U{n}; end
%     end    
    
    % Iterate over all N modes of the tensor
%     lambda = ones(R,1);
    for n = updateorder(1:end) %[1 3 2] %
        U{n} = bsxfun(@times,U{n},lambda.');lambda = ones(R,1);
        % Calculate Unew = X_(n) * mkhatrirao(all U except n, 'r').
        if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
            G = mttkrp(X,U,n);
        elseif isa(X,'tensor') || isa(X,'sptensor')
            if (n == ns) || (n == ns+1)
                [G,Pmat] = cp_gradient(U,n,X);
            else
                [G,Pmat] = cp_gradient(U,n,Pmat);
            end
        end

        
        Gamma = (prod(UtU(:,:,[1:n-1 n+1:N]),3).');
        %Fq = chol(Gamma);% fq'*fq = Qn
        [ug,eg] = eig(Gamma);
        eg = diag(eg);[eg,ide] = sort(eg,'ascend');ug = ug(:,ide);
        eg = eg.';
         
        % debug   
        F = G*ug;
       
        % norm(X- full(ktensor(U))^2 - (norm(X(:))^2+ trace((U{n}*ug) *
        % diag(eg) * (U{n}*ug)') - 2 * trace(F*(U{n}*ug)'))
        % min  delta^2 * trace(Z * Sigma * Z') - 2 * delta * trace(F*Z')
        %  subject to  Z'*Z < 1
        %
        
        % current cost 
        if 0
            norm(X - full(ktensor(U)))^2
            
            Z = (U{n}*ug);
            trace(Z * diag(eg) * Z') - 2 * trace(F*Z') + normX^2
            
            % current cost  in vector form
            z = Z(:);
            (kron(eg,ones(1,I(n))))*z.^2 - 2 * F(:)'*z(:) + normX^2
            %z' * diag(kron(eg,ones(1,I(n)))) * z - 2 * F(:)'*z(:) + normX^2
        end
        
%         zcvx1= cvx_sqp_diag(kron(eg,ones(1,I(n)))',F(:),delta);
%         z = zcvx1(:);
%         z' * diag(kron(eg,ones(1,I(n)))) * z - 2 * F(:)'*z(:) + normX^2
%         
%         zcvx2= cvx_sqp_diag(eg',sqrt(sum(F.^2))',delta);
%         z = F*diag(zcvx2'./sqrt(sum(F.^2)));
%         z = z(:);
%         z' * diag(kron(eg,ones(1,I(n)))) * z - 2 * F(:)'*z(:) + normX^2
        
%         % minimize using cvx
%         zcvx2= cvx_sqp_diag(eg',sqrt(sum(F.^2))',delta);
%         z = F*diag(zcvx2'./sqrt(sum(F.^2)));
%         xn = z;
%         z = z(:);
        %z' * diag(kron(eg,ones(1,I(n)))) * z - 2 * F(:)'*z(:) + normX^2
        
%         % normalization and shift
            cvxss = 0;
            if cvxss
                es = eg'*delta/norm(F,'fro');
                cn = sqrt(sum(F.^2))'/norm(F,'fro');
                zcvx2= cvx_sqp_diag(es,cn,1);
                z = F*diag(zcvx2'./sqrt(sum(F.^2)));
                z = z(:)*delta;
                (kron(eg,ones(1,I(n))))*z.^2 - 2 * F(:)'*z(:) + normX^2
                %z' * diag(kron(eg,ones(1,I(n)))) * z - 2 * F(:)'*z(:) + normX^2
            end
        
        
%         [z,fval,lda2] = qp_sphere_diagq(delta*[0;eg'], -[0;sqrt(sum(F.^2))']);
%         z = z(2:end)*delta;
%         
%         
%         [z,fval,lda2] = qp_sphere_diagq(delta*[0; kron(eg(:),ones(I(n),1))], -[0;F(:)]);
%         z = z(2:end)*delta;
        
        % MIN   1/2* x'*diag(s)*x + c'*x        subject to x'*x <=1
        % MEthOD 1: move the factor matrices to have a large lambda         
        %s = delta*eg/norm(F(:)); c = -F/norm(F(:));
        s = delta*eg; c = -F;
        %s = (s-s(1))/norm(F(:))+1;
        cn = -sqrt(sum(c.^2))';
        
        if param.enforce_equal_norm 
            [z,fval,lda2] = qp_sphere_diagq(s(:),cn);
            
        else
            % add into z one more variable to give a new vector z'*z = 1
            % this expands s and c by a zero and solve the QP in sphere by a SQP
            [z,fval,lda2] = qp_sphere_diagq([0;s(:)],[0;cn]);
            %z = bsxfun(@rdivide,c,(lda2-s));
             z = z(2:end,1); 
%              if lda2~=1
%                  1;
%              end 
        end
        
        normF = sqrt(sum(F.^2));
        innz = normF~=0;
%         if any(normF==0)
%             1
%         end
        F(:,innz) = F(:,innz)*diag(z(innz)'./normF(innz));
        
        % OR enforce z on the circle  z'*z = 1
        
        xn = F*delta;
        
        Un = xn*ug'; 
        
        % norm(Un-x(:))
        
        %[norm(U{n}(:)) norm(Un(:))]
        U{n} = reshape(Un,[],R);
        
        
        % Innerproduct for fast computation of approximation error
        if param.TraceFit && (n == updateorder(end))
            innXXhat = sum(sum(U{updateorder(end)}.*conj(G)));
        end
        
        % Normalize each vector to prevent singularities in coefmatrix
%         if iter == 1
            lambda = sqrt(sum(abs(U{n}).^2,1)).'; %2-norm % fixed for complex tensors
%         else
%             lambda = max( max(abs(U{n}),[],1), 1 ).'; %max-norm % fixed for complex tensors
%         end
        U{n} = bsxfun(@rdivide,U{n},lambda.');
        UtU(:,:,n) = U{n}'*U{n};
        
        if ~IsReal % complex conjugate of U{n} for fast CP-gradient
            U{n} = conj(U{n});
        end
        
        % cost = |lambda|^2
        lambda_ = [lambda_ ;lambda(:).'];
%         cost_curr = 
%         cost_ = [cost_; cost_curr];
    end
    U{updateorder(1)} = bsxfun(@times,U{updateorder(1)},lambda.');lambda = ones(R,1);
    UtU(:,:,updateorder(1)) = U{updateorder(1)}'*U{updateorder(1)};
    UtU(:,:,updateorder(1)) = conj(UtU(:,:,updateorder(1))); % fixed complex conjugate due to conj(U{n})
    
    if param.verify_convergence==1
        %if param.TraceFit 
            P = ktensor(U);
            normresidual = sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod((P),X)));
%             normresidual = sqrt(normX^2 + sum(sum(prod(UtU,3))) - 2*real(innXXhat));
            
            fit = 1 - (normresidual/ normX); %fraction explained by model
            cost_ = [cost_; fit];
            cost_diff = abs(fitold - fit);
%             cost_diff = abs(cost_(end)-cost_(end-N))/cost_(end-N);
            if (iter > 1) && (cost_diff < param.tol) % Check for convergence
                flagtol = flagtol + 1;
            else
                flagtol = 0;
            end
            stop(1) = flagtol >= 10;
            
            
            %stop(3) = fit >= param.fitmax;
            
            
            if nargout >=2
                output.Fit = [output.Fit; iter fit];
                %output.cost = [output.cost; iter norm(lambda_(end,:))^2];
            end
        %end
        
%         if param.TraceMSAE 
%             msae = SAE(U,Uold);
%             msaechange = abs(msaeold - msae); % SAE changes
%             stop(2) = msaechange < param.tol*abs(msaeold);
%             if nargout >=2
%                 output.MSAE = [output.MSAE; msae];
%             end 
%         end
        
        if mod(iter,param.printitn)==0
            fprintf(' Iter %2d: ',iter);
%             fprintf('|lambda|^2 = %e diff = %7.1e ', cost_(end), cost_diff);
%             if param.TraceFit
                fprintf('fit = %e fitdelta = %7.1e ', fit, cost_diff);
%             end
%         if param.TraceMSAE
%             fprintf('msae = %e delta = %7.1e', msae, msaechange);
%         end
            fprintf('\n');
        end
        
        % Check for convergence
        if (iter > 1) && any(stop)
            break;
        end
    end
end

%% Clean up final result
% Convert conj(U{n}) back to U{n}
if ~IsReal % complex conjugate of U{n} for fast CP-gradient
    U = cellfun(@conj,U,'uni',0);
end

% Arrange the final tensor so that the columns are normalized.
P = ktensor(lambda,U);

% Normalize factors and fix the signs
% P = arrange(P);P = fixsigns(P);

% Display final fit
if param.printitn>0
    %innXXhat = sum(sum(U{updateorder(end)}.*conj(G)));
    %normresidual = sqrt(normX^2 + sum(sum(prod(UtU,3))) - 2*real(innXXhat));
    normresidual = sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod(X,(P)) ));
    fit = 1 - (normresidual / normX); %fraction explained by model
    fprintf('Final fit = %e \n', fit);
end

% Rearrange dimension of the estimation tensor 
if ~isempty(p_perm)
    P = ipermute(P,p_perm);
end
if nargout >=2
    output.NoIters = iter;
    output.cost = cost_;
    output.lambda = lambda_;
end

%% CP Gradient with respect to mode n
    function [G,Pmat] = cp_gradient(A,n,Pmat)
        persistent KRP_right0;
        right = N:-1:n+1; left = n-1:-1:1;
        % KRP_right =[]; KRP_left = [];
        if n <= ns
            if n == ns
                if numel(right) == 1
                    KRP_right = A{right};
                elseif numel(right) > 2
                    [KRP_right,KRP_right0] = mkhatrirao(A(right));
                elseif numel(right) > 1
                    KRP_right = mkhatrirao(A(right));
                else
                    KRP_right = 1;
                end
                
                if isa(Pmat,'tensor')
                    Pmat = reshape(Pmat.data,[],prod(I(right))); % Right-side projection
                elseif isa(Pmat,'sptensor')
                    Pmat = reshape(Pmat,[prod(size(Pmat))/prod(I(right)),prod(I(right))]); % Right-side projection
                    Pmat = spmatrix(Pmat);
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
                KRP_left = mkhatrirao(A(left));
%                 if (isempty(KRP_2) && (numel(left) > 2))
%                     [KRP_left,KRP_2] = mkhatrirao(A(left));
%                 elseif isempty(KRP_2)
%                     KRP_left = mkhatrirao(A(left));
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
                    G = (KRP_left.'*T).';
                end
            else
                %G = squeeze(Pmat);
                G = reshape(Pmat,[],R);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left}.';
                elseif numel(left) > 1
                    KRP_left = khatrirao_t(A(left));
                    %KRP_left = mkhatrirao(A(left));KRP_left = KRP_left';
                else 
                    KRP_left = 1;
                end
                if isa(Pmat,'tensor')
                    T = reshape(Pmat.data,prod(I(left)),[]);
                elseif isa(Pmat,'sptensor')
                    T = reshape(Pmat,[prod(I(left)) prod(size(Pmat))/prod(I(left))]); % Right-side projection
                    T = spmatrix(T);
                else
                    T = reshape(Pmat,prod(I(left)),[]);
                end
                %
                Pmat = KRP_left * T;   % Left-side projection
            else
                if R>1
                    Pmat = reshape(Pmat,R,I(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}.');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,I(left(1)),[]);
                    Pmat = A{left(1)}.'* Pmat;
                end
            end
            
            if ~isempty(right)
                T = reshape(Pmat,[],I(n),prod(I(right)));
                
                if (n == (ns+1)) && (numel(right)>=2)
                    %KRP_right = KRP_right0;
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right0.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T, R,[]).';
                    else
                        %G = squeeze(T) * KRP_right0;
                        G = reshape(T,[],prod(I(right))) * KRP_right0;
                    end
                else
                    KRP_right = mkhatrirao(A(right));
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T,R,[]).';        % Right-side projection
                    else
                        %G = squeeze(T) * KRP_right;
                        G = reshape(T,I(n),[]) * KRP_right;
                    end
                end
            else
                %G = squeeze(Pmat)';
                G = reshape(Pmat,R,[]).';
            end
            
        end
        
        %         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
        %             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
        %             size(KRP_left,1),size(KRP_left,2),...
        %             size(KRP_right,1),size(KRP_right,2))
    end


%% CP Gradient with respect to mode n
    function [G,Pmat] = cp_gradient3(A,n,Pmat)
        persistent KRP_right0;
        right = N:-1:n+1; left = n-1:-1:1;
        % KRP_right =[]; KRP_left = [];
        if n <= ns
            if n == ns
%                 if numel(right) == 1
%                     KRP_right = A{right};
%                 elseif numel(right) > 2
%                     [KRP_right,KRP_right0] = mkhatrirao(A(right));
%                 elseif numel(right) > 1
%                     KRP_right = mkhatrirao(A(right));
%                 else
%                     KRP_right = 1;
%                 end
%                 
%                 if isa(Pmat,'tensor')
%                     Pmat = reshape(Pmat.data,[],prod(I(right))); % Right-side projection
%                 elseif isa(Pmat,'sptensor')
%                     Pmat = reshape(Pmat,[prod(size(Pmat))/prod(I(right)),prod(I(right))]); % Right-side projection
%                     Pmat = spmatrix(Pmat);
%                 else
%                     Pmat = reshape(Pmat,[],prod(I(right))); % Right-side projection
%                 end
%                 Pmat = Pmat * KRP_right ;
                

                [Pmat,Pmat2] = tenxkr(Pmat.data,A(right),right,1);
             
            else
                Pmat = reshape(Pmat,[],I(right(end)),R);
                if R>1
                    Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],I(right(end)),R));
                    Pmat = sum(Pmat,2);    % fast Right-side projection
                else
                    Pmat = Pmat * A{right(end)};
                end
                Pmat = reshape(Pmat,[I(left) I(n) R]);
            end
            
            if ~isempty(left)       % Left-side projection
                G = tenxkr((Pmat),A(left),left,2);
            else
                G = reshape(Pmat,[],R);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left}.';
                elseif numel(left) > 1
                    KRP_left = khatrirao_t(A(left));
                    %KRP_left = mkhatrirao(A(left));KRP_left = KRP_left';
                else 
                    KRP_left = 1;
                end
                if isa(Pmat,'tensor')
                    T = reshape(Pmat.data,prod(I(left)),[]);
                elseif isa(Pmat,'sptensor')
                    T = reshape(Pmat,[prod(I(left)) prod(size(Pmat))/prod(I(left))]); % Right-side projection
                    T = spmatrix(T);
                else
                    T = reshape(Pmat,prod(I(left)),[]);
                end
                %
                Pmat = KRP_left * T;   % Left-side projection
%                   Pmat = tenxkr(Pmat.data,A(left),left,1);  
            else
                if R>1
                    Pmat = reshape(Pmat,R,I(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}.');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,I(left(1)),[]);
                    Pmat = A{left(1)}.'* Pmat;
                end
                Pmat = reshape(Pmat,[R I(n) I(right)]);
            end
            
            if ~isempty(right)
                T = reshape(Pmat,[],I(n),prod(I(right)));
                
                if (n == (ns+1)) && (numel(right)>=2)
                    KRP_right0 = mkhatrirao(A(right));

                    %KRP_right = KRP_right0;
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right0.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T, R,[]).';
                    else
                        %G = squeeze(T) * KRP_right0;
                        G = reshape(T,[],prod(I(right))) * KRP_right0;
                    end
                else
                    KRP_right = mkhatrirao(A(right));
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T,R,[]).';        % Right-side projection
                    else
                        %G = squeeze(T) * KRP_right;
                        G = reshape(T,I(n),[]) * KRP_right;
                    end
                end
            else
                %G = squeeze(Pmat)';
                G = reshape(Pmat,R,[]).';
            end
            
        end
        
        %         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
        %             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
        %             size(KRP_left,1),size(KRP_left,2),...
        %             size(KRP_right,1),size(KRP_right,2))
    end


% %% CP Gradient with respect to mode n
%     function [G,Pmat] = cp_gradient2(A,n,Pmat)
%         persistent KRP_right0 KRP_2;
%         right = N:-1:n+1; left = n-1:-1:1;
%         % KRP_right =[]; KRP_left = [];
%         if n <= ns
%             if n == ns
%                 if numel(right) == 1
%                     KRP_right = A{right};
%                 elseif numel(right) > 2
%                     [KRP_right,KRP_right0] = mkhatrirao(A(right));
%                 elseif numel(right) > 1
%                     KRP_right = mkhatrirao(A(right));
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
%                 % KRP_left2 = mkhatrirao(A(left));
%                 if (rem(ns-n,2)==0) && (numel(left)>2)
%                     [KRP_left,KRP_2] = khatrirao_r2l(A(left));
%                 elseif rem(ns-n,2)==1 && (numel(left)>=2)
%                     KRP_left = KRP_2;
%                 else
%                     KRP_left = khatrirao_r2l(A(left));
%                 end
%                 
%                 T = reshape(Pmat,prod(I(left)),I(n),[]);
%                 
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
%                     %KRP_left = mkhatrirao(A(left));KRP_left = KRP_left';
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
%                     %KRP_right2 = mkhatrirao(A(right));
%                     if (rem(n-ns,2)==0) && (numel(right)>2)
%                         [KRP_right,KRP_2] = mkhatrirao(A(right));
%                     elseif (rem(n-ns,2)==1) && (numel(right)>1)
%                         KRP_right = KRP_2;
%                     else
%                         KRP_right = mkhatrirao(A(right));
%                     end
%                     
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

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = mkhatrirao(A)
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
function [K,K2] = khatrirao_r2l(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{end};
if nargout == 1
    for i = numel(A)-1:-1:1
        K = bsxfun(@times,reshape(K,[],1,R),reshape(A{i},1,[],R));
    end
elseif numel(A) > 2
    for i = numel(A)-1:-1:2
        K = bsxfun(@times,reshape(K,[],1,R),reshape(A{i},1,[],R));
    end
    K2 = reshape(K,[],R);
    K = bsxfun(@times,reshape(K,[],1,R),reshape(A{1},1,[],R));
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

param.addParamValue('enforce_equal_norm',false,@islogical);


param.addOptional('normX',[]); 

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end


%%

function [Pnew,param_ls] = cp_linesearch(X,P,U0,param_ls)
% Simple line search adapted from Rasmus Bro's approach.

alpha = param_ls.alpha^(1/param_ls.acc_pow);
U = P.U;
U{1} = U{1} * diag(P.lambda);

Unew = cellfun(@(u,uold) uold + (u-uold) * alpha,U,U0,'uni',0);
Pnew = ktensor(Unew);
mod_newerr = norm(Pnew)^2 - 2 * innerprod(X,Pnew);
            
if mod_newerr>param_ls.mod_err
    param_ls.acc_fail=param_ls.acc_fail+1;
    Pnew = false;
    if param_ls.acc_fail==param_ls.max_fail,
        param_ls.acc_pow=param_ls.acc_pow+1+1;
        param_ls.acc_fail=0;
    end
else
    param_ls.mod_err = mod_newerr;
end

end

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


function [X,X2] = tenxkr(X,U,modes,type)
% X_{1:n} * KRP(U{n+1},U{n+2},...,U{N})
% modes should be [k:N] or [1:m]

N = ndims(X);In = size(X);
X2 = [];
K = numel(U);    R = size(U{1},2);

% modes = N-numel(U)+1:N;
[modes,iio] = sort(modes);U = U(iio);
if modes(1) ~= 1 % modes = k:N % right side
    if type == 1  % right first
        X = reshape(X,[],In(end));
        X = X * U{end};
        In(end) = R;
        
        if K > 1
            for k = K-1:-1:2
                X = reshape(X,[prod(In(1:modes(k)-1)) In(modes(k)) R]);
                X = bsxfun(@times,X,reshape(U{k},1,[],R));
                X = sum(X,2);
            end
            X2 = X;
            k = 1;
            X = reshape(X,[prod(In(1:modes(k)-1)) In(modes(k)) R]);
            X = bsxfun(@times,X,reshape(U{k},1,[],R));
            X = sum(X,2);
        else
            X2 = [];
        end
        X = reshape(X,[In(1:N-K) R]);
    else % type == 2 % left first then right
        for k = K:-1:1
            X = reshape(X,[R, prod(In(2:modes(k)-1)) In(modes(k))]);
            X = bsxfun(@times,X,reshape(U{k}',R,1,[]));
            X = sum(X,2);
        end
        X = reshape(X,[R In(1:N-K)]);
    end

else % left side
    if type == 1
        X = reshape(X,In(1),[]);
        X = U{1}'*X;
        In(1) = R;
        
        if K > 1
            for k = 2:K-1
                X = reshape(X,[R, In(modes(k)) prod(In(modes(k)+1:end))]);
                X = bsxfun(@times,X,U{k}');
                X = sum(X,2);
            end
            X2 = X;
            k = K;
            X = reshape(X,[R, In(modes(k)) prod(In(modes(k)+1:end))]);
            X = bsxfun(@times,X,U{k}');
            X = sum(X,2);
        else
            X2 = [];
        end
        X = reshape(X,R,[]);
    else % right first then left
        for k = 1:K
            X = reshape(X,[In(modes(k)) prod(In(modes(k)+1:end-1)) R]);
            X = bsxfun(@times,X,reshape(U{k},[],1,R));
            X = sum(X,1);
        end
        X = reshape(X,[],R);
    end
end
end