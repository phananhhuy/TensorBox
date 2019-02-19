function [P,output] = cp_ttconv(X,R,opts)
% Fast algorithm for CP factorizes the N-way tensor X into factors of R
% components.
%
% The algorithm is based on TT-decomposition Xtt of the tensor X into N cores.
% Then a K-tensor fit to the TT-tensor Xtt
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
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki,
% See also: cp_als
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.
% 2012, Fast CP gradient
% 2013, Fix fast CP gradients for sparse tensor
% 2013, Extend GRAM intialization to higher order CPD



%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

% if param.linesearch
%     param.TraceFit = true;
% end

maxiters = param.maxiters;
% When components B are highly collinear, the matrices B^T is
% ill-condition, the damping paramter mu should be non-zero.
% This is oberserved when the objective function does not always decrease
%
% mu = 1e-6; % a damping parameter
mu = param.damping;

N = ndims(X); SzX = size(X);

%%
if param.printitn ~=0
    fprintf('\nTT-Conversion for CPD:\n');
end

if nargout >=2
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%% Stage 1: compression of the data tensor by a TT-tensor of rank-(R,...,R)
% If X is not a TT-tensor, first fit it by a TT-tensor of rank <= R.

% Xtt1 = tt_stensor(double(X),1e-6,SzX(:)',[1 R*ones(1,N-1) 1]');
% Xtt1 = tt_tensor(Xtt1);
% % Xtt = round(Xtt,1e-6,[1 R*ones(1,N-1) 1]');
% tttol = (1-.999)*norm(Xtt1)/sqrt(prod(SzX));
% Xtt = round(Xtt1,tttol,[1 R*ones(1,N-1) 1]');
% Xtt = tt_als(Xtt1,Xtt,10);

tt_opts = struct('tol',1e-9,'maxiters',100);

if ~isa(X,'tt_tensor')
    
    %Xtt = tt_stensor(double(X),1e-8,size(X)',[1 R*ones(1,N-1) 1]');
    %Xtt = tt_tensor(Xtt);
    Xtt1 = tt_tensor(double(X),tt_opts.tol,SzX,[1 R*ones(1,N-1) 1]');
    
    %Xtt = Xtt1;
    
    Xtt = round(Xtt1,tt_opts.tol,[1 R*ones(1,N-1) 1]');
    %     Xtt = tt_als(Xtt1,Xtt,tt_opts.maxiters);
    % %     tt_opts = tt_a2cu;
    tt_opts.init = Xtt;
    %     [Xtt,output] = tt_a2cu(double(X),[1 R*ones(1,N-1) 1]',tt_opts);
    [Xtt,output] = ttmps_a2cu(Xtt1,[1 R*ones(1,N-1) 1]',tt_opts);
    Xtt = TTeMPS_to_TT(Xtt);
else
    % if input is a TT-tensor, round its rank to R, or not exceed R
    % round the input TT-tensor to rank R
    Xtt = round(X,tt_opts.tol,[1 R*ones(1,N-1) 1]');
    % Further improvement by ALS with 10 sweeps
    %Xtt = tt_als(X,Xtt,tt_opts.maxiters);
    tt_opt = ttmps_a2cu;
    tt_opt.init = Xtt;
    [Xtt,output] = ttmps_a2cu(X,[1 R*ones(1,N-1) 1]',tt_opt);
    Xtt = TTeMPS_to_TT(Xtt);
end

%% Stage 2: Initialize factor matrices and conversion from TT to K-tensor
%if (ischar(param.init) && strcmp(param.init,'exact_tt_to_cp')) || isa(X,'tt_tensor')
if (ischar(param.init) && strcmp(param.init,'exact_tt_to_cp'))  
    Binit = exact_tt_to_cp(Xtt,R);
else
    param.cp_func = @cp_fastals;
    Binit = cp_init(X,R,param);
end
B = Binit;

%% Stage 3: % Fit a TT tensor of B to TTtensor of G
BtB = cellfun(@(x) x'*x,B,'uni',0);
BtB = reshape(cell2mat(BtB(:)'),[R,R,N]);

normX = norm(Xtt)^2;
% Get core tensors of Xtt
Ux = core2cell(Xtt);
Ux{1} = squeeze(Ux{1});
rankX = Xtt.r;


% % Pre-Compute Phi_left % matrices of size R x R, products of all cores form
% % 1:n-1 between two tensor Xtt and Btt
Phi_left = cell(N,1);
Phi_left{1} = 1;

% Pre-Compute Phi_right % matrices of size R x R, products of all cores form
% n+1:N between two tensor Xtt and Btt
Phi_right = cell(N,1);
Phi_right{N} = 1;
Phi_right{N-1} = Ux{N}*conj(B{N}); %fixed for complex data
for n = N-2:-1:1
    %dir = 'RL';upto = n+1; % upto = 2, 3, 4
    %Phi_right{n} = innerprod(Xtte,Btte, dir, upto ); % contraction between cores k of X and B where k = n+1:N
    Phi_right{n} = fmttkrp(Ux{n+1},conj(B{n+1}),Phi_right{n+1},1); %fixed for complex data
end

%%
err_iter = zeros(maxiters,2*N-2);
err = zeros(1,N);
err2= zeros(N,2);
lambda = ones(R,1);

debug = false;
if debug
    curr_err = [];
    Xttf = full(Xtt);
end

for kiter = 1:maxiters
    
    for update_dir = 1:2
        switch update_dir
            case {1 'LR'}
                dim_update = 1:N-1;
            case {2 'RL'}
                dim_update = N:-1:2;
        end
        
        for n = dim_update
            if debug
                Bo = B;
                Phi_lefto = Phi_left;
                Phi_righto = Phi_right;
                BtBo = BtB;
            end
            
            %%
            XQb = prod(BtB(:,:,[1:n-1 n+1:end]),3);
            
            XQleft = Phi_left{n};
            XQright = Phi_right{n};
            
            %% Update B{n} and assess Cost value
            if n == 1 % right
                % Objective function
                % f = |X - B|_F^2
                %   = |X|_F^2 + |B|_F^2 - 2 * Real(<X,B>)
                %   = |X|_F^2 + trace(Bn * Gamma.' * Bn') - 2 * Real(<X<n * Xn * X>n, [B_<n , Bn, B>n]))
                %   = |X|_F^2 + trace(Bn * Gamma.' * Bn') ...
                %      -2 * Real(trace(Xn_(2) * kron(X>n.', X<n).' * conj(krp(B>n,B<n))*B{n}')
                %   = |X|_F^2 + trace(Bn * Gamma.' * Bn') ...
                %      -2 * Real(trace(Xn_(2) * krp(X>n* conj(B>n), X<n.' * conj(B<n)) * B{n}')
                %   = |X|_F^2 + trace(Bn * Gamma.' * Bn') ...
                %      -2 * Real(trace(Xn_(2) * krp(Phi_>n, Phi<n) * B{n}')
                %  where
                %     Phi_>n = X>n * conj(B>n)
                %     Phi_<n = X<n.' * conj(B<n)
                %
                % The term "squeeze(Xtte.U{n}) * XQright" is mttkrp(X_cp,B,n);
                % debug
                
                if debug
                    Grad_n = Ux{n} * XQright;
                    e1 = norm(Xttf - khatrirao(B,'r') * lambda)^2;
                    e2 = normX + norm(ktensor(lambda,B))^2 - 2 * real(trace(Grad_n*diag(lambda)*B{n}'));
                    if abs(e1 - e2)/e1 > 1e-4
                        1;
                    end
                end
                
                if mu ~= 0
                    XQb = XQb + mu * eye(size(XQb));
                end
                 
                B{n} = Ux{n} * (XQright/XQb.');
                
                
                %err(n) = normX + norm(ktensor(B))^2 - 2 * real(trace((Ux{n} * XQright)'*B{n}));
                %err(n) = norm(Xttf - sum(khatrirao(B,'r'),2))^2;
                
                if R < SzX(n)
                    err(n) = normX - real(trace(B{n}'*Ux{n}*XQright));
                else
                    err(n) = normX - real(trace(Ux{n}*XQright*B{n}'));
                end
                
                if debug
                    if ~isempty(curr_err) &&  ((err(n)-curr_err)>1e-9)
                        1
                    end
                end
                
            elseif n == N
                % THe term squeeze(Xtte.U{n})'*XQleft is mttkrp(X_cp,B,n);
                if debug
                    Grad_n = Ux{N}.' * XQleft;
                    e1 = norm(Xttf - khatrirao(B,'r') * lambda)^2;
                    e2 = normX + norm(ktensor(lambda,B))^2 - 2 * real(trace(Grad_n*diag(lambda)*B{n}'));
                    if abs(e1 - e2)/e1 > 1e-4
                        1;
                    end
                end
                
                if mu ~= 0
                    XQb = XQb + mu * eye(size(XQb));
                end
                B{n} = Ux{N}.' * (XQleft/XQb.');
                
                %err(n) = normX + norm(ktensor(B))^2 - 2 * real(trace(Ux{n}.' * XQleft*B{n}'));
                %err(n) = norm(Xttf - sum(khatrirao(B,'r'),2))^2;
                
                if R > SzX(n)
                    err(n)  = normX - real(trace(Ux{N}.'*XQleft * B{n}'));
                else
                    err(n)  = normX - real(trace(B{n}'*Ux{N}.'*XQleft));
                end
                
                if debug
                    if ~isempty(curr_err) &&  ((err(n)-curr_err)>1e-9)
                        1
                    end
                end
            else
                % This term is  mttkrp(X_cp,B,n)
                %Grad_n = mttkrp(tensor(Ux{n}),{XQleft,B{n},XQright},2);
                
                Grad_n = fmttkrp(Ux{n},XQleft,XQright,2);
                
                
                if debug                    
                    e1 = norm(Xttf - khatrirao(B,'r') * lambda)^2;
                    e2 = normX + norm(ktensor(lambda,B))^2 - 2 * real(trace(Grad_n*diag(lambda)*B{n}'));
                    if abs(e1 - e2)/e1 > 1e-4
                        1;
                    end
                end
                
                if mu ~= 0
                    XQb = XQb + mu * eye(size(XQb));
                end
                B{n} = Grad_n/XQb.';
                
                %err(n) = normX + norm(ktensor(B))^2 - 2 * real(trace((Grad_n)'*B{n}));
                %err(n) = norm(Xttf - sum(khatrirao(B,'r'),2))^2;
                if R < SzX(n)
                   err(n)= normX  - real(trace(B{n}'*Grad_n));
                else
                   err(n) = normX  - real(trace(Grad_n*B{n}'));
                end
                if debug
                    if ~isempty(curr_err) &&  ((err(n)-curr_err)>1e-9)
                        1
                    end
                end
                
                
            end
            %err(n)-curr_err
            if debug
                curr_err = err(n);
            end
            
            %BtB(:,:,n) = B{n}'*B{n};
            
            %% Normalization % may not be necessary
            %             if kiter == 1
            lambda = sqrt(sum(abs(B{n}).^2,1)).'; %2-norm %
            %             else
            %                 lambda = max( max(abs(B{n}),[],1), 1 ).'; %max-norm
            %             end
            B{n} = bsxfun(@rdivide,B{n},lambda.');
            BtB(:,:,n) = B{n}'*B{n};
            
            %% Update Phi_left and Phi_right for computing the gradients
            % for the next update B{n}
            
            % Update Phi_left only when going from left to right 1, 2, ...
            if update_dir == 1 % Left-to-Right
                if n == 1
                    Phi_left{n+1} = Ux{1}.'*conj(B{n});
                elseif n<N
                    Phi_left{n+1} = fmttkrp(Ux{n},Phi_left{n},conj(B{n}),3);
                end
            end
            
            % Update Phi_right only when going from right to left, N,N-1, ...
            if update_dir == 2 % RIght-to-Left
                if n == N
                    Phi_right{n-1} = Ux{N}*conj(B{N});
                elseif n>1
                    Phi_right{n-1} = fmttkrp(Ux{n},conj(B{n}),Phi_right{n},1);
                end
            end
            
            % check error
            if debug
                Phi_left2 = cell(N,1);
                Phi_left2{1} = 1;
                
                Phi_right2 = cell(N,1);
                Phi_right2{N} = 1;
                
                for n2 = 1:N-1
                    if n2 == 1
                        Phi_left2{n2+1} = Ux{1}.'*conj(B{n2});
                    elseif n2<N
                        Phi_left2{n2+1} = fmttkrp(Ux{n2},Phi_left2{n2},conj(B{n2}),3);
                    end
                end
                
                for n2 = N:-1:2
                    if n2 == N
                        Phi_right2{n2-1} = Ux{N}*conj(B{N});
                    elseif n2>1
                        Phi_right2{n2-1} = fmttkrp(Ux{n2},conj(B{n2}),Phi_right2{n2},1);
                    end
                end
                
                e1 = norm(Xttf - khatrirao(B,'r') * lambda)^2;
                e1b = zeros(N,1);
                normB = norm(ktensor(lambda,B))^2;
                for n2 = 1:N
                    XQleft2 = Phi_left2{n2};
                    XQright2 = Phi_right2{n2};
                    
                    if n2 == 1
                        Grad_n = Ux{n2} * XQright2;
                    elseif n2 == N
                        Grad_n = Ux{N}.' * XQleft2;
                    else
                        Grad_n = fmttkrp(Ux{n2},XQleft2,XQright2,2);
                    end
                    e1b(n2) = normX + normB - 2 * real(trace(Grad_n*diag(lambda)*B{n2}'));
                end
                
                
                if update_dir == 2
                    if n>1
                        [norm(e1b - e1)  norm(reshape(Phi_right{n-1} - Phi_right2{n-1},[],1))]
                    end
                else
                    if n<N
                        [norm(e1b - e1) norm(reshape(Phi_left2{n+1} - Phi_left{n+1},[],1))]
                    end
                end
            end
            
            fprintf('ITer %d, Factor %d, Err %d\n',kiter,n,err(n)/normX)
            err2(n,update_dir) = err(n);
        end
        
    end
    err_iter(kiter,:) = [err2(1:N-1,1); err2(N:-1:2,2)]';
    %     if kiter>1
    %         err_iter(kiter-1,1) = err(end); % fixed err(1)
    %     end
    
    
    %if (kiter >1) && abs(sum(err_iter(kiter,:)) - sum(err_iter(kiter-1,:))) < param.tol*N
    if (kiter >1) && abs(mean(sqrt(err_iter(kiter,:)/normX)) - mean(sqrt(err_iter(kiter-1,:)/normX))) < param.tol
        %stop_check = true;
        break
    end
    
end

P = ktensor(lambda,B);
% Normalize factors and fix the signs
P = arrange(P);P = fixsigns(P);

if nargout >=2
    err_iter = err_iter(1:kiter,:);
    output = struct('Uinit',{Binit},'Fit',1-sqrt(err_iter/normX),'NoIters',kiter,'Xtt',Xtt);
end

end



function G = fmttkrp(X,A,B,mode_n)
%  X_(n)^T * khatrirao(B,A)
%
szX = size(X);R = size(A,2);
switch mode_n
    case 1
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[1 szX(2) R]));
        G = squeeze(sum(G,2));
    case 2
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[szX(1) 1 R]));
        G = squeeze(sum(G,1));
    case 3
        tmp = A.'*reshape(X,[szX(1)  szX(2)*szX(3)]);
        G = bsxfun(@times, reshape(tmp,[R szX(2) szX(3)]), B.');
        G = squeeze(sum(G,2)).';
end

end


%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','exact_tt_to_cp',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'exac'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('linesearch',true);
param.addOptional('damping',0); % damping parameter for the term + mu/2 * norm(theta)^2
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addOptional('normX',[]);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end


%%
function Ah = exact_tt_to_cp(Xtt,R)
% Exact conversion from a TT-tensor Xtt of rank-R to a rank-R tensor.
%   Xtt is often obtained by fit a TT-tensor to a noiseless (nor nearly
%          noisefree) rank-R tensor
%   Ah : cell arrray of factor matrices of the rank-R tensor.
%
% Phan Anh Huy, 2016

G = core2cell(Xtt);
N = ndims(Xtt);
Pn_ = cell(N,1);

% Decomposition of each core G{n} by a rank-R tensor.
% The second factor matrices are factor matrices of the rank-R tensor.

if R <= 20
    if isreal(Xtt{1})
        cp_func = @cp_fLMa;
    else
        cp_func = @cpx_fLMa;
    end
else
    cp_func = @cp_fastals;
end


cp_opts = cp_func();
cp_opts.maxiters = 2000;
cp_opts.init = {'nvec' 'dtld'};
cp_opts.printitn = 0;
cp_opts.maxboost = 0;

for n = 2: N-1
    Pn = cp_func(tensor(G{n}),R,cp_opts);
    Pn_{n} = Pn;
end

% matching order and scaling between
% Pn_{n}.u{3} and Pn_{n+1}.u{1}
% For the exact case, the following relation holds
%   Pn_{n}.u{3}' * inv(Pn_{n+1}.u{1}) = diag(dn)
%
% The following normalization normalizes Pn_{n}.u{3} *and Pn_{n+1}.u{1}
% so that
%   Pn_{n}.u{3}' * inv(Pn_{n+1}.u{1}) = I_R

for n = 2:N-2
    C = (Pn_{n}.u{3}.'*Pn_{n+1}.u{1});
    [foe,ix] = max(abs(C),[],2);
    
    %
    Pn_{n+1}.u = cellfun(@(x) x(:,ix),Pn_{n+1}.u,'uni',0);
    Pn_{n+1}.lambda = Pn_{n+1}.lambda(ix);
    
    C = (Pn_{n}.u{3}.'*Pn_{n+1}.u{1});
    al = diag(C);
    
    Pn_{n}.u{3} = Pn_{n}.u{3}*diag(1./al);
    Pn_{n}.lambda = bsxfun(@times,Pn_{n}.lambda,al);
    
end

% Approximate to the factor matrices of X
Ah = cell(N,1);
Ah{1} = squeeze(G{1})*Pn_{2}.u{1} ;
Ah{N} = (squeeze(G{N}).')*(Pn_{N-1}.u{3});
Ah(2:N-1) = cellfun(@(x) x.u{2} * diag(x.lambda),Pn_(2:N-1),'uni',0);

end