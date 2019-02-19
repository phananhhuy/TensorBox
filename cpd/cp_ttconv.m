function [P,output] = cp_ttconv(X,R,opts)
% Fast algorithm for CP factorizes the N-way tensor X into factors of R
% components.
%
% The algorithm is based on TT-decomposition Xtt of the tensor X into N cores.
% Then a K-tensor is fit to the TT-tensor Xtt
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
    tinit = tic;
    Xtt1 = tt_tensor(double(X),tt_opts.tol,SzX,[1 R*ones(1,N-1) 1]');
    Xtt = round(Xtt1,tt_opts.tol,[1 R*ones(1,N-1) 1]');
    Xtt = tt_als(Xtt1,Xtt,tt_opts.maxiters);
    tinit = toc(tinit);
else
    % if input is a TT-tensor, round its rank to R, or not exceed R
    % round the input TT-tensor to rank R
    tinit = tic;
    Xtt = round(X,tt_opts.tol,[1 R*ones(1,N-1) 1]');
    % Further improvement by ALS with 10 sweeps
    Xtt = tt_als(X,Xtt,tt_opts.maxiters);
    tinit = toc(tinit);
end

%% Stage 2: Initialize factor matrices and conversion from TT to K-tensor
if (ischar(param.init) && strcmp(param.init,'exact_tt_to_cp')) || isa(X,'tt_tensor')
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
Ux{1} = reshape(Ux{1},size(Ux{1},2),[]);
rankX = Xtt.r;


% % Pre-Compute Phi_left % matrices of size R x R, products of all cores form
% % 1:n-1 between two tensor Xtt and Btt
Phi_left = cell(N,1);
Phi_left{1} = 1;

% Pre-Compute Phi_right % matrices of size R x R, products of all cores form
% n+1:N between two tensor Xtt and Btt
Phi_right = cell(N,1);
Phi_right{N} = 1;
%Phi_right{N-1} = Ux{N}*conj(B{N}); %fixed for complex data 
Phi_right{N-1} = Ux{N}*(B{N}); %fixed for complex data 
for n = N-2:-1:1
    %dir = 'RL';upto = n+1; % upto = 2, 3, 4
    %Phi_right{n} = innerprod(Xtte,Btte, dir, upto ); % contraction between cores k of X and B where k = n+1:N
    %Phi_right{n} = fmttkrp(Ux{n+1},conj(B{n+1}),Phi_right{n+1},1); %fixed for complex data 
    Phi_right{n} = fmttkrp(Ux{n+1}, (B{n+1}),Phi_right{n+1},1); %fixed for complex data 
end

%%
err_iter = zeros(maxiters,N);
err = zeros(1,N);
err2= zeros(2,N);
lambda = ones(R,1);

for kiter = 1:maxiters
    
    for update_dir = 1:2
        switch update_dir
            case {1 'LR'}
                dim_update = 1:N-1;
            case {2 'RL'}
                dim_update = N:-1:2;
        end
        
        for n = dim_update
            %%
            XQb = prod(BtB(:,:,[1:n-1 n+1:end]),3);
            
            XQleft = Phi_left{n};
            XQright = Phi_right{n};
            
            %% Update B{n} and assess Cost value
            if n == 1 % right
                % The term "squeeze(Xtte.U{n}) * XQright" is mttkrp(X_cp,B,n);
                B{n} = Ux{n} * (XQright/XQb);
                
                if R < SzX(n)
                    err(n) = normX - trace(B{n}'*Ux{n}*XQright);
                else
                    err(n) = normX - trace(Ux{n}*XQright*B{n}');
                end
                
            elseif n == N
                % THe term squeeze(Xtte.U{n})'*XQleft is mttkrp(X_cp,B,n);
                B{n} = Ux{N}' * (XQleft/XQb);
                                
                if R > SzX(n)
                    err(n)  = normX - trace(B{n}*XQleft' * Ux{N});
                else
                    err(n)  = normX - trace(XQleft' * Ux{N} * B{n});
                end
                
            else
                % This term is  mttkrp(X_cp,B,n)
                %Grad_n = mttkrp(tensor(Ux{n}),{XQleft,B{n},XQright},2);
                Grad_n = fmttkrp(Ux{n},XQleft,XQright,2);
                
                B{n} = Grad_n/XQb;
                                
                if R < SzX(n)
                    err(n)= normX  - trace(B{n}'*Grad_n);
                else
                    err(n) = normX  - trace(Grad_n*B{n}');
                end
            end
            
            BtB(:,:,n) = B{n}'*B{n};
            
            %% Normalization % may not be necessary
            if kiter == 1
                lambda = sqrt(sum(abs(B{n}).^2,1)).'; %2-norm %
            else
                lambda = max( max(abs(B{n}),[],1), 1 ).'; %max-norm
            end
            B{n} = bsxfun(@rdivide,B{n},lambda.');
            BtB(:,:,n) = B{n}'*B{n};
            
            %% Update Phi_left and Phi_right for computing gradients
            % for the next update B{n}
            
            % Update Phi_left only when go from left to right 1, 2, ...
            if update_dir == 1 % Left-to-Right 
                if n == 1
                    Phi_left{n+1} = Ux{1}'*B{n};
                elseif n<N
                    Phi_left{n+1} = fmttkrp(Ux{n},Phi_left{n},B{n},3);
                end
            end
            
            % Update Phi_right only when go from right to left, N,N-1, ...
            if update_dir == 2 % RIght-to-Left
                if n == N
                    Phi_right{n-1} = Ux{N}*B{N};
                elseif n>1
                    Phi_right{n-1} = fmttkrp(Ux{n},B{n},Phi_right{n},1);
                end
            end
            
            fprintf('ITer %d, Factor %d, Err %d\n',kiter,n,err(n)/normX)
             
        end
    end
    err_iter(kiter,:) = err;
    if kiter>1 
        err_iter(kiter-1,1) = err(1); % fixed err(1)        
    end
    
    
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
    output = struct('Uinit',{Binit},'Fit',err_iter,'NoIters',kiter,'Xtt',Xtt,'Init_time',tinit);
end

end



function G = fmttkrp(X,A,B,mode_n)
%  X_(n)^T * khatrirao(B,A)
%
szX = size(X);
if numel(szX)<3, szX(3) = 1;end
R = size(A,2);
switch mode_n
    case 1
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[1 szX(2) R]));
        G = squeeze(sum(G,2));
    case 2
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[szX(1) 1 R]));
        G = reshape(sum(G,1),size(G,2),[]);
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
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addOptional('normX',[]);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end