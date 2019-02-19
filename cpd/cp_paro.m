function [P,output] = cp_paro(T,R,opts)
%
% PArallel Rank-One Tensor Update Algorithm for CPD 
%
% The algorithm updates rank-1 tensors simultaneously 
%   x_r = best_rank1(xr + err)  for r = 1, ..., R
% 
%
% INPUT:
%   T:  N-way data which can be a tensor or a ktensor.
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
% REF:
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.    
% Phan Anh Huy, 2017


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

N = ndims(T); 
szT = size(T);

%% Initialize factors U
param.cp_func = str2func(mfilename);
Uinit = cp_init(T,R,param); U = Uinit;
P = ktensor(U);

%% Output
if param.printitn ~=0
    fprintf('\nCP-PARO:\n');
end

if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
%     if param.TraceFit
        output.Fit = [];
%     end
end

if isempty(param.normX)
    normX = norm(T);
else
    normX = param.normX;
end

adjust_mu = param.adjust_gamma;
check_gamma = param.check_gamma;

gamma = param.gamma;

lambda = P.lambda;
U = P.U;
x = khatrirao(U,'r')*diag(lambda);
 
T = T(:);

% For fast version
xb = sum(x,2);
mu = 1/gamma;
err = (T - xb); % the mean error 
eb = err/(mu + R);
curr_norm = norm(err);


% for best rank-1 algorithm
r1_opts = bestrank1tensorapprox();
r1_opts.maxiters = 100;
r1_opts.r1_alg = param.r1_alg;
r1_opts.init = 'nvec';

%% Mainloop
flagtol =0; normchange = 0;
pre_iter_adjust_mu = 0;
accepted_mu_ = mu;
chk_stop = 0;


%%
for iter = 1:param.maxiters
    pre_norm = curr_norm;

    [Upre,lambda_pre] = deal(U,lambda);
    
    % xr = argmin cpd ( x_r + 1/(1/gamma + R) * (t - yb - xb))
    % Parallel loop here to estimate best rank-1 tensors x_r
    Pr_ = cell(1,R);
%     for r = 1:R 
    parfor r = 1:R
        Txr = x(:,r) + eb;
        Txr= tensor(Txr,szT);
        
%         cp_opts = cp_fLMa;
%         cp_opts.maxiters = 1000;
%         cp_opts.dimorder = 1:N;        
%         %Pr = mtucker_als(Txr,ones(1,N),cp_opts,cellfun(@(x) x(:,r),U,'uni',0));
%         %cp_opts.init = {'nvec' cellfun(@(x) x(:,r),U,'uni',0)};
%         cp_opts.init = 'nvec';
        %Pr = cp_fastals(Txr,1,r1_opts);
        
        
        %r1_opts.r1_alg = @cp_fastals;
        %r1_opts.r1_alg = @cp_r1LM;
        %r1_opts.r1_alg = @cp_r1lm_optmu;
        % best rank-1 of Txr
        
        % init 1
        [Pr,out_] = bestrank1tensorapprox(Txr,r1_opts);
        
        
        %% init 2
        if 0
            [Pr2] = mtucker_als(Txr,ones(1,N),cp_opts,'nvecs');
            
            %% fLM - stage 2
            cp_opts = cp_fLMa;
            cp_opts.maxiters = 1000;
            cp_opts.printitn = 0;
            cp_opts.tol = 1e-8;
            cp_opts.init = {Pr Pr2};
            
            Pr = cp_fLMa(Txr,1,cp_opts);
        end
        %%
        
        
        Pr_{r} = Pr;
    end
    
    for r = 1:R
        Pr = Pr_{r};
        for n = 1:numel(U)
            U{n}(:,r) = Pr.u{n};
        end
        lambda(r) = Pr.lambda;
    end
    
    xnew = khatrirao(U,'r')*diag(lambda);
    xb_new = sum(xnew,2);
    
    % update the error
    err = T-xb_new;
    
    % evaluate cost function 
    curr_norm = norm(err);
    fit = 1 - (curr_norm/ normX); %fraction explained by model
     
    if iter >= 1
        if (pre_norm < curr_norm) && (adjust_mu || check_gamma)
            
            if ( iter > 5) && check_gamma
                [U,lambda] = deal(Upre,lambda_pre);
                % function is non-decreasing Stop and run again with higher gamma
                break;
            end
            if adjust_mu
                flagtol = 0;
                % adjust mu % increase mu and eb
                eb(:) = 0;
                %             eb = eb*(mu + R)/(accepted_mu_(end) + R);
                mu = accepted_mu_(end);
                [U,lambda] = deal(Upre,lambda_pre);
                curr_norm = pre_norm;
            end
        else 
            % update the mean residue
            %  when mu = R, the average mean eb = err/R;
            % eb = err/R; % mu = R
            %
            % eb = T - 2*xb_new + xb + R/(mu + R) *eb;
            eb = (err - xb_new + xb + R *eb)/(mu + R);
            
            %eb = (err - xb_new + xb + R *eb)/(mu + R);
            
            x = xnew;
            xb = xb_new;
                
            normchange = abs(pre_norm - curr_norm);
            if (iter > 1) && (normchange < param.tol) % Check for convergence
                flagtol = flagtol + 1;
            else
                flagtol = 0;
            end
            chk_stop(1) = flagtol >= 10;
            
            %chk_stop(3) = fit >= param.fitmax;
            if nargout >=2
                output.Fit = [output.Fit; iter fit];
            end 
            
            if (adjust_mu == 1) && (iter - pre_iter_adjust_mu)>20
                % we can lower mu to be smaller if the algorithm works well
                % for the current mu
                accepted_mu_ = [accepted_mu_ mu];
                
                mu = mu / 1.15;
                pre_iter_adjust_mu = iter;
            end
        end
    end
    
    if mod(iter,param.printitn)==0
        fprintf(' Iter %2d: ',iter);
        fprintf('fit = %e fitdelta = %7.1e , mu % d', fit, normchange,mu);
        fprintf('\n');
    end
    
    
    % Check for convergence
    if (iter > 1) && any(chk_stop)
        break;
    end
end

P = ktensor(lambda,U);

if nargout >=2
    output.NoIters = iter;
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
% param.addParamValue('TraceFit',true,@islogical);
% param.addParamValue('TraceMSAE',false,@islogical);

% param.addParamValue('TraceRank1Norm',false,@islogical);
% param.addParamValue('max_rank1norm',inf);

param.addOptional('gamma',1e-3);
param.addOptional('adjust_gamma',false,@islogical);
param.addOptional('normX',[]);

param.addOptional('check_gamma',true,@islogical);

param.addOptional('r1_alg',@cp_roro);

param.parse(opts);
param = param.Results;
% param.verify_convergence = param.TraceFit || param.TraceMSAE;
end