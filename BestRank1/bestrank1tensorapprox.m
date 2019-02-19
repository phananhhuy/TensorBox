function [P,out] = bestrank1tensorapprox(Y,opts)
% Best rank-1 tensor approximation to a tensor X
%
% TENSORBOX, 2018

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end


N = ndims(Y);

permYs = [];
if strcmp(param.init,'sqpr') 
   % sequential projection and truncation
   permYs = perms(1:N); 
end
r1_alg = param.r1_alg;


best_err = inf;
for kp = 1:size(permYs,1)+1
  
    if kp<=size(permYs,1)
        % Sequential Projection and Truncation
    
        permY = permYs(kp,:);
        Y0 = permute(Y,permY);
        
        Usqprj = seqproj_trunc(tensor(Y0),1);
    else
        % SVD for the last initilization method
        Usqprj = cp_init(tensor(Y),1,struct('init','nvecs'));
        permY = 1:N;
    end
    
    %% optimize the rank-1 norm lda
    u = Usqprj;
    lda = ttv(Y,u);
    yu = normalize(ktensor(lda,u));
    u = cellfun(@(x) x* (yu.lambda)^(1/(N)),yu.u,'uni',0);
    
    Yt = ktensor(u);
    Yt = normalize(Yt);
    Yt = ipermute(Yt,permY);
    
    % first pass through ALS
    cp_opts = cp_fastals();
    %     cp_opts.dimorder = 1:N;
    %     cp_opts.printitn = 1;
    cp_opts.tol = param.tol;
    
    cp_opts.init = Yt;
    cp_opts.maxiters = 10;
    
    [Pxh_0,outh] = cp_fastals(Y,1,cp_opts);
    
    Pxh_0 = normalize(Pxh_0);
    
    %Err_perm3(kp) = normY^2 + norm(Pxh_0)^2 - 2 * innerprod(Y,Pxh_0);
    
    %% Best rank-1 algorithm
    cp_opts.maxiters = 10;
    cp_opts.linesearch = 0;
    cp_opts.init = Pxh_0.u;
    
    % RORO rank-1 rotational algorithm
    try 
        [P_r1ru,out_r1ru] = r1_alg(Y,cp_opts);
    catch 
        [P_r1ru,out_r1ru] = r1_alg(Y,1,cp_opts);
    end
    
    err_r1ru  = 1-out_r1ru.Fit(end,2);
    %Err_r1ru{kp} = 1-out_r1ru.Fit(:,2);
    
    if best_err > err_r1ru
        best_err = err_r1ru;
        P_best = P_r1ru;
    end
    %% error of all algorithms
end


%% STAGE 2
cp_opts.maxiters = param.maxiters;
% cp_opts.linesearch = 0;
cp_opts.init = P_best;

% RORO rank-1 rotational algorithm
try
    [P,out] = r1_alg(Y,cp_opts);
catch
    [P,out] = r1_alg(Y,1,cp_opts);
end


% err_r1ru  = 1-out_r1ru.Fit(end,2);
% Err_r1ru{kp} = 1-out_r1ru.Fit(:,2);

end




%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'sqpr'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('r1_alg',@cp_roro); % cp_fastals 

param.parse(opts);
param = param.Results;
end