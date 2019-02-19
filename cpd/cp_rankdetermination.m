function [P,R,apxrank] = cp_rankdetermination(Y,delta,Rinit,opts)
%% RANK Detection with a prescribed noise level
% The method is based on the EPC method
% 
%
% TENSORBOX, 2018
%
if ~exist('opts','var'), opts = struct; end
opt_cp = parseInput(opts);
if nargin == 0
    P = opt_cp; return
end

%% accuracy = C*noise_level*sqrt(numel(Y)); % C = 1.01
accuracy = delta;

apxrank = [];

doing_estimation = true;
best_result = [];

Rcp = Rinit;

opts = opt_cp;

while doing_estimation
    fprintf('Rank %d ...\n',Rcp);
    

    %%
    try
        fprintf('step 1');
        [Pbx,outputx] = cp_anc(Y,Rcp,accuracy,opt_cp);
        
        %%
        fprintf('step 2');
        
        opts.maxiters = 1000;
        opts.init = Pbx;
    catch
        opts = opt_cp;
    end
    [tt_approx] = cp_nc_sqp(Y,Rcp,accuracy,opts);
    tt_approx = normalize(tt_approx);
    tt_approx = arrange(tt_approx);
    
    err_cp = norm(Y - full(tt_approx));
    norm_lda = norm(tt_approx.lambda(:));
    
    %%
    if err_cp <= accuracy + 1e-5
        % the estimated result seems to be good
        if isempty(best_result) || (Rcp < best_result(1))
            best_tt_approx = tt_approx;
            best_result = [Rcp err_cp norm_lda];
        end
        
        if (Rcp > 1)   % try the estimation with a lower rank
            Rcp_new = Rcp-1;
        else
            doing_estimation = false;
        end
    else
        Rcp_new = Rcp+1;
    end
    
    apxrank = [apxrank ; [Rcp  err_cp norm_lda]];
    if any(apxrank(:,1) == Rcp_new)
        doing_estimation = false;
    end
    Rcp = Rcp_new;
end

%%
R = best_result(1);
P = best_tt_approx;
fprintf('Estimated Rank %d\n',R);

end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init',[repmat({'rand'},1,20) 'nvec'],@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));
param.addOptional('maxiters',1000);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('linesearch',true);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addParamValue('TraceRank1Norm',false,@islogical);
param.addParamValue('max_rank1norm',inf);

param.addOptional('normX',[]);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end


