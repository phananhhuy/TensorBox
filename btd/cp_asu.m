 function [P,output] = cp_rankonereduce(X,R,opts)
% Rank-R CPD of tensor X through R rank-1 tensor extractions. 
%
% INPUT:
%   X:  N-way data.
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
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "DEFLATION METHOD FOR
% CANDECOMP/PARAFAC TENSOR DECOMPOSITION", ICASSP 2014.
% [2] A.-H. Phan, P. Tichavsky, A. Cichocki, "Tensor Deflation for
% CANDECOMP/PARAFAC. Part 1: Algorithms", IEEE Transaction on Signal
% Processing, 63(12), pp. 5924-5938, 2015.
% [3] A.-H. Phan, P. Tichavsky, A. Cichocki, "Tensor deflation for
% CANDECOMP/PARAFAC. Part 2: Initialization and Error Analysis?, IEEE
% Transaction on Signal Processing, 63(12), pp. 5939-5950, 2015.

% See also: cp_als
%
% TENSOR BOX, v1. 2012

%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

%%
N = ndims(X); I = size(X);

normX = norm(X);

if isa(X,'tensor')
    IsReal = isreal(X.data);
elseif isa(X,'ktensor') || isa(X,'ttensor')
    IsReal = all(cellfun(@isreal,X.u));
end

%% Output
fprintf('\n CP Alternating Subspace Update:\n');

%% BLock term ALS update on the second block

Rm = R;
err = zeros(Rm-1,1);
Xs = [];
output = struct;
P = cell(2,1);

%% 11/06/2014
kstage = 1;
while kstage < Rm
    fprintf('Stage %d\n',kstage);
    Re = Rm-kstage+1;
    Xsold = Xs;
    
    
    ts = tic;
    switch param.reducetype
        case 'secondblock'
            if kstage ==1
                X2 = X;
            else
                X2 = Xs{2}.core;
            end
            
            % deflation rank should not exceed the tensor size.
            Re = min([Re size(X2)]);

            
            [ah,Uh,lambda,errbtd] = bcdLp1_asu3(X2,Re,param.cp_param);
            errbtd = errbtd(:,end);
            G = ttm(X2,Uh,'t') - full(ttm(ktensor(lambda,ah),Uh,'t'));
            P{1} = ktensor(lambda,ah);
            if all(size(G) == 1)
                P{2} = ktensor(double(G),Uh);
            else
                P{2} = ttensor(G,Uh);
            end
            %%
            if kstage == 1
                Xs{1} = P{1};
            else
                Xs{1} = Xs{1} + ttm(P{1},Xs{2}.U);
            end
            %err(kstage) = norm(X - full(Xs{1}))/normX;
            
            if isa(P{2},'ktensor')
                if Rm > 2
                    Xs{1} = Xs{1} + ttm(P{2},Xs{2}.U);
                else
                    Xs{1} = Xs{1} + P{2};
                end
                %err(kstage+1) = norm(X - full(Xs{1}))/normX;
            else
                if kstage ==1
                    Xs{2} = P{2};
                else
                    Xs{2} = ttm(P{2},Xs{2}.U);
                end
            end
            
            
        case 'residue'
            
            if kstage ==1
                X2 = X;
            else
                X2 = X - full(Xs{1});
            end
            
            X2c = tucker_als(X2,Re,struct('tol',param.cp_param.tol,'printitn',0));
            
            % Stage 1
            %[P,errbtd] = b2d_rR_old(X2,Rb,opts);
            %[P,errbtd] = b2d_rR(X2,Rb,param);
            [ah,Uh,lambda,errbtd] = bcdLp1_asu3(X2c.core,Re,param.cp_param);
            %                     [ah,Uh,lambda,errbtd] = bcdLp1_asu3(X2,R,opts);
            errbtd = errbtd(:,end);
            G = ttm(X2c.core,Uh,'t') - full(ttm(ktensor(lambda,ah),Uh,'t'));
            P{1} = ktensor(lambda,ah);
            if all(size(G) == 1)
                P{2} = ktensor(double(G),Uh);
            else
                P{2} = ttensor(G,Uh);
            end
            
            P{1} = ttm(P{1},X2c.u);
            P{2} = ttm(P{2},X2c.u);
            
            if kstage ==1
                Xs{1} = P{1};
            else
                Xs{1} = Xs{1} + P{1};
            end
            %             err(kstage) = norm(Y - full(Xs{1}))/normY;
            
            if isa(P{2},'ktensor')
                Xs{1} = Xs{1} + P{2};
                %                 err(kstage+1) = norm(Y - full(Xs{1}))/normY;
            else
                Xs{2} = P{2};
            end
    end
    tkstage = toc(ts);
    
    %% Verify angular and error condition
    
    rho_h = cell2mat(cellfun(@(x,y) diag(x'*y(:,1)),ah,Uh,'uni',0)');
    rank1err = sqrt(normX^2 + norm(Xs{1})^2 - 2*innerprod(X,Xs{1}))/normX;
    
    
    if ((sum(rho_h>0.995)>=(N-1)) || (prod(rho_h)>0.993)) && (rank1err >= 1-1e-4)
        %aref = ah; Uref = Uh;
        warning('an and un are very close. Reduce rank of the deflation.\n')
        Rm = Rm-1;
        Xs = Xsold;
        continue;
    else
        kstage = kstage + 1;
    end
    
    
    %%
    fprintf('Stage % d, No. iterations %d, Error %.4f\n',kstage, numel(errbtd),errbtd(end))
    output.ErrorvsIter{kstage} = errbtd;
    output.NoIters(kstage) = numel(errbtd);
    output.Error(kstage) = rank1err; %errbtd(end);
    output.Exectime(kstage) = tkstage;
    
    if (kstage > 1) && param.keep_residue
        output.Residue{kstage} = X2;
    end
end

%%

% %%
% for kstage = 1:Rm-1
%     fprintf('Stage %d\n',kstage);
%     Rb = [ones(1,N)
%         (Rm-kstage)*ones(1,N)];
%     
%     ts = tic;
%     switch opts.reducetype
%         case 'secondblock'
%             if kstage ==1
%                 X2 = X;
%             else
%                 X2 = Xs{2}.core;
%             end
%             
%             % Stage 1
%             %%
%             %                     [P,errbtd] = b2d_rR(X2,Rb,opts);
%             
%             %%
%             [ah,Uh,lambda,errbtd] = bcdLp1_asu3(X2,sum(Rb(:,1)),param.cp_param);
%             errbtd = errbtd(:,end);
%             G = ttm(X2,Uh,'t') - full(ttm(ktensor(lambda,ah),Uh,'t'));
%             P{1} = ktensor(lambda,ah);
%             if all(size(G) == 1)
%                 P{2} = ktensor(double(G),Uh);
%             else
%                 P{2} = ttensor(G,Uh);
%             end
%             %%
%             if kstage ==1
%                 Xs{1} = P{1};
%             else
%                 Xs{1} = Xs{1} + ttm(P{1},Xs{2}.U);
%             end
%             %err(kstage) = norm(X - full(Xs{1}))/normX;
%             
%             if isa(P{2},'ktensor')
%                 if R > 2
%                     Xs{1} = Xs{1} + ttm(P{2},Xs{2}.U);
%                 else
%                     Xs{1} = Xs{1} + P{2};
%                 end
%                 %err(kstage+1) = norm(X - full(Xs{1}))/normX;
%             else
%                 if kstage ==1
%                     Xs{2} = P{2};
%                 else
%                     Xs{2} = ttm(P{2},Xs{2}.U);
%                 end
%             end
%             
%         case 'residue'
%             
%             if kstage ==1
%                 X2 = X;
%             else
%                 X2 = X - full(Xs{1});
%             end
%             
%             X2c = tucker_als(X2,sum(Rb(:,1)),struct('tol',tol,'printitn',0));
%             
%             % Stage 1
%             %[P,errbtd] = b2d_rR_old(X2,Rb,opts);
%             %[P,errbtd] = b2d_rR(X2,Rb,param);
%             [ah,Uh,lambda,errbtd] = bcdLp1_asu3(X2c.core,sum(Rb(:,1)),param.cp_param);
%             %                     [ah,Uh,lambda,errbtd] = bcdLp1_asu3(X2,R,opts);
%             errbtd = errbtd(:,end);
%             G = ttm(X2c.core,Uh,'t') - full(ttm(ktensor(lambda,ah),Uh,'t'));
%             P{1} = ktensor(lambda,ah);
%             if all(size(G) == 1)
%                 P{2} = ktensor(double(G),Uh);
%             else
%                 P{2} = ttensor(G,Uh);
%             end
%             
%             P{1} = ttm(P{1},X2c.u);
%             P{2} = ttm(P{2},X2c.u);
%             
%             if kstage ==1
%                 Xs{1} = P{1};
%             else
%                 Xs{1} = Xs{1} + P{1};
%             end
%             %             err(kstage) = norm(Y - full(Xs{1}))/normY;
%             
%             if isa(P{2},'ktensor')
%                 Xs{1} = Xs{1} + P{2};
%                 %                 err(kstage+1) = norm(Y - full(Xs{1}))/normY;
%             else
%                 Xs{2} = P{2};
%             end
%     end
%     tkstage = toc(ts);
%     
%     fprintf('Stage % d, No. iterations %d, Error %.4f\n',kstage, numel(errbtd),errbtd(end))
%     output.ErrorvsIter{kstage} = errbtd;
%     output.NoIters(kstage) = numel(errbtd);
%     output.Error(kstage) = errbtd(end);
%     output.Exectime(kstage) = tkstage;
%     
%     if (kstage > 1) && param.keep_residue
%         output.Residue{kstage} = X2;
%     end
% end


%% Clean up final result
P = Xs{1};
% % Normalize factors and fix the signs
% P = arrange(P);P = fixsigns(P);

for r = 1:numel(P.lambda)
    Pr = ktensor(P.lambda([1:r]),cellfun(@(x) x(:,[1:r]),P.U,'uni',0));
   
    err(r) = normX^2 + norm(Pr)^2 - 2*innerprod(X,Pr);
end

output.Err = sqrt(err)/normX;
% if nargout >=2
%     output.NoIters = iter;
% end
end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;

param.addOptional('reducetype','secondblock',@(x) ismember(x,{'secondblock' 'residue'}));
param.addOptional('keep_residue',false);

% CP function
param.addOptional('cp_func',@bcdLp1_asu3,@(x) isa(x,'function_handle'));
asu_param = bcdLp1_asu3;
param.addOptional('cp_param',asu_param);

param.parse(opts);
param = param.Results;

end