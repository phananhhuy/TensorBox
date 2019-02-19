function [P,out] = cp_roro(Y,opts)
% Rotational update algorithm for best Rank-One tensor approximation
%
% Phan Anh-Huy, August, 2017,
%
% TENSORBOX, 2018

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end


N = ndims(Y);
err = 0;
normY = norm(Y);

%%
% param.cp_func = str2func(mfilename);
Uinit = cp_init(Y,1,param); u = Uinit;
P = normalize(ktensor(u));
u = P.u;

%% Output
if param.printitn ~=0
    fprintf('\Rotational algorithm for best Rank-One tensor approximation:\n');
end

%% NIcholson ALS - new form  works as ALS
%
cgn = zeros(N,1);
clear err;
v = cell(N,1);

for kiter = 1:param.maxiters
    if 0 %mod(kiter,5) == 1% run als
        for n = 1:N
            tn = double(ttv(Y,u,-n));
            u{n} = tn/norm(tn);
        end
        
    else
        tn_ = cp_gradients(Y,u);
        gn = cell(N,1);
        
        for n = 1:N
            tn = tn_{n};
            xi = tn'*u{n};
            
            %gn{n} = (-tn + u{n}*xi)*xi;
            % due to the normalization xi is omitted
            gn{n} = (-tn + u{n}*xi);
            cgn(n) = norm(gn{n});
            
            if cgn(n) < 1e-10
                cgn(n) = 0;
                gn{n}(:) = 0;
            else
                gn{n} = gn{n}/cgn(n);
            end
        end
        
        %         gn = cell(N,1);
        %
        %         for n = 1:N
        %             tn = double(ttv(Y,u,-n));
        %             xi = tn'*u{n};
        %
        %             %gn{n} = (-tn + u{n}*xi)*xi;
        %             % due to the normalization xi is omitted
        %             gn{n} = (-tn + u{n}*xi);
        %             cgn(n) = norm(gn{n});
        %
        %             if cgn(n) < 1e-10
        %                 cgn(n) = 0;
        %                 gn{n}(:) = 0;
        %             else
        %                 gn{n} = gn{n}/cgn(n);
        %             end
        %         end
        
        zc2 = find(cgn > 0);
        %     numel(zc2)
        
        ug = u;
        ug(zc2) = cellfun(@(x,y) [x y], u(zc2),gn(zc2),'uni',0);
        
        %% best rank-1 of tensor 2x2x2
        % Yug = ttm(Y,ug,'t'); slow because of permute and ipermute
        Yug = fullprojection(Y,ug);
        
        Yug = squeeze(tensor(Yug));
        N2 =  ndims(Yug);
        %v = tucker_als(Yug,1,struct('init','nvecs','tol',1e-8));
        %v = mtucker_als(Yug,ones(1,ndims(Yug)),struct('init','nvecs','tol',1e-8,'maxiter',1000),'nvecs');
        %
        
        if N2 > 4
            
            cp_opts2.maxiters = 1000;
            cp_opts2.dimorder = 1:N;
            cp_opts2.printitn = 0;
            cp_opts2.tol = 1e-12;
            cp_opts2.alsinit = 0;
            cp_opts2.init = {'dtld' 'nvec'};
            
            v = cp_fastals(Yug,1,cp_opts2);
            cp_opts2.init = {'dtld' 'nvec' v};
            v = cp_r1LM(Yug,1,cp_opts2);
            v = normalize(v);
            lv = v.lambda;
            v = v.u;
            
        elseif N2 == 4
            [p4,cost] = bestrank1_2222(Y,opts);
            v = p4.u;
            
        elseif N2 == 3
            % closed form to find best rank-1 tensor of 2x2x2
            %     y3 = squeeze(double(c(:,:,1))*cos(x) + double(Yug(:,:,2))*sin(x));
            %
            %     s1 = sum(y3(:).^2);
            %     s2 = sqrt((y3(1)^2 + y3(3)^2 - y3(2)^2 - y3(4)^2)^2 + 4 * (y3(1)*y3(2)+y3(3)*y3(4))^2);
            %
            %     sigma2 = (s1+s2)/2; % = svd(y3)^2
            %     [xs,fv] = fminbnd(@(v) -double(subs(sigma2,x,v)),0,pi);
            
            [v{1},v{2},v{3}] = bestrank1_222(double(Yug));
            
            %[v2,sigma] = bestrank1_222_iter(double(Yug));
            %if ttv(tensor(Yug),v) < ttv(tensor(Yug),v2)-1e-10
            %    1
            %    break
            %end
            
        elseif N2 == 2
            [uy,sy,vy] = svd(double(Yug));
            sy = diag(sy); [~,is] = max(sy);
            v{1} = uy(:,is);
            v{2} = vy(:,is);
        else % ndims(Yug) == 1
            v{1} = double(Yug)/norm(Yug);
        end
        %     norm(lv - sqrt(-fv))
        
        %%
        
        unew = u;
        for n = 1:numel(zc2)
            %unew{zc2(n)} = ug{zc2(n)} * (v{n}*sign(v{n}(1)));
            unew{zc2(n)} = ug{zc2(n)} * v{n};
            unew{zc2(n)} = unew{zc2(n)}/norm(unew{zc2(n)});
        end
        
        u = unew;
        
    end
    
    % lda = ttv(Y,u);
    lda = fullprojection(Y,u);
    err(kiter) = sqrt(normY^2 - double(lda)^2)/normY;
    
    if mod(kiter,param.printitn)==0
        fprintf(' Iter %2d: ',kiter);
        if kiter>1
            fprintf('fit = %e fitdelta = %7.1e \n', 1-err(kiter), err(kiter)-err(kiter-1));
        else
            fprintf('fit = %e   \n', 1-err(kiter));
        end
    end
    
    if (kiter>1) && abs(err(kiter)-err(kiter-1))< param.tol
        break
    end
end
P = ktensor(lda,u);
out.Fit = [(1:numel(err))' 1-err(:)];


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

param.addParamValue('TraceRank1Norm',false,@islogical);
param.addParamValue('max_rank1norm',inf);

param.addParamValue('gamma',1e-3);
param.addParamValue('adjust_gamma',false,@islogical);
param.addOptional('normX',[]);


param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end


function X = fullprojection(Y,u)
szY = size(Y);
szYnew = cellfun(@(x) size(x,2),u,'uni',1);
X = reshape(Y.data,[],szY(end))*u{end};
X = X'; % RN x I1...I(N-1)
for n = ndims(Y)-1:-1:1
    X = reshape(X,[],szY(n))*u{n};  % R(n+1)...RN I1...I(n-1) x Rn
    X = X'; % Rn R(n+1)...RN I1...I(n-1)
end
X = reshape(X,szYnew');
end