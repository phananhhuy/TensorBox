function [P,out] = cp_r1lm_optmu(Y,opts)
% LM algorithm for best rank-1 tensor approximation 
% 
% Phan Anh Huy 2017

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

%%
% param.cp_func = str2func(mfilename);
Uinit = cp_init(Y,1,param); u = Uinit;
% P = ktensor(U);

%% Output
if param.printitn ~=0
    fprintf('\LM algorithm for best Rank-One tensor approximation\n');
end

N = ndims(Y);
err = 0;

% u = Pxh_0.u;
% u = Pxh.u;
gamma_ = zeros(N,1);
tn_ = u;

for kiter = 1:param.maxiters
    
    for n = 1:N
        gamma_(n) = u{n}'*u{n};
    end
    gamma = prod(gamma_);
    lda = ttv(Y,u)/gamma;
    if lda<0
        u{1} = -u{1};
        lda = -lda;
    end
    
    alpha = gamma^(1/N/2) * lda^(1/N);
    alpha2 = alpha^2;
    for n = 1:N
        u{n} = u{n}/sqrt(gamma_(n)) * alpha;
    end
    gamma = lda^2*gamma;
    %mu = gamma;
    
    
    % mu = 0 
    mu_opt = 2;  % try mu = [0 alpha^
    if mu_opt == 0
        mu = 0;
        %mu = alpha2^(N-1);
%         mu =  alpha2^(N-2);
        for n = 1:N
            tn = double(ttv(Y,u,-n));
            %unew{n} = tn/alpha2^(N-1);
            %u{n} = tn/alpha2^(N-1);
%             unew{n} = (mu* u{n} + tn)/(mu + alpha2^(N-1));
            u{n} = (mu* u{n} + tn)/(mu + alpha2^(N-1));
        end
%         u = unew;

    elseif mu_opt == 1
        % mu = inf
        % current error
            current_err = -gamma; % + normY^2 =  norm(Y-full(ktensor(u)));

        for n = N:-1:1
            tn = double(ttv(Y,u,-n));            
            tn_{n} = tn;
        end
        
        % mu = 0
        Ptn =ktensor(1/gamma^(N-1),tn_);
        muzero_err = norm(Ptn)^2 - 2 * innerprod(Y,Ptn);
        
        % mu = alpha2^(N-1);
        mu = alpha2^(N-2);
        unew = cellfun(@(x,y) (x*mu+y)/(mu + alpha2^(N-1)),u,tn_,'uni',0);
        Ptn2 =ktensor(unew);
        mu2_err = norm(Ptn2)^2 - 2 * innerprod(Y,Ptn2);
        
        if mu2_err<muzero_err
            Ptn = Ptn2;
            muzero_err = mu2_err;
        end
        
        if muzero_err< current_err
            u = Ptn.u;
            u{1} = u{1}*Ptn.lambda; 
        else
            1;
        end
         
    else % choose the optimal damping parameter 
        %
        gamma_ns1 = alpha2^(N-1);
        %%
        v = u;cn = zeros(N,1);
        for n = N:-1:1
            tn = double(ttv(Y,u,-n));
            v{n} = tn - gamma_ns1 * u{n};
            cn(n) = norm(v{n});
%             if cn(n) > 1e-8
%                 v{n} = v{n}/cn(n);
%             else
%                 v{n}(:) = 0;
%                 cn(n) = 0;
%             end
            
            uv{n} = [u{n} v{n}];
        end
        
        Yuv = double(ttm(Y,uv,'t')); % 2x2x...x2
        
        % generate polynomial of norm of Yx = [[uv{n} * [1 ; eta] ]]
        % f1 : of degree 2N
        %syms eta; 
        % f1 = prod(alpha2+cn.^2*eta^2);        
        for n = 1:N
            if n == 1
                f1 = [cn(n).^2 0 alpha2];
            else
                f1 = conv(f1,[cn(n).^2 0 alpha2]);
            end
        end

        % Polynomial of the second term of degree N
        f2 = gen_poly_kron_1x(Yuv(:),N);
%         f2 = [1; eta];
%         for n = 1:N-1
%             f2 = kron(f2, [1; eta]);
%         end
%         f2 = Yuv(:)'*f2;
        
        fcost = f1;
        fcost(N+1:end) = fcost(N+1:end) - 2*f2;
        df = polyder(fcost);
        etas = roots(df);
        % eta should be in [0 1/alpha^(N-1)]
        etas = etas(abs(imag(etas))<1e-8);
        eta = etas((0<=etas)&(etas<1/alpha^(N-1))); 
        eta = [eta; 0 ;1/alpha^(N-1)];
        if numel(eta)>1
            fcost_etas = polyval(fcost,eta);
            [~,ietas] = min(fcost_etas);
            eta = eta(ietas);
        end
        
        %eta = fminbnd(@(x) double(subs(fcost,eta,x)),0,1/gamma_ns1);
        unew = u;
        for n = 1:N
            unew{n} = u{n} + v{n}*eta;
        end
        u = unew;
        
        %%
        if 0
            mu = alpha2^(N-1);
            syms nu;
            
            KUT = 1;c2 = 1;
            
            f2 = 1;
            for n = N:-1:1
                tn = double(ttv(Y,u,-n));
                
                tn_{n} = tn;
                if n == N
                    KUT = [u{n} tn];
                else
                    KUT = kron(KUT,[u{n} tn]);
                end
                
                c2 = kron(c2,[nu ;1]);
                
                f2 = f2 *([nu 1]*([alpha2 gamma ; gamma tn'*tn]) * [nu ; 1]);
            end
            c1 = Y(:)'*KUT;
            
            fcost = f2/(nu + alpha2^(N-1))^(2*N) - 2*(c1*c2) * 1/(nu + alpha2^(N-1))^N;
            
            while 1
                [mu2,FVAL,EXITFLAG] = fmincon(@(x) double(subs(fcost,nu,x)),mu,[],[],[],[],0,2*mu);
                if EXITFLAG == 1
                    break
                else
                    mu = 2*mu;
                end
            end
            mu = double(mu2);
            
            for n = 1:N
                unew{n} = (mu* u{n} + tn_{n})/(mu + alpha2^(N-1));
            end
%             u = unew;
        %fcost2 = double(subs(fcost,nu,mu))+normY2; 
        end
    end
    
     
    err(kiter) = norm(Y-full(ktensor(u)));
    
    fprintf('%d err %d \n',kiter,err(kiter))
    if (kiter>1) && abs(err(kiter)-err(kiter-1))< 1e-8
        break
    end
end

P = ktensor(u);
out.Fit = [(1:numel(err))' 1-err(:)/norm(Y)];


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

param.addOptional('normX',[]);


param.parse(opts);
param = param.Results;

end