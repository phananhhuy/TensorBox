function [unew,mu,g,d] = lmrank1_update(Y,u,mu,mu_opt)
% LM update for best rank-1 tensor approximation
%
% TENSORBOX, 2018

if nargin < 4
    mu_opt = 0;
end
% lda_opt = true;
lda_opt = false;

%% normalize
szY = size(Y);

N = numel(u);

gamma_ = zeros(N,1);
for n = 1:N
    gamma_(n) = u{n}'*u{n};
end
gamma = prod(gamma_);


if lda_opt  
    lda = ttv(Y,u)/gamma;
else
    lda = 1;
end

alpha = gamma^(1/N/2) * lda^(1/N);
alpha2 = alpha^2;
for n = 1:N
    u{n} = u{n}/sqrt(gamma_(n)) * alpha;
end



%%
beta = mu +  N * alpha2^(N-1);

% xi = ttv(Y,u);

csz = cumsum([0 szY]);
t = zeros(sum(szY),1);
for n = 1:N
    tn = double(ttv(Y,u,-n));
    t(csz(n)+1:csz(n+1)) = (tn);
end
xi = u{N}'*tn;

theta = cell2mat(u(:));
%%
% unew = 1/beta * ((alpha2^(N-1) +  (N-1)*xi/mu) * theta - t);

% d = alpha2^(N-2)/beta * (alpha2 + (N-1)*xi/(alpha2^(N-1)+mu)) * theta  - 1/(alpha2^(N-1)+mu) * t ;

%unew = (1 - alpha2^(N-2)/beta * (alpha2 + (N-1)*xi/(alpha2^(N-1)+mu))) * theta  + 1/(alpha2^(N-1)+mu) * t ;

unew = 1/(alpha2^(N-1)+mu) * ((mu + (N-1)*alpha2^(N-2)*(alpha2^N-xi)/beta) * theta  + t );

%%
debug = 0;
if debug
    Z = blkdiag(u{1},u{2},u{3});
    H =  (alpha2^(N-1)+mu) * eye(sum(szY)) + Z*(ones(N) - eye(N))*Z' * alpha2^(N-2);
    norm(H - (F+ theta*theta'*alpha2^(N-2)),'fro')
    F = (alpha2^(N-1)+mu) * eye(sum(szY)) - Z*Z'*alpha2^(N-2);
    norm(inv(F) - 1/(alpha2^(N-1)+mu) * (eye(size(H))  +   alpha2^(N-2)/mu * Z*Z'),'fro')
    norm((inv(F)*theta )  - 1/mu* theta)
    norm(inv(H) - (inv(F) - alpha2^(N-2)/(mu^2 + alpha2^(N-1)*mu * N) * theta * theta'),'fro')
    norm(inv(H) - (1/(alpha2^(N-1)+mu) * (eye(size(H))  +   alpha2^(N-2)/mu *  Z*Z')),'fro')
    inv(H)*t - 1/(alpha2^(N-1)+mu) * (t - alpha2^(N-2)*xi*(N-1)/beta * theta)
    inv(H)*theta - 1/beta * theta
end
g = -theta * alpha2^(N-1) + t;
%
% H = cp_hessian(u);
% unew2 = -H\g;

%% optimize mu

if mu_opt
    syms nu;
    beta_nu = nu +  N * alpha2^(N-1);
    b = 1/(alpha2^(N-1)+nu);
    a = (nu + (N-1)*alpha2^(N-2)*(alpha2^N-xi)/beta_nu) * b;
    
    KUT = 1;c2 = 1;
    
    f2 = 1;
    for n = N:-1:1
        tn = double(ttv(Y,u,-n));
        if n == N
            KUT = [u{n} tn];
        else
            KUT = kron(KUT,[u{n} tn]);
        end
        c2 = kron(c2,[a ;b]);
        
        %f2 = f2*([a b]*([u{n} tn]'*[u{n} tn]) * [a ; b]);
        f2 = f2*([a b]*([alpha2 xi ; xi tn'*tn]) * [a ; b]);
    end
    c1 = Y(:)'*KUT;
    
 
    % [un,tn]*[a; b]
     
    fcost = f2 -2*c1*c2 ;
    mu = fmincon(@(x) double(subs(fcost,nu,x)),1000*mu,[],[],[],[],0,mu*1000);
    mu = double(mu);
    beta = mu +  N * alpha2^(N-1);
    unew = 1/(alpha2^(N-1)+mu) *  ( (mu + (N-1)*alpha2^(N-2)*(alpha2^N-xi)/beta) * theta  + t );
end

d = unew - theta;
unew = mat2cell(unew,szY);
end
