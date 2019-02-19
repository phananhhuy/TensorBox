function [CRB_a,CRB_U,CRIB_a,CRB,CRBaf] =crb_bcdLp1(a,U,lambda,G,crlb_opts)
% Cramer-Rao Lower Bound on estimation parameters for the rank-1 +
% multilinear rank-(R-1,...,R-1) BCD
% Y = lambda * [[a]] + G x {U1,U2,...,UN}
%
% Constraints are imposed on a, U and the core tensor
% G(2:end,2:end,...,2:end)
% Output :
%   CRB_a: CRB on squares errors on estimation of a{n}
%   CRB_U: CRB on squares errors on estimation of U{n}
%   CIRB_a: CRB on squares angular errors on estimation of a{n}
%%

if nargin < 5
    crlb_opts = 1; %1- with_constraintsG, %2- without _constraintsG
end
N = numel(a); R = size(U{1},2)+1;


X{1} = ktensor(lambda,a);
X{2} = ttensor(tensor(G),U);X0 = X;
X = rotatefactor(X);
a = X{1}.u; lambda = X{1}.lambda;
U = X{2}.u; G = double(X{2}.core);
if lambda< 0 
    lambda = -lambda;
    a{1} = -a{1};
    U{1}(:,1) = -U{1}(:,1);
    ss = ones(R-1,1);ss(1) = -1;
    G = ttm(tensor(G),diag(ss),1);
    G = double(G);
end


%%
CRB_U = [];CRBaf =[];
% CRLB with constraints on a, U and G
switch crlb_opts
    case 1 %'with_constraintsG'
        % % Rotate core G(2:end,...,2:end)
        opts = struct('init','nvecs','printitn',0);
        if R > 3
            Gc = tucker_als(tensor(G(2:end,2:end,2:end)),R-2,opts);
            U = cellfun(@(x,y) [x(:,1) x(:,2:end)*y],U,Gc.U,'uni',0);
            Q = cellfun(@(x) blkdiag(1, x),Gc.U,'uni',0);
            G = double(ttm(tensor(G),Q,'t'));
        end


        [CRB_a,CRB_U,CRIB_a,CRB] = crb_ctr_aUG(a,U,lambda,G);
    
    case 2 %'with_constraints on wn and Un '
        [CRB_a,CRB_U,CRIB_a,CRB] = crb_ctr_aU(a,U,lambda,G);
        
    case 3 % concentrated CRLB with constraints on wn, un and alpha_n =[xi_n, rho_n]
        %[CRB_a,CRIB_a,CRB] = crb_ctr_concentrated(a,U,lambda,G);
        [CRB_a,CRIB_a,CRB,CRBaf] = crb_ctr_concentrated_fast(a,U,lambda,G);    
        
    case 4
        [CRB_a,CRIB_a,CRB] = crb_ctr_concentrated_fullform(a,U,lambda,G);
        
    otherwise
        %[CRB_a,CRIB_a,CRB] = crb_ctr_concentrated(a,U,lambda,G);    
        
        [CRB_a,CRIB_a,CRB] = crb_ctr_concentrated_fast(a,U,lambda,G);    
end


end


function [CRB_a,CRB_U,CRIB_a,CRB] = crb_ctr_aUG(a,U,lambda,G)

N = numel(a); R = size(U{1},2)+1;
%% Prepare for computing Cramer Rao Lower Bound
u = cellfun(@(x) x(:,1),U,'uni',0);
rhoc = cellfun(@(x,y) x'*y,a,u,'uni',1);
rho = prod(rhoc);
w = cell(N,1);
for n = 1:N
    w{n} = a{n}-rhoc(n)*u{n};
    w{n} = w{n}/norm(w{n});
end
alpha = cell(N,1);
for n = 1:N
    alpha{n} = [sqrt(1 - rhoc(n)^2);rhoc(n)];
end

% alpha = [sqrt(1-rhoc.^2) rhoc]';
% alpha = mat2cell(alpha,2,ones(1,N));

%% Jacobian of the constraint functions
% J_constraints = [JwuU
%                       Jalpha
%                              Jg_2:R-1]

% Expand the contraints function by some dummy variables for entries of G
% without any constraints (entries on the first slices)

% fgdummy = [G_(idg_withoutconstraint) - etha)]
%
% new parameters is theta_new = [theta; etha]
%
%
% Jacobian of the new expanded function
% J_constraints_expand = [JwuU
%                             Jalpha
%                                  Jg_2:R-1
%                                         Iatidg_withoutconstraint -I]
%
% Note that Jc_g_2:R-1 comprises zero columns at g_withoutconstraint .
% Orthogonal complement of J_constraints
% Jc_constraints = [Jc_wuU
%                         Jc_alpha
%                                  Jc_g_2:R-1
%                                  Iatidg_withoutconstraint/sqrt(2)   I/sqrt(2)]
%
%
%  FIM matrix for the expanded parameters
%
%   FIM_expand = [FIM  0
%                 0    0];
%
%  ->  J_constraints_expand'* FIM * J_constraints_expand =
%     = [J_constraints'*FIM_1 *J_constraints   J_constraints'*FIM_2/sqrt(2)
%        FIM_2'*J_constraints/sqrt(2)             FIM_3/2]
%
% FIM = [FIM_1 FIM_2
%        FIM_2' FIM_3]
%
%
% CRLB for [wn Un] are located in top_left block of Inv(J_constraints_expand'* FIM * J_constraints_expand)


%% Orthogonal complement to Jacobian of the constraints functions for [wn, Un, rho_n
JFuc = [];
% id of entries on upper left of matrix R x R
id = tril(reshape(1:R^2,R,R)',-1);id(id==0)= [];
id = [1:R+1:R^2 id]';
ix = ind2sub_full([R R],id);
for n = 1:N
    V = [w{n} U{n}];
    B1c = (zeros(R^2,R*(R-1)/2));
    for k = R+1:numel(id)
        B1c((ix(k,1)-1)*R+1:ix(k,1)*R,k-R) = V(:,ix(k,2))/sqrt(2);
        B1c((ix(k,2)-1)*R+1:ix(k,2)*R,k-R) = -V(:,ix(k,1))/sqrt(2);
    end
    JFuc = blkdiag(JFuc,B1c);
end
for n = 1:N
    JFuc = blkdiag(JFuc, [-alpha{n}(2); alpha{n}(1)]);
end


%% Orthogonal complement to Jacobian of the constraint function for G
% orthogonality G_n(2:end,:) * G_n(2:end,:) = I_{R-2)
idg = tril(reshape(1:(R-2)^2,(R-2),(R-2))',-1);idg(idg==0)= [];
% idg = [1:R-1:(R-2)^2 idg]';
idg_sub = ind2sub_full([R-2 R-2],idg);

% Jacobian of the constrain function wrt [wn;un;alphan]

JFg = [];
Gi = reshape(1:(R-1)^3,R-1,R-1,R-1);
Gi2 = Gi(2:end,2:end,2:end);
for n = 1:N
    Gi2n = double(tenmat(Gi2,n));
    g = (zeros((R-1)^3,(R-2)*(R-3)/2));
    %     for r = 1:R-2
    %         g(Gi2n(r,:),r) = 2*G(Gi2n(r,:));
    %     end
    for r = 1:numel(idg)
        g(Gi2n(idg_sub(r,2),:),r) = G(Gi2n(idg_sub(r,1),:));
        g(Gi2n(idg_sub(r,1),:),r) = G(Gi2n(idg_sub(r,2),:));
        
    end
    JFg = [JFg ; g.'];
end
JFgv =JFg;
% idg_withoutconstraint = setdiff(1:(R-1)^3,Gi2(:)); %(R-1)^3 - (R-2)^3
idg_withconstraint = Gi2(:);

[qq,rr] = qr(JFgv(:,idg_withconstraint)');
Jfgcv = qq(:,size(JFgv)+1:end);

Jfgcvfull = zeros(size(JFgv,2),size(Jfgcv,2));
Jfgcvfull(idg_withconstraint,:) = Jfgcv;


% Complement to Jacobian of the constraints function
% JFucv = subs(JFuc,theta,thetav);
JFucv = JFuc;
JFcv = blkdiag(JFucv,Jfgcvfull);


%% CONSTRUCTION of Hessian H = [J_V^T J_V] where V = [wn Un]
e1 = zeros(R-1,1);e1(1) = 1;
H = zeros(N*R^2);
for n = 1:N
    for m = n:N
        if n ==m
            Gn = double(tenmat(G,n));
            gn1= Gn(:,1);
            
            Kn = [0 alpha{n}(1)/alpha{n}(2) alpha{n}(1)
                gn1 e1*[1 alpha{n}(2)]] * ...
                [0 lambda*rho 0 ; lambda*rho 0 0; 0 0 lambda^2] *  [0 alpha{n}(1)/alpha{n}(2) alpha{n}(1);gn1 e1*[1 alpha{n}(2)]]';
            Kn(2:end,2:end) = Kn(2:end,2:end) + Gn*Gn';
            
            Jnm = kron(Kn,eye(R));
            
        else
            Gnm = double(tenmat(G,[n m]));
            Gnm = reshape(Gnm,R-1,R-1,[]);
            Gmn = permute(Gnm,[2 1 3]);
            
            %
            Jwn_wm = lambda^2 * alpha{n}(1)*alpha{m}(1) * a{n} * a{m}';
            Jwn_Um = lambda * alpha{n}(1)*rho/(alpha{n}(2)*alpha{m}(2)) * U{n} * Gnm(:,:,1) * kron(eye(R-1),a{m})';
            Jwn_Um(:,1:R) = Jwn_Um(:,1:R) + lambda^2 * alpha{n}(1) * alpha{m}(2) * a{n} * a{m}';
            
            JUn_wm = lambda * alpha{m}(1)*rho/(alpha{n}(2)*alpha{m}(2)) * kron(eye(R-1),a{n})*Gnm(:,:,1)*U{m}';
            JUn_wm(1:R,:) = JUn_wm(1:R,:) + lambda^2 * alpha{m}(1) * alpha{n}(2) * a{n} * a{m}';
            
            % JUn_wm  - [subs(Juc{n},theta,thetav) subs(JUc{n},theta,thetav)]'*subs(Jwc{m},theta,thetav)
            
            
            % clear err;
            for s = 1:R-1
                for r = 1:R-1
                    xx2 = U{n}*squeeze(Gnm(:,r,:)) * squeeze(Gmn(:,s,:))'*U{m}';
                    hh2 = xx2;
                    
                    if s == 1
                        gnm_r = Gnm(:,r,1);
                        hh2 = hh2 + lambda * rho/alpha{m}(2) * U{n} * gnm_r *a{m}';
                    end
                    
                    if r == 1
                        gmn_s = Gmn(:,s,1);
                        hh2 = hh2 + lambda * rho/alpha{n}(2) * a{n} * (U{m} * gmn_s)';
                    end
                    
                    if (r == 1) &&(s == 1)
                        hh2 = hh2 + lambda^2 * alpha{n}(2)*alpha{m}(2)* a{n}*a{m}';
                    end
                    
                    Jun_um((s-1)*R+1:s*R,(r-1)*R+1:r*R) = hh2;
                end
            end
            
            Jnm = [Jwn_wm  Jwn_Um
                JUn_wm  Jun_um];
        end
        
        H((n-1)*R^2+1:n*R^2,(m-1)*R^2+1:m*R^2) = Jnm;
        if n ~= m
            H((m-1)*R^2+1:m*R^2,(n-1)*R^2+1:n*R^2) = Jnm';
        end
    end
end


% J_alpha^T J_alpah
Halpha = zeros(2*N,2*N);
for n = 1:N
    for m = n:N
        if m == n
            Halpha((n-1)*2+1:2*n,(n-1)*2+1:2*n)= lambda^2 * eye(2);
        else
            Halpha((n-1)*2+1:2*n,(m-1)*2+1:2*m)= lambda^2 * alpha{n}*alpha{m}';
            Halpha((m-1)*2+1:2*m,(n-1)*2+1:2*n)= lambda^2 * alpha{m}*alpha{n}';
        end
    end
end
H(N*R^2+1:N*R^2+2*N,N*R^2+1:N*R^2+2*N) = Halpha;

% J_[w,U]^T * J_alpha
for n = 1:N
    Gn = double(tenmat(G,n));
    
    for m =1:N
        if m == n
            
            Jwn_alpham = lambda^2 * alpha{n}(1) * [w{n} u{m}];
            
            Jun_alpham = [w{n} u{n}] * (lambda^2 * alpha{n}(2)+lambda*rho/(alpha{n}(2))*G(1));
            JUn_alpham = kron(Gn(2:end,1),[w{n} u{n}])*lambda*rho/alpha{n}(2);
            
        else
            Jwn_alpham = lambda^2 * alpha{n}(1) * a{n}*alpha{m}';
            
            Gnm = double(tenmat(G,[n m]));
            Gnm = reshape(Gnm,R-1,R-1,[]);
            %             Gmn = permute(Gnm,[2 1 3]);
            
            Jun_alpham = a{n} * [(lambda^2 * alpha{n}(2) * alpha{m}(1)) ...
                lambda^2 * alpha{n}(2)*alpha{m}(2)+ lambda*rho/(alpha{n}(2)*alpha{m}(2))*G(1)];
            
            %Jun_alpham = a{n} * [0 lambda*rho/(alpha{n}(2)*alpha{m}(2))*Gnm(r,1,1)];
            JUn_alpham = zeros(R*(R-2),2);
            JUn_alpham(:,2) = kron(Gnm(2:end,1,1),a{n}) * lambda * rho/(alpha{n}(2)*alpha{m}(2)) ;
        end
        
        Jvn_Jalpham = [Jwn_alpham; Jun_alpham;JUn_alpham];
        H((n-1)*R^2+1:n*R^2,N*R^2+(m-1)*2+1:N*R^2+m*2) = Jvn_Jalpham;
        H(N*R^2+(m-1)*2+1:N*R^2+m*2,(n-1)*R^2+1:n*R^2) = Jvn_Jalpham';
    end
end

Hvg = [];e11 = zeros((R-1)^(N-1),1);e11(1) = 1;
for n = 1:N
    Gn = double(tenmat(G,n));
    %     gnr1 = Gn(1,:);
    Hvng = kron([lambda*alpha{n}(1) * rho/alpha{n}(2)*e11'
        lambda* rho *e11' + Gn(1,:)
        Gn(2:end,:)], U{n});
    
    Pn = permute_vec_new((R-1)*ones(1,N),n);
    Hvg = [Hvg; Hvng*Pn];
end

Hrhog = zeros((R-1)^N,2*N);
Hrhog(1,2:2:end) = lambda * rho./cellfun(@(x) x(2) ,alpha);

Hvlda = [];
for n = 1:N
    Gn = double(tenmat(G,n));
    Hvlda = [Hvlda
        kron([lambda * alpha{n}(1)
        lambda * alpha{n}(2) * e1 + rho/alpha{n}(2) * Gn(:,1)],a{n})];
end

Hrholda = lambda*cell2mat(alpha(:));

H2=  [Hvg     Hvlda
    Hrhog'  Hrholda];

H3 = eye((R-1)^(N)+1);
H3(end,1) = rho;
H3(1,end) = rho;

FIM = [H      H2
    H2'    H3];

% H(N*R^2+1:N*R^2+2*N,N*R^2+1:N*R^2+2*N) = lambda^2*eye(N*2);
% Jv = subs(J,theta,thetav);
% Jv = J;
% FIM = Jv'*Jv;
% norm(FIM - H,'fro')

%%  CRLB
% H = Jfc_ex'*FIM*  Jfc_ex

H = [JFcv'*FIM(1:size(JFcv,1),1:size(JFcv,1))* JFcv  JFcv'*FIM(1:size(JFcv,1),size(JFcv,1)+1:end)/sqrt(2)
    FIM(size(JFcv,1)+1:end,1:size(JFcv,1))*JFcv/sqrt(2)     FIM(size(JFcv,1)+1:end,size(JFcv,1)+1:end)/(2)];

iH = inv(H);

CRB = JFcv*iH(1:size(JFcv,2),1:size(JFcv,2))*JFcv.';

% Jacobian of transfer function an = [wn un] * alpha_n
Ja_wnun = [];Ja_alpha = [];
for n = 1:N
    Ja_wnun = blkdiag(Ja_wnun,...
        [alpha{n}(1)*eye(R) alpha{n}(2)*eye(R) zeros(R,R^2-2*R) ]);
    
    Ja_alpha = blkdiag(Ja_alpha,[w{n} u{n}]);
end
Ja_wnun = [Ja_wnun Ja_alpha];
CRB_af = Ja_wnun*CRB(1:N*R^2+2*N,1:N*R^2+2*N)*Ja_wnun';
CRB_a = diag(CRB_af);
%
CRB_U = diag(CRB(1:N*R^2,1:N*R^2));
CRB_U = reshape(CRB_U,R^2,N);
CRB_U = CRB_U(R+1:end,:);
CRB_U = CRB_U(:);

CRIB_a = zeros(N,1);
for n = 1:N
    CRIB_a(n) = trace(CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)) - a{n}'*CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)*a{n};
end

end

%% Constrained CRB with constraints on a, U  and alpha{n} = [xi_n; rho_n]
function [CRB_a,CRB_U,CRIB_a,CRB] = crb_ctr_aU(a,U,lambda,G)

N = numel(a); R = size(U{1},2)+1;

%% Prepare for computing Cramer Rao Lower Bound
u = cellfun(@(x) x(:,1),U,'uni',0);
rhoc = cellfun(@(x,y) x'*y,a,u,'uni',1);
rho = prod(rhoc);
w = cell(N,1);
for n = 1:N
    w{n} = a{n}-rhoc(n)*u{n};
    w{n} = w{n}/norm(w{n});
end
alpha = cell(N,1);
for n = 1:N
    alpha{n} = [sqrt(1 - rhoc(n)^2);rhoc(n)];
end

% alpha = [sqrt(1-rhoc.^2) rhoc]';
% alpha = mat2cell(alpha,2,ones(1,N));

%% Jacobian of the constraint functions
% J_constraints = [JwuU
%                       Jalpha]

% Expand the contraints function by some dummy variables for entries of G
% without any constraints (entries on the first slices)

% fgdummy = [[G,lambda] - etha)]
%
% new parameters is theta_new = [theta; etha]
%
%
% Jacobian of the new expanded function
% J_constraints_expand = [JwuU
%                             Jalpha
%                                  I_Glda -I]
%
% Orthogonal complement of J_constraints
% Jc_constraints = [Jc_wuU
%                         Jc_alpha
%                                  I/sqrt(2)   I/sqrt(2)]
%
%
%  FIM matrix for the expanded parameters
%
%   FIM_expand = [FIM  0
%                 0    0];
%
%  ->  J_constraints_expand'* FIM * J_constraints_expand =
%     = [J_constraints'*FIM_1 *J_constraints   J_constraints'*FIM_2/sqrt(2)
%        FIM_2'*J_constraints/sqrt(2)             FIM_3/2]
%
% FIM = [FIM_1 FIM_2
%        FIM_2' FIM_3]
%
%
% CRLB for [wn Un] are located in top_left block of Inv(J_constraints_expand'* FIM * J_constraints_expand)


%% Orthogonal complement to Jacobian of the constraints functions for [wn, Un, rho_n]
JFuc = [];
% id of entries on upper left of matrix R x R
id = tril(reshape(1:R^2,R,R)',-1);id(id==0)= [];
id = [1:R+1:R^2 id]';
ix = ind2sub_full([R R],id);
for n = 1:N
    V = [w{n} U{n}];
    B1c = (zeros(R^2,R*(R-1)/2));
    for k = R+1:numel(id)
        B1c((ix(k,1)-1)*R+1:ix(k,1)*R,k-R) = V(:,ix(k,2))/sqrt(2);
        B1c((ix(k,2)-1)*R+1:ix(k,2)*R,k-R) = -V(:,ix(k,1))/sqrt(2);
    end
    JFuc = blkdiag(JFuc,B1c);
end
for n = 1:N
    JFuc = blkdiag(JFuc, [-alpha{n}(2); alpha{n}(1)]);
end

JFcv = JFuc;


%% CONSTRUCTION of Hessian H = [J_V^T J_V] where V = [wn Un]
e1 = zeros(R-1,1);e1(1) = 1;
H = zeros(N*R^2);
for n = 1:N
    for m = n:N
        if n ==m
            Gn = double(tenmat(G,n));
            gn1= Gn(:,1);
            
            Kn = [0 alpha{n}(1)/alpha{n}(2) alpha{n}(1)
                gn1 e1*[1 alpha{n}(2)]] * ...
                [0 lambda*rho 0 ; lambda*rho 0 0; 0 0 lambda^2] *  [0 alpha{n}(1)/alpha{n}(2) alpha{n}(1);gn1 e1*[1 alpha{n}(2)]]';
            Kn(2:end,2:end) = Kn(2:end,2:end) + Gn*Gn';
            
            Jnm = kron(Kn,eye(R));
            
        else
            Gnm = double(tenmat(G,[n m]));
            Gnm = reshape(Gnm,R-1,R-1,[]);
            Gmn = permute(Gnm,[2 1 3]);
            
            %
            Jwn_wm = lambda^2 * alpha{n}(1)*alpha{m}(1) * a{n} * a{m}';
            Jwn_Um = lambda * alpha{n}(1)*rho/(alpha{n}(2)*alpha{m}(2)) * U{n} * Gnm(:,:,1) * kron(eye(R-1),a{m})';
            Jwn_Um(:,1:R) = Jwn_Um(:,1:R) + lambda^2 * alpha{n}(1) * alpha{m}(2) * a{n} * a{m}';
            
            JUn_wm = lambda * alpha{m}(1)*rho/(alpha{n}(2)*alpha{m}(2)) * kron(eye(R-1),a{n})*Gnm(:,:,1)*U{m}';
            JUn_wm(1:R,:) = JUn_wm(1:R,:) + lambda^2 * alpha{m}(1) * alpha{n}(2) * a{n} * a{m}';
            
            % JUn_wm  - [subs(Juc{n},theta,thetav) subs(JUc{n},theta,thetav)]'*subs(Jwc{m},theta,thetav)
            
            
            % clear err;
            for s = 1:R-1
                for r = 1:R-1
                    xx2 = U{n}*squeeze(Gnm(:,r,:)) * squeeze(Gmn(:,s,:))'*U{m}';
                    hh2 = xx2;
                    
                    if s == 1
                        gnm_r = Gnm(:,r,1);
                        hh2 = hh2 + lambda * rho/alpha{m}(2) * U{n} * gnm_r *a{m}';
                    end
                    
                    if r == 1
                        gmn_s = Gmn(:,s,1);
                        hh2 = hh2 + lambda * rho/alpha{n}(2) * a{n} * (U{m} * gmn_s)';
                    end
                    
                    if (r == 1) &&(s == 1)
                        hh2 = hh2 + lambda^2 * alpha{n}(2)*alpha{m}(2)* a{n}*a{m}';
                    end
                    
                    Jun_um((s-1)*R+1:s*R,(r-1)*R+1:r*R) = hh2;
                end
            end
            
            Jnm = [Jwn_wm  Jwn_Um
                JUn_wm  Jun_um];
        end
        
        H((n-1)*R^2+1:n*R^2,(m-1)*R^2+1:m*R^2) = Jnm;
        if n ~= m
            H((m-1)*R^2+1:m*R^2,(n-1)*R^2+1:n*R^2) = Jnm';
        end
    end
end


% J_alpha^T J_alpah
Halpha = zeros(2*N,2*N);
for n = 1:N
    for m = n:N
        if m == n
            Halpha((n-1)*2+1:2*n,(n-1)*2+1:2*n)= lambda^2 * eye(2);
        else
            Halpha((n-1)*2+1:2*n,(m-1)*2+1:2*m)= lambda^2 * alpha{n}*alpha{m}';
            Halpha((m-1)*2+1:2*m,(n-1)*2+1:2*n)= lambda^2 * alpha{m}*alpha{n}';
        end
    end
end
H(N*R^2+1:N*R^2+2*N,N*R^2+1:N*R^2+2*N) = Halpha;

% J_[w,U]^T * J_alpha
for n = 1:N
    Gn = double(tenmat(G,n));
    
    for m =1:N
        if m == n
            
            Jwn_alpham = lambda^2 * alpha{n}(1) * [w{n} u{m}];
            
            Jun_alpham = [w{n} u{n}] * (lambda^2 * alpha{n}(2)+lambda*rho/(alpha{n}(2))*G(1));
            JUn_alpham = kron(Gn(2:end,1),[w{n} u{n}])*lambda*rho/alpha{n}(2);
            
        else
            Jwn_alpham = lambda^2 * alpha{n}(1) * a{n}*alpha{m}';
            
            Gnm = double(tenmat(G,[n m]));
            Gnm = reshape(Gnm,R-1,R-1,[]);
            %             Gmn = permute(Gnm,[2 1 3]);
            
            Jun_alpham = a{n} * [(lambda^2 * alpha{n}(2) * alpha{m}(1)) ...
                lambda^2 * alpha{n}(2)*alpha{m}(2)+ lambda*rho/(alpha{n}(2)*alpha{m}(2))*G(1)];
            
            %Jun_alpham = a{n} * [0 lambda*rho/(alpha{n}(2)*alpha{m}(2))*Gnm(r,1,1)];
            JUn_alpham = zeros(R*(R-2),2);
            JUn_alpham(:,2) = kron(Gnm(2:end,1,1),a{n}) * lambda * rho/(alpha{n}(2)*alpha{m}(2)) ;
        end
        
        Jvn_Jalpham = [Jwn_alpham; Jun_alpham;JUn_alpham];
        H((n-1)*R^2+1:n*R^2,N*R^2+(m-1)*2+1:N*R^2+m*2) = Jvn_Jalpham;
        H(N*R^2+(m-1)*2+1:N*R^2+m*2,(n-1)*R^2+1:n*R^2) = Jvn_Jalpham';
    end
end

Hvg = [];e11 = zeros((R-1)^(N-1),1);e11(1) = 1;
for n = 1:N
    Gn = double(tenmat(G,n));
    %     gnr1 = Gn(1,:);
    Hvng = kron([lambda*alpha{n}(1) * rho/alpha{n}(2)*e11'
        lambda* rho *e11' + Gn(1,:)
        Gn(2:end,:)], U{n});
    
    Pn = permute_vec_new((R-1)*ones(1,N),n);
    Hvg = [Hvg; Hvng*Pn];
end

Hrhog = zeros((R-1)^N,2*N);
Hrhog(1,2:2:end) = lambda * rho./cellfun(@(x) x(2) ,alpha);

Hvlda = [];
for n = 1:N
    Gn = double(tenmat(G,n));
    Hvlda = [Hvlda
        kron([lambda * alpha{n}(1)
        lambda * alpha{n}(2) * e1 + rho/alpha{n}(2) * Gn(:,1)],a{n})];
end

Hrholda = lambda*cell2mat(alpha(:));

H2=  [Hvg     Hvlda
    Hrhog'  Hrholda];

% H3 = eye((R-1)^(N)+1);
% H3(end,1) = rho;
% H3(1,end) = rho;

%%  CRLB
% FIM = [H      H2
%        H2'    H3];
% 
% % H = Jfc_ex'*FIM*  Jfc_ex
% 
% H = [JFcv'*FIM(1:size(JFcv,1),1:size(JFcv,1))* JFcv  JFcv'*FIM(1:size(JFcv,1),size(JFcv,1)+1:end)/sqrt(2)
%     FIM(size(JFcv,1)+1:end,1:size(JFcv,1))*JFcv/sqrt(2)     FIM(size(JFcv,1)+1:end,size(JFcv,1)+1:end)/(2)];
% 
% % H = [H1  H2
% %      H2' H3]
% %  H3 is of size (R-1)^N+1 x   (R-1)^N+1
% %  H3 = I + rho * [e1 eK]   [eK e1]^T;
% % inv(H3) = I - rho^2/(1-rho^2) * [e1 e2] * [-1  1/rho; 1/rho -1] * [e1 e2]^T
% %
% % inv(H1) = inv(H1 - H2*inv(H3)*H2')
% 
% % iH = inv(H);
% H1 = H(1:size(JFcv,2),1:size(JFcv,2));
% H2 = H(1:size(JFcv,2),size(JFcv,2)+1:end);
% % H3 = H(size(JFc,2)+1:end,size(JFc,2)+1:end);
% % iH3 = inv(H3);
% % iH1 = pinv(H1 - H2*iH3*H2');
% iH1 = pinv(H1 - H2*H2' + (rho/(1-rho^2)) * H2(:,[1 end]) * [-rho  1; 1 -rho] * H2(:,[1 end])');

%% CRLB
H1 = JFcv'*H* JFcv;
H2 = JFcv'*H2/sqrt(2);
iH1 = pinv(H1 - H2*H2' + (rho/(1-rho^2)) * H2(:,[1 end]) * [-rho  1; 1 -rho] * H2(:,[1 end])');


%%
CRB = JFcv*iH1*JFcv.';

% Jacobian of transfer function an = [wn un] * alpha_n
Ja_wnun = [];Ja_alpha = [];
for n = 1:N
    Ja_wnun = blkdiag(Ja_wnun,...
        [alpha{n}(1)*eye(R) alpha{n}(2)*eye(R) zeros(R,R^2-2*R) ]);
    
    Ja_alpha = blkdiag(Ja_alpha,[w{n} u{n}]);
end
Ja_wnun = [Ja_wnun Ja_alpha];
CRB_af = Ja_wnun*CRB(1:N*R^2+2*N,1:N*R^2+2*N)*Ja_wnun';
CRB_a = diag(CRB_af);
%
CRB_U = diag(CRB(1:N*R^2,1:N*R^2));
CRB_U = reshape(CRB_U,R^2,N);
CRB_U = CRB_U(R+1:end,:);
CRB_U = CRB_U(:);

CRIB_a = zeros(N,1);
for n = 1:N
    CRIB_a(n) = trace(CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)) - a{n}'*CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)*a{n};
end

end


%% Concentrated Constrained CRB with constraints on w_n, u_n  and alpha{n} = [xi_n; rho_n]
function [CRB_a,CRIB_a,CRB] = crb_ctr_concentrated(a,U,lambda,G)

N = numel(a); R = size(U{1},2)+1;

%% Prepare for computing Cramer Rao Lower Bound
u = cellfun(@(x) x(:,1),U,'uni',0);
rhoc = cellfun(@(x,y) x'*y,a,u,'uni',1);
rho = prod(rhoc);
w = cell(N,1);
for n = 1:N
    w{n} = a{n}-rhoc(n)*u{n};
    w{n} = w{n}/norm(w{n});
end
alpha = cell(N,1);
for n = 1:N
    alpha{n} = [sqrt(1 - rhoc(n)^2);rhoc(n)];
end
rhov = rhoc;
rhoav = prod(rhov);
xiv = sqrt(1-rhov.^2);

% alpha = [sqrt(1-rhoc.^2) rhoc]';
% alpha = mat2cell(alpha,2,ones(1,N));

%% Jacobian of the constraint functions
% J_constraints = [Jwu
%                       Jalpha ]

% Expand the contraints function by some dummy variables for Un(:,2:end), G
% and lambda

% fgdummy = [[Un(2:end), G,lambda] - etha)]
%
% new parameters is theta_new = [theta; etha]
%
%
% Jacobian of the new expanded function
% J_constraints_expand = [JwuU
%                             Jalpha
%                                  I_Glda -I]
%
% Orthogonal complement of J_constraints
% Jc_constraints = [Jc_wuU
%                         Jc_alpha
%                                  I/sqrt(2)   I/sqrt(2)]
%
%
%  FIM matrix for the expanded parameters
%
%   FIM_expand = [FIM  0
%                 0    0];
%
%  ->  J_constraints_expand'* FIM * J_constraints_expand =
%     = [J_constraints'*FIM_1 *J_constraints   J_constraints'*FIM_2/sqrt(2)
%        FIM_2'*J_constraints/sqrt(2)             FIM_3/2]
%
% FIM = [FIM_1 FIM_2
%        FIM_2' FIM_3]
%
%
% CRLB for [wn Un] are located in top_left block of Inv(J_constraints_expand'* FIM * J_constraints_expand)

%% CLEAN and compact version for CONSTRUCTION of Hessian with at ERR = 0
% function x = b
rhoev = rhoav./rhov;

Hwu =  (zeros(R*2*N));
Hrho_wu =  (zeros(2*N,R*2*N));
Hlda_wu =  (zeros(1,R*2*N));
Hlda_rho =  (zeros(1,2*N));

Ir = eye(R);

for n = 1:N
    %zn = double(ttv(Yv,u,-n));%
    Gn = double(tenmat(G,n));
    zn = lambda * rhoev(n) * a{n} + U{n}*Gn(:,1);
    
    for m = n:N
       
        %%
        if n == m 
            %Znvv = ttm(Yv,Wv,-n);
            %Qn = double(ttt(Znvv,Yv,setdiff(1:N,n)));
            Qn = lambda^2 * rhoev(n)^2 * a{n}*a{n}' + U{n}*(Gn*Gn')*U{n}'  ...
                + lambda * rhoev(n) * U{n}*Gn(:,1)*a{n}' ...
                + lambda * rhoev(n) * a{n} * (U{n}*Gn(:,1))';
              
            Hwnwm = lambda^2*xiv(n)^2 * (1 - rhoev(n)^2) * Ir  + Qn;
            
            %Hwnwm = (je_wn.'*je_wn);
            Hunum = lambda^2 * (rhov(n)^2 - rhoav^2) * Ir;% %je_un.'*je_un;
            Hwnum = lambda^2 *xiv(n) * rhov(n) * (1 - rhoev(n)^2) * Ir;% je_wn.'*je_un;
            Hunwm = Hwnum; %je_un.'*je_wn;
            
            Hwu_nm = [full(Hwnwm)  full(Hwnum)
                full(Hunwm) full(Hunum)];
            
            Hwu((n-1)*2*R+1:n*2*R,(m-1)*2*R+1:m*2*R) = Hwu_nm;
            
        else
            %% Hwnwm.terms{4}
            Gnm = double(tenmat(G,[n m]));
            Gnm1 = reshape(Gnm(:,1),R-1,R-1);
            rhoe_nm = rhoev(n)/rhov(m);
            %Znm = double(ttv(Yv,u,-[n m]));
            Znm = lambda * rhoe_nm * a{n}*a{m}' + U{n} * Gnm1 * U{m}';
   
            Hwnwm = lambda^2 * xiv(n)*xiv(m) * a{n} * a{m}' ...
                + lambda^2*xiv(n)*xiv(m) * rhoe_nm^2 * ...
                (2*xiv(n)*xiv(m) * w{n} * w{m}' - rhov(n) * rhov(m) * u{n} * u{m}')...
                -2*lambda*xiv(n)*xiv(m) * rhoe_nm * Znm;

            %  Note that Znm == (lambda * rhoe_nm * a{n} * a{m}' + U{n} * double(squeeze(G(:,:,1))) * U{m}')
            
            %%  Hwnum je_wn.'*je_um;
            Hwnum = lambda^2 * xiv(n) * rhov(m) * (1 - rhoe_nm^2)* a{n} * a{m}' ...
                +lambda^2*xiv(n)* xiv(m) * rhov(m)*rhoe_nm^2 * (2*a{n}-rhov(n)* u{n}) *w{m}' ...
                - lambda * xiv(n) * rhoev(n) * Znm;

            %% Hunwm Hunwm = je_un.'*je_wm;
            Hunwm  = lambda^2 * xiv(m) * rhov(n) *(1-rhoe_nm^2)* a{n} * a{m}' ... 
                +lambda^2 * xiv(n)*xiv(m)*rhov(n)* rhoe_nm^2 * w{n} *(2*a{m} -rhov(m)* u{m})' ...
                -lambda * rhov(n)* rhoe_nm * xiv(m) * Znm  ;
                
            %% Hunum            
            Hunum = lambda^2 * [a{n} u{n}] * ...
                        [rhov(n)*rhov(m) -rhoav*rhoev(n);
                         -rhoav*rhoev(m)  rhoav^2] ...
                        * [a{m} u{m}].';% je_un.'*je_um;
 
            Hwu_nm = [full(Hwnwm)  full(Hwnum)
                full(Hunwm)  full(Hunum)];
            
            Hwu((n-1)*2*R+1:n*2*R,(m-1)*2*R+1:m*2*R) = Hwu_nm;
            Hwu((m-1)*2*R+1:m*2*R,(n-1)*2*R+1:n*2*R) = Hwu_nm.';
        end
    end
     
    %% Hrho_wu
    
    
    for m = 1:N
        % Hrho_wu
        if n == m
            %  Hrhon_wm = je_rhon.'*je_wn; Hxin_wm =  je_xin.'*je_wn; Hrhon_um =  je_rhon.'*je_un; Hxin_um = je_xin.'*je_un;
            Hrhon_um = lambda^2 * rhov(n) * (1 - rhoev(n)^2) * u{n}';
            Hxin_um = lambda^2 * rhov(n) * (1 - rhoev(n)^2) * w{n}';
            
            %Hxin_wm = lambda^2 *xiv(n) * w{n} - lambda * rhoev(n) * ((lambda*rhoev(n)*xiv(n)) * w{n} +  zn);
            Hxin_wm = lambda^2 *xiv(n) * (1 - rhoev(n)^2) * w{n} - lambda * rhoev(n) *zn;
            Hxin_wm = Hxin_wm';
            Hrhon_wm = lambda^2 * xiv(n) * (1 - rhoev(n)^2) * u{n}; Hrhon_wm = Hrhon_wm';
            
            Hrhon_wum = [full(Hxin_wm)   full(Hxin_um)
                full(Hrhon_wm)  full(Hrhon_um)];
            
            Hrho_wu((n-1)*2+1:n*2,(m-1)*2*R+1:m*2*R) = Hrhon_wum;
            
        else 
              
            % Hrhon_wm =  je_rhon.'*je_wm; Hxin_wm =  je_xin.'*je_wm; % Hrhon_um =  je_rhon.'*je_um; Hxin_um =   je_xin.'*je_um;
%             zm = double(ttv(Yv,u,-m));
            Gm = double(tenmat(G,m));
            zm = lambda * rhoev(m) * a{m} + U{m}*Gm(:,1);
            rhoe_nm = rhoev(n)/rhov(m);
%             Hrhon_wm = (lambda^2 * xiv(m) * rhov(n) * a{m} ...
%                 -lambda^2 * xiv(m) * rhoev(n) * rhoev(m)*   u{m} ...
%                 -lambda * xiv(m) * rhoev(n)/rhov(m) *( (lambda*xiv(m) * rhoev(m))* w{m} + zm))';
% 

            Hrhon_wm = (lambda^2 * rhov(n)*xiv(m)*(1-rhoe_nm^2)*a{m} ...
                - lambda * xiv(m) * rhoe_nm * zm)';

            Hxin_wm = lambda^2 * xiv(n) * xiv(m) * a{m}';
            Hxin_um = lambda^2 * rhov(m) * xiv(n) *a{m}';
            Hrhon_um = lambda^2 * (rhov(m) * rhov(n) - rhoav^2/(rhov(m) * rhov(n))) * a{m}';
            
            Hrhon_wum = ([full(Hxin_wm)   full(Hxin_um)
                full(Hrhon_wm)  full(Hrhon_um)]);
            
            Hrho_wu((n-1)*2+1:n*2,(m-1)*2*R+1:m*2*R) = Hrhon_wum;
        end
    end
     
    %% Hlda_wu %Hlda_wn =  je_lda.'*je_wn; Hlda_un =  je_lda.'*je_un;
    
    Hlda_wn = (lambda * xiv(n)* a{n} ...
                - lambda * xiv(n)*rhoav^2/rhov(n)* u{n} ...
                - xiv(n) * rhoev(n) * ((lambda*rhoev(n)*xiv(n)) * w{n} +  zn))';
            
    Hlda_un = lambda* rhov(n)* (1-rhoev(n)^2)* a{n}'; 
    
    Hlda_wu((n-1)*2*R+1:(n-1)*2*R+R) =  (Hlda_wn);
    Hlda_wu((n-1)*2*R+R+1:n*2*R) =  (Hlda_un);
    
    % Hlda_rho   Hlda_xin =  je_lda.'*je_xin; Hlda_rhon =  je_lda.'*je_rhon;
    Hlda_xin = lambda * xiv(n);
    Hlda_rhon = lambda * rhov(n)*(1-rhoev(n)^2);
 
    Hlda_rho((n-1)*2+1) = full(Hlda_xin);
    Hlda_rho(2*n) = full(Hlda_rhon);
     
end

Hlda_lda = 1- rhoav^2;

rhoev = rhoav./rhov;
alph = blkdiag(alpha{1},alpha{2},alpha{3});
Hrho_rho = lambda^2*(alph* (ones(N) - eye(N)) * alph' + eye(2*N) - kron(rhoev*rhoev',[0 0 ; 0 1]));

H = [Hwu     Hrho_wu.'  Hlda_wu.'
    Hrho_wu  Hrho_rho   Hlda_rho.'
    Hlda_wu  Hlda_rho   Hlda_lda];
 

JFcv = [];
for n = 1:N
    %[qq,rr] = qr([wv{n} uv{n}]);
    %Uc = qq(:,3:end);
    Uc = U{n}(:,2:end);
    Uc = [blkdiag(Uc,Uc)  [u{n};-w{n}]/sqrt(2)];
    %Uc = blkdiag([blkdiag(Uc,Uc)  [uv{n};-wv{n}]/sqrt(2)], [-rhov(n); xiv(n)]);
    JFcv = blkdiag(JFcv,Uc);
end
for n = 1:N
    JFcv = blkdiag(JFcv, [-rhov(n); xiv(n)]);
end
% JFcv = blkdiag(JFcv,1/sqrt(2));

%% CRLB
FIM = min(H,H'); % when sigma = 1
H1 = FIM(1:size(JFcv,1),1:size(JFcv,1));
H2 = FIM(1:size(JFcv,1),size(JFcv,1)+1:end);
H3 = FIM(size(JFcv,1)+1:end,size(JFcv,1)+1:end); % H3 is a scalar 
iH1 = pinv(JFcv.'*H1*JFcv - JFcv.'*H2*H2'*JFcv/H3);

CRB = JFcv*(iH1)*JFcv.';

% FIM = min(H,H'); % when sigma = 1
% CRB = JFcv*pinv(JFcv'*FIM*JFcv)*JFcv.';
% Jacobian of transfer function an = [wn un] * alpha_n
Ja_wnun = []; Ja_rho= [];
for n = 1:N
    Ja_wnun = blkdiag(Ja_wnun,...
        [xiv(n)*eye(R) rhov(n)*eye(R)]);
    Ja_rho = blkdiag(Ja_rho,[w{n} u{n}]);
end
Ja_wnun = [Ja_wnun Ja_rho];

CRB_af = Ja_wnun*CRB(1:size(Ja_wnun,2),1:size(Ja_wnun,2))*Ja_wnun';
CRB_a = double(diag(CRB_af));

CRIB_a = zeros(N,1);
for n = 1:N
    CRIB_a(n) = trace(CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)) - a{n}'*CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)*a{n};
end

end

%%  FAST concentrated CRB for tensor deflation withou construction the Hessian explicitly
% 
%   Concentrated Constrained CRB with constraints on w_n, u_n  and alpha{n} = [xi_n; rho_n]
%
%   The code computes JFcv'*H1 * JFcv and JFcv'*H2
%   where H = [H1   h2
%              h2'  h_lambda_lambda]
% 
%   H1 :  Hessian of the error function w.r.t ([wn,un,xi,rhon])
%   H2 :  second order derivatives of the error function w.r.t
%   ([wn,un,xi,rhon]) [lambda]

function [CRB_a,CRIB_a,CRB,CRB_af] = crb_ctr_concentrated_fast(a,U,lambda,G)
% CRB_a:  Cramer-Rao lower bound on a{n}, only diagonal of the full CRB CRB_af
% CRIB_a: Cramer-Rao lower bound on a{n}
% CRB :  matrix of Cramer-Rao lower bound for the parameters 
% [wn,un,..., rhon,xn,...,lambda]

N = numel(a); R = size(U{1},2)+1;

%% Prepare for computing Cramer Rao Lower Bound
u = cellfun(@(x) x(:,1),U,'uni',0);
rhoc = cellfun(@(x,y) x'*y,a,u,'uni',1);
rho = prod(rhoc);
w = cell(N,1);
for n = 1:N
    w{n} = a{n}-rhoc(n)*u{n};
    w{n} = w{n}/norm(w{n});
end
alpha = cell(N,1);
for n = 1:N
    alpha{n} = [sqrt(1 - rhoc(n)^2);rhoc(n)];
end
rhov = rhoc;
rhoav = prod(rhov);
xiv = sqrt(1-rhov.^2);

% alpha = [sqrt(1-rhoc.^2) rhoc]';
% alpha = mat2cell(alpha,2,ones(1,N));

%% Jacobian of the constraint functions
% J_constraints = [Jwu
%                       Jalpha ]

% Expand the contraints function by some dummy variables for Un(:,2:end), G
% and lambda

% fgdummy = [[Un(2:end), G,lambda] - etha)]
%
% new parameters is theta_new = [theta; etha]
%
%
% Jacobian of the new expanded function
% J_constraints_expand = [JwuU
%                             Jalpha
%                                  I_Glda -I]
%
% Orthogonal complement of J_constraints
% Jc_constraints = [Jc_wuU
%                         Jc_alpha
%                                  I/sqrt(2)   I/sqrt(2)]
%
%
%  FIM matrix for the expanded parameters
%
%   FIM_expand = [FIM  0
%                 0    0];
%
%  ->  J_constraints_expand'* FIM * J_constraints_expand =
%     = [J_constraints'*FIM_1 *J_constraints   J_constraints'*FIM_2/sqrt(2)
%        FIM_2'*J_constraints/sqrt(2)             FIM_3/2]
%
% FIM = [FIM_1 FIM_2
%        FIM_2' FIM_3]
%
%
% CRLB for [wn Un] are located in top_left block of Inv(J_constraints_expand'* FIM * J_constraints_expand)

%% CLEAN and compact version for CONSTRUCTION of Hessian with at ERR = 0
% function x = b
rhoev = rhoav./rhov; 

JFcv = [];
for n = 1:N
    %[qq,rr] = qr([wv{n} uv{n}]);
    %Uc = qq(:,3:end);
    Uc = U{n}(:,2:end);
    Uc = [blkdiag(Uc,Uc)  [u{n};-w{n}]/sqrt(2)];
    %Uc = blkdiag([blkdiag(Uc,Uc)  [uv{n};-wv{n}]/sqrt(2)], [-rhov(n); xiv(n)]);
    JFcv = blkdiag(JFcv,Uc);
end
for n = 1:N
    JFcv = blkdiag(JFcv, [-rhov(n); xiv(n)]);
end
% JFcv = blkdiag(JFcv,1/sqrt(2));

%% CRLB
%% Fast construction JFcv.'*H_VV*JFcv
JHwuJ = zeros(N*2*(R-2)+2*N);
Ir2 = eye(R-2);

for n = 1:N
    Gn = double(tenmat(G,n));
    
    for m = n:N
        if m == n
            Psin = Gn*Gn';
            
            d1 = (Psin(1,2:end) + lambda * rho * Gn(2:end,1)')/sqrt(2) ;%u{n}(:,1)'* Hwnwm*U{n}(:,2:end) = u{1}'* (Qn) * U{1}(:,2:end)
            d5 = (Psin(1) + 2*lambda*rhoav * Gn(1) + lambda^2* (1- xiv(n)^2*rhoev(n)^2)) /2;
            JnHJm =  [lambda^2*xiv(n)^2*(1-rhoev(n)^2)*Ir2+Psin(2:end,2:end)  lambda^2*xiv(n)*rhov(n)*(1-rhoev(n)^2)*Ir2   d1'
                lambda^2*xiv(n)*rhov(n)*(1-rhoev(n)^2)*Ir2  lambda^2*(rhov(n)^2 - rhoav^2)*Ir2    zeros(R-2,1)
                d1 zeros(1,R-2)  d5];
        else
            Gnm = double(tenmat(G,[n m]));
            Gnm1 = reshape(Gnm(:,1),R-1,R-1);
            rhoe_nm = rhoev(n)/rhov(m);
            
            co2 =  [2*xiv(n)*xiv(m)  xiv(n)*rhov(m)
                xiv(m)*rhov(n)   0];

            JnHJm = -lambda * rhoe_nm *  [ kron(co2, Gnm1(2:end,2:end))    kron(xiv(m)/sqrt(2)*[2*xiv(n) ;rhov(n)],Gnm1(2:end,1))
                                           kron(xiv(n)* [2*xiv(m) rhov(m)],Gnm1(1,2:end))/sqrt(2)  xiv(n)*xiv(m)/2*(2*Gnm1(1,1)+lambda*rhoav)];
        end
        
        JHwuJ((2*R-3)*(n-1)+1:(2*R-3)*n,(2*R-3)*(m-1)+1:(2*R-3)*m) = JnHJm;
        JHwuJ((2*R-3)*(m-1)+1:(2*R-3)*m,(2*R-3)*(n-1)+1:(2*R-3)*n) = JnHJm';
    end
end


%% JFcv.'*H_w_rho
 
JHwrho = zeros(N*(2*R-3),N);
for n = 1:N
    Gn = tenmat(G,n);
    for m = 1:N
        if n == m
            JHwn_rhom = lambda * rhoav*Gn(2:end,1);
            
            jHwn_rhom = (lambda^2*(1 - xiv(n)^2 * rhoev(n)^2) + lambda * rhoav * G(1))/sqrt(2);

        else
            rhoe_nm = rhoev(n)/rhov(m);
            
            JHwn_rhom = -lambda * xiv(n)*xiv(m)*rhoe_nm * Gn(2:end,1);
            
            
            jHwn_rhom = -lambda * xiv(n)*xiv(m) * rhoe_nm * (lambda * rhoav + G(1))/sqrt(2);
        end
        % JHun_rhom = 0;
        JHwrho((2*R-3)*(n-1)+1:(2*R-3)*(n-1)+R-2,m) = JHwn_rhom;
        JHwrho((2*R-3)*(n-1)+2*R-3,m) = jHwn_rhom;
    end
end

JHwuJ(1:N*(2*R-3),N*(2*R-3)+1:end) = JHwrho;
JHwuJ(N*(2*R-3)+1:end,1:N*(2*R-3)) = JHwrho';

for n = 1:N
    for m = n:N
        if n == m
            JHwuJ(N*(2*R-3)+n,N*(2*R-3)+m) = lambda^2*(1 - xiv(n)^2 * rhoev(n)^2);
        else
            JHwuJ(N*(2*R-3)+n,N*(2*R-3)+m) = -lambda^2*xiv(n)*xiv(m)*rhoev(n)*rhoev(m);
            JHwuJ(N*(2*R-3)+m,N*(2*R-3)+n) = -lambda^2*xiv(n)*xiv(m)*rhoev(n)*rhoev(m);
        end
    end
end

JH2 = zeros(N*(2*R-3)+N,1);
for n = 1:N
    Gn = tenmat(G,n);
    JH2((2*R-3)*(n-1)+1:(2*R-3)*(n-1)+R-2) = - xiv(n) * rhoev(n) * Gn(2:end,1);
    JH2((2*R-3)*n) = - xiv(n) * rhoev(n) * (lambda * rhoav +G(1)) /sqrt(2);
    
    JH2(N*(2*R-3)+n) = -lambda * xiv(n) * rhoev(n) * rhoav;
end

%%
H3 = 1- rhoav^2; % Hlda_lda;
iH1 = pinv(JHwuJ - JH2*JH2'/H3);

CRB = JFcv*(iH1)*JFcv.';

sz = cellfun(@(x) size(x,1),w);
% FIM = min(H,H'); % when sigma = 1
% CRB = JFcv*pinv(JFcv'*FIM*JFcv)*JFcv.';
% Jacobian of transfer function an = [wn un] * alpha_n
Ja_wnun = []; Ja_rho= [];
for n = 1:N
    
    Ja_wnun = blkdiag(Ja_wnun,...
       [xiv(n)*eye(sz(n)) rhov(n)*eye(sz(n))]); % fixed on January 6, 2016 for factors with different lengths
    %Ja_wnun = blkdiag(Ja_wnun,...
    %    [xiv(n)*eye(R) rhov(n)*eye(R)]);
    Ja_rho = blkdiag(Ja_rho,[w{n} u{n}]);
end
Ja_wnun = [Ja_wnun Ja_rho];

CRB_af = Ja_wnun*CRB(1:size(Ja_wnun,2),1:size(Ja_wnun,2))*Ja_wnun';
CRB_a = double(diag(CRB_af));

CRIB_a = zeros(N,1);
csz = cumsum([0 ;sz]);
for n = 1:N
    % fixed on January 6, 2016 for factors with different lengths
    CRIB_a(n) = trace(CRB_af(csz(n)+1:csz(n+1),csz(n)+1:csz(n+1))) - a{n}'*CRB_af(csz(n)+1:csz(n+1),csz(n)+1:csz(n+1))*a{n};
    %CRIB_a(n) = trace(CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)) - a{n}'*CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)*a{n};
end

end



%% 
%% Concentrated Constrained CRB with constraints on w_n, u_n  and alpha{n} = [xi_n; rho_n]
function [CRB_a,CRIB_a,CRB] = crb_ctr_concentrated_fullform(a,U,lambda,G)

N = numel(a); R = size(U{1},2)+1;

%% Prepare for computing Cramer Rao Lower Bound
u = cellfun(@(x) x(:,1),U,'uni',0);
rhoc = cellfun(@(x,y) x'*y,a,u,'uni',1);
rho = prod(rhoc);
w = cell(N,1);
for n = 1:N
    w{n} = a{n}-rhoc(n)*u{n};
    w{n} = w{n}/norm(w{n});
end
alpha = cell(N,1);
for n = 1:N
    alpha{n} = [sqrt(1 - rhoc(n)^2);rhoc(n)];
end
rhov = rhoc;
rhoav = prod(rhov);
xiv = sqrt(1-rhov.^2);

Yv = full(ktensor(lambda,a)) + full(ttensor(tensor(G),U));
y0v = Yv(:);
lambdav = lambda;
uv = u;wv = w;av = a;alphav = alpha;
Uv = U;

% Wv = cellfun(@(x) eye(R) - x*x',wv,'uni',0);
Wv = cellfun(@(x) eye(size(x,1)) - x*x',wv,'uni',0); % fixed on January 6, 2016

%% Second order or Hessian for double variables

%     errv = skronsum(y0v) - lambdav * skron(av(end:-1:1)) + lambdav * rhoav * skron(uv(end:-1:1)) - skron(Wv(end:-1:1))*y0v;
% errv = skronsum(y) - lambdav * skron(av(end:-1:1)) + lambdav * rhoav * skron(uv(end:-1:1)) - skron(Wv(end:-1:1))*y;


Hwu =  (zeros(R*2*N));
Hrho_wu =  (zeros(2*N,R*2*N));
Hrho_rho =  (zeros(2*N,2*N));
Hlda_wu =  (zeros(1,R*2*N));
Hlda_rho =  (zeros(1,2*N));

Ir = eye(R);

% COnstruct the Hessian with full expressions

for n = 1:N
    
    an = av;an{n} = eye(R);
    un = uv;un{n} = eye(R);
    aun = av;aun{n} = uv{n};
    awn = av;awn{n} = wv{n};
    
    Pn = permute_vec_new(R*ones(1,N),n);
    Wn = Wv; Wn{n} = eye(R);
    %Z = krona(Wn(end:-1:1))*y;
    Z = skron(Wn(end:-1:1)) * y0v;
    Z = reshape(Z,R*ones(1,N));
    Z = permute(Z,[n 1:n-1 n+1:N]);
    Znv = reshape(Z,R,[]);
    
    kan = skron(an(end:-1:1));
    kun =  skron(un(end:-1:1));
    ka = skron(av(end:-1:1));
    ku = skron(uv(end:-1:1));
    kawn = skron(awn(end:-1:1));
    kaun = skron(aun(end:-1:1));
    
    je_un = -lambdav * rhov(n)*kan + lambdav * rhoav * kun;
    je_wn = -lambdav * xiv(n)*kan + Pn'* kron(Znv.',eye(R)) * (kron(wv{n},eye(R))+ kron(eye(R),wv{n}));
    je_lda = -ka + rhoav * ku;
    je_xin = -lambdav * kawn;
    je_rhon = -lambdav * kaun + lambdav * rhoav/rhov(n) * ku;
    
    %%
    for m = n:N
        
        %% Hwu
        if m ~=n
            an_m = av;an_m{n} = eye(R);an_m{m} = eye(R);
            un_m = uv;un_m{n} = eye(R);un_m{m} = eye(R);
            awn_m = av;awn_m{n} = wv{n};awn_m{m} = eye(R);
            aun_m = av;aun_m{n} = uv{n};aun_m{m} = eye(R);
            awnwm = av;awnwm{n} = wv{n};awnwm{m} = wv{m};
            awnum = av;awnum{n} = wv{n};awnum{m} = uv{m};
            
            aunwm = av;aunwm{n} = uv{n};aunwm{m} = wv{m};
            aunum = av;aunum{n} = uv{n};aunum{m} = uv{m};
            
            kan_m = skron(an_m(end:-1:1));
            kun_m =  skron(un_m(end:-1:1));
            
            kawn_m = skron(awn_m(end:-1:1));
            kaun_m = skron(aun_m(end:-1:1));
            kawnwm = skron(awnwm(end:-1:1));
            kawnum =  skron(awnum(end:-1:1));
            kaunum =  skron(aunum(end:-1:1));
            kaunwm =  skron(aunwm(end:-1:1));
            
            am = av;am{m} = eye(R);
            um = uv;um{m} = eye(R);
            kam = skron(am(end:-1:1));
            kum =  skron(um(end:-1:1));
            
            Pm = permute_vec_new(R*ones(1,N),m);
            Wm = Wv; Wm{m} = eye(R);
            Z = skron(Wm(end:-1:1))*y0v;
            Z = reshape(Z,R*ones(1,N));
            Z = permute(Z,[m 1:m-1 m+1:N]);
            Zmv = reshape(Z,R,[]);
            
            je_wm = -lambdav * xiv(m)*kam + Pm'* kron(Zmv.',eye(R)) * (kron(wv{m},eye(R))+ kron(eye(R),wv{m}));
            je_um = -lambdav * rhov(m)*kam + lambdav * rhoav * kum;
        end
        
        %%
        if n == m
            
            Hwnwm = je_wn.'*je_wn;
            Hunum = je_un.'*je_un;
            
            % je_wn_un = 0
            Hwnum = je_wn.'*je_un;
            Hunwm = je_un.'*je_wn;
            
            Hwu_nm = [full(Hwnwm)  full(Hwnum)
                full(Hunwm) full(Hunum)];
            
            Hwu((n-1)*2*R+1:n*2*R,(m-1)*2*R+1:m*2*R) = Hwu_nm;
            
        else
            
            Hwnwm = je_wn.'*je_wm;
            Hunum = je_un.'*je_um;
            Hwnum = je_wn.'*je_um;
            Hunwm = je_un.'*je_wm;
            
            Hwu_nm = [full(Hwnwm)  full(Hwnum)
                full(Hunwm)  full(Hunum)];
            %             Hwu_nm = (Hwu_nm);
            Hwu((n-1)*2*R+1:n*2*R,(m-1)*2*R+1:m*2*R) = Hwu_nm;
            Hwu((m-1)*2*R+1:m*2*R,(n-1)*2*R+1:n*2*R) = Hwu_nm.';
        end
        
        %% Hrho_rho
        if n == m
            
            Hxin_xin = je_xin.'*je_xin;
            Hxin_rhon = je_xin.'*je_rhon;
            Hrhon_rhon = je_rhon.'*je_rhon;
            
            Hrho_nm = ([full(Hxin_xin)  full(Hxin_rhon)
                full(Hxin_rhon) full(Hrhon_rhon)]);
            Hrho_rho((n-1)*2+1:n*2,(m-1)*2+1:m*2) = Hrho_nm;
            
        else
            
            awm = av;awm{m} = wv{m};
            kawm = skron(awm(end:-1:1));
            aum = av;aum{m} = uv{m};
            kaum = skron(aum(end:-1:1));
            
            
            je_xim = -lambdav * kawm;
            je_rhom = -lambdav * kaum + lambdav * rhoav/rhov(m) * ku;
            
            Hxin_xim =  je_xin.'*je_xim;
            Hxin_rhom =  je_xin.'*je_rhom;
            Hrhon_xim=  je_rhon.'*je_xim;
            Hrhon_rhom =   je_rhon.'*je_rhom;
            
            Hrho_nm = [full(Hxin_xim)  full(Hxin_rhom)
                full(Hrhon_xim) full(Hrhon_rhom)];
            
            Hrho_rho((n-1)*2+1:n*2,(m-1)*2+1:m*2) = Hrho_nm;
            Hrho_rho((m-1)*2+1:m*2,(n-1)*2+1:n*2) = Hrho_nm';
        end
    end
    
    
    %% Hrho_wu
    
    for m = 1:N
        % Hrho_wu
        if n == m
            
            
            %Hrhon_wm = reshape(je_rhon_wn.' * err,R,R) + je_rhon.'*je_wn;
            Hrhon_wm = je_rhon.'*je_wn;
            Hxin_wm =  je_xin.'*je_wn;
            
            Hrhon_um =  je_rhon.'*je_un;
            %Hxin_um = reshape(je_xin_un.' * err,R,R) + je_xin.'*je_un;
            Hxin_um = je_xin.'*je_un;
            
            
            Hrhon_wum = [full(Hxin_wm)   full(Hxin_um)
                full(Hrhon_wm)  full(Hrhon_um)];
            
            Hrho_wu((n-1)*2+1:n*2,(m-1)*2*R+1:m*2*R) = Hrhon_wum;
            
        else%if m>n
            
            awn_m = av;awn_m{n} = wv{n};awn_m{m} = eye(R);
            aun_m = av;aun_m{n} = uv{n};aun_m{m} = eye(R);
            
            kawn_m = skron(awn_m(end:-1:1));
            kaun_m = skron(aun_m(end:-1:1));
            
            am = av;am{m} = eye(R);
            um = uv;um{m} = eye(R);
            kam = skron(am(end:-1:1));
            kum =  skron(um(end:-1:1));
            
            Pm = permute_vec_new(R*ones(1,N),m);
            Wm = Wv; Wm{m} = eye(R);
            Z = skron(Wm(end:-1:1))*y0v;
            Z = reshape(Z,R*ones(1,N));
            Z = permute(Z,[m 1:m-1 m+1:N]);
            Zmv = reshape(Z,R,[]);
            
            je_wm = -lambdav * xiv(m)*kam + Pm'* kron(Zmv.',eye(R)) * (kron(wv{m},eye(R))+ kron(eye(R),wv{m}));
            je_um = -lambdav * rhov(m)*kam + lambdav * rhoav * kum;
            
            
            
            Hrhon_wm =   je_rhon.'*je_wm;
            Hxin_wm =  je_xin.'*je_wm;
            
            Hrhon_um =  je_rhon.'*je_um;
            Hxin_um =  je_xin.'*je_um;
            
            
            Hrhon_wum = ([full(Hxin_wm)   full(Hxin_um)
                full(Hrhon_wm)  full(Hrhon_um)]);
            
            Hrho_wu((n-1)*2+1:n*2,(m-1)*2*R+1:m*2*R) = Hrhon_wum;
            
        end
    end
    
    
    %% Hlda_wu
    
    Hlda_wn =  je_lda.'*je_wn;
    Hlda_un =  je_lda.'*je_un;
    Hlda_wu((n-1)*2*R+1:(n-1)*2*R+R) = full(Hlda_wn);
    Hlda_wu((n-1)*2*R+R+1:n*2*R) = full(Hlda_un);
    
    % Hlda_rho
    Hlda_xin =  je_lda.'*je_xin;
    Hlda_rhon =  je_lda.'*je_rhon;
    
    Hlda_rho((n-1)*2+1) = full(Hlda_xin);
    Hlda_rho(2*n) = full(Hlda_rhon);
    
    
    %%
end

Hlda_lda = 1- rhoav^2;

H = [Hwu     Hrho_wu.'  Hlda_wu.'
    Hrho_wu  Hrho_rho   Hlda_rho.'
    Hlda_wu  Hlda_rho   Hlda_lda];


    %%

JFcv = [];
for n = 1:N
    %[qq,rr] = qr([wv{n} uv{n}]);
    %Uc = qq(:,3:end);
    Uc = U{n}(:,2:end);
    Uc = [blkdiag(Uc,Uc)  [u{n};-w{n}]/sqrt(2)];
    %Uc = blkdiag([blkdiag(Uc,Uc)  [uv{n};-wv{n}]/sqrt(2)], [-rhov(n); xiv(n)]);
    JFcv = blkdiag(JFcv,Uc);
end
for n = 1:N
    JFcv = blkdiag(JFcv, [-rhov(n); xiv(n)]);
end
JFcv = blkdiag(JFcv,1);

%% CRLB
FIM = min(H,H');
CRB = JFcv*pinv(JFcv'*FIM*JFcv)*JFcv.';

% added on January 7
sz = cellfun(@(x) size(x,1),w);

% Jacobian of transfer function an = [wn un] * alpha_n
Ja_wnun = []; Ja_rho= [];
for n = 1:N
    Ja_wnun = blkdiag(Ja_wnun,...
        [xiv(n)*eye(sz(n)) rhov(n)*eye(sz(n))]); % fixed on January 6, 2016 for factors with different lengths
    %Ja_wnun = blkdiag(Ja_wnun,...
    %    [xiv(n)*eye(R) rhov(n)*eye(R)]);
    Ja_rho = blkdiag(Ja_rho,[w{n} u{n}]);
end
Ja_wnun = [Ja_wnun Ja_rho];

CRB_af = Ja_wnun*CRB(1:end-1,1:end-1)*Ja_wnun';
CRB_a = double(diag(CRB_af));

CRIB_a = zeros(N,1);
for n = 1:N
    % fixed on January 6, 2016 for factors with different lengths
    CRIB_a(n) = trace(CRB_af(csz(n)+1:csz(n+1),csz(n)+1:csz(n+1))) - a{n}'*CRB_af(csz(n)+1:csz(n+1),csz(n)+1:csz(n+1))*a{n};
    %CRIB_a(n) = trace(CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)) - a{n}'*CRB_af((n-1)*R+1:n*R,(n-1)*R+1:n*R)*a{n};
end

end





%% ********************************************************************
function Xp = rotatefactor(Xp)
% Rotate factors to be orthogonal and decorrelate
% Rotate factors
P = 2; N = ndims(Xp{1});
for p2 = 1:P
    for n =1:N
        [temp,rr] = qr(Xp{p2}.U{n},0);
        Xp{p2}.U{n} = temp;
        if isa(Xp{p2},'ttensor')
            Xp{p2}.core = ttm(Xp{p2}.core,rr,n);
        elseif isa(Xp{p2},'ktensor')
            Xp{p2}.lambda = Xp{p2}.lambda * rr;
        end
    end
end
% Rotate factors to be orthogonal
for n2 = 1:N
    [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
    
    Xp{1}.U{n2} = Xp{1}.U{n2} * u;
    Xp{2}.U{n2} = Xp{2}.U{n2} * v;
    
    if isa(Xp{1},'ttensor')
        Xp{1}.core = ttm(Xp{1}.core,u',n2);
    elseif isa(Xp{1},'ktensor')
        Xp{1}.lambda = Xp{1}.lambda * u';
    end
    
    
    if isa(Xp{2},'ttensor')
        Xp{2}.core = ttm(Xp{2}.core,v',n2);
    elseif isa(Xp{2},'ktensor')
        Xp{2}.lambda = Xp{2}.lambda * v';
    end
end
end
