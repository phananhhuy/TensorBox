function [H] = cp_hessian(U)
% Generate the Hessian of the CP model of the factor matrices Un
% H_mu = J'*J + \mu * I
%
% Phan Anh Huy
%
%
%% Jacobian
N = numel(U);
In = cellfun(@(x) size(x,1),U,'uni',1);
R = size(U{1},2);

%% Hessian
% H = G + Z*K*Z'
% or

H = hessian_form2;

%%
 
    function [H,J] = hessian_jacobian
        % H = J'*J
        J = [];
        for n = 1: N
            Pn = permute_vec(In,n);
            Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(In(n)));
            J = [J Pn' * Jn];
        end
        H = J'*J;
    end


    function H = hessian_form1
        % H = G + Z * K * Z'
        Z = [];
        for n = 1: N
            Z = blkdiag(Z,kron(eye(R),U{n}));
        end
        
        K = zeros(N*R^2);
        for n = 1: N
            for m = setdiff(1:N,n)
                gmn = prod(Gamma_n(:,:,setdiff(1:N,[n,m])),3);
                K((n-1)*R^2+1:(n)*R^2,(m-1)*R^2+1:(m)*R^2) = Prr*diag(gmn(:));
            end
        end
        H = G + Z * K * Z';
    end


    function H = hessian_form2
        % H = G + Z*K0*Z' + V * K1 * V'
        %
        Prr = per_vectrans(R,R);
        
        Gamma_n = zeros(R,R,N);
        for n = 1: N
            Gamma_n(:,:,n) = U{n}'*U{n};
        end
        
        G =[];
        for n = 1:N
            gnn = prod(Gamma_n(:,:,[1:n-1 n+1:N]),3);
            G = blkdiag(G,kron(gnn, eye(In(n))));
        end
        
        Z = [];
        for n = 1: N
            Z = blkdiag(Z,kron(eye(R),U{n}));
        end
        
        Gamma = prod(Gamma_n,3);
        F = diag(Gamma(:));
        
        K0 = -diag(reshape(bsxfun(@rdivide,Gamma,Gamma_n.^2),[],1));
        K0 = kron(eye(N),Prr) * K0;
        
        %  kron(eye(N),Prr) *D = D * Prr;
        Zd = [];
        for n = 1:N
            Dn = diag(1./reshape(Gamma_n(:,:,n),[],1));
            temp = kron(eye(R),U{n}) * Dn;
            Zd = [Zd;temp];
        end
        Hblkdiag = G + Z * K0 * Z';
        H = Hblkdiag + Zd * (Prr* F) * Zd';
    end
end
