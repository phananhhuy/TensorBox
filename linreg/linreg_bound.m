function [X,cone_d] = linreg_bound(y,A,delta,method)
% Solving the linear regresion
%
%    min   \| X \|_F^2
%
%  subject to   \| y - A*vec(X)\| < delta
%
%  where y is a vector of length K,
%  A is required to be full column rank matrix of size K x I, K>=I
%
%  The bound delta must satisfy
%    norm(y)^2 - norm(F'*y)^2 <= delta^2  <= norm(y)^2
%
%  where F is an orthonormal basis of A
%
% Phan Anh Huy, 19/05/2017

if nargin<4
    method = 'closeform';
end
sz = size(A);
delta = max(0,delta);
%% Compression when K exceeds number of elements of X
%  A = Q*F
%  \| y - A*x\| = \|Q'*y - F*x\|
compression = sz(1) > sz(2);
if compression
    y0 = y;normy2 = norm(y0)^2;
    delta0 = delta;

    % Find subspace of A
    eig_compression = true;
    if eig_compression
        % this method can be cheaper since no need compute the basis of A
        Q = A'*A;
        [Vq,Sq] = eig(Q);
        Sq = max(diag(Sq),0);
        Sq = sqrt(Sq);
        %Uq = A*Vq'/Sq;
        
        if min(Sq) == 0
           warning('A should be a full column rank matrix.\nUse "linreg_lrmx" to solve \min |X|_* + mu * |X|_F^2   subject to  | Y - A(X)|_F^2  < delta')
           iss = find(Sq>1e-12);
           y = (Vq(:,iss)'*(A'*y))./Sq(iss); % compressed y
           U = diag(Sq(iss))*Vq(:,iss)';
        else
            y = (Vq'*(A'*y))./(Sq);
            U = diag(Sq)*Vq';
        end
    else
        % QR-based compression of A
        [F,U] = qr(A,0); 
        y = F'*y0;
    end
    gamma = normy2-norm(y)^2;
    delta2 = delta^2;
    % gamma < delta^2 < norm(y)
    if (delta2 < gamma) 
        error('delta is now smaler than norm(y - A*A\y)\n. It must be in the range [norm(y - A*A\y), norm(y)]')
    end
    if (delta > normy2)
        error('delta is greater than norm(y)\n. It must be in the range [norm(y - A*A\y), norm(y)]')
    end
    
    delta2 = delta2 - gamma;
    delta = sqrt(delta2);
else
    U = A;
end


if (size(U,1) < size(U,2)) || strcmp(method,'cvx')
    % non square matrix case
    cvx_begin
        cvx_precision best
        variable X(sz(2),1)
        minimize   sum_square(X)
        subject to
            norm(U*(X) - y ) <= delta
    cvx_end
    

%     Aop = linop_matrix(U,'R2R');
%     tf_opts = [];
%     tf_opts.alg    = 'AT'; % gradien descend 'AT'  'LLM'; 'GRA' N83
%     tf_opts.maxIts = 5000;
%     tf_opts.continuation = 0;
%     [X,out] = tfocs_SCD( [], { Aop, -y }, prox_l2(delta), 1, [], [], tf_opts);
    
else
    % closed form 
    B = inv(U);
    b = B*y; % == A\y
    % x = pinv(U)*(y2+delta2*r) = b + delta2* inv(U)*r , r^T * r = 1
    % min |b + delta2 * B*r|^2 s.t. r^T*r = 1
    %
    % min  r'*(B'*B)*r + 2/delta2*b'*B*r   s.t. r^T*r = 1
    Q = B'*B;
    [r,fval,lda] = squadprog(Q,B'*b/delta);
    
    X = b + delta*B*r;
end
X = X(:,1);
cone_d = norm(y - U*X);
if compression
    cone_d = sqrt(cone_d^2 + gamma);
end
 
%%




function op = prox_l2( q )

%PROX_L2    L2 norm.
%    OP = PROX_L2( q ) implements the nonsmooth function
%        OP(X) = q * norm(X,'fro').
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar.
%    If Q is a vector or matrix of the same size and dimensions as X,
%       then this uses an experimental code to compute the proximity operator
%       of OP(x) = norm( q.*X, 'fro' )
%    In the limit q --> 0, this function acts like prox_0 (aka proj_Rn)
% Dual: proj_l2.m
% For the proximity operator of the l2 squared norm (that is, norm(X,'fro')^2)
%   use smooth_quad.m (which can be used in either a smooth gradient-based fashion
%   but also supports proximity usage). Note smooth_quad() is self-dual.
% See also proj_l2, prox_0, proj_Rn

% Feb '11, allowing for q to be a vector
%       This is complicated, so not for certain
%       A safer method is to use a linear operator to scale the variables

if nargin == 0,
	q = 1;
% elseif ~isnumeric( q ) || ~isreal( q ) || numel( q ) ~= 1 || q <= 0,
elseif ~isnumeric( q ) || ~isreal( q ) %||  any( q < 0 ) || all(q==0)
	error( 'Argument must be positive.' );
end
if isscalar(q)
    if any( q <= 0 )
        error('Scaling argument must be positive, real scalar. If q=0, use prox_0 instead');
    end
    op = @(varargin)prox_l2_q( q, varargin{:} );
else
    if all(q==0), error('Argument must be nonzero'); end
    warning('TFOCS:experimental','Using experimental feature of TFOCS');
    op = @(varargin)prox_l2_vector( q, varargin{:} );
end

function [ v, x ] = prox_l2_q( q, x, t )
if nargin < 2,
	error( 'Not enough arguments.' );
end
v = sqrt( tfocs_normsq( x ) ); 
if nargin == 3,
	s = 1 - 1 ./ max( v / ( t * q ), 1 ); 
   
	x = x * s;
	v = v * s; % version A
elseif nargout == 2,
	error( 'This function is not differentiable.' );
end
v = q * v; % version A


% --------- experimental code -----------------------
function [ v, x ] = prox_l2_vector( q, x, t )
if nargin < 2,
	error( 'Not enough arguments.' );
end
v = sqrt( tfocs_normsq( q.*x ) ); % version B
if nargin == 3,
%{
   we need to solve for a scalar variable s = ||q.*x|| 
      (where x is the unknown solution)
    
   we have a fixed point equation:
        s = f(s) := norm( q.*x_k ) where x_k = x_0/( 1 + t*q/s )
   
   to solve this, we'll use Matlab's "fzero" to find the zero
    of the function F(s) = f(s) - s
    
   Clearly, we need s >= 0, since it is ||q.*x||
    
   If q is a scalar, we can solve explicitly: s = q*(norm(x0) - t)
    
%}
    
    xk = @(s) x./( 1 + t*(q.^2)/s );
    f = @(s) norm( q.*xk(s) );
%     F = @(s) f(s) - s;
    tq2 = t*(q.^2);
    F = @(s) norm( (q.*x)./( 1 + tq2/s ) ) - s;
    [s,sVal] = fzero( F, 1);
    if abs( sVal ) > 1e-4
        error('cannot find a zero'); 
    end
    if s <= 0
        x = 0*x;
    else
        x = xk(s);
    end
    v = sqrt( tfocs_normsq( q.*x ) ); % version B
elseif nargout == 2,
	error( 'This function is not differentiable.' );
end

% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
