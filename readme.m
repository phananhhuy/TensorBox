%% TENSORBOX 
% TENSORBOX is a Matlab package containing state-of-the-art algorithms
% for decompositions of multiway array data into rank-1 tensors such as
%   - CANDECOMP/PARAFAC
%   - Tucker decomposition
%   - Generalized Kronecker tensor decomposition
%   - Tensor deconvolution
%   - Tensor train decomposition
%   - Best rank-1 tensor approximation
%   - Tensor decomposition with given error bound (or the denoising
%   problem)
%   - Application for BSS, harmonic retrieval, image denoising, image
%   completion, dictionary learning, ... 
% 
%% Algorithms for CANDECOMP/PARAFAC decomposition (CPD)
%
% FastALS :   fast ALS algorithm employs the fast method to compute CP
%             gradients. 
%             Other alternating and all-at-once algorithms for CPD can be
%             accelerated using the similar method. 
%             See HALS, QALS, MLS, fLM and subfunction of FastALS:
%             cp_gradient. 
%
% HALS        hierarchical ALS algorithm for nonnegative CPD (NCPD).
% QALS        recursive ALS algorithm for nonnegative CPD (NCPD).
% MLS         multiplicative algorithm for NCPD.
% fLM         fast damped Gauss-Newton or Levenberg-Marquardt algorithm for
%             CPD and NCPD
%             (see cp_fLMa and cpx_fLMa, ncp_fLM)  
% FCP         fast algorithm for higher order CPD through tensor unfolding.
% XGRAM       extended generalized rank annihilation method (GRAM) and
%             direct trilinear decomposition (DLTD) to higher order CPD.
%             
% CPO-ALS1    ALS algorithm for CPD with column-wise orthogonal factor.
% CPO-ALS2    ALS algorithm for CPD with column-wise orthogonal factor.
%
% CRIB       Cramer-Rao Induced Bound for CPD.

%% Algorithms for Tensor Deflation and Rank-1 tensor extraction 
% ASU         Alternating Subspace update. The algorithm extracts a rank-1
%             tensor from a rank-R tensor, i.e., deflation. It can be used
%             to sequentially decompose a rank-R tensor over R rank-1
%             tensor extraction.
%
% CRB for the tensor deflation
%
%% Algorithms for Tucker decomposition (TD)
% HALS       hierarchical ALS algorithm for nonnegative TD.
% LM         Levenberg-Marquardt algorithm with log-barrier function.
% LMPU       A simplified version of the LM algorithm which updates only
%            one factor matrix and a core tensor at a time.
% O2LB       A simplified version of the LM algorithm which updates only
%            one factor matrix or a core tensor at a time.
% CrNc       Crank-Nicholson algorithm for orthogonal Tucker decomposition.
%
%% Algorithms for Error Preserving Correction method
% which solves the optimization problem
%       min   sum_r  |norm_of_rank-1_tensor_r-th|_F^2
%       s.t.  |Y - X|_F^2 <= error_bound
%        
% ANC       alternating correction algorithm
% SQP       sequential QP algorithm
% ITP       Iterior point algorithm
%
%% Algorithms for CPD with bound constraint
% which solves the optimization problem
%       min   |Y - X|_F^2
%       s.t.  sum_r  |norm_of_rank-1_tensor_r-th|_F^2 <= error_bound
%        
% ANC       alternating correction algorithm
% SQP       sequential QP algorithm
% ITP       Iterior point algorithm
%
%% Algorithms for best rank-1 tensor approximation
%  Closed form expression to find best rank-1 for tensor 2x2x2
%  Closed form expression to find best rank-1 for tensor 2x2x2x2
%  LM       iterative algorithm with optimal damping parameter
%
%  RORO     rotational rank-1 tensor approximation
%  
%% Algorithms for rank-(L o M) block term decompositions
%  bcdLoR_als
%
%% Algorithms for tensor denoising 
%  i.e.   min rank(C)   s.t.  |Y - X|_F^2 <= error_bound
%  
%% Algorithms for Tensor train decomposition
% ASCU     alternating single core update
% ADCU     alternating double core update
% A3CU     alternating trible core update
%
%% Algorithms for Tensor to CPD conversion
% Exact conversion between CPS and TT tensors
% Iterative algorithms for fiting a CP tensor to a TT-tensor
%
%% Algorithms for BSS based on Tensor network decomposition
% Exact conversion between CPS and TT tensors
% Iterative algorithms for fiting a CP tensor to a TT-tensor
%
%% Algorithms for Rank-1 Tensor Deconvolution

%% Algorithms for generalized Kronecker Tensor decomposition
%  with low-rank constraint and sparsity constraints
%  
% Examples for image denoising and completion
%
%% Algorithm for Quadratic Programming over sphere (QPS)
%
%
%% 1- TENSORBOX and the Matlab Tensor toolbox 
% 
% TENSORBOX requires the Matlab Tensor toolbox (at least ver 2.4.), and
% can decompose multiway data in different data type and classes defined in
% the Matlab Tensor toolbox including tensor, k-tensor (tensor in Kruskal
% form), t-tensor (Tucker form), sparse tensor.
%
% Algorithms in TENSORBOX can decompose real-valued and complex-valued
% tensors. Because the Matlab Tensor toolbox is developed primarily for
% real-valued tensors, there will be errors when decompositing
% complex-valued tensors. The errors are mostly due to computing 
% innerproduct between two tensors. 
% 
% To this end, we modified some routines of the Matlab Tensor toolbox ver
% 2.4, and provided them in TENSORBOX. It may happen that these
% corrections may not work properly with the newer version of the Matlab
% Tensor toolbox.
%
% Moreover, these additional files might be removed in future version if
% the authors of the Matlab Tensor toolbox do not allow. In this case, we
% will provide instructions how to fix the errors.
%
%
%% 2- How to use algorithms in TENSORBOX
% 
% Algorithms in TENSORBOX are very similar in usage. Basically, an
% algorithm can be called as follows 
%
%       [P,output] = algorithm(Y, R, opts);
% 
% Y:    an N-way tensor, K-tensor or T-tensor. If data Y is a double array,
%       it needs to be converted to a tensor such as  Y = tensor(Y);
% 
% R :   number of components of factor matrices
%
% opts: optional parameters with default values given by calling the
% function without any input, e.g.,
%
%       opts = cp_fastals;
% 
% opts =  
%                   init: 'random'     
%             linesearch: 1
%               maxiters: 200            
%                  normX: []
%               printitn: 0
%                    tol: 1.0000e-06
%               TraceFit: 1
%              TraceMSAE: 0
%     verify_convergence: 1
%     
%
% Output of the decomposition is returned to an Kruskal tensor for CPD or 
% a t-tensor for Tucker decomposition. See documents of the Matlab tensor
% toolbox for k-tensor and t-tensor objects.
%
% The second output contains other results such as Fit, number of
% iterations.
%
% For CP decomposition, see a simple demo in "demo_CPD.m" and others 
% 
% -demo_CPD  : a simple example shows decomposition of a 3-way Amino acids
%              fluorescence data using the FastALS algorithm for CPD. 
% -demo_CPD_2: this example compares FastALS, fLM for CPD, HALS and mLS for
%              nonnegative CPD (NCPD).
% -demo_CPD_3: compares intialization methods using in CP algorithms.
% -demo_CPD_4: CPD of a complex-valued tensor using the FastALS, fLM and
%              FCP algorithms. 
% -demo_CPD_5: CPD of a complex-valued tensor with one column-wise
%              orthogonal factor matrix using the CPO-ALS1, CPO-ALS2 and 
%              FCP algorithms. 
% -demo_fcp:   a simple example shows how to use the FCP algorithm to
%              decompose an order-5 tensor.
% -demo_fcp_2: illustrates the FCP algorithm for decomposition of a higher
%              order tensor using "good" and "bad" tensor unfoldings. 
% -demo_fcp_3: similar to demo_fcp_2. This example will show how the
%              rank-one FCP (R1FCP) is sentive to unfolding rules. With a
%              ``bad'' unfolding, R1FCP may completely fail to estimate
%              latent components from the unfolded tensor. 
%              The example confirms (low-rank) FCP much better than R1FCP.
% -demo_fcp_ortho: illustrates CPD with one column-wise orthogonal factor
%              matrix using the FCP algorithm.
%
%              The example compares the CPO-ALS2 algorithm and the FCP
%              algorithm.
%
% -demo_ntd_1:  compares algorithms for NTD.
% 
% -demo_ntd_2:  compares initialization methods for NTD.
%
% Some other demos 
% -demo_tkd    demo for the CrNc algorithm
% -demo_rank1_extraction   example for rank-1 tensor extraction.
%              
%% 3- How to initialize a decomposition
% 
% The following guidelines are provided for CPD with/without constraints.
%  
% TENSORBOX implements several simple and efficient methods to initialize
% factor matrices.
%   - Random initialization which is always the simplest and fastest, but
%    not the most efficient method.
%
%   - SVD-based initialization which initializes factor matrices by
%    leading singular vectors of mode-n unfoldings of the tensor. The
%    method is similar to higher order SVD or multilinear SVD.
%    This method is often better than using random values in term of
%    convergence speed.
%    However, this method is less efficient when factor matrices comprise
%    highly collinear components. 
%   
%   - Direct trilinear decomposition (DTLD) and its extended version for
%   higher CPD using the FCP algorithm are recommended for most CPD. This
%   initialization may consume time due to tensor compression, but its
%   initial values are always much better than those using other methods.
%     
%   - Initialization based on selected fibers is another suggested method.
%   The method employs the Nystrom or CUR decomposition.
%   
%   - Multi-initialization with some small number of iterations are often
%   performed. The component matrices with the lowest approximation
%   error are selected.
%
% See comparison of initialization methods in "demo_CPD_3.m".
% 
% Factor matrices can be initialized outside algorithms using the routine
% "cp_init" 
%   
%   opts = cp_init;  
%   opts.init = 'dtld';
%   Uinit = cp_init(X,R,opts);
%
% then passed into the algorithm through the optional parameter "init". For
% example, 
%
%   opts = cp_fastals;
%   opts.init = {Uinit}; % or can directly set  opts.init = 'dtld';
%   P = cp_fastals(X,R,opts);
%
% For the fLM algorithm, ALS with small runs can be employed before the main
% algorithm by setting
%   opts.alsinit = 1;
%
% For nonnegative CPD, one needs to set
%    opts.nonnegative = true
%
% For Nonnegative Tucker decomposition, one can use NMFs to initialize
% factor matrices. See Details in ntd_init.
%
%% 4- How to efficiently decompose large-scale tensor
%
% There are many efficient methods to deal with relatively large-scale
% tensor. Compression and block tensor decomposition are some of them. In
% TENSORBOX, for higher order CPD, the FCP algorithm is recommended.
% Instead of directly decomposing a relatively large-scale higher order
% tensor, FCP decomposes a compressed tensor unfolded from the original
% data in a lower order, e.g., order 3.
%
% For example, in order to decompose an order-5 tensor of size 50 x 50 x 50
% x 50 x 50 into 50 rank-one tensor, using FCP with an unfolding
% [1,2,(3,4,5)], one only needs to decompose an order-3 tensor of size 50 x
% 50 x 50.
%
% The performance of FCP may depend on the unfolding rule. One can try
% several unfoldings, and choose the solutions with the best fit.
% The loss of accuracy can be significant when combining modes with low
% collinearity degree.
% 
% See examples for the FCP algorithm with/without constraints for
% real-valued and complex-valued tensors, e.g., demo_fcp, demo_fcp_3,
% demo_fcp_ortho, demo_CPD_5. 
%
%% 5- How to cite algorithms
%
% References of algorithms are listed inside Matlab routines. 
% Please cite our publications if you use the algorithms and the Matlab
% codes.
% Some publications are available online at ArXiv.org. 
% 
% 
% In addition, If you find this software useful and wish to cite it, please
% use the following citation 
%  
% Anh-Huy Phan, Petr Tichavsky and Andrzej Cichocki, "TENSORBOX: a Matlab
% package for tensor decomposition", 2013, available online at
% http://www.bsp.brain.riken.jp/~phan/tensorbox.php 
%
%% 6- What to do if error appears 
% 
%  We would be grateful to receive any feedback on the TENSORBOX and
%  algorithms.
%  Please send email to Phan Anh-Huy 
%                       a.phan@skoltech.ru
%   
%  The RIKEN email address (phan@brain.riken.jp) is no longer in use.
%
%  or to authors of algorithms.
% 
% August, 2018
%
% All rights reserved.
%
% History
% 2018/08  major update with many new algorithms for CPD, best rank-1
%          tensor approximation, Tensor train decomposition, tensor based
%          denoising method, Tensor-train to CPD conversion, Error
%          preserving correction method, Parallel rank-1 tensor update,
%
%
% 2014/12  add the CrNc algorithm for orthogonal Tucker algorithm, the ASU
%          algorithm for rank-1 tensor extraction.
%          Routine to compute the Cramer-Rao induced bound for CPD.
% 2014/02  update fLM algorithms for CPD and NCPD.
% 2013/04  fix ntd_hals for sparse tensor.
% 2013/03  Tensorbox supports complex-valued tensor. New version of FCP, FastALS, fLM, GRAM or XGRAM, CPO_ALS.
% 2012/11  first release, includes FastALS, fLM, FCP, HALS for CPD, nCPD and NTD.
% 2012/04  FastALS and fastCP gradient were uploaded online
% 2011     fLM was uploaded online.