# TensorBox
TENSORBOX is a Matlab package containing state-of-the-art algorithms
for decompositions of multiway array data into rank-1 tensors such as
  - CANDECOMP/PARAFAC
  - Tucker decomposition
  - Generalized Kronecker tensor decomposition
  - Tensor deconvolution
  - Tensor train decomposition
  - Best rank-1 tensor approximation
  - Tensor decomposition with given error bound (or the denoising problem)
  - Application for BSS, harmonic retrieval, image denoising, image
  completion, dictionary learning, ... 

#  Algorithms for CANDECOMP/PARAFAC decomposition (CPD)

-FastALS :   fast ALS algorithm employs the fast method to compute CP
            gradients. 
            Other alternating and all-at-once algorithms for CPD can be
            accelerated using the similar method. 
            See HALS, QALS, MLS, fLM and subfunction of FastALS:
            cp_gradient. 

-HALS        hierarchical ALS algorithm for nonnegative CPD (NCPD).
-QALS        recursive ALS algorithm for nonnegative CPD (NCPD).
-MLS         multiplicative algorithm for NCPD.
-fLM         fast damped Gauss-Newton or Levenberg-Marquardt algorithm for
            CPD and NCPD
            (see cp_fLMa and cpx_fLMa, ncp_fLM)  
-FCP         fast algorithm for higher order CPD through tensor unfolding.
-XGRAM       extended generalized rank annihilation method (GRAM) and
            direct trilinear decomposition (DLTD) to higher order CPD.
            
-CPO-ALS1    ALS algorithm for CPD with column-wise orthogonal factor.
-CPO-ALS2    ALS algorithm for CPD with column-wise orthogonal factor.

-CRIB       Cramer-Rao Induced Bound for CPD.

#  Algorithms for Tensor Deflation and Rank-1 tensor extraction 
-ASU         Alternating Subspace update. The algorithm extracts a rank-1
            tensor from a rank-R tensor, i.e., deflation. It can be used
            to sequentially decompose a rank-R tensor over R rank-1
            tensor extraction.

-CRB for the tensor deflation

#  Algorithms for Tucker decomposition (TD)
-HALS       hierarchical ALS algorithm for nonnegative TD.
-LM         Levenberg-Marquardt algorithm with log-barrier function.
-LMPU       A simplified version of the LM algorithm which updates only
           one factor matrix and a core tensor at a time.
-O2LB       A simplified version of the LM algorithm which updates only
           one factor matrix or a core tensor at a time.
-CrNc       Crank-Nicholson algorithm for orthogonal Tucker decomposition.

#  Algorithms for Error Preserving Correction method
which solves the optimization problem
      min   sum_r  |norm_of_rank-1_tensor_r-th|_F^2
      s.t.  |Y - X|_F^2 <= error_bound
       
-ANC       alternating correction algorithm
-SQP       sequential QP algorithm
-ITP       Iterior point algorithm

#  Algorithms for CPD with bound constraint
which solves the optimization problem
      min   |Y - X|_F^2
      s.t.  sum_r  |norm_of_rank-1_tensor_r-th|_F^2 <= error_bound
       
-ANC       alternating correction algorithm
-SQP       sequential QP algorithm
-ITP       Iterior point algorithm

#  Algorithms for best rank-1 tensor approximation
 -Closed form expression to find best rank-1 for tensor 2x2x2
 -Closed form expression to find best rank-1 for tensor 2x2x2x2
 -LM       iterative algorithm with optimal damping parameter

 -RORO     rotational rank-1 tensor approximation
 
 #  Algorithms for rank-(L o M) block term decompositions
 -bcdLoR_als

#  Algorithms for tensor denoising 
 i.e.   min rank(C)   s.t.  |Y - X|_F^2 <= error_bound
 
#  Algorithms for Tensor train decomposition
-ASCU     alternating single core update
-ADCU     alternating double core update
-A3CU     alternating trible core update

#  Algorithms for Tensor to CPD conversion
-Exact conversion between CPS and TT tensors
-Iterative algorithms for fiting a CP tensor to a TT-tensor

#  Algorithms for BSS based on Tensor network decomposition
-Exact conversion between CPS and TT tensors
-Iterative algorithms for fiting a CP tensor to a TT-tensor

#  Algorithms for Rank-1 Tensor Deconvolution

#  Algorithms for generalized Kronecker Tensor decomposition
 with low-rank constraint and sparsity constraints
 
Examples for image denoising and completion

#  Algorithm for Quadratic Programming over sphere (QPS)
