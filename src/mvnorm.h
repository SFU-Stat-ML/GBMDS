// #ifndef CHOLESKY_TRIDIAGONAL_H
// #define CHOLESKY_TRIDIAGONAL_H
// 
// #ifndef FORWARD_ALGORITHM_H
// #define FORWARD_ALGORITHM_H
// 
// #ifndef BACKWARD_ALGORITHM_H
// #define BACKWARD_ALGORITHM_H
// 
// #ifndef RMVNORM_ARMA_STOCHVOL_H
// #define RMVNORM_ARMA_STOCHVOL_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

List cholesky_tridiagonal(const vec& omega_diag, 
                          const double& omega_offdiag);

vec forward_algorithm(const vec& chol_diag, const vec& chol_offdiag, 
                      const vec& covector);

vec backward_algorithm(const vec& chol_diag, const vec& chol_offdiag, 
                       const vec& htmp);

// [[Rcpp::export]]  

mat rmvnorm_arma_stochvol(int n, mat precision, vec location);

// #endif

