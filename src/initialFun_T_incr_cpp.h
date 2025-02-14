#ifndef INITIALFUN_T_INCR_CPP_H
#define INITIALFUN_T_INCR_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Initialize parameters
Rcpp::List initialFun_T_incr_cpp(arma::mat prev_result, arma::mat dist_mat, 
                                 String metric, Rcpp::List hyperparList);

#endif