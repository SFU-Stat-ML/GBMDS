#ifndef INITIALFUN_SN_CPP_H
#define INITIALFUN_SN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Initialize parameters
Rcpp::List initialFun_SN_cpp(arma::mat cmds_result, arma::mat dist_mat, 
                             String metric, Rcpp::List hyperparList);

#endif