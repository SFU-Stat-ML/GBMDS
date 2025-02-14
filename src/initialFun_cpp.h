#ifndef INITIALFUN_CPP_H
#define INITIALFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Initialize parameters
Rcpp::List initialFun_cpp(arma::mat cmds_result, arma::mat dist_mat, 
                          String metric, Rcpp::List hyperparList);

#endif