#ifndef DPROPOSALFUN_CPP_H
#define DPROPOSALFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// dproposal function
double dproposalFun_cpp(arma::mat dist_mat,
                        Rcpp::List para_result_l, Rcpp::List para_result_r,
                        String metric, Rcpp::List hyperparList);

#endif