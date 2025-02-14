#ifndef PROPOSALFUN_CPP_H
#define PROPOSALFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// proposal function
Rcpp::List proposalFun_cpp(arma::mat dist_mat, Rcpp::List currentVal, 
                           arma::mat prevX, double annealingPar,
                           String metric, Rcpp::List hyperparList,
                           double upper_bound, 
                           arma::vec update_list, arma::vec update_list_x);

#endif