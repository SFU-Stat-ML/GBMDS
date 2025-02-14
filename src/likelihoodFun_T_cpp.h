#ifndef LIKELIHOODFUN_T_CPP_H
#define LIKELIHOODFUN_T_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// likelihood function
Rcpp::List likelihoodFun_T_cpp(arma::mat dist_mat, double upper_bound,
                               Rcpp::List proposal_result, 
                               String metric, Rcpp::List hyperparList);

#endif