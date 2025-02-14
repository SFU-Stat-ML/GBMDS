#ifndef SSRFUN_CPP_H
#define SSRFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Calculate the SSR value

double SSRFun_cpp(arma::mat d_mat, arma::mat delta_mat);

#endif