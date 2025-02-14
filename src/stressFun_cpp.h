#ifndef STRESSFUN_CPP_H
#define STRESSFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Calculate the stress value

double stressFun_cpp(arma::mat d_mat, arma::mat delta_mat);

#endif

