#ifndef REFERENCE_D_X_CPP_H
#define REFERENCE_D_X_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// [[Rcpp::export]]

// Calculate the density of x
arma::vec reference_d_x_cpp(arma::mat x, arma::mat prev_x);

#endif