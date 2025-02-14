#ifndef PSNORM_CPP_H
#define PSNORM_CPP_H

// #include <RcppArmadillo.h>

using namespace Rcpp;
// using namespace arma;
// // [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

SEXP psnorm_cpp(double q, double mean, double sd, double psi);

#endif