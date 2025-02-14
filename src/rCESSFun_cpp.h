#ifndef RCESSFUN_CPP_H
#define RCESSFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Calculate relative conditional effective sample size
double rCESSFun_cpp(arma::vec W, arma::vec logL, double num, double phi);

#endif