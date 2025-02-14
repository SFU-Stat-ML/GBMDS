#ifndef DISTRCPP_H
#define DISTRCPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

Rcpp::NumericMatrix distRcpp(Rcpp::NumericMatrix X, String metric);
  
#endif