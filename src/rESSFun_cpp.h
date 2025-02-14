#ifndef RESSFUN_CPP_H
#define RESSFUN_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Calculate relative effective sample size
double rESSFun_cpp(const std::vector<double>& logW);

#endif

