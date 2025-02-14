#ifndef DINVGAMMA_CPP_H
#define DINVGAMMA_CPP_H

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Calculate the next annealing parameter
Rcpp::NumericVector dinvgamma_cpp(Rcpp::NumericVector x,
                                  double shape,
                                  double rate,
                                  bool log);

#endif