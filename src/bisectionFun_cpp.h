#ifndef BISECTIONFUN_CPP_H
#define BISECTIONFUN_CPP_H

#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]
// #include "rCESSFun_cpp.h"
using namespace Rcpp;

// [[Rcpp::export]]

// Calculate the next annealing parameter
double bisectionFun_cpp(double low, double high, 
                        arma::vec W, arma::vec logL, double phi);

#endif
