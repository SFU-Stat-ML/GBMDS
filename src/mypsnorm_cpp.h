#ifndef MYPSNORM_CPP_H
#define MYPSNORM_CPP_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

double mypsnorm_cpp(double q, double mean, double sd, double psi);

#endif