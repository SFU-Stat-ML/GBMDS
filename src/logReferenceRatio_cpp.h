#ifndef LOGREFERENCERATIO_CPP_H
#define LOGREFERENCERATIO_CPP_H

#include <RcppArmadillo.h>
#include "reference_d_x_cpp.h"

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// Calculate the reference ratio in log scale

double logReferenceRatio_cpp(arma::mat new_x, arma::mat current_x,
                             arma::mat prev_x);

#endif
