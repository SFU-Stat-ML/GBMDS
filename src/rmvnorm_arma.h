#ifndef RMVNORM_ARMA_H
#define RMVNORM_ARMA_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

arma::mat rmvnorm_arma(int n, arma::vec mu, arma::mat sigma);

#endif