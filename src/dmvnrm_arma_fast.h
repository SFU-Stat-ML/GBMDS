// #ifndef DMVNRN_ARMA_FAST_H
// #define DMVNRN_ARMA_FAST_H

#include <RcppArmadillo.h>
// #include "inplace_tri_mat_mult.h"

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat);


// [[Rcpp::export]]
arma::vec dmvnrm_arma_fast(arma::mat const &x,  
                           arma::rowvec const &mean,  
                           arma::mat const &sigma, 
                           bool const logd = false);

// #endif