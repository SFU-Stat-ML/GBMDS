#include <RcppArmadillo.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
#include "dmvnrm_arma_fast.h"

using namespace Rcpp;
// using namespace arma;

// [[Rcpp::export]]

// Calculate the density of x
arma::vec reference_d_x_cpp(arma::mat x, arma::mat prev_x){
  
  int n = x.n_rows;
  int p = x.n_cols;
 
  arma::vec v(p, arma::fill::value(0.01));
  arma::mat D_mat = diagmat(v);
  
  arma::vec d_x_vec;
  arma::vec d_x(n);

  for (int i = 0; i < n; i++){
  
    arma::mat x_mat = x.row(i);
    //arma::rowvec prev_x_vec = prev_x.row(i);
    
    d_x_vec = dmvnrm_arma_fast(x_mat, prev_x.row(i), D_mat, true);
    //d_x_vec.print("d_x_vec:");
    
    d_x[i] = as_scalar(d_x_vec);
    //d_x.print("d_x:");
  }
  
  // d_x = vectorise(d_x);
  // D_mat.print("sd:");
  
  return(d_x);
  
}

