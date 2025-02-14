#include <RcppArmadillo.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


// [[Rcpp::export]]
// Calculate the SSR value

double SSRFun_cpp(arma::mat d_mat, arma::mat delta_mat){
  
  d_mat.diag().fill(0);
  delta_mat.diag().fill(0);
  
  arma::mat A = pow(d_mat - delta_mat, 2);
  
  double SSR = 0.5*(accu(A));
  
  return SSR;
}
