#include <RcppArmadillo.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


// [[Rcpp::export]]

// Calculate the stress value

double stressFun_cpp(arma::mat d_mat, arma::mat delta_mat){
  
  d_mat.diag().fill(0);
  delta_mat.diag().fill(0);
  
  arma::mat A = pow(d_mat - delta_mat, 2);
  arma::mat B = pow(d_mat, 2);
  
  double stress = sqrt(accu(A) / accu(B));
  
  return stress;
}
