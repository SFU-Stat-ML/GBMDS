#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::NumericVector dinvgamma_cpp(Rcpp::NumericVector x,
                              double shape,
                              double rate,
                              bool log) {
  Rcpp::NumericVector log_f = Rcpp::dgamma(1.0/x, shape, rate, true) - 2 * Rcpp::log(x);
  if (log) {
    return log_f;
  } else{
    return Rcpp::exp(log_f);
  }
}
