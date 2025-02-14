#ifndef MULTINOMIALRESAMPLEFUN_CPP_H
#define MULTINOMIALRESAMPLEFUN_CPP_H

using namespace Rcpp;

// [[Rcpp::export]]

// Obtain resampling index using multinomial resampling
Rcpp::NumericVector multinomialResampleFun_cpp(const std::vector<double>& W);

#endif