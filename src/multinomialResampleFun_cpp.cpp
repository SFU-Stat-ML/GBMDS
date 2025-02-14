#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]

// Obtain resampling index using multinomial resampling
Rcpp::NumericVector multinomialResampleFun_cpp(const std::vector<double>& W) {
  int N = W.size();
  //std::vector<int> index(N);
  
  // Obtaining namespace of Matrix package
  // Environment pkg = Environment::namespace_env("base");
  
  // Picking up Matrix() function from Matrix package
  // Function f = pkg["sample.int"];
  
  // calling rnorm()
  Function f("sample.int");   
  
  return f(Named("n", N), Named("size", N), Named("replace", true), Named("prob", W));
  // return index;
}

