#include <RcppArmadillo.h>
#include "dmvnrm_arma_fast.h"
#include "reference_d_x_cpp.h"
#include "rmvnorm_arma.h"
#include "distRcpp.h"
#include "SSRFun_cpp.h"
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
// [[Rcpp::export]]

// Initialize parameters
Rcpp::List initialFun_T_cpp(arma::mat cmds_result, arma::mat dist_mat, 
                            String metric, Rcpp::List hyperparList){
  
  int n_obj = dist_mat.n_rows;
  int p = cmds_result.n_cols;
  int m = n_obj * (n_obj - 1) / 2;
  arma::mat reference_x_sd = hyperparList["reference_x_sd"];
  
  // initialize particles
  // x_i (from reference distribution)
  arma::mat x_initial(n_obj, p, arma::fill::zeros);
  for (int i = 0; i < cmds_result.n_rows; i++){
    arma::rowvec temp0 = cmds_result.row(i);
    NumericVector temp1 = NumericVector(temp0.begin(), temp0.end());
    x_initial.row(i) = rmvnorm_arma(1, temp1, reference_x_sd);
    //x_initial.row(i).print();
  }
  
  // calculate delta matrix and d matrix
  arma::mat d_mat = dist_mat;
  Rcpp::NumericMatrix delta_mat_rcpp = distRcpp(wrap(x_initial), metric);
  arma::mat delta_mat = Rcpp::as<Mat<double>>(delta_mat_rcpp);
  // Rcout << "delta_mat " << delta_mat;
  
  // calculate SSR initial
  // double SSR_initial = SSRFun_cpp(d_mat, delta_mat);
  // Rcout << "SSR_initial" << SSR_initial;

  // extract hyperparameters
  double a = hyperparList["a"];
  double b = hyperparList["b"];
  double alpha = hyperparList["alpha"];
  arma::vec beta = hyperparList["beta"];
  double df_nu = hyperparList["df"];
  
  // sigma2 (from prior distribution)
  double sigma2_initial = 1/arma::randg(distr_param(a,1/b));
  // Rcout << "sigma2_initial" << sigma2_initial;
  
  // lambda (from prior distribution)
  arma::vec lambda_initial_diag(p);
  // arma::mat lambda_initial(p, p, arma::fill::zeros);
  for (int i = 0; i < p; i++){
    lambda_initial_diag(i) = 1/arma::randg(distr_param(alpha,1/(beta[i])));
  }
  arma::mat lambda_initial = diagmat(lambda_initial_diag);
  
  // g (from prior distribution)
  arma::mat g_initial(n_obj, n_obj, arma::fill::zeros);
  arma::uvec ug_idx = arma::trimatu_ind(arma::size(g_initial), 1);
  NumericVector temp_u = rgamma(m, df_nu, 1/df_nu);
  //Rcout << "temp_u" << temp_u;
  arma::vec temp_uu = as<arma::vec>(wrap(temp_u));
  g_initial.elem(ug_idx) = temp_uu;
  //g_initial.print();
  
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("x")=x_initial,
                                         Rcpp::Named("sigma2")=sigma2_initial,
                                         Rcpp::Named("lambda")=lambda_initial,
                                         Rcpp::Named("g")=g_initial);
  
  return(output);
}


/*** R

# initialFun_T_cpp(cmds_result, dist_mat, metric, hyperparList)


*/
