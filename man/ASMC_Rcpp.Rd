\name{ASMC_Rcpp}
\alias{ASMC_Rcpp}
\title{run ASMC for MDS with a fixed dimension using integration of R and C++ to boost computing performance}
\usage{
ASMC_Rcpp(model, dist.mat, tuningparList, n.core, cmds.result, metric, upper_bound, n.update, n.update.x)
}
\arguments{
\item{model}{likelihood model and hyperparameters}
\item{dist.mat}{distance matrix}
\item{tuningparList}{ASMC tuning parameters}
\item{n.core}{number of cores}
\item{cmds.result}{results from the classical MDS}
\item{metric}{distance metric}
\item{upper_bound}{upper bound in modeling the dissimilarity}
\item{n.update}{number of parameters to be updated in each ASMC iteration}
\item{n.update.x}{number of observations to be updated in each ASMC iteration}
}
\value{
results of weighted particles, marginal likelihood estimates
}
\description{
Perform annealed SMC algorithm for the multidimensional scaling problem with a fixed data dimension. Various models can be implemented. Computing performance is improved by rewriting key functions in C++.
}