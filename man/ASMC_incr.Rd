\name{ASMC_incr}
\alias{ASMC_incr}
\title{run ASMC for MDS with an incremental dimension}
\usage{
ASMC_incr(model, dist.mat, tuningparList, n.core, prev.result, metric)
}
\arguments{
\item{model}{likelihood model and hyperparameters}
\item{dist.mat}{distance matrix}
\item{tuningparList}{ASMC tuning parameters}
\item{n.core}{number of cores}
\item{prev.result}{results from the previous ASMC}
\item{metric}{distance metric}
}
\value{
results of weighted particles, marginal likelihood estimates
}
\description{
Perform annealed SMC algorithm for the multidimensional scaling problem with an incremental data dimension. Various models can be implemented.
}