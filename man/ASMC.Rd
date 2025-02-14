\name{ASMC}
\alias{ASMC}
\title{run ASMC for MDS with a fixed dimension}
\usage{
ASMC(model, dist.mat, tuningparList, n.core, cmds.result, metric)
}
\arguments{
\item{model}{likelihood model and hyperparameters}
\item{dist.mat}{distance matrix}
\item{tuningparList}{ASMC tuning parameters}
\item{n.core}{number of cores}
\item{cmds.result}{results from the classical MDS}
\item{metric}{distance metric}
}
\value{
results of weighted particles, marginal likelihood estimates
}
\description{
Perform annealed SMC algorithm for the multidimensional scaling problem with a fixed data dimension. Various models can be implemented.
}