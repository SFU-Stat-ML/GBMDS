# This example is for section 5.1 NIPS text data with incremental dimensions

library(parallel)
library(doMC)
library(fGarch)
library(GBMDS)


data("NIPS_data")

data <- NIPS_data

## define dissimilarity metric
dist.metric <-"cosine"

## set incremental size
n.t1 <- 10      # correspond to n0
incr.step <- 5  # correspond to n1

# calculate Cosine distances for text
dis <- 1-philentropy::distance(t(data), method = "cosine", mute.message = TRUE)

## select initial dis and incremental dis
dis.t1 <- dis[1:n.t1, 1:n.t1]
dis.t2 <- dis[1:(n.t1+incr.step), 1:(n.t1+incr.step)]

## MDS with annealed SMC
p = 2 # dims in lower space

# use cmds to get initial values
t1.cmds.res <- cmdscale(d = dis.t1, k = p,
                        eig = FALSE, add = FALSE, x.ret = FALSE)

t2.cmds.res <- cmdscale(d = dis.t2, k = p,
                        eig = FALSE, add = FALSE, x.ret = FALSE)

## set hyperparameters
constant.multiple <- 2.38^2

n.t1 <- nrow(dis.t1)

sim.a.initial.t1 <- 5
SSR.initial.t1 <- SSRFun(d.mat = dis.t1, delta.mat = as.matrix(dist(t1.cmds.res)))
sim.m.t1 <- n.t1 * (n.t1 - 1)/2
sim.b.initial.t1 <- SSR.initial.t1/sim.m.t1

sim.alpha.initial.t1 <- 1/2
sample.cov.t1 <- cov(t1.cmds.res)
sim.beta.initial.t1 <- (1/2)*diag(sample.cov.t1)

df.initial.t1 <- 5
c.initial.t1 <- -2
d.initial.t1 <- 2

reference.x.sd <- diag(rep(0.01, p))

hyperpars.t1 <- list(a = sim.a.initial.t1, b = sim.b.initial.t1,
                     alpha = sim.alpha.initial.t1, beta = sim.beta.initial.t1, df = df.initial.t1,
                     c = c.initial.t1, d = d.initial.t1, constant_multiple = constant.multiple,
                     reference_x_sd = reference.x.sd)

n.t2 <- nrow(dis.t2)

sim.a.initial.t2 <- 5
SSR.initial.t2 <- SSRFun(d.mat = dis.t2, delta.mat = as.matrix(dist(t2.cmds.res)))
sim.m.t2 <- n.t2 * (n.t2 - 1)/2
sim.b.initial.t2 <- SSR.initial.t2/sim.m.t2

sim.alpha.initial.t2 <- 1/2
sample.cov.t2 <- cov(t2.cmds.res)
sim.beta.initial.t2 <- (1/2)*diag(sample.cov.t2)

df.initial.t2 <- 5
c.initial.t2 <- -2
d.initial.t2 <- 2

hyperpars.t2 <- list(a = sim.a.initial.t2, b = sim.b.initial.t2,
                     alpha = sim.alpha.initial.t2, beta = sim.beta.initial.t2, df = df.initial.t2,
                     c = c.initial.t2, d = d.initial.t2, constant_multiple = constant.multiple,
                     reference_x_sd = reference.x.sd)

tuningparList1 <- list(K = 100, phi = 0.90, eps = 0.5)
n.core <- detectCores() - 1

## set 1 (original part) only
model <- truncatedT(hyperparList = hyperpars.t1, p, reference.x.sd)

n <- nrow(dis.t1)

# Use integration of R and C++ to boost computing performance
t1.asmc.result <- ASMC_Rcpp(model = model,
                            dist.mat = dis.t1,
                            tuningparList = tuningparList1, 
                            n.core, cmds.result = t1.cmds.res,
                            metric = dist.metric,
                            upper_bound = 1, n.update = 4, 
                            n.update.x = nrow(dis.t1))
# # Use original R code
# t1.asmc.result <- ASMC(model = model,
#                        dist.mat = dis.t1,
#                        tuningparList = tuningparList1, n.core,
#                        cmds.result = t1.cmds.res,
#                        metric = dist.metric)


# posterior inference
index.asmc.t1 <- which.min(t1.asmc.result$SSR.output)
t1.asmc.res <- t1.asmc.result$xi.output[[index.asmc.t1]]

stressFun(d.mat = dis.t1,
          delta.mat = 1-philentropy::distance(t1.asmc.res,
                                              method = dist.metric, mute.message = TRUE))


# store results
t1.asmc.res.all <- list(xi = t1.asmc.res, sigma2 = t1.asmc.result$sigma2.output[[index.asmc.t1]],
                        lambda = t1.asmc.result$lambda.output[[index.asmc.t1]],
                        psi = t1.asmc.result$psi.output[[index.asmc.t1]],
                        g = t1.asmc.result$g.output[[index.asmc.t1]])

## set 2 (original + incremental part) only
n <- nrow(dis.t2)

# Use integration of R and C++ to boost computing performance
t2.asmc.result <- ASMC_Rcpp(model = model,
                            dist.mat = dis.t2,
                            tuningparList = tuningparList1, 
                            n.core, cmds.result = t2.cmds.res,
                            metric = dist.metric,
                            upper_bound = 1, n.update = 4, 
                            n.update.x = nrow(dis.t2))
# # Use original R code
# t2.asmc.result <- ASMC(model = model,
#                        dist.mat = dis.t2,
#                        tuningparList = tuningparList1, n.core,
#                        cmds.result = t2.cmds.res,
#                        metric = dist.metric)

# posterior inference
index.asmc.t2 <- which.min(t2.asmc.result$SSR.output)
t2.asmc.res <- t2.asmc.result$xi.output[[index.asmc.t2]]

stressFun(d.mat = dis.t2,
          delta.mat = 1-philentropy::distance(t2.asmc.res,
                                              method = dist.metric, mute.message = TRUE))
## set 1 and 2 combine
hyperpars.t12 <- list(a = sim.a.initial.t2, b = sim.b.initial.t2,
                      alpha = sim.alpha.initial.t2, beta = sim.beta.initial.t2, df = df.initial.t2,
                      c = c.initial.t2, d = d.initial.t2, constant_multiple = constant.multiple,
                      reference_x_sd = cov(t1.asmc.res.all$xi))

model <- truncatedT_incr(hyperparList = hyperpars.t2, p,
                         reference.x.sd = cov(t1.asmc.res.all$xi))

n.incr <- nrow(dis.t2) - nrow(dis.t1)
tuningparList2 <- list(K = 100, 
                       phi = 0.90, eps = 0.5)

# Use integration of R and C++ to boost computing performance
t12.asmc.result <- ASMC_incr_Rcpp(model = model,
                                  dist.mat = dis.t2,
                                  tuningparList =  tuningparList2, n.core, 
                                  prev.result = t1.asmc.res.all$xi,
                                  metric = dist.metric,
                                  upper_bound = 1, n.update = 4, 
                                  n.update.x = nrow(dis.t2))
# # Use original R code
# t12.asmc.result <- ASMC_incr(model = model,
#                              dist.mat = dis.t2,
#                              tuningparList =  tuningparList2, n.core,
#                              prev.result = t1.asmc.res.all,
#                              metric = dist.metric)

# posterior inference
index.asmc.t12 <- which.min(t12.asmc.result$SSR.output)
t12.asmc.res <- t12.asmc.result$xi.output[[index.asmc.t12]]

stressFun(d.mat = dis.t2,
          delta.mat = 1-philentropy::distance(t12.asmc.res,
                                              method = dist.metric, 
                                              mute.message = TRUE))


