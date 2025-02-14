# This example is for section 5.3 NIH text data

library(parallel)
library(doMC)
library(fGarch)
library(GBMDSpackage)
library(tidyverse)
library(textmineR)


data("RePORTER_PRJABS_C_FY2022")
data("RePORTER_PRJ_C_FY2022")

data1 <- RePORTER_PRJABS_C_FY2022
data2 <- RePORTER_PRJ_C_FY2022

## define dissimilarity metric
dist.metric <- "euclidean"
#dist.metric <- "cosine"

data <- data2 %>%
  dplyr::filter(ORG_CITY == "SAINT LOUIS") %>%
  left_join(data1, by = "APPLICATION_ID")

# create a document term matrix
dtm <- CreateDtm(doc_vec = data$ABSTRACT_TEXT, # character vector of documents
                 doc_names = data$APPLICATION_ID, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo
                 cpus = 2) # default is all available cpus on the system

# construct the matrix of term counts to get the IDF vector
tf_mat <- TermDocFreq(dtm)

# TF-IDF and cosine similarity
tfidf <- t(dtm[ , tf_mat$term ]) * tf_mat$idf
tfidf <- t(tfidf)
csim <- tfidf / sqrt(rowSums(tfidf * tfidf))
csim <- csim %*% t(csim)
cdist <- as.dist(1 - csim)
dis <- as.matrix(cdist)
dis.df <- data.frame(dis = dis[upper.tri(dis)])


## general settings
p <- 2
n <- nrow(dis)

cmds.result <- cmdscale(d = dis, k = p,
                        eig = TRUE, add = FALSE, x.ret = FALSE)
class(cmds.result) <- append(class(cmds.result), "CMDS")


## set hyperparameters
sim.a.initial <- 5
SSR.initial <- SSRFun(d.mat = dis, delta.mat = as.matrix(dist(cmds.result$points)))
sim.m <- n * (n - 1)/2
sim.b.initial <- SSR.initial/sim.m

sim.alpha.initial <- 1/2
sample.cov <- cov(cmds.result$points)
sim.beta.initial <- (1/2)*diag(sample.cov)

df.initial <- 5
c.initial <- -2
d.initial <- 2

constant.multiple <- 2.38^2/5

reference.x.sd <- diag(rep(0.01, p))

hyperparList <- list(a = sim.a.initial, b = sim.b.initial,
                     alpha = sim.alpha.initial, beta = sim.beta.initial,
                     df = df.initial,
                     c = c.initial, d = d.initial, constant_multiple = constant.multiple,
                     reference_x_sd = reference.x.sd)


tuningparList <- list(K = 20, phi = 0.8, eps = 0.5)
n.core <- detectCores()-1

## run GBMDS-ASMC
#model <- truncatedT(hyperparList, p, reference.x.sd)
model <- truncatedSkewedN(hyperparList, p, reference.x.sd)

asmc.result <- ASMC_Rcpp(model = model, dist.mat = dis,
                         tuningparList, n.core=1, 
                         cmds.result = cmds.result$points,
                         metric = dist.metric,
                         upper_bound = 1, n.update = 4, 
                         n.update.x = nrow(dis))

# posterior inference
index.asmc <- which.min(asmc.result$SSR.output)
asmc.res <- asmc.result$xi.output[[index.asmc]]

## compute STRESS values
cmds.stress <- stressFun(d.mat = dis,
                         delta.mat = philentropy::distance(cmds.result$points, method = "euclidean"))
asmc.stress <- stressFun(d.mat = dis,
                         delta.mat = philentropy::distance(asmc.res, method = "euclidean"))

