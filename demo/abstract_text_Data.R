# This example is for section 5.4 Abstract text data

library(parallel)
library(doMC)
library(fGarch)
library(GBMDS)
library(tidyverse)
library(textmineR)
library(stringdist)
library(cluster)
library(GGally)

data("abstract_text")

data <- abstract_text

# Jaccard distance between q-gram profiles
distmatrix.jaccard <- stringdistmatrix(tolower(data$text_combined), 
                                       useNames=FALSE, 
                                       method = "jaccard", q = 2)
dis.jaccard.df <- data.frame(dis = distmatrix.jaccard[upper.tri(distmatrix.jaccard)])

ggplot(dis.jaccard.df, aes(x = dis)) +
  geom_histogram(bins = 50) +
  labs(x = "Jaccard dissimilarity") +
  theme_bw()

dis <- as.matrix(distmatrix.jaccard)

## general settings
p <- 5
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

reference.x.sd <- diag(rep(0.0001, p))

hyperparList <- list(a = sim.a.initial, b = sim.b.initial,
                     alpha = sim.alpha.initial, beta = sim.beta.initial,
                     df = df.initial,
                     c = c.initial, d = d.initial, 
                     constant_multiple = constant.multiple,
                     reference_x_sd = reference.x.sd)


tuningparList <- list(K = 200, phi = 0.9, eps = 0.5)
n.core <- detectCores()-1

## run GBMDS-ASMC
model <- truncatedSkewedN(hyperparList, p, reference.x.sd)

asmc.result <- ASMC_Rcpp(model = model, dist.mat = dis,
                         tuningparList, n.core, cmds.result = cmds.result$points,
                         metric = "euclidean", upper_bound = 1, 
                         n.update = 4, n.update.x = nrow(dis))

index.asmc <- which.min(asmc.result$SSR.output)
ASMC.res <- asmc.result$xi.output[[index.asmc]]

ASMC.cluster.res <- pam(x = ASMC.res, k = 3, 
                        keep.data = FALSE)
ASMC.clustering <- ASMC.cluster.res$clustering

ASMC.res.df <- as.data.frame(ASMC.res)
ASMC.clustering.cat <- as.factor(ASMC.clustering)

ggpairs(ASMC.res.df, aes(colour = ASMC.clustering.cat, alpha = 0.4),
        axisLabels = "none",
        columnLabels = paste0("x", seq(1, p)),
        upper = list(continuous = wrap("density", alpha = 0.5), combo = "box_no_facet")) +
  theme_bw()

clustering = ASMC.clustering

# create a document term matrix
dtm <- CreateDtm(doc_vec = data$text_combined, # character vector of documents
                 doc_names = data$doi, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo
                 cpus = 2) # default is all available cpus on the system


p_words <- colSums(dtm) / sum(dtm)

cluster_words <- lapply(unique(clustering), function(x){
  rows <- dtm[ clustering == x , ]
  
  # for memory's sake, drop all words that don't appear in the cluster
  rows <- rows[ , colSums(rows) > 0 ]
  
  colSums(rows) / sum(rows) - p_words[ colnames(rows) ]
})

cluster_summary <- data.frame(cluster = unique(clustering),
                              size = as.numeric(table(clustering)),
                              top_words = sapply(cluster_words, function(d){
                                paste(names(d)[ order(d, decreasing = TRUE) ][ 1:50 ], 
                                      collapse = ", ")
                              }),
                              stringsAsFactors = FALSE)
cluster_summary

