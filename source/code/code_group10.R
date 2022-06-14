
## Group 10

## Names ~~ SNR ~~ ANR 
## Julia van Bon ~~ 2014511 ~~ 872721
## Ernani Hazbolatow ~~ 2023708 ~~ 304318
## Andrey Peshev ~~ 2023638 ~~ 138297
## Sarah Via ~~ 2025640 ~~ 135598

###########################################
## Unsupervised learning Assignment: PCA ##
###########################################



## Clearing variables
rm(list=ls())

## Installing libraries 
install.packages('corrplot')

## Loading libraries 
library(foreign)
library(car)
library(psych)
library(GPArotation)
library(corrplot)
library(clValid)
library(factoextra)
library(dplyr)
library(magrittr)

## Loading the data 
dataDir <- '../data/'
fileName <- "PCAdata.sav"


PCAdata <- read.spss(paste0(dataDir, fileName), to.data.frame=TRUE)


################################################################
#~~~~~~~~~~~~~~~Testing the linearity of the data ~~~~~~~~~~~~~#
################################################################

##Creating a matrix plot to
PCAdata_cor <- cor(PCAdata)
corrplot(PCAdata_cor, method = "ellipse")

## Bartlett test for sphericity 
cortest.bartlett(PCAdata, n=nrow(PCAdata))

## KMO test 
KMO(PCAdata)


###########################################
#~~~~~~~~~~~~~~~~PCA Tasks~~~~~~~~~~~~~~~~#
###########################################

## First PCA with no rotation. 
pca_1 <- princomp(PCAdata, cor=TRUE)
pca_1

## Summary of PCA with no rotation. 
summary(pca_1)

## Loadings for the PCA with no rotation. 
loadings_pca_1 <- loadings(pca_1)
loadings_pca_1

## Calculating the eigenvalues 
eigenvalues <- eigen(cor(PCAdata))$values

##Plotting the eigenvalues on a scree plot 
plot(eigenvalues, 
     type = 'b')

##Calculating the component scores for each item 
pca_1$scores


## Based on the scree plot, we will do the rotations with 5 components. 

## Running a PCA  with the VARIMAX rotation  
pca_varimax <- principal(PCAdata, 
                         nfactors = 5,
                         rotate = 'varimax')
pca_varimax

## Loadings for the VARIMAX rotation. 
loadings(pca_varimax)

## Running a PCA the OBLIMIN rotation  
pca_oblimin <- principal(PCAdata, 
                         nfactors = 5,
                         rotate = 'oblimin')

pca_oblimin

## Loadings for the OBLIMIN rotation.  
loadings(pca_oblimin)

##################################################
## Unsupervised learning Assignment: Clustering ##
##################################################

## Loading in data for clustering 
dataDir1 <- "../data/"
fileName1 <- "clusterdata.sav"

data <- read.spss(paste0(dataDir1, fileName1), to.data.frame=TRUE)

## Standardize data and add row names
data_standardized <- scale(data[-1])

## In order for clValid to run, we respecify the dataframe. Not doing this causes clValid to break (???).
data_standardized <- data.frame(data_standardized)

## This prevents the row.name warning from occuring
row.names(data_standardized) <- data[,1]

######################################################
#~~~~~~~~~~~~~~~ K-means clustering ~~~~~~~~~~~~~~~~~#
######################################################
## Clustering with K-means
out_kmeans <- kmeans(data_standardized, 2, nstart = 20)

out_kmeans$cluster

## Plot Clusters
plot(data_standardized, col=out_kmeans$cluster, pch=20, cex=2)

## Define a function, wssplot that plots the within sum of squares for n = 2:6
wssplot <- function(data, nc=15, seed=200420){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}

## Run function wssplot
wssplot(data_standardized, n=6)

######################################################
#~~~~~~~~~~~~~~~~~ PAM clustering ~~~~~~~~~~~~~~~~~~~#
######################################################
## Clustering with PAM with 2, 3, 4 clusters
out_pam <- pam(data_standardized, 2)
out_pam2 <- pam(data_standardized, 3)
out_pam3 <- pam(data_standardized, 4)

## Inspect medoids
out_pam$medoids
out_pam2$medoids
out_pam3$medoids

## Plot medoids
fviz_cluster(out_pam)
fviz_cluster(out_pam2)
fviz_cluster(out_pam3)

######################################################
##~~~~~~~~~~~~ Hierarchical clustering ~~~~~~~~~~~~~~#
######################################################
## Run HC 3 times for complete, average, and single method
hc_complete <- hclust(dist(data_standardized), method="complete")
hc_average <- hclust(dist(data_standardized), method="average")
hc_single <- hclust(dist(data_standardized), method="single")

## Plot HC results
par(mfrow=c(1,3))
plot(hc_complete,main="Complete Linkage", xlab="", sub="", cex=.9)
plot(hc_average,main="Average Linkage", xlab="", sub="", cex=.9)
plot(hc_single,main="Single Linkage", xlab="", sub="", cex=.9)
par(mfrow=c(1,1))


######################################################
##~~~~~~~~~~~ Comparison of clustering ~~~~~~~~~~~~~~#
######################################################
## Compute stability and internal measures for our three clustering approaches 
intern <- clValid(obj = data_standardized, nClust = 2:6, 
                  clMethods = c("hierarchical", "kmeans", "pam"), 
                  validation = "internal")

stab <-  clValid(obj = data_standardized, nClust = 2:6, 
                 clMethods = c("hierarchical", "kmeans", "pam"), 
                 validation = "stability")

## Take a look at the ouput
intern
stab

## Optimal scores for internal and stability measures
optimalScores(intern)
optimalScores(stab)

######################################################
##~~~~~~~~~~~~~ Cluster investigation ~~~~~~~~~~~~~~~#
######################################################
# Define clusters based on hierarachical plot
index_cluster1 <- c(1, 2, 3, 4, 8, 10)
index_cluster2 <- c(5, 6, 7, 9, 11, 12)

# Split dataframe into clusters
cluster1_dog <- data_standardized[index_cluster1, ]
cluster2_dog <- data_standardized[index_cluster2, ]
cluster1_dog
cluster2_dog

## Find mean for each column
means_cluster1 <- colMeans(cluster1_dog)
means_cluster2 <- colMeans(cluster2_dog)
means_cluster1
means_cluster2
