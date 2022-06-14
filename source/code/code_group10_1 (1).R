####################################
## Supervised learning Assignment ##
####################################

## Group 10

## Names ~~ SNR ~~ ANR 
## Julia van Bon ~~ 2014511 ~~ 872721
## Ernani Hazbolatow ~~ 2023708 ~~ 304318
## Andrey Peshev ~~ 2023638 ~~ 138297
## Sarah Via ~~ 2025640 ~~ 135598

## Variable selection ##
## 2.1: Decision making
## 2.2: Punctuality

#######################################################
#~~~~~~~~~~~~~~~~General Data prepping~~~~~~~~~~~~~~~~#
#######################################################

## Clear prevous variables
rm(list = ls(all = TRUE))

## Load libraries
library(MLmetrics)
library(glmnet)
library(nnet)

## Loading in data
dataDir <- "../data/"
fileName1 <- "community_sample.rds"
fileName2 <- "candidates.rds"

regclassData <- readRDS(paste0(dataDir, fileName1))
forecasData <- readRDS(paste0(dataDir, fileName2))

## Partition Data
# 1. Set seed to 200420
# 2. Create a permutation
# 3. Split data into mutually exclusive and exhaustive sets for training and testing
set.seed(200420)
perm <- sample(1 : nrow(regclassData))
trainData <- regclassData[perm[1 : 800], ]
testData <- regclassData[perm[801 : nrow(regclassData)], ]

##########################################################################
#~~~~~~~~~~~~~~~Data prepping for RIDGE/LASSO in reg/class~~~~~~~~~~~~~~~#
##########################################################################

## Regression: Split data into IV/DV using vectors/matrices for both test and train data
# 1. Make Y into a vector, X into a matrix
# 2. Store both in a list 

trainData_regRL <- list(y = trainData$Decision.making,
                     X = model.matrix(Decision.making ~., trainData)[ , -1]
)

testData_regRL <- list(y = testData$Decision.making,
                    X = model.matrix(Decision.making ~., testData)[ , -1]
)

## Classification: Split data into IV/DV using vectors/matrices for both test and train data
# 1. Make Y into a vector, X into a matrix
# 2. Store both in a list 

trainData_classRL <- list(y = trainData$Punctuality,
                 X = model.matrix(Punctuality ~., trainData)[ , -1]
)

testData_classRL <- list(y = testData$Punctuality,
                X = model.matrix(Punctuality ~., testData)[ , -1]
)

###############################################################
#~~~~~~~~~~~~~~~~Data prepping for Forecasting~~~~~~~~~~~~~~~~#
###############################################################

## Split data into IV/DV using vectors/matrices for test data
# 1. Make Y into a vector, X into a matrix
# 2. Store both in a list 
testData_forecastDM <- list(y = forecasData$Decision.making,
                          X = model.matrix(Decision.making ~., forecasData)[ , -1]
)


testData_forecastPUN <- list(y = forecasData$Punctuality,
                          X = model.matrix(Punctuality ~., forecasData)[ , -1]
)


######################################################
#~~~~~~~~~~~~~~~~2.1: Regression task~~~~~~~~~~~~~~~~#
######################################################

## Create unpenalized LR model 
model_unpLR <- lm(Decision.making ~., data = trainData)
summary_unpLR<- summary(model_unpLR)
print(summary_unpLR)

## Create ridge LR model
model_ridgeLR <- cv.glmnet(y = trainData_regRL$y,
                           x = trainData_regRL$X, 
                           alpha = 0, 
                           nfolds = 10)

summary_ridgeLR <- summary(model_ridgeLR)
print(summary_ridgeLR)

## Create LASSO LR model
model_lassoLR <- cv.glmnet(y = trainData_regRL$y, 
                           x = trainData_regRL$X, 
                           alpha = 1, 
                           nfolds = 10)

summary_lassoLR <- summary(model_lassoLR)
print(summary_lassoLR)

## Question 1
## Compute CVE for ridge LR model
model_ridgeLR$cvm
cve_ridgeLR <- with(model_ridgeLR , cvm[lambda == lambda.min])

## Compute CVE for LASSO LR model
model_lassoLR$cvm
cve_lassoLR <- with(model_lassoLR , cvm[lambda == lambda.min])

## Question 2
## Compute MSE for unp. LR model
predict_unpLR <- predict(model_unpLR , newdata = testData)
MSE_unpLR <- MSE(y_pred = predict_unpLR, 
                 y_true = testData$Decision.making)

## Compute MSE for ridge LR model
predict_ridgeLR <- predict(model_ridgeLR, 
                           newx = testData_regRL$X, 
                           s = 'lambda.min')

MSE_ridgeLR <- MSE(y_pred = predict_ridgeLR,
                   y_true = testData_regRL$y)


## Compute MSE for LASSO LR model
predict_lassoLR <- predict(model_lassoLR, 
                           newx = testData_regRL$X,
                           s = 'lambda.min')

MSE_lassoLR <- MSE(y_pred = predict_lassoLR,
                   y_true = testData_regRL$y)



## Print and compare MSE
print(paste("MSE Unpenalized Linear Regression:", MSE_unpLR))
print(paste("MSE Ridge Linear Regression:", MSE_ridgeLR))
print(paste("MSE Lasso Linear Regression:", MSE_lassoLR))

## Compare models
MSE_comparison <- c(MSE_unpLR, MSE_ridgeLR, MSE_lassoLR)
names(MSE_comparison) <- c("Unpenalized Linear Regression",
                           "Ridge Linear Regression", 
                           "Lasso Linear Regression")

paste0("Best Model: ", 
       names(MSE_comparison)[which.min(MSE_comparison)], 
       ", ",
       "MSE: ", min(MSE_comparison))

## Print and compare CVE
print(paste("CVE Ridge Linear Regression:", cve_ridgeLR))
print(paste("CVE Lasso Linear Regression:", cve_lassoLR))

cve_comparison_LR <- c(cve_ridgeLR, cve_lassoLR)
names(cve_comparison_LR) <- c("Ridge Linear Regression",
                              "Lasso Linear Regression")

paste0("Best Model: ", 
       names(cve_comparison_LR)[which.min(cve_comparison_LR)], 
       ", ",
       "CVE LR: ", min(cve_comparison_LR))

######################################################
#~~~~~~~~~~~~~~2.2: Classification Task~~~~~~~~~~~~~~#
######################################################
      
## Create multinomial logistic regression model
model_unpmnLogR <- multinom(Punctuality ~ ., data = trainData)
summary_unpmnLogR <- summary(model_unpmnLogR)
summary_unpmnLogR 

## Create ridge logistic regression model
model_RmnLogR <- cv.glmnet(y = trainData_classRL$y, 
                           x = trainData_classRL$X, 
                           family = "multinomial", 
                           alpha = 0, 
                           nfolds = 10)

## Create LASSO logistic regression model
model_LmnLogR <- cv.glmnet(y = trainData_classRL$y, 
                           x = trainData_classRL$X, 
                           family = "multinomial", 
                           alpha = 1, 
                           nfolds = 10)


## Question 4
## Compute CVE for ridge logistic regression
model_RmnLogR$cvm
cve_RmnLogR <- with(model_RmnLogR, cvm[lambda == lambda.min])

## Compute CVE for LASSO logistic regression
model_LmnLogR$cvm
cve_LmnLogR <- with(model_LmnLogR, cvm[lambda == lambda.min])

## Question 5
## Compute CEE for unpenalized multinomial logistic regression
predict_unpLRclass <- predict(model_unpmnLogR, 
                              newdata = testData,
                              type = "probs")

cee_unpmnLogR <- MultiLogLoss(y_pred = predict_unpLRclass, 
                              y_true = testData$Punctuality)

## Compute CEE for ridge logistic regression
predict_RmnLogR  <- predict(model_RmnLogR, 
                            newx = testData_classRL$X, 
                            s = 'lambda.min', 
                            type = "response")


cee_RmnLogR <- MultiLogLoss(y_pred = predict_RmnLogR[ , , 1], 
                            y_true = testData_classRL$y)


## Compute CEE for LASSO logistic regression
predict_LmnLogR  <- predict(model_LmnLogR, 
                            newx = testData_classRL$X, 
                            s = 'lambda.min', 
                            type = "response")

cee_LmnLogR <- MultiLogLoss(y_pred = predict_LmnLogR[ , , 1], 
                            y_true = testData_classRL$y)


## Print and compare CEE
print(paste("CEE Unpenalized Linear Regression:", cee_unpmnLogR))
print(paste("CEE Ridge Linear Regression:", cee_RmnLogR))
print(paste("CEE Lasso Linear Regression:", cee_LmnLogR))

cee_comparison <- c(cee_unpmnLogR, cee_RmnLogR, cee_LmnLogR)
names(cee_comparison) <- c("Unpenalized Linear Regression",
                           "Ridge Linear Regression", 
                           "Lasso Linear Regression")

paste0("Best Model: ", 
       names(cee_comparison)[which.min(cee_comparison)], 
       ", ",
       "CEE: ", min(cve_comparison_LR))

## Print and compare CVE
print(paste("CVE Ridge Linear Regression:", cve_RmnLogR))
print(paste("CVE Lasso Linear Regression:", cve_LmnLogR))

cve_comparison_Log <- c(cve_RmnLogR, cve_LmnLogR)
names(cve_comparison_Log) <- c('Ridge-penalized multinomial log. regression',
                              'LASSO-penalized multinomial log. regression')


paste0("Best Multinom. Log Model: ", 
       names(cve_comparison_Log)[which.min(cve_comparison_Log)], 
       ", ",
       "CVE: ", min(cve_comparison_Log))

######################################################
#~~~~~~~~~~~~~~3.1: Forecasting Task~~~~~~~~~~~~~~~~~#
######################################################

## Predict response for N=10 on decision making
forecast_DM <- predict(model_lassoLR, 
                        newx = testData_forecastDM$X,  
                        s = 'lambda.min', 
                        type = "response" )
forecast_DM

## Question 7 
## Predict class for N=10 on punctuality
forecast_PUN <- predict(model_RmnLogR, 
                       newx = testData_forecastPUN$X,  
                       s = 'lambda.min', 
                       type = "class" )
forecast_PUN

## Question 8 
## Calculating class membership probabilities 
forecast_PUNprobs <- predict(model_RmnLogR, 
                             newx = testData_forecastPUN$X, 
                             type = "response")
forecast_PUNprobs

## Question 9
## Draw table contrasting predict v. true values for punctuality
table_predtrue <- table(pred = forecast_PUN, true=testData_forecastPUN$y)

## Question 10 - Multinomial ridge log. regression probabilities of class membership  
predict_RmnLogR

######################################################
#~~~~~~~~~~~~~~~~~~~Miscellaneous~~~~~~~~~~~~~~~~~~~~#
######################################################
#Display MSE
MSE_comparison

#Display CEE
cee_comparison

#Plot MSE per model
plot_MSE <- barplot(MSE_comparison, 
                    main = "Mean Squared Errors of Linear Regression Models", 
                    ylab = "MSE", 
                      xlab = "Model",
                    col = "red",
                    border = "black",
                    space = 0.7)

text(x = plot_MSE, y = MSE_comparison, 
     label = MSE_comparison, 
     pos = 3, 
     cex = 1.1, col = "black")        

#Plot CVE per model
plot_CEE <- barplot(cee_comparison, 
                    main = "Cross-Entropy Errors of Linear Regression Models", 
                    ylab = "CEE", 
                    xlab = "Model",
                    col = "blue",
                    border = "black",
                    space = 0.7)

text(x = plot_CEE, y = cee_comparison, 
     label = cee_comparison, 
     pos = 3, 
     cex = 1.1, col = "black")
