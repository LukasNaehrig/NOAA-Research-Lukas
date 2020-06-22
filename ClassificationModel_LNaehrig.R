## Lukas Naehrig
## 6/19/2020
## Caret, Neural Network, Support Vector Machine
## ADA Boost


## Questions: Need indeces?


setwd("C:/Users/lnaeh/Desktop/NOAA Research")
library(dplyr)
library(caret)
library(doParallel)
library(neuralnet)
library(fastAdaboost)
library(e1071)


## Loading data:
geoClass <- read.csv('geo_data.csv')
geoClass <-
  geoClass %>%
  rename('Classification' = 'Classname')

head(geoClass)
tail(geoClass)
n <- nrow(geoClass)
set.seed(123)
s <- sample(n, 0.9 * n, replace = F)


## -- Splitting data into train and test: -- #

#set.seed(998)
#inTraining <- createDataPartition(geoClass$Classname, p = .9, list = FALSE)
#training <- Sonar[ inTraining,]
#testing  <- Sonar[-inTraining,]

training <- geoClass[s,]
testing  <- geoClass[-s,]

## -- Section 1: Neural Network -- #

# set.seed(825)
nnetModel <- train(Classification ~ ., data = training, 
                 method = 'nnet', 
                 #trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
nnetModel
accuracy1 <- predict(nnetModel, newdata = training)
accuracy2 <- predict(nnetModel, newdata = testing)

## Paramter Tuning Option:
# fitControl <- trainControl(## 10-fold CV
#   method = "repeatedcv",
#   number = 10,
#   ## repeated ten times
#   repeats = 10)
# 
# set.seed(825)
# gbmFit1 <- train(Class ~ ., data = training, 
#                  method = "gbm", 
#                  trControl = fitControl,
#                  ## This last option is actually one
#                  ## for gbm() that passes through
#                  verbose = FALSE)
# gbmFit1
# 




# -- Section 2: Support Vector Machine -- #
svmModel <- train(Classification ~ ., data = training, 
                   method = '', 
                   #trControl = fitControl,
                   ## This last option is actually one
                   ## for gbm() that passes through
                   verbose = FALSE)
svmModel
accuracy3 <- predict(svmModel, newdata = training)
accuracy4 <- predict(svmModel, newdata = testing)

# -- Section 3: ADA Boost -- # 
adaModel <- train(Classification ~ ., data = training, 
                   method = 'adaboost', 
                   #trControl = fitControl,
                   ## This last option is actually one
                   ## for gbm() that passes through
                   verbose = FALSE)
adaModel
accuracy5 <- predict(adaModel, newdata = training)
accuracy6 <- predict(adaModel, newdata = testing)







####### Different attempts:

nnMod <- neuralnet(Classification ~ ., training, hidden = 3 , linear.output = T )
nnMod


#dat = data.frame(Classification, y = as.factor(y))
svmfit = svm(Classification ~ ., data = training, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)

